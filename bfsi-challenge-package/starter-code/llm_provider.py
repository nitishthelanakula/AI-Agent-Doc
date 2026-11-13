"""OpenAI-only chat wrapper. THIS FILE HAS NO OBSERVABILITY OF ANY KIND -
no spans, and `usage` below is read from the API response and then
immediately discarded. The token counts are right there in `response.usage`
on every single call; nothing converts them into a cost or records them
anywhere. That's Step 5 of the outline: add pricing.py, and both a
lab.cost_usd span attribute AND a proper cost metric - once Step 1 has
given you a tracer and a meter to record them with."""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from opentelemetry import metrics, trace
from opentelemetry.metrics import get_meter

from pricing import calculate_cost_usd

ENV_FILE = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_FILE)

DEFAULT_MODEL = "gpt-4o"
_client: Optional[OpenAI] = None

meter = get_meter("bfsi-assistant.llm")
llm_cost_usd = meter.create_histogram(
    name="lab.llm_cost_usd",
    description="USD cost of OpenAI API calls",
    unit="USD",
)


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"OPENAI_API_KEY is not set. Check your .env file at: {ENV_FILE}"
            )
        _client = OpenAI(api_key=api_key)
    return _client


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class AgentTurn:
    content: Optional[str]
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    raw: object = None


def chat_raw(messages, tools=None, model: str = DEFAULT_MODEL,
             temperature: Optional[float] = None) -> AgentTurn:
    client = _get_client()

    kwargs = {"model": model, "messages": messages}
    if tools:
        kwargs["tools"] = tools
    if temperature is not None:
        kwargs["temperature"] = temperature

    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]

    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    cost_usd = calculate_cost_usd(prompt_tokens, completion_tokens, model=model)

    current_span = trace.get_current_span()
    if current_span.is_recording():
        current_span.set_attribute("lab.cost_usd", cost_usd)
        current_span.set_attribute("llm.model", model)
        current_span.set_attribute("llm.prompt_tokens", prompt_tokens)
        current_span.set_attribute("llm.completion_tokens", completion_tokens)

    llm_cost_usd.record(cost_usd, attributes={"llm.model": model})

    # response.usage.prompt_tokens / completion_tokens exist right here, on
    # every call - and nothing in this file does anything with them. Look
    # for this comment again once you're on Step 5.

    tool_calls = []
    for tc in (choice.message.tool_calls or []):
        tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments))

    return AgentTurn(
        content=choice.message.content,
        tool_calls=tool_calls,
        finish_reason=choice.finish_reason,
        raw=response,
    )
