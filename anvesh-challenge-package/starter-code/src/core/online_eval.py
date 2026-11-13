import json
import os
import random
from typing import Any, Optional

from src.core.llm_provider import chat_raw
from src.core.otel_setup import get_tracer

tracer = get_tracer("src.core.online_eval")

MAX_HISTORY = 100
DEFAULT_SAMPLE_RATE = 0.1
recent_history: list[dict[str, Any]] = []


def _sample_rate() -> float:
    raw = os.getenv("ONLINE_EVAL_SAMPLE_RATE", str(DEFAULT_SAMPLE_RATE))
    try:
        sample_rate = float(raw)
    except ValueError:
        sample_rate = DEFAULT_SAMPLE_RATE
    return max(0.0, min(1.0, sample_rate))


def rolling_average_score(history: Optional[list[dict[str, Any]]] = None) -> float:
    values = history if history is not None else recent_history
    if not values:
        return 0.0

    scores = [float(item.get("score", 0)) for item in values if isinstance(item, dict)]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def fail_verdicts(history: Optional[list[dict[str, Any]]] = None) -> list[dict[str, Any]]:
    values = history if history is not None else recent_history
    return [
        item for item in values
        if isinstance(item, dict)
        and (bool(item.get("low_agreement")) or int(item.get("score", 5) or 5) <= 2)
    ]


def maybe_grade(gathered_evidence: dict[str, Any], sar_narrative: str, investigation_id: Optional[str] = None) -> Optional[dict[str, Any]]:
    """Sample a fraction of investigations and judge the SAR narrative against the evidence."""
    sample_rate = _sample_rate()
    if sample_rate <= 0.0 or random.random() > sample_rate:
        return None

    with tracer.start_as_current_span("online_eval") as span:
        span.set_attribute("sample_rate", sample_rate)
        span.set_attribute("evidence_present", bool(gathered_evidence))
        span.set_attribute("sar_narrative_present", bool(sar_narrative))
        if investigation_id is not None:
            span.set_attribute("investigation_id", investigation_id)

        judge_prompt = (
            "You are an impartial compliance judge. Compare the SAR narrative only against the provided evidence dict. "
            "Use the evidence dict as the source of truth. Return JSON only with keys: score, reason, low_agreement. "
            "score must be an integer from 1 to 5 where 5 means the SAR narrative is strongly supported by the evidence and 1 means it conflicts with or ignores the evidence. "
            "reason must be a single line explaining the score in <= 120 characters. low_agreement must be true only when the narrative omits material evidence, adds unsupported claims, or contradicts the evidence."
        )
        payload = {
            "evidence": gathered_evidence,
            "sar_narrative": sar_narrative,
        }

        result = chat_raw(
            messages=[
                {"role": "system", "content": judge_prompt},
                {"role": "user", "content": json.dumps(payload, indent=2, default=str)},
            ],
            model="gpt-4o-mini",
            temperature=0.0,
        )

        raw_text = (result.content or "").strip()
        json_text = raw_text.replace("```json", "").replace("```", "").strip()
        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError:
            parsed = {"score": 3, "reason": "Judge returned non-JSON output", "low_agreement": True}

        score = int(parsed.get("score", 3) or 3)
        score = max(1, min(5, score))
        reason = str(parsed.get("reason", "No reason provided")).strip() or "No reason provided"
        low_agreement = bool(parsed.get("low_agreement", score <= 2))

        entry = {
            "investigation_id": investigation_id,
            "score": score,
            "reason": reason,
            "low_agreement": low_agreement,
            "sample_rate": sample_rate,
        }
        recent_history.append(entry)
        if len(recent_history) > MAX_HISTORY:
            recent_history[:] = recent_history[-MAX_HISTORY:]

        span.set_attribute("online_eval.score", score)
        span.set_attribute("online_eval.low_agreement", bool(low_agreement))

        return entry
