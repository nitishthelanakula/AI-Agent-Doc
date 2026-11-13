"""Simple token pricing utilities for the LLM model used in this project."""

MODEL_TOKEN_COSTS = {
    "gpt-4o": {
        "input_per_1m_usd": 2.50,
        "output_per_1m_usd": 0,
    }
}


def calculate_cost_usd(prompt_tokens: int, completion_tokens: int, model: str = "gpt-4o") -> float:
    """Return the estimated USD cost for a model call using token prices per 1M tokens."""
    pricing = MODEL_TOKEN_COSTS.get(model, MODEL_TOKEN_COSTS["gpt-4o"])
    input_cost = (prompt_tokens / 1_000_000) * pricing["input_per_1m_usd"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output_per_1m_usd"]
    return round(input_cost + output_cost, 8)
