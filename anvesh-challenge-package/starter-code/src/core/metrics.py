from __future__ import annotations

from typing import Any

from src.core.otel_setup import get_meter

meter = get_meter("src.core.metrics")

request_counter = meter.create_counter(
    "anvesh.investigation.requests",
    description="Number of investigation runs started",
    unit="{request}",
)

duration_histogram = meter.create_histogram(
    "anvesh.investigation.duration",
    description="Investigation run duration in seconds",
    unit="s",
)

cost_histogram = meter.create_histogram(
    "anvesh.investigation.cost_usd",
    description="Total USD cost of an investigation run",
    unit="USD",
)

turn_counter = meter.create_counter(
    "anvesh.investigation.turns",
    description="Number of model turns in an investigation",
    unit="{turn}",
)

tool_failure_counter = meter.create_counter(
    "anvesh.tool.failures",
    description="Number of tool-call failures",
    unit="{failure}",
)


def record_investigation_turn(
    customer_id: str,
    *,
    turn_count: int,
    total_cost_usd: float,
    duration_seconds: float,
    model: str,
    tool_failures: int = 0,
    finish_reason: str | None = None,
    **_: Any,
) -> None:
    """Record a single finished investigation run's aggregate metrics."""
    request_counter.add(1, {"customer_id": customer_id, "model": model})
    duration_histogram.record(duration_seconds, {"customer_id": customer_id, "model": model})
    cost_histogram.record(total_cost_usd, {"customer_id": customer_id, "model": model})
    turn_counter.add(turn_count, {"customer_id": customer_id, "model": model})
    if tool_failures:
        tool_failure_counter.add(tool_failures, {"customer_id": customer_id, "model": model, "finish_reason": finish_reason or "unknown"})
