from __future__ import annotations

import os
from collections import deque
from typing import Any, Sequence

from src.core.drift import observe as observe_drift, status as drift_status
from src.core.online_eval import recent_history, rolling_average_score
from src.core.otel_setup import get_tracer

tracer = get_tracer("src.core.alerting")

MAX_ALERTS = 100
COST_BASELINE_WINDOW = int(os.getenv("ALERT_COST_BASELINE_WINDOW", "20"))
COST_SPIKE_MULTIPLIER = float(os.getenv("ALERT_COST_SPIKE_MULTIPLIER", "2.0"))
LOW_SCORE_THRESHOLD = float(os.getenv("ALERT_LOW_SCORE_THRESHOLD", "3.0"))

_alerts: list[dict[str, Any]] = []
_cost_history: deque[float] = deque(maxlen=COST_BASELINE_WINDOW)


def get_alerts() -> list[dict[str, Any]]:
    return list(_alerts)


def _append_alert(kind: str, message: str, **details: Any) -> dict[str, Any]:
    alert = {
        "kind": kind,
        "message": message,
        "details": details,
    }
    _alerts.append(alert)
    if len(_alerts) > MAX_ALERTS:
        del _alerts[:-MAX_ALERTS]
    return alert


def _model_mismatch(requested_model: str, served_model: str) -> bool:
    return bool(requested_model) and bool(served_model) and requested_model != served_model


def _cost_spike(cost_usd: float) -> tuple[bool, str, float | None]:
    _cost_history.append(float(cost_usd))
    if len(_cost_history) < 2:
        return False, "cost baseline not yet established", None

    baseline = sum(_cost_history) / len(_cost_history)
    if baseline <= 0:
        return False, "baseline cost is zero", baseline

    spike_factor = float(cost_usd) / baseline
    if spike_factor > COST_SPIKE_MULTIPLIER:
        return True, f"cost spike: {cost_usd:.4f} USD vs {baseline:.4f} USD baseline", baseline
    return False, "cost within baseline", baseline


def _low_online_eval_score() -> tuple[bool, str, float]:
    avg = rolling_average_score()
    if avg < LOW_SCORE_THRESHOLD:
        return True, f"online eval score below threshold: {avg:.2f} < {LOW_SCORE_THRESHOLD:.2f}", avg
    return False, f"online eval score within threshold: {avg:.2f}", avg


def observe_and_alert(
    requested_model: str,
    served_model: str,
    cost_usd: float,
    sar_narrative: str,
    tool_sequence: Sequence[str] | None,
) -> list[dict[str, Any]]:
    """Check model drift, cost, evaluation score, and narrative drift and emit alerts when triggered."""
    fired: list[dict[str, Any]] = []

    with tracer.start_as_current_span("alerting") as span:
        span.set_attribute("requested_model", requested_model)
        span.set_attribute("served_model", served_model)
        span.set_attribute("cost_usd", float(cost_usd))
        span.set_attribute("sar_present", bool(sar_narrative))

        if _model_mismatch(requested_model, served_model):
            alert = _append_alert(
                "model_mismatch",
                f"requested model {requested_model!r} differed from served model {served_model!r}",
                requested_model=requested_model,
                served_model=served_model,
            )
            fired.append(alert)
            span.set_attribute("alert.model_mismatch", True)

        cost_alert, cost_message, baseline = _cost_spike(cost_usd)
        if cost_alert:
            alert = _append_alert(
                "cost_spike",
                cost_message,
                cost_usd=float(cost_usd),
                baseline_cost=baseline,
            )
            fired.append(alert)
            span.set_attribute("alert.cost_spike", True)

        low_score, low_score_message, avg_score = _low_online_eval_score()
        if low_score:
            alert = _append_alert(
                "low_online_eval_score",
                low_score_message,
                rolling_average_score=avg_score,
                threshold=LOW_SCORE_THRESHOLD,
            )
            fired.append(alert)
            span.set_attribute("alert.low_online_eval_score", True)

        drift_state = None
        if sar_narrative:
            drift_state = observe_drift(sar_narrative, list(tool_sequence or []))
            drift = drift_status()
            if drift.get("drift_alert"):
                alert = _append_alert(
                    "narrative_drift",
                    "SAR narrative drift exceeds the configured threshold",
                    rolling_mean_cosine_distance=drift.get("rolling_mean_cosine_distance"),
                    total_variation_distance=drift.get("total_variation_distance"),
                    drift_details=drift_state,
                )
                fired.append(alert)
                span.set_attribute("alert.narrative_drift", True)

        span.set_attribute("alert_count", len(fired))

    return fired
