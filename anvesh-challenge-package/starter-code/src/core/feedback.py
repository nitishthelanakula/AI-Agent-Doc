from __future__ import annotations

from collections import OrderedDict
from typing import Any, Optional

from src.core.online_eval import fail_verdicts, recent_history

_feedback: OrderedDict[str, dict[str, Any]] = OrderedDict()


def record_feedback(investigation_id: str, rating: int, note: str) -> dict[str, Any]:
    """Record manual feedback for an investigation. A thumbs-down rating is treated as a fail verdict."""
    normalized_rating = int(rating)
    payload = {
        "investigation_id": investigation_id,
        "type": "manual",
        "rating": normalized_rating,
        "note": note,
        "fail_verdict": normalized_rating <= 2,
        "source": "user",
    }
    _feedback[investigation_id] = payload
    return dict(payload)


def annotation_queue() -> list[dict[str, Any]]:
    """Return merged manual feedback and sampled low-scoring online-eval results, deduplicated by investigation_id."""
    merged: OrderedDict[str, dict[str, Any]] = OrderedDict()

    for entry in list(_feedback.values()):
        investigation_id = str(entry.get("investigation_id") or "")
        if investigation_id:
            merged[investigation_id] = {
                **entry,
                "type": "manual",
                "source": "user",
            }

    for item in fail_verdicts():
        investigation_id = str(item.get("investigation_id") or "")
        if not investigation_id:
            continue
        existing = merged.get(investigation_id)
        if existing is None:
            merged[investigation_id] = {
                "investigation_id": investigation_id,
                "type": "online_eval",
                "score": item.get("score"),
                "reason": item.get("reason"),
                "low_agreement": item.get("low_agreement", False),
                "source": "online_eval",
                "fail_verdict": True,
            }
        else:
            existing["source"] = "merged"
            existing["fail_verdict"] = True
            existing.setdefault("reason", item.get("reason"))
            existing.setdefault("score", item.get("score"))
            existing.setdefault("low_agreement", item.get("low_agreement", False))

    return list(merged.values())
