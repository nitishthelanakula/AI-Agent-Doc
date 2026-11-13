from __future__ import annotations

import math
import os
from collections import Counter
from typing import Iterable, Sequence

from src.core.llm_provider import _get_client

BASELINE_SIZE = max(1, int(os.getenv("DRIFT_BASELINE_SIZE", "10")))
ROLLING_WINDOW = max(1, int(os.getenv("DRIFT_ROLLING_WINDOW", "20")))
EMBEDDING_MODEL = os.getenv("DRIFT_EMBEDDING_MODEL", "text-embedding-3-small")

_narratives: list[str] = []
_embeddings: list[list[float]] = []
_tool_history: list[Counter[str]] = []
_recent_cosine_distances: list[float] = []
_baseline_centroid: list[float] | None = None


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 1.0
    na = _norm(a)
    nb = _norm(b)
    if na == 0.0 or nb == 0.0:
        return 1.0
    similarity = _dot(a, b) / (na * nb)
    return max(0.0, 1.0 - float(similarity))


def _mean_vector(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    total = [0.0 for _ in range(dim)]
    for vector in vectors:
        if len(vector) != dim:
            continue
        for idx, value in enumerate(vector):
            total[idx] += value
    count = float(len(vectors))
    return [value / count for value in total]


def _embed_text(text: str) -> list[float]:
    client = _get_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def _total_variation_distance(left: Counter[str], right: Counter[str]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    total_left = sum(left.values())
    total_right = sum(right.values())
    if total_left == 0 and total_right == 0:
        return 0.0
    left_distribution = {k: left.get(k, 0) / total_left if total_left else 0.0 for k in keys}
    right_distribution = {k: right.get(k, 0) / total_right if total_right else 0.0 for k in keys}
    return 0.5 * sum(abs(left_distribution[k] - right_distribution[k]) for k in keys)


def observe(narrative: str, tool_sequence: Sequence[str] | None) -> dict:
    """Observe a new SAR narrative and its tool sequence for drift monitoring."""
    normalized_narrative = narrative or ""
    observed_tools = list(tool_sequence or [])

    embedding = _embed_text(normalized_narrative)
    _narratives.append(normalized_narrative)
    _embeddings.append(embedding)
    _tool_history.append(Counter(observed_tools))

    baseline_history = _tool_history[:BASELINE_SIZE]
    baseline_counter = Counter()
    for entry in baseline_history:
        baseline_counter.update(entry)

    rolling_start = max(0, len(_tool_history) - ROLLING_WINDOW)
    rolling_counter = Counter()
    for entry in _tool_history[rolling_start:]:
        rolling_counter.update(entry)

    total_variation_distance = _total_variation_distance(baseline_counter, rolling_counter)
    current_cosine_distance = None

    global _baseline_centroid
    if len(_embeddings) >= BASELINE_SIZE:
        if _baseline_centroid is None:
            _baseline_centroid = _mean_vector(_embeddings[:BASELINE_SIZE])
        current_cosine_distance = _cosine_distance(embedding, _baseline_centroid)
        _recent_cosine_distances.append(current_cosine_distance)

    result = {
        "narrative_count": len(_narratives),
        "baseline_size": BASELINE_SIZE,
        "baseline_centroid_ready": _baseline_centroid is not None,
        "current_cosine_distance": current_cosine_distance,
        "rolling_cosine_distances": _recent_cosine_distances[-ROLLING_WINDOW:],
        "rolling_mean_cosine_distance": (
            sum(_recent_cosine_distances[-ROLLING_WINDOW:]) / len(_recent_cosine_distances[-ROLLING_WINDOW:])
            if _recent_cosine_distances[-ROLLING_WINDOW:]
            else 0.0
        ),
        "baseline_tool_usage": dict(baseline_counter),
        "rolling_tool_usage": dict(rolling_counter),
        "total_variation_distance": total_variation_distance,
    }
    return result


def status() -> dict:
    """Return the current drift state for the observed investigations."""
    if not _narratives:
        return {
            "narrative_count": 0,
            "baseline_size": BASELINE_SIZE,
            "baseline_centroid_ready": False,
            "baseline_centroid": None,
            "recent_cosine_distances": [],
            "rolling_mean_cosine_distance": 0.0,
            "baseline_tool_usage": {},
            "rolling_tool_usage": {},
            "total_variation_distance": 0.0,
            "drift_alert": False,
        }

    baseline_counter = Counter()
    for entry in _tool_history[:BASELINE_SIZE]:
        baseline_counter.update(entry)

    rolling_start = max(0, len(_tool_history) - ROLLING_WINDOW)
    rolling_counter = Counter()
    for entry in _tool_history[rolling_start:]:
        rolling_counter.update(entry)

    total_variation_distance = _total_variation_distance(baseline_counter, rolling_counter)
    rolling_mean = (
        sum(_recent_cosine_distances[-ROLLING_WINDOW:]) / len(_recent_cosine_distances[-ROLLING_WINDOW:])
        if _recent_cosine_distances[-ROLLING_WINDOW:]
        else 0.0
    )
    drift_alert = rolling_mean > 0.25 or total_variation_distance > 0.35

    return {
        "narrative_count": len(_narratives),
        "baseline_size": BASELINE_SIZE,
        "baseline_centroid_ready": _baseline_centroid is not None,
        "baseline_centroid": _baseline_centroid,
        "recent_cosine_distances": _recent_cosine_distances[-ROLLING_WINDOW:],
        "rolling_mean_cosine_distance": rolling_mean,
        "baseline_tool_usage": dict(baseline_counter),
        "rolling_tool_usage": dict(rolling_counter),
        "total_variation_distance": total_variation_distance,
        "drift_alert": drift_alert,
    }
