"""Shared percentile / distribution-summary helpers.

Used by ``diagnostics`` (adapter-reported pre-Phase-1 fields) and
``latency_views`` (Layer-1 canonical wall_ms). They were duplicated
across both modules; this is the single source of truth.
"""
from __future__ import annotations

from statistics import mean
from typing import Iterable, List, Optional


def percentile(sorted_values: List[float], pct: float) -> float:
    """Linear-interp percentile. ``sorted_values`` must already be sorted."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    k = (len(sorted_values) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = k - lo
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac)


def summarize(values: Iterable[float | None]) -> Optional[dict]:
    """Return ``{n, mean, p50, p95, max}`` or ``None`` when no samples.

    bool is a subclass of int in Python; exclude it explicitly so a
    field accidentally set to True/False cannot contribute 1.0/0.0
    datapoints to a latency distribution.
    """
    nums = [
        float(v)
        for v in values
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    if not nums:
        return None
    nums.sort()
    return {
        "n": len(nums),
        "mean": float(mean(nums)),
        "p50": percentile(nums, 50),
        "p95": percentile(nums, 95),
        "max": float(nums[-1]),
    }
