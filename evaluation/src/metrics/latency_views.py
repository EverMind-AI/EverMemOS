"""
Layer-1 latency views: derive four distributions + reliability signals
from a LatencyRecorder's CallRecord log.

This is the canonical cross-adapter latency report. All four views are
computed at the adapter-boundary ``wall_ms`` layer, which is harness-
measured and identical across every adapter. See
docs/latency-alignment.md for the contract and
evaluation/src/core/benchmark_context.py for the data model.

Views:

* ``realistic``       — wall_ms of every call. User-perceived latency.
* ``clean``           — wall_ms of calls with attempts==1, not failed,
                        not fallback. "Nothing went wrong" latency.
* ``first_attempt``   — duration_ms of attempts[0] across every call.
                        Core first-attempt cost, retries stripped.
* ``successful_attempt`` — duration_ms of each attempt whose outcome
                           is ``success``. Cost of a successful attempt
                           regardless of prior failures.

Reliability signals: retry_rate, fallback_rate, failed_rate, plus
retry_by_class bucketing retried attempts by outcome (http_5xx / 429
/ timeout / ...).
"""
from __future__ import annotations

from collections import Counter
from typing import List, Optional

from evaluation.src.core.benchmark_context import (
    CallRecord,
    OUTCOME_SUCCESS,
)
from evaluation.src.metrics.distributions import summarize as _percentiles


def aggregate_stage(records: List[CallRecord], op: str) -> Optional[dict]:
    """Layer-1 summary for one stage (op == 'add' | 'search' | 'answer').

    Returns ``None`` when no call records exist for that op. Otherwise
    returns ``{"wall_ms": {view: percentiles, ...}, "reliability": {...}}``.
    """
    filtered = [r for r in records if r.op == op]
    if not filtered:
        return None

    realistic = [r.wall_ms for r in filtered]
    clean = [
        r.wall_ms
        for r in filtered
        if len(r.attempts) == 1 and not r.failed and not r.fallback
    ]
    first_attempt = [r.attempts[0].duration_ms for r in filtered if r.attempts]
    successful_attempt = [
        a.duration_ms
        for r in filtered
        for a in r.attempts
        if a.outcome == OUTCOME_SUCCESS
    ]

    n = len(filtered)
    retry_rate = sum(1 for r in filtered if len(r.attempts) > 1) / n
    fallback_rate = sum(1 for r in filtered if r.fallback) / n
    failed_rate = sum(1 for r in filtered if r.failed) / n

    retry_by_class: Counter[str] = Counter()
    for r in filtered:
        for a in r.attempts:
            if a.outcome != OUTCOME_SUCCESS:
                retry_by_class[a.outcome] += 1

    return {
        "n_calls": n,
        "wall_ms": {
            "realistic": _percentiles(realistic),
            "clean": _percentiles(clean),
            "first_attempt": _percentiles(first_attempt),
            "successful_attempt": _percentiles(successful_attempt),
        },
        "reliability": {
            "retry_rate": retry_rate,
            "fallback_rate": fallback_rate,
            "failed_rate": failed_rate,
            "retry_by_class": dict(retry_by_class),
        },
    }


def aggregate_all(records: List[CallRecord]) -> dict:
    """Run aggregate_stage for every stage that has at least one record.

    Returns ``{"add": {...}, "search": {...}, "answer": {...}}``. Stages
    with zero records are omitted; ``e2e_query_ms`` composes search+
    answer per-unit when both are present.
    """
    ops = sorted({r.op for r in records})
    # Walrus avoids calling aggregate_stage twice per op (once for the
    # filter guard, once to populate the value).
    out: dict = {op: s for op in ops if (s := aggregate_stage(records, op))}

    # Derive per-question end-to-end (search + answer sum) when we have
    # matched unit_ids on both sides. This is purely a function of
    # Layer-1 measurements; we never re-time.
    search_by_uid = {r.unit_id: r.wall_ms for r in records if r.op == "search"}
    answer_by_uid = {r.unit_id: r.wall_ms for r in records if r.op == "answer"}
    shared = sorted(set(search_by_uid).intersection(answer_by_uid))
    if shared:
        e2e = [search_by_uid[u] + answer_by_uid[u] for u in shared]
        out["e2e_query_ms"] = {
            "n_calls": len(shared),
            "wall_ms": {
                "realistic": _percentiles(e2e),
            },
        }

    return out


def records_to_jsonl(records: List[CallRecord]) -> List[dict]:
    """Flatten CallRecord list into JSON-serializable rows for disk."""
    rows: List[dict] = []
    for r in records:
        rows.append(
            {
                "op": r.op,
                "unit_id": r.unit_id,
                "wall_ms": r.wall_ms,
                "failed": r.failed,
                "fallback": r.fallback,
                "subphase_ms": r.subphase_ms,
                "attempts": [
                    {
                        "attempt_n": a.attempt_n,
                        "duration_ms": a.duration_ms,
                        "outcome": a.outcome,
                        "wait_ms_before_next": a.wait_ms_before_next,
                    }
                    for a in r.attempts
                ],
            }
        )
    return rows
