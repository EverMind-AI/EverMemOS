"""
Phase 3 invariant checks for Layer-1 latency data.

These make the alignment contract machine-verifiable. Violations are
returned as a list of ``InvariantViolation`` rows so the pipeline can
decide whether each one is fatal (abort the run) or just a warning
(surface in the report).

Contract (see docs/latency-alignment.md):

1. ``wall_ms ≈ Σ attempt.duration_ms + Σ wait_ms_before_next``.
   Gap >5% → the adapter under-reported attempts. warning.
2. ``N(add)`` == ``len(conversations)``, ``N(search) == N(answer) ==
   len(qa)``. Violation aborts the run: work-unit accounting is wrong.
3. If ``retry_policy == strict_no_retry``, every CallRecord must have
   exactly one attempt. Violation → adapter ignored the policy.
4. If sub-phases were recorded for a call, ``Σ subphase_ms <= wall_ms``.
   Violation → subphase decomposition is self-inconsistent. warning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from evaluation.src.core.benchmark_context import CallRecord


@dataclass(frozen=True)
class InvariantViolation:
    code: str                 # "wall_mismatch" | "work_unit" | "strict_policy" | "subphase_over_wall"
    severity: str             # "warning" | "error"
    op: str
    unit_id: str
    message: str


def _check_wall_matches_attempts(
    records: Sequence[CallRecord], tolerance: float = 0.05
) -> List[InvariantViolation]:
    """Wall time ≈ sum of attempt durations + inter-attempt waits.

    Anything above ``tolerance`` (default 5 %) is a warning: the
    adapter or stage didn't report every attempt. Synthesized
    single-attempt records (no adapter wiring yet) trivially satisfy
    this because duration == wall_ms by construction.
    """
    out: List[InvariantViolation] = []
    for r in records:
        if not r.attempts:
            continue
        expected = sum(a.duration_ms + a.wait_ms_before_next for a in r.attempts)
        if r.wall_ms <= 0:
            continue
        rel_gap = abs(r.wall_ms - expected) / r.wall_ms
        if rel_gap > tolerance:
            out.append(
                InvariantViolation(
                    code="wall_mismatch",
                    severity="warning",
                    op=r.op,
                    unit_id=r.unit_id,
                    message=(
                        f"wall_ms={r.wall_ms:.1f} but Σ(attempts+waits)="
                        f"{expected:.1f} (gap {rel_gap:.1%}); "
                        f"adapter may have missed record_attempt calls."
                    ),
                )
            )
    return out


def _check_work_unit_counts(
    records: Sequence[CallRecord],
    n_conversations: int,
    n_qa_pairs: int,
) -> List[InvariantViolation]:
    """Each stage must produce exactly the expected number of records.

    Phase 1's add_stage currently wraps the whole batch under unit_id
    "all", so n_add is expected to be 1 per pipeline run (not
    n_conversations). Phase 3 is where we un-batch; this invariant
    accommodates both shapes: exactly 1 (batched) or exactly
    n_conversations (per-conv) are acceptable.
    """
    counts: dict[str, int] = {"add": 0, "search": 0, "answer": 0}
    for r in records:
        if r.op in counts:
            counts[r.op] += 1

    out: List[InvariantViolation] = []

    if counts["add"] not in (0, 1, n_conversations):
        out.append(
            InvariantViolation(
                code="work_unit",
                severity="error",
                op="add",
                unit_id="*",
                message=(
                    f"add produced {counts['add']} records; expected 1 "
                    f"(batched) or {n_conversations} (per-conv)."
                ),
            )
        )

    if counts["search"] not in (0, n_qa_pairs):
        out.append(
            InvariantViolation(
                code="work_unit",
                severity="error",
                op="search",
                unit_id="*",
                message=(
                    f"search produced {counts['search']} records; "
                    f"expected {n_qa_pairs} (one per qa)."
                ),
            )
        )

    if counts["answer"] not in (0, n_qa_pairs):
        out.append(
            InvariantViolation(
                code="work_unit",
                severity="error",
                op="answer",
                unit_id="*",
                message=(
                    f"answer produced {counts['answer']} records; "
                    f"expected {n_qa_pairs} (one per qa)."
                ),
            )
        )

    return out


def _check_strict_policy(
    records: Sequence[CallRecord], retry_policy: str
) -> List[InvariantViolation]:
    """strict_no_retry demands every call has exactly one attempt."""
    if retry_policy != "strict_no_retry":
        return []
    out: List[InvariantViolation] = []
    for r in records:
        if len(r.attempts) > 1:
            out.append(
                InvariantViolation(
                    code="strict_policy",
                    severity="error",
                    op=r.op,
                    unit_id=r.unit_id,
                    message=(
                        f"retry_policy=strict_no_retry but observed "
                        f"{len(r.attempts)} attempts; adapter or stage "
                        f"ignored the policy."
                    ),
                )
            )
    return out


def _check_subphase_within_wall(
    records: Sequence[CallRecord], tolerance: float = 0.01
) -> List[InvariantViolation]:
    """Adapter-reported subphases can't sum to more than wall_ms.

    ``tolerance`` is small because subphases are inherently additive
    when run sequentially; a violation here means the adapter is
    double-counting or reporting overlapping parallel work without
    labelling it.
    """
    out: List[InvariantViolation] = []
    for r in records:
        if not r.subphase_ms:
            continue
        total = sum(r.subphase_ms.values())
        cap = r.wall_ms * (1 + tolerance)
        if total > cap:
            out.append(
                InvariantViolation(
                    code="subphase_over_wall",
                    severity="warning",
                    op=r.op,
                    unit_id=r.unit_id,
                    message=(
                        f"Σ subphase_ms={total:.1f} > wall_ms={r.wall_ms:.1f} "
                        f"(over by {total - r.wall_ms:.1f} ms); likely "
                        f"parallel work reported as sequential."
                    ),
                )
            )
    return out


def check_all(
    records: Sequence[CallRecord],
    *,
    n_conversations: int,
    n_qa_pairs: int,
    retry_policy: str,
) -> List[InvariantViolation]:
    """Run all Phase 3 invariants and return the flat violation list."""
    return (
        _check_wall_matches_attempts(records)
        + _check_work_unit_counts(records, n_conversations, n_qa_pairs)
        + _check_strict_policy(records, retry_policy)
        + _check_subphase_within_wall(records)
    )


def summarize_violations(
    violations: Sequence[InvariantViolation],
) -> dict:
    """Group violations by code/severity for JSON-friendly reporting."""
    by_code: dict[str, int] = {}
    by_severity: dict[str, int] = {"error": 0, "warning": 0}
    for v in violations:
        by_code[v.code] = by_code.get(v.code, 0) + 1
        if v.severity in by_severity:
            by_severity[v.severity] += 1
    return {
        "count": len(violations),
        "by_code": by_code,
        "by_severity": by_severity,
        "items": [
            {
                "code": v.code,
                "severity": v.severity,
                "op": v.op,
                "unit_id": v.unit_id,
                "message": v.message,
            }
            for v in violations
        ],
    }
