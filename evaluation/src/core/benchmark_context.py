"""
Pipeline-owned latency measurement context.

This module is the Phase-1 deliverable of the latency-alignment plan
(docs/latency-alignment.md). It moves responsibility for when a timer
starts and stops from the adapter into the pipeline layer, so every
memory-system adapter - current and future - contributes to a single
canonical ``wall_ms`` measurement at the adapter boundary regardless
of what it does internally.

Three pieces:

* ``AttemptRecord`` — one execution attempt of a single adapter call.
  Every retry is a separate attempt.
* ``CallRecord`` — the full log of one adapter boundary call, including
  the harness-measured ``wall_ms`` and zero-or-more attempts.
* ``BenchmarkContext`` — per-call handle passed into stage code.
  Harness and (Phase 2+) adapters use it to record attempts, sub-phase
  timings, and fallback signals without needing direct access to the
  recorder.
* ``LatencyRecorder`` — pipeline-owned collection of CallRecord across
  all stages. ``recorder.measure(op, unit_id)`` is the async context
  manager that wraps a single adapter call.

Adapters that never call ``record_attempt`` still get exactly one
synthesized attempt whose duration equals ``wall_ms``; Layer-1 views
therefore work from day one without requiring adapter changes.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional


# Closed enum for Phase 2. Kept here so both harness and adapters import
# from one place and so latency_views.py can bucket retries by class.
# Any value not in this set is accepted but reported under "other".
OUTCOME_SUCCESS = "success"
OUTCOME_HTTP_5XX = "http_5xx"
OUTCOME_HTTP_429 = "http_429"
OUTCOME_TIMEOUT = "timeout"
OUTCOME_UPSTREAM_UNAVAILABLE = "upstream_unavailable"
OUTCOME_INVALID_RESPONSE = "invalid_response"
OUTCOME_QUOTA_EXCEEDED = "quota_exceeded"
OUTCOME_FAILED_OTHER = "failed_other"


@dataclass(frozen=True)
class AttemptRecord:
    """One execution attempt. A call without retries has exactly one."""

    attempt_n: int                 # 1-based
    duration_ms: float             # attempt's own wall time, excluding backoff wait
    outcome: str                   # one of OUTCOME_* constants above
    wait_ms_before_next: float = 0.0  # backoff/throttle delay between this and the next attempt


@dataclass
class CallRecord:
    """Full log of a single adapter boundary call.

    Four-view derivations in ``latency_views.py`` consume this shape:

    * ``wall_ms`` — harness measurement, always present.
    * ``attempts`` — non-empty. If the adapter didn't record anything,
      recorder.measure() synthesizes a single success/failure attempt
      whose duration == wall_ms.
    * ``failed`` / ``fallback`` — Boolean call-level signals.
    * ``subphase_ms`` — Layer-2 diagnostic, adapter-specific, optional.
    """

    op: str                        # "add" | "search" | "answer"
    unit_id: str                   # conv_id (add) or question_id (search/answer)
    wall_ms: float
    attempts: List[AttemptRecord] = field(default_factory=list)
    failed: bool = False
    fallback: bool = False
    subphase_ms: Dict[str, float] = field(default_factory=dict)


class BenchmarkContext:
    """Per-call handle. Callers record attempts / subphases / fallback.

    Design: the harness creates one BenchmarkContext per adapter call
    via ``LatencyRecorder.measure()``. The context is intentionally
    cheap (no locks, no async); pipeline stages already run one adapter
    call per task so there is no contention inside a single context.
    """

    def __init__(
        self,
        op: str,
        unit_id: str,
        retry_policy: str = "realistic",
        deadline_ms: Optional[float] = None,
    ):
        self.op = op
        self.unit_id = unit_id
        self.retry_policy = retry_policy
        self.deadline_ms = deadline_ms
        self._attempts: List[AttemptRecord] = []
        self._subphases: Dict[str, float] = {}
        self._fallback = False

    def record_attempt(
        self,
        attempt_n: int,
        duration_ms: float,
        outcome: str,
        wait_ms_before_next: float = 0.0,
    ) -> None:
        self._attempts.append(
            AttemptRecord(
                attempt_n=attempt_n,
                duration_ms=float(duration_ms),
                outcome=str(outcome),
                wait_ms_before_next=float(wait_ms_before_next),
            )
        )

    def record_subphase(self, name: str, duration_ms: float) -> None:
        """Accumulate per-subphase duration; names are adapter-specific."""
        self._subphases[name] = self._subphases.get(name, 0.0) + float(duration_ms)

    def record_fallback(self) -> None:
        self._fallback = True

    # Convenience for pipeline-level code that already tracks wall time
    # and just needs to log a single attempt of known duration.
    def record_success_once(self, duration_ms: float) -> None:
        self.record_attempt(
            attempt_n=len(self._attempts) + 1,
            duration_ms=duration_ms,
            outcome=OUTCOME_SUCCESS,
        )


class LatencyRecorder:
    """Pipeline-owned collection of CallRecord across stages.

    Usage from a stage:

        async with recorder.measure("search", qa.question_id) as ctx:
            result = await adapter.search(...)

    After the pipeline run completes, ``recorder.records`` is the
    authoritative dataset. ``latency_views.aggregate_stage(records, op)``
    derives Layer-1 four-view distributions + reliability signals.
    """

    def __init__(
        self,
        retry_policy: str = "realistic",
        deadline_ms: Optional[float] = None,
    ):
        self.retry_policy = retry_policy
        self.deadline_ms = deadline_ms
        self.records: List[CallRecord] = []

    @asynccontextmanager
    async def measure(self, op: str, unit_id: str) -> AsyncIterator[BenchmarkContext]:
        ctx = BenchmarkContext(
            op=op,
            unit_id=unit_id,
            retry_policy=self.retry_policy,
            deadline_ms=self.deadline_ms,
        )
        t0 = time.perf_counter()
        failed = False
        try:
            yield ctx
        except BaseException:
            failed = True
            raise
        finally:
            wall_ms = (time.perf_counter() - t0) * 1000.0
            attempts = list(ctx._attempts)
            if not attempts:
                # Synthesize a single attempt so Layer-1 views are
                # computable even for adapters that do not yet adopt
                # the record_attempt() callback. duration_ms == wall_ms
                # because there's no way to distinguish sub-attempt
                # breakdown without adapter cooperation.
                outcome = OUTCOME_FAILED_OTHER if failed else OUTCOME_SUCCESS
                attempts = [
                    AttemptRecord(
                        attempt_n=1,
                        duration_ms=wall_ms,
                        outcome=outcome,
                        wait_ms_before_next=0.0,
                    )
                ]
            self.records.append(
                CallRecord(
                    op=op,
                    unit_id=unit_id,
                    wall_ms=wall_ms,
                    attempts=attempts,
                    failed=failed,
                    fallback=ctx._fallback,
                    subphase_ms=dict(ctx._subphases),
                )
            )


# Module-level no-op recorder for tests / callers that don't want
# measurement. Using .measure() still yields a valid BenchmarkContext
# so adapter code can be written against BenchmarkContext
# unconditionally once Phase 2 lands.
class _NullRecorder(LatencyRecorder):
    @asynccontextmanager
    async def measure(self, op: str, unit_id: str) -> AsyncIterator[BenchmarkContext]:
        yield BenchmarkContext(op=op, unit_id=unit_id, retry_policy=self.retry_policy)


NULL_RECORDER = _NullRecorder()
