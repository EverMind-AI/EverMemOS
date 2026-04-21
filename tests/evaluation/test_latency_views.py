"""Layer-1 latency views + BenchmarkContext / LatencyRecorder."""
import asyncio
import time

import pytest

from evaluation.src.core.benchmark_context import (
    AttemptRecord,
    CallRecord,
    LatencyRecorder,
    NULL_RECORDER,
    OUTCOME_HTTP_5XX,
    OUTCOME_SUCCESS,
    OUTCOME_TIMEOUT,
)
from evaluation.src.metrics.latency_views import (
    aggregate_all,
    aggregate_stage,
    records_to_jsonl,
)


# ---------- LatencyRecorder behaviour ----------


def test_recorder_synthesizes_success_attempt_when_adapter_silent():
    """Adapters that don't call record_attempt still get one attempt
    equal to wall_ms. Otherwise Layer-1 views break for any adapter
    that has not yet been migrated."""
    recorder = LatencyRecorder()

    async def run():
        async with recorder.measure("search", "q1") as _ctx:
            await asyncio.sleep(0.01)

    asyncio.run(run())
    assert len(recorder.records) == 1
    rec = recorder.records[0]
    assert rec.op == "search" and rec.unit_id == "q1"
    assert not rec.failed and not rec.fallback
    assert len(rec.attempts) == 1
    assert rec.attempts[0].outcome == OUTCOME_SUCCESS
    assert rec.attempts[0].duration_ms == pytest.approx(rec.wall_ms, rel=0.01)


def test_recorder_marks_failed_when_body_raises():
    """Wall time is still recorded on exceptions; outcome downgraded."""
    recorder = LatencyRecorder()

    async def run():
        with pytest.raises(RuntimeError):
            async with recorder.measure("answer", "q2") as _ctx:
                await asyncio.sleep(0.005)
                raise RuntimeError("boom")

    asyncio.run(run())
    assert len(recorder.records) == 1
    rec = recorder.records[0]
    assert rec.failed is True
    assert rec.wall_ms > 0
    assert rec.attempts[0].outcome != OUTCOME_SUCCESS


def test_recorder_preserves_adapter_attempts():
    """If the adapter/stage calls record_attempt explicitly, the
    recorder stores exactly those attempts and does NOT synthesize a
    fake one on top."""
    recorder = LatencyRecorder()

    async def run():
        async with recorder.measure("search", "q3") as ctx:
            ctx.record_attempt(1, 120.0, OUTCOME_HTTP_5XX, wait_ms_before_next=1000)
            ctx.record_attempt(2, 110.0, OUTCOME_SUCCESS)

    asyncio.run(run())
    rec = recorder.records[0]
    assert [a.attempt_n for a in rec.attempts] == [1, 2]
    assert [a.outcome for a in rec.attempts] == [OUTCOME_HTTP_5XX, OUTCOME_SUCCESS]
    assert rec.attempts[0].wait_ms_before_next == 1000.0


def test_null_recorder_is_no_op():
    """NULL_RECORDER is usable in tests / legacy code paths without
    producing CallRecord state."""

    async def run():
        async with NULL_RECORDER.measure("add", "c0") as ctx:
            assert ctx.op == "add"

    asyncio.run(run())
    assert NULL_RECORDER.records == []


def test_subphase_record_accumulates():
    recorder = LatencyRecorder()

    async def run():
        async with recorder.measure("search", "q4") as ctx:
            ctx.record_subphase("bm25", 10.0)
            ctx.record_subphase("rerank", 200.0)
            ctx.record_subphase("bm25", 5.0)  # accumulate, not overwrite

    asyncio.run(run())
    assert recorder.records[0].subphase_ms == {"bm25": 15.0, "rerank": 200.0}


def test_fallback_flag_propagates():
    recorder = LatencyRecorder()

    async def run():
        async with recorder.measure("search", "q5") as ctx:
            ctx.record_fallback()

    asyncio.run(run())
    assert recorder.records[0].fallback is True


# ---------- Four-view aggregation ----------


def _mk(op, uid, wall_ms, attempts, *, failed=False, fallback=False):
    """Direct CallRecord construction for view tests."""
    return CallRecord(
        op=op,
        unit_id=uid,
        wall_ms=wall_ms,
        attempts=[AttemptRecord(**a) for a in attempts],
        failed=failed,
        fallback=fallback,
    )


def test_aggregate_stage_returns_none_for_empty():
    assert aggregate_stage([], "search") is None


def test_aggregate_stage_four_views_distinct():
    """A realistic dataset with a mix of success, retries, failures.
    Exercises all four views to confirm they partition the data the
    way docs/latency-alignment.md specifies."""
    records = [
        # A — single success
        _mk("search", "qA", 100.0, [
            {"attempt_n": 1, "duration_ms": 100.0, "outcome": OUTCOME_SUCCESS},
        ]),
        # B — retry success (3 attempts total, 2 failed then 1 success)
        _mk("search", "qB", 5000.0, [
            {"attempt_n": 1, "duration_ms": 500.0, "outcome": OUTCOME_HTTP_5XX,
             "wait_ms_before_next": 1000.0},
            {"attempt_n": 2, "duration_ms": 500.0, "outcome": OUTCOME_HTTP_5XX,
             "wait_ms_before_next": 2000.0},
            {"attempt_n": 3, "duration_ms": 1000.0, "outcome": OUTCOME_SUCCESS},
        ]),
        # C — single failure
        _mk("search", "qC", 200.0, [
            {"attempt_n": 1, "duration_ms": 200.0, "outcome": OUTCOME_TIMEOUT},
        ], failed=True),
    ]

    out = aggregate_stage(records, "search")
    assert out["n_calls"] == 3

    # realistic: all three wall times
    assert out["wall_ms"]["realistic"]["n"] == 3
    assert out["wall_ms"]["realistic"]["max"] == 5000.0

    # clean: only A qualifies (B has retries, C failed)
    assert out["wall_ms"]["clean"]["n"] == 1
    assert out["wall_ms"]["clean"]["mean"] == 100.0

    # first_attempt: one per call
    assert out["wall_ms"]["first_attempt"]["n"] == 3
    # durations are 100 / 500 / 200
    assert out["wall_ms"]["first_attempt"]["max"] == 500.0

    # successful_attempt: A (100), B final (1000) — C contributes nothing
    assert out["wall_ms"]["successful_attempt"]["n"] == 2
    assert out["wall_ms"]["successful_attempt"]["mean"] == 550.0

    # Reliability
    rel = out["reliability"]
    assert rel["retry_rate"] == pytest.approx(1 / 3)   # only B retried
    assert rel["failed_rate"] == pytest.approx(1 / 3)  # only C
    assert rel["fallback_rate"] == 0.0
    # http_5xx appears twice (B attempt 1+2), timeout once (C)
    assert rel["retry_by_class"] == {OUTCOME_HTTP_5XX: 2, OUTCOME_TIMEOUT: 1}


def test_aggregate_all_derives_e2e_when_both_stages_present():
    """e2e_query_ms = search_wall + answer_wall per unit, derived
    without re-timing. Only appears when search and answer share unit_ids."""
    records = [
        _mk("search", "qA", 100.0, [
            {"attempt_n": 1, "duration_ms": 100.0, "outcome": OUTCOME_SUCCESS},
        ]),
        _mk("search", "qB", 300.0, [
            {"attempt_n": 1, "duration_ms": 300.0, "outcome": OUTCOME_SUCCESS},
        ]),
        _mk("answer", "qA", 800.0, [
            {"attempt_n": 1, "duration_ms": 800.0, "outcome": OUTCOME_SUCCESS},
        ]),
        _mk("answer", "qB", 1200.0, [
            {"attempt_n": 1, "duration_ms": 1200.0, "outcome": OUTCOME_SUCCESS},
        ]),
    ]
    out = aggregate_all(records)
    assert "search" in out and "answer" in out
    assert "e2e_query_ms" in out
    e2e = out["e2e_query_ms"]
    assert e2e["n_calls"] == 2
    # qA: 100+800=900, qB: 300+1200=1500 → mean 1200
    assert e2e["wall_ms"]["realistic"]["mean"] == 1200.0


def test_aggregate_all_skips_e2e_when_only_one_stage_present():
    records = [
        _mk("search", "qA", 100.0, [
            {"attempt_n": 1, "duration_ms": 100.0, "outcome": OUTCOME_SUCCESS},
        ]),
    ]
    out = aggregate_all(records)
    assert "search" in out
    assert "e2e_query_ms" not in out


def test_records_to_jsonl_roundtrippable_shape():
    records = [
        _mk("add", "c0", 1500.0, [
            {"attempt_n": 1, "duration_ms": 1500.0, "outcome": OUTCOME_SUCCESS},
        ]),
    ]
    rows = records_to_jsonl(records)
    assert rows == [
        {
            "op": "add",
            "unit_id": "c0",
            "wall_ms": 1500.0,
            "failed": False,
            "fallback": False,
            "subphase_ms": {},
            "attempts": [
                {
                    "attempt_n": 1,
                    "duration_ms": 1500.0,
                    "outcome": OUTCOME_SUCCESS,
                    "wait_ms_before_next": 0.0,
                }
            ],
        }
    ]


def test_bool_excluded_from_percentiles():
    """bool <: int in Python; a mis-typed duration must not slip in."""
    records = [
        _mk("search", "qA", True, [  # type: ignore[arg-type]
            {"attempt_n": 1, "duration_ms": 100.0, "outcome": OUTCOME_SUCCESS},
        ]),
        _mk("search", "qB", 200.0, [
            {"attempt_n": 1, "duration_ms": 200.0, "outcome": OUTCOME_SUCCESS},
        ]),
    ]
    out = aggregate_stage(records, "search")
    # realistic excludes True; only 200.0 remains
    assert out["wall_ms"]["realistic"]["n"] == 1
    assert out["wall_ms"]["realistic"]["mean"] == 200.0
