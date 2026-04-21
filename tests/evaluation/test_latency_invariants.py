"""Phase 3 invariant checks on Layer-1 latency data."""
from evaluation.src.core.benchmark_context import (
    AttemptRecord,
    CallRecord,
    OUTCOME_HTTP_5XX,
    OUTCOME_SUCCESS,
)
from evaluation.src.metrics.latency_invariants import (
    check_all,
    summarize_violations,
)


def _mk(op, uid, wall_ms, attempts, *, subphases=None, failed=False, fallback=False):
    return CallRecord(
        op=op,
        unit_id=uid,
        wall_ms=wall_ms,
        attempts=[AttemptRecord(**a) for a in attempts],
        subphase_ms=subphases or {},
        failed=failed,
        fallback=fallback,
    )


# --- wall_mismatch ---


def test_wall_matches_attempts_passes_within_tolerance():
    """wall_ms == sum(attempt_durations + waits), perfectly aligned."""
    r = _mk("search", "q1", 500.0, [
        {"attempt_n": 1, "duration_ms": 200.0, "outcome": OUTCOME_HTTP_5XX,
         "wait_ms_before_next": 100.0},
        {"attempt_n": 2, "duration_ms": 200.0, "outcome": OUTCOME_SUCCESS},
    ])
    out = check_all([r], n_conversations=0, n_qa_pairs=1, retry_policy="realistic")
    assert not [v for v in out if v.code == "wall_mismatch"]


def test_wall_mismatch_flags_big_gap():
    """wall_ms much bigger than reported attempts → adapter missed attempts."""
    r = _mk("search", "q1", 1000.0, [
        {"attempt_n": 1, "duration_ms": 100.0, "outcome": OUTCOME_SUCCESS},
    ])
    out = check_all([r], n_conversations=0, n_qa_pairs=1, retry_policy="realistic")
    mismatches = [v for v in out if v.code == "wall_mismatch"]
    assert len(mismatches) == 1
    assert mismatches[0].severity == "warning"


# --- work_unit ---


def test_work_unit_counts_accept_both_batched_and_per_conv_add():
    """add may be batched (1 record) or per-conv (N records); either ok."""
    batched = _mk("add", "all", 100.0, [
        {"attempt_n": 1, "duration_ms": 100.0, "outcome": OUTCOME_SUCCESS},
    ])
    out = check_all([batched], n_conversations=10, n_qa_pairs=0,
                    retry_policy="realistic")
    assert not [v for v in out if v.code == "work_unit"]

    per_conv = [
        _mk("add", f"c{i}", 10.0, [
            {"attempt_n": 1, "duration_ms": 10.0, "outcome": OUTCOME_SUCCESS},
        ])
        for i in range(10)
    ]
    out = check_all(per_conv, n_conversations=10, n_qa_pairs=0,
                    retry_policy="realistic")
    assert not [v for v in out if v.code == "work_unit"]


def test_work_unit_errors_on_mismatched_search_count():
    """search count must equal len(qa). Anything else is an error."""
    srs = [
        _mk("search", "q1", 10.0, [
            {"attempt_n": 1, "duration_ms": 10.0, "outcome": OUTCOME_SUCCESS},
        ]),
    ]
    out = check_all(srs, n_conversations=0, n_qa_pairs=3, retry_policy="realistic")
    errs = [v for v in out if v.code == "work_unit" and v.severity == "error"]
    assert len(errs) == 1
    assert "search produced 1" in errs[0].message


# --- strict_policy ---


def test_strict_no_retry_violation_caught():
    """retry_policy=strict_no_retry but call has 2 attempts → error."""
    r = _mk("search", "q1", 200.0, [
        {"attempt_n": 1, "duration_ms": 100.0, "outcome": OUTCOME_HTTP_5XX,
         "wait_ms_before_next": 0.0},
        {"attempt_n": 2, "duration_ms": 100.0, "outcome": OUTCOME_SUCCESS},
    ])
    out = check_all([r], n_conversations=0, n_qa_pairs=1,
                    retry_policy="strict_no_retry")
    strict = [v for v in out if v.code == "strict_policy"]
    assert len(strict) == 1
    assert strict[0].severity == "error"


def test_strict_no_retry_happy_path():
    r = _mk("search", "q1", 100.0, [
        {"attempt_n": 1, "duration_ms": 100.0, "outcome": OUTCOME_SUCCESS},
    ])
    out = check_all([r], n_conversations=0, n_qa_pairs=1,
                    retry_policy="strict_no_retry")
    assert not [v for v in out if v.code == "strict_policy"]


# --- subphase_over_wall ---


def test_subphase_over_wall_flags_doublecounting():
    """Reported subphases sum higher than wall → self-inconsistent."""
    r = _mk("search", "q1", 100.0, [
        {"attempt_n": 1, "duration_ms": 100.0, "outcome": OUTCOME_SUCCESS},
    ], subphases={"bm25": 80.0, "rerank": 90.0})  # 170 > 100
    out = check_all([r], n_conversations=0, n_qa_pairs=1, retry_policy="realistic")
    violations = [v for v in out if v.code == "subphase_over_wall"]
    assert len(violations) == 1
    assert violations[0].severity == "warning"


def test_subphase_under_wall_ok():
    r = _mk("search", "q1", 100.0, [
        {"attempt_n": 1, "duration_ms": 100.0, "outcome": OUTCOME_SUCCESS},
    ], subphases={"bm25": 20.0, "rerank": 40.0})  # 60 < 100
    out = check_all([r], n_conversations=0, n_qa_pairs=1, retry_policy="realistic")
    assert not [v for v in out if v.code == "subphase_over_wall"]


# --- summarize_violations ---


def test_summarize_violations_groups_by_code_and_severity():
    r_strict = _mk("search", "q1", 200.0, [
        {"attempt_n": 1, "duration_ms": 100.0, "outcome": OUTCOME_HTTP_5XX,
         "wait_ms_before_next": 0.0},
        {"attempt_n": 2, "duration_ms": 100.0, "outcome": OUTCOME_SUCCESS},
    ])
    r_mismatch = _mk("search", "q2", 1000.0, [
        {"attempt_n": 1, "duration_ms": 50.0, "outcome": OUTCOME_SUCCESS},
    ])
    # strict_no_retry policy — the q1 case is a violation there,
    # plus the wall-mismatch on q2.
    out = check_all([r_strict, r_mismatch], n_conversations=0, n_qa_pairs=2,
                    retry_policy="strict_no_retry")
    summary = summarize_violations(out)
    assert summary["count"] == 2
    assert summary["by_code"]["strict_policy"] == 1
    assert summary["by_code"]["wall_mismatch"] == 1
    assert summary["by_severity"]["error"] == 1
    assert summary["by_severity"]["warning"] == 1
