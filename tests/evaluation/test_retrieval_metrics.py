"""
Task 6: session-level retrieval metrics + diagnostics + summary.
"""
from evaluation.src.core.data_models import QAPair, SearchResult
from evaluation.src.metrics.retrieval_metrics import (
    evaluate_retrieval_for_question,
    evaluate_retrieval_metrics,
    normalize_gold_sessions,
)
from evaluation.src.metrics.diagnostics import aggregate_diagnostics
from evaluation.src.metrics.benchmark_summary import build_benchmark_summary


def test_normalize_gold_sessions_accepts_both_formats():
    assert normalize_gold_sessions(["D3:11", "D3:12"]) == ["S3"]
    assert normalize_gold_sessions(["S1", "S3"]) == ["S1", "S3"]
    assert normalize_gold_sessions(["D1:0", "S3", "D3:5"]) == ["S1", "S3"]


def test_normalize_gold_sessions_fails_on_unknown_format():
    import pytest

    with pytest.raises(ValueError):
        normalize_gold_sessions(["session_1"])


def test_retrieval_metrics_use_session_level_projection():
    qa = QAPair(
        question_id="q1",
        question="",
        answer="",
        evidence=["D3:11", "D3:12"],
        metadata={"conversation_id": "c0"},
    )
    sr = SearchResult(
        query="",
        conversation_id="c0",
        results=[
            {"metadata": {"source_sessions": ["S1"]}},
            {"metadata": {"source_sessions": ["S3"]}},
        ],
        retrieval_metadata={},
    )
    metrics = evaluate_retrieval_for_question(qa, sr, k=2)
    assert metrics["evidence_hit_at_k"] == 1.0
    assert metrics["mrr"] == 0.5
    assert metrics["evidence_recall_at_k"] == 1.0  # 1/1 gold session covered


def test_retrieval_metrics_zero_when_no_match():
    qa = QAPair(
        question_id="q2",
        question="",
        answer="",
        evidence=["D3:11"],
        metadata={"conversation_id": "c0"},
    )
    sr = SearchResult(
        query="",
        conversation_id="c0",
        results=[
            {"metadata": {"source_sessions": ["S1"]}},
            {"metadata": {"source_sessions": ["S2"]}},
        ],
        retrieval_metadata={},
    )
    metrics = evaluate_retrieval_for_question(qa, sr, k=2)
    assert metrics["evidence_hit_at_k"] == 0.0
    assert metrics["mrr"] == 0.0
    assert metrics["evidence_recall_at_k"] == 0.0


def test_evaluate_retrieval_metrics_batch_aggregates_mean():
    qas = [
        QAPair(question_id="q1", question="", answer="", evidence=["D0:0"],
               metadata={"conversation_id": "c0"}),
        QAPair(question_id="q2", question="", answer="", evidence=["D1:0"],
               metadata={"conversation_id": "c0"}),
    ]
    srs = [
        SearchResult("", "c0", [{"metadata": {"source_sessions": ["S0"]}}], {}),
        SearchResult("", "c0", [{"metadata": {"source_sessions": ["S9"]}}], {}),
    ]
    metrics = evaluate_retrieval_metrics(qas, srs, k=1)
    assert metrics["per_question"][0]["evidence_hit_at_k"] == 1.0
    assert metrics["per_question"][1]["evidence_hit_at_k"] == 0.0
    assert metrics["evidence_hit_at_k_mean"] == 0.5
    assert metrics["evidence_recall_at_k_mean"] == 0.5


def test_aggregate_diagnostics_tolerates_missing_fields():
    # Emulate a non-openclaw adapter that doesn't populate latency metadata.
    search_results = [
        SearchResult("q", "c0", [{}], {"retrieval_latency_ms": 10.0,
                                         "retrieval_route": "search_only",
                                         "backend_mode": "fts_only"}),
        SearchResult("q", "c0", [], {}),  # empty retrieval
    ]
    answer_results_meta = [
        {"answer_latency_ms": 50.0, "final_context_tokens": 100},
        {"answer_latency_ms": None, "final_context_tokens": 0},
    ]
    diag = aggregate_diagnostics(search_results, answer_results_meta)
    assert diag["retrieval_latency_ms_mean"] == 10.0  # only one valid
    assert diag["answer_latency_ms_mean"] == 50.0
    assert diag["empty_retrieval_rate"] == 0.5
    assert diag["final_context_tokens_mean"] == 50.0
    assert diag["retrieval_route_distribution"] == {"search_only": 1}
    assert diag["backend_mode_distribution"] == {"fts_only": 1}


def test_build_benchmark_summary_shape():
    eval_result = {"accuracy": 0.6}
    retrieval_metrics = {
        "evidence_hit_at_k_mean": 0.5,
        "evidence_recall_at_k_mean": 0.5,
        "mrr_mean": 0.5,
        "ndcg_at_k_mean": 0.5,
        "per_question": [],
    }
    answer_aux_metrics = {
        "f1_mean": 0.4,
        "bleu1_mean": 0.2,
    }
    diagnostics = {
        "add_latency_ms_mean": 1000.0,
        "time_to_visible_ms_mean": 500.0,
        "retrieval_latency_ms_mean": 30.0,
        "answer_latency_ms_mean": 120.0,
        "final_context_tokens_mean": 100.0,
    }
    summary = build_benchmark_summary(
        system="openclaw", dataset="locomo",
        eval_result=eval_result,
        retrieval_metrics=retrieval_metrics,
        answer_aux_metrics=answer_aux_metrics,
        diagnostics=diagnostics,
        k=5,
    )
    assert summary["system"] == "openclaw"
    assert summary["dataset"] == "locomo"
    assert summary["answer_level"]["accuracy"] == 0.6
    assert summary["answer_level"]["f1_mean"] == 0.4
    assert summary["answer_level"]["bleu1_mean"] == 0.2
    assert summary["retrieval_level"]["evidence_hit_at_5"] == 0.5
    assert summary["retrieval_level"]["mrr"] == 0.5
    assert summary["diagnostics"]["retrieval_latency_ms_mean"] == 30.0
