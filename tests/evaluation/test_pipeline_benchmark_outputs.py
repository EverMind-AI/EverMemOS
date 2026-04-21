"""
Task 7: pipeline writes benchmark-specific artifact files.

We don't spin up a full Pipeline here - that would need LLMProvider + real
evaluator + adapter etc. Instead we exercise the write-side helpers added
to Pipeline (``_write_retrieval_metrics_artifact`` and friends) directly,
which is what pipeline.py now calls at each stage boundary.
"""
import json
from pathlib import Path

from evaluation.src.core.data_models import (
    AnswerResult,
    Dataset,
    EvaluationResult,
    QAPair,
    SearchResult,
)
from evaluation.src.core.pipeline import Pipeline
from evaluation.src.metrics.answer_aux_metrics import build_answer_aux_metrics


def test_build_answer_aux_metrics_basic():
    ar = [
        AnswerResult(
            question_id="q1",
            question="",
            answer="Chicago",
            golden_answer="She moved from Chicago.",
            conversation_id="c0",
        ),
        AnswerResult(
            question_id="q2",
            question="",
            answer="unknown",
            golden_answer="Seattle",
            conversation_id="c0",
        ),
    ]
    metrics = build_answer_aux_metrics(ar)
    assert metrics["count"] == 2
    assert 0.0 <= metrics["f1_mean"] <= 1.0
    assert 0.0 <= metrics["bleu1_mean"] <= 1.0
    assert metrics["per_question"][0]["question_id"] == "q1"


def _make_stub_pipeline(tmp_path):
    """Construct a Pipeline without touching LLMProvider or any real deps."""
    # Skip __init__ since it instantiates LLMProvider etc; seed only the
    # attributes the write helpers actually read.
    p = Pipeline.__new__(Pipeline)
    p.output_dir = Path(tmp_path)
    from evaluation.src.utils.saver import ResultSaver

    p.saver = ResultSaver(p.output_dir)
    p.adapter = None
    p.checkpoint = None
    import logging

    p.logger = logging.getLogger("stub_pipeline")
    # Phase 1: stub out the LatencyRecorder so helpers that write
    # latency_views.json don't AttributeError on Pipeline.__new__-style
    # fixtures.
    from evaluation.src.core.benchmark_context import LatencyRecorder

    p.latency_recorder = LatencyRecorder()
    p.retry_policy = "realistic"
    p.deadline_ms = None
    return p


def test_pipeline_writes_retrieval_metrics(tmp_path):
    p = _make_stub_pipeline(tmp_path)

    qas = [
        QAPair(
            question_id="q1",
            question="where?",
            answer="Chicago",
            evidence=["D3:11"],
            metadata={"conversation_id": "c0"},
        )
    ]
    srs = [
        SearchResult(
            query="where?",
            conversation_id="c0",
            results=[{"metadata": {"source_sessions": ["S3"]}}],
            retrieval_metadata={"retrieval_latency_ms": 10.0},
        )
    ]

    p._write_retrieval_metrics_artifact(qas, srs, k=5)

    path = Path(tmp_path) / "retrieval_metrics.json"
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["evidence_hit_at_k_mean"] == 1.0


def test_pipeline_writes_answer_aux_metrics(tmp_path):
    p = _make_stub_pipeline(tmp_path)

    ars = [
        AnswerResult(
            question_id="q1",
            question="",
            answer="Chicago",
            golden_answer="Chicago",
            conversation_id="c0",
        )
    ]
    p._write_answer_aux_metrics_artifact(ars)

    path = Path(tmp_path) / "answer_aux_metrics.json"
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["f1_mean"] == 1.0


def test_pipeline_writes_diagnostics_and_summary(tmp_path):
    p = _make_stub_pipeline(tmp_path)

    srs = [
        SearchResult(
            "q",
            "c0",
            [{}],
            {"retrieval_latency_ms": 10.0, "retrieval_route": "search_only",
             "backend_mode": "fts_only"},
        )
    ]
    ars = [
        AnswerResult(
            question_id="q1",
            question="",
            answer="ok",
            golden_answer="ok",
            conversation_id="c0",
            metadata={"answer_latency_ms": 50.0, "final_context_tokens": 100},
        )
    ]
    eval_result = EvaluationResult(
        total_questions=1, correct=1, accuracy=0.5, detailed_results=[]
    )
    retrieval_metrics = {"evidence_hit_at_k_mean": 0.5, "evidence_recall_at_k_mean": 0.5,
                         "mrr_mean": 0.5, "ndcg_at_k_mean": 0.5, "per_question": []}
    answer_aux_metrics = {"f1_mean": 0.8, "bleu1_mean": 0.3, "count": 1, "per_question": []}

    p._write_diagnostics_and_summary_artifact(
        system_name="openclaw",
        dataset_name="locomo",
        search_results=srs,
        answer_results=ars,
        eval_result=eval_result,
        retrieval_metrics=retrieval_metrics,
        answer_aux_metrics=answer_aux_metrics,
        index=None,
        k=5,
    )

    diag_path = Path(tmp_path) / "diagnostics.json"
    summary_path = Path(tmp_path) / "benchmark_summary.json"
    assert diag_path.exists()
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text())
    assert summary["system"] == "openclaw"
    assert summary["dataset"] == "locomo"
    assert summary["answer_level"]["f1_mean"] == 0.8
    # Phase 6 moved session-level metrics under adapter_specific_retrieval.
    assert summary["adapter_specific_retrieval"]["evidence_hit_at_5"] == 0.5
    # Phase 1 wrote latency_views.json; the stub has no recorder events,
    # so it should serialize as an empty dict but exist on disk.
    assert (Path(tmp_path) / "latency_views.json").exists()
    assert (Path(tmp_path) / "latency_records.json").exists()
