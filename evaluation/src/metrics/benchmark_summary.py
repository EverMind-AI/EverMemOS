"""
Compose answer-level / retrieval-level / diagnostics into one summary dict
for benchmark_summary.json.

Missing upstream metrics are preserved as ``None`` (not coerced to 0.0)
so reports can distinguish "the system scored zero" from "this source
didn't run / never emitted the field". The report generator in
pipeline.py renders None as "n/a".
"""
from __future__ import annotations

from typing import Any, Optional


def _get(source: Optional[dict], key: str) -> Any:
    if not source:
        return None
    return source.get(key)


def build_benchmark_summary(
    *,
    system: str,
    dataset: str,
    eval_result: Optional[dict],
    retrieval_metrics: Optional[dict],
    answer_aux_metrics: Optional[dict],
    diagnostics: Optional[dict],
    k: int = 5,
) -> dict:
    return {
        "system": system,
        "dataset": dataset,
        "answer_level": {
            "accuracy": _get(eval_result, "accuracy"),
            "f1_mean": _get(answer_aux_metrics, "f1_mean"),
            "bleu1_mean": _get(answer_aux_metrics, "bleu1_mean"),
        },
        "retrieval_level": {
            f"evidence_hit_at_{k}": _get(retrieval_metrics, "evidence_hit_at_k_mean"),
            f"evidence_recall_at_{k}": _get(retrieval_metrics, "evidence_recall_at_k_mean"),
            "mrr": _get(retrieval_metrics, "mrr_mean"),
            f"ndcg_at_{k}": _get(retrieval_metrics, "ndcg_at_k_mean"),
        },
        "diagnostics": {
            # add_latency covers the full add() span: ingest markdown +
            # openclaw memory index --force + status settle wait. We do
            # NOT alias it as time_to_visible because the two are not
            # separately measured; a future change can split them if
            # needed. For now, the single field honestly described is
            # better than a fake dual-report.
            "add_latency_ms_mean": _get(diagnostics, "add_latency_ms_mean"),
            "retrieval_latency_ms_mean": _get(diagnostics, "retrieval_latency_ms_mean"),
            "answer_latency_ms_mean": _get(diagnostics, "answer_latency_ms_mean"),
            "final_context_tokens_mean": _get(diagnostics, "final_context_tokens_mean"),
        },
    }
