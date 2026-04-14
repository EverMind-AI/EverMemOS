"""
Compose answer-level / retrieval-level / diagnostics into one summary dict
for benchmark_summary.json.

Schema is stable so the report generator + external dashboards don't break
between runs; missing sources degrade to 0.0 rather than raising.
"""
from __future__ import annotations

from typing import Any, Optional


def _g(source: Optional[dict], key: str, default: Any = 0.0) -> Any:
    if not source:
        return default
    value = source.get(key)
    return default if value is None else value


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
            "accuracy": _g(eval_result, "accuracy"),
            "f1_mean": _g(answer_aux_metrics, "f1_mean"),
            "bleu1_mean": _g(answer_aux_metrics, "bleu1_mean"),
        },
        "retrieval_level": {
            f"evidence_hit_at_{k}": _g(retrieval_metrics, "evidence_hit_at_k_mean"),
            f"evidence_recall_at_{k}": _g(retrieval_metrics, "evidence_recall_at_k_mean"),
            "mrr": _g(retrieval_metrics, "mrr_mean"),
            f"ndcg_at_{k}": _g(retrieval_metrics, "ndcg_at_k_mean"),
        },
        "diagnostics": {
            "add_latency_ms_mean": _g(diagnostics, "add_latency_ms_mean"),
            "time_to_visible_ms_mean": _g(diagnostics, "time_to_visible_ms_mean"),
            "retrieval_latency_ms_mean": _g(diagnostics, "retrieval_latency_ms_mean"),
            "answer_latency_ms_mean": _g(diagnostics, "answer_latency_ms_mean"),
            "final_context_tokens_mean": _g(diagnostics, "final_context_tokens_mean"),
        },
    }
