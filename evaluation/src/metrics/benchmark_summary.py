"""
Compose answer-level / retrieval-level / diagnostics into one summary dict
for benchmark_summary.json.

Missing upstream metrics are preserved as ``None`` (not coerced to 0.0)
so reports can distinguish "the system scored zero" from "this source
didn't run / never emitted the field". The report generator in
pipeline.py renders None as "n/a".

Retrieval-level contract:
- ``content_overlap_at_{k}`` is the canonical cross-adapter retrieval
  quality scalar. Every adapter that returns text in
  ``SearchResult.results[*].content`` contributes to it with zero extra
  integration work.
- Session-level metrics (``evidence_hit_at_{k}`` / ``evidence_recall``
  / ``mrr`` / ``ndcg_at_{k}``) move under ``adapter_specific_retrieval``
  because they rely on adapter-emitted ``source_sessions`` and
  dataset-specific evidence-id projection; they are kept for diagnostic
  use but are NOT comparable across adapters.
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
    content_overlap: Optional[dict] = None,
    latency_views: Optional[dict] = None,
    retry_policy: str = "realistic",
    latency_invariants: Optional[dict] = None,
) -> dict:
    return {
        "system": system,
        "dataset": dataset,
        "retry_policy": retry_policy,
        "answer_level": {
            "accuracy": _get(eval_result, "accuracy"),
            "f1_mean": _get(answer_aux_metrics, "f1_mean"),
            "bleu1_mean": _get(answer_aux_metrics, "bleu1_mean"),
        },
        "retrieval_level": {
            # Canonical — present for any adapter.
            f"content_overlap_at_{k}": _get(
                content_overlap, "content_overlap_at_k_mean"
            ),
            f"content_overlap_precision_at_{k}": _get(
                content_overlap, "content_overlap_precision_mean"
            ),
            f"content_overlap_recall_at_{k}": _get(
                content_overlap, "content_overlap_recall_mean"
            ),
        },
        "adapter_specific_retrieval": {
            # Diagnostic only, not cross-adapter-comparable.
            # Populated when the adapter emits source_sessions.
            f"evidence_hit_at_{k}": _get(retrieval_metrics, "evidence_hit_at_k_mean"),
            f"evidence_recall_at_{k}": _get(
                retrieval_metrics, "evidence_recall_at_k_mean"
            ),
            "mrr": _get(retrieval_metrics, "mrr_mean"),
            f"ndcg_at_{k}": _get(retrieval_metrics, "ndcg_at_k_mean"),
        },
        "diagnostics": {
            # Legacy scalar means — kept for backward compat with reports
            # that only print single numbers. p50/p95/max live in *_stats.
            # Layer-1 canonical view lives in ``latency`` below; these
            # fields are pre-alignment adapter-reported values and should
            # NOT be used for cross-adapter comparison.
            "add_latency_ms_mean": _get(diagnostics, "add_latency_ms_mean"),
            "retrieval_latency_ms_mean": _get(diagnostics, "retrieval_latency_ms_mean"),
            "answer_latency_ms_mean": _get(diagnostics, "answer_latency_ms_mean"),
            "final_context_tokens_mean": _get(diagnostics, "final_context_tokens_mean"),
            "add_latency_ms_stats": _get(diagnostics, "add_latency_ms_stats"),
            "retrieval_latency_ms_stats": _get(diagnostics, "retrieval_latency_ms_stats"),
            "answer_latency_ms_stats": _get(diagnostics, "answer_latency_ms_stats"),
            "final_context_tokens_stats": _get(diagnostics, "final_context_tokens_stats"),
            "add_retry_rate": _get(diagnostics, "add_retry_rate"),
            "add_fallback_rate": _get(diagnostics, "add_fallback_rate"),
            "add_failed_rate": _get(diagnostics, "add_failed_rate"),
            "add_samples": _get(diagnostics, "add_samples"),
        },
        # Layer-1 canonical latency — harness-measured at the adapter
        # boundary, always comparable across adapters. See
        # docs/latency-alignment.md. Each stage (add/search/answer)
        # carries four views (realistic/clean/first_attempt/
        # successful_attempt) plus reliability signals.
        "latency": latency_views or {},
        # Phase 3 self-check on the Layer-1 data. count > 0 means one
        # of the alignment invariants (wall ≈ Σ attempts, N equals
        # work units, strict_no_retry respected, subphases within wall)
        # was violated; full details live in latency_invariants.json.
        "latency_invariants": latency_invariants
        or {"count": 0, "by_code": {}, "by_severity": {"error": 0, "warning": 0}, "items": []},
    }
