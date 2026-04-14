"""
Cross-stage diagnostics aggregator.

Reads per-question retrieval_metadata (from search_results) and per-question
metadata (from answer_results), plus optional per-conversation add_summary
files, and emits averages / distributions.

Every aggregation ignores None values so adapters that don't emit the field
(e.g. mem0/memos on latency) don't poison the mean.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Iterable, Optional


def _safe_mean(values: Iterable[float | None]) -> Optional[float]:
    valid = [v for v in values if isinstance(v, (int, float))]
    if not valid:
        return None
    return float(mean(valid))


def _distribution(values: Iterable[str | None]) -> dict[str, int]:
    return dict(Counter(v for v in values if isinstance(v, str) and v))


def aggregate_diagnostics(
    search_results, answer_results_metadata, index: Optional[dict] = None
) -> dict:
    """Pull time-series across stages into a single summary dict."""
    retrieval_latencies = [
        sr.retrieval_metadata.get("retrieval_latency_ms") for sr in search_results
    ]
    scheduler_waits = [
        sr.retrieval_metadata.get("scheduler_wait_ms") for sr in search_results
    ]
    routes = [sr.retrieval_metadata.get("retrieval_route") for sr in search_results]
    backends = [sr.retrieval_metadata.get("backend_mode") for sr in search_results]

    empty_hits = sum(1 for sr in search_results if not sr.results)
    empty_rate = empty_hits / len(search_results) if search_results else 0.0

    answer_latencies = [m.get("answer_latency_ms") for m in answer_results_metadata]
    context_tokens = [m.get("final_context_tokens") for m in answer_results_metadata]
    context_chars = [m.get("final_context_chars") for m in answer_results_metadata]

    # Optional: read per-conversation add_summary.json for lifecycle timings
    add_latencies: list[float] = []
    flush_count = 0
    index_settle_total = 0.0
    if index and index.get("type") == "openclaw_sandboxes":
        for sandbox in (index.get("conversations") or {}).values():
            summary_path = Path(sandbox.get("metrics_dir", "")) / "add_summary.json"
            if not summary_path.exists():
                continue
            try:
                summary = json.loads(summary_path.read_text())
            except Exception:
                continue
            lat = summary.get("add_latency_ms")
            if isinstance(lat, (int, float)):
                add_latencies.append(lat)
            flush_count += int(summary.get("flush_triggered_count", 0))
            settle = summary.get("index_settle_latency_ms") or 0
            if isinstance(settle, (int, float)):
                index_settle_total += settle

    return {
        "add_latency_ms_mean": _safe_mean(add_latencies),
        "time_to_visible_ms_mean": _safe_mean(add_latencies),  # alias in this build
        "retrieval_latency_ms_mean": _safe_mean(retrieval_latencies),
        "scheduler_wait_ms_mean": _safe_mean(scheduler_waits),
        "answer_latency_ms_mean": _safe_mean(answer_latencies),
        "empty_retrieval_rate": empty_rate,
        "final_context_tokens_mean": _safe_mean(context_tokens),
        "final_context_chars_mean": _safe_mean(context_chars),
        "retrieval_route_distribution": _distribution(routes),
        "backend_mode_distribution": _distribution(backends),
        "flush_triggered_count_total": flush_count,
        "index_settle_latency_ms_total": index_settle_total,
    }
