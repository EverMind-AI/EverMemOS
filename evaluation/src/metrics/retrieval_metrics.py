"""
Session-level retrieval metrics.

The manifest + source_sessions convention lets us score retrieval on the
same granularity across all systems: a hit covers a set of session ids,
gold evidence is projected to the same session ids, and we compute
hit@k / recall@k / MRR / nDCG@k on that shared vocabulary.
"""
from __future__ import annotations

import math
from typing import Iterable

from evaluation.src.adapters.openclaw_manifest import project_message_id_to_session_id


def normalize_gold_sessions(evidence: Iterable[str]) -> list[str]:
    """Project gold evidence down to sorted unique session ids.

    Accepts both ``D<s>:<m>`` (LoCoMo / LongMemEval after converter) and
    bare ``S<s>`` formats. Anything else raises ValueError - benchmark A
    runs fail-closed so no silent evidence drops inflate recall numbers.
    """
    out: list[str] = []
    for item in evidence:
        s = str(item)
        if s.startswith("S") and s[1:].isdigit():
            out.append(s)
        elif s.startswith("D"):
            out.append(project_message_id_to_session_id(s))
        else:
            raise ValueError(f"Unsupported evidence format: {item!r}")
    return sorted(set(out))


def _hit_sessions(item: dict) -> set[str]:
    return set(item.get("metadata", {}).get("source_sessions") or [])


def evaluate_retrieval_for_question(qa, search_result, k: int) -> dict:
    """Compute per-question retrieval metrics against ``k`` top hits.

    qa must expose ``.evidence``; search_result must expose ``.results``.
    Hits without source_sessions count as zero-recall but still occupy a
    rank slot.
    """
    gold_sessions = set(normalize_gold_sessions(qa.evidence))
    top_results = list(search_result.results[:k])

    first_rank = None
    covered: set[str] = set()
    gains: list[int] = []
    for rank, item in enumerate(top_results, start=1):
        item_sessions = _hit_sessions(item)
        matched = item_sessions & gold_sessions
        if matched and first_rank is None:
            first_rank = rank
        new_covered = matched - covered
        covered |= matched
        gains.append(1 if new_covered else 0)

    hit_at_k = 1.0 if first_rank is not None else 0.0
    mrr = 1.0 / first_rank if first_rank is not None else 0.0
    recall_at_k = len(covered) / len(gold_sessions) if gold_sessions else 0.0

    # nDCG with binary gains (1 when a new session is covered at that rank)
    dcg = sum(g / math.log2(idx + 1) for idx, g in enumerate(gains, start=1))
    ideal_gains = min(len(gold_sessions), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_gains + 1))
    ndcg_at_k = dcg / idcg if idcg > 0 else 0.0

    return {
        "question_id": getattr(qa, "question_id", None),
        "k": k,
        "gold_sessions": sorted(gold_sessions),
        "first_rank": first_rank,
        "evidence_hit_at_k": hit_at_k,
        "evidence_recall_at_k": recall_at_k,
        "mrr": mrr,
        "ndcg_at_k": ndcg_at_k,
    }


def evaluate_retrieval_metrics(qa_pairs, search_results, k: int = 5) -> dict:
    """Batch wrapper: pairs qa[i] with search_results[i] and aggregates."""
    per_question = []
    for qa, sr in zip(qa_pairs, search_results):
        per_question.append(evaluate_retrieval_for_question(qa, sr, k=k))
    n = len(per_question) or 1
    return {
        "k": k,
        "per_question": per_question,
        "evidence_hit_at_k_mean": sum(p["evidence_hit_at_k"] for p in per_question) / n,
        "evidence_recall_at_k_mean": sum(p["evidence_recall_at_k"] for p in per_question) / n,
        "mrr_mean": sum(p["mrr"] for p in per_question) / n,
        "ndcg_at_k_mean": sum(p["ndcg_at_k"] for p in per_question) / n,
    }
