"""
Adapter-agnostic retrieval quality metric.

``content_overlap@k`` is the token F1 between a question's gold answer
and the concatenation of the top-k retrieved documents' text. It needs
only two things the evaluation framework already has for every adapter:

  * ``QAPair.answer`` — the gold answer.
  * ``SearchResult.results[i]["content"]`` — the retrieved chunk text.

No adapter-specific projection (source_sessions, manifest files,
message-id -> session-id mapping) is required, so new memory systems
plug in without any extra integration work. See
docs/latency-alignment.md for the broader "measurement lives at the
adapter boundary, not inside the adapter" principle; this metric is the
retrieval-quality analogue of that stance.

Session-level metrics (evidence_hit_at_k / evidence_recall_at_k / MRR /
nDCG@k) remain available in ``retrieval_metrics.py`` but are now treated
as adapter-specific diagnostics that only apply to runs where evidence
is in ``D<s>:<m>`` form and retrieved docs carry ``source_sessions``.
"""
from __future__ import annotations

from typing import Iterable

from evaluation.src.metrics.text_overlap import simple_tokenize


def _concat_top_k_text(search_result, k: int) -> str:
    """Join the ``content`` field of the top-k retrieved docs."""
    pieces: list[str] = []
    for item in list(search_result.results[:k]):
        text = item.get("content") if isinstance(item, dict) else None
        if text:
            pieces.append(str(text))
    return " \n ".join(pieces)


def compute_content_overlap_at_k(qa, search_result, k: int) -> dict:
    """Per-question token-F1 between gold answer and top-k retrieved text.

    Returns the full (precision, recall, f1) triple so callers that want
    to distinguish "retrieved text is too noisy" (low precision) from
    "gold tokens aren't in the retrieved text" (low recall) can.
    The canonical scalar is ``content_overlap_at_k`` == f1.
    """
    gold_tokens = set(simple_tokenize(str(qa.answer)))
    ctx_tokens = set(simple_tokenize(_concat_top_k_text(search_result, k)))
    if not gold_tokens or not ctx_tokens:
        return {
            "question_id": getattr(qa, "question_id", None),
            "k": k,
            "content_overlap_precision": 0.0,
            "content_overlap_recall": 0.0,
            "content_overlap_at_k": 0.0,
        }
    common = gold_tokens & ctx_tokens
    if not common:
        return {
            "question_id": getattr(qa, "question_id", None),
            "k": k,
            "content_overlap_precision": 0.0,
            "content_overlap_recall": 0.0,
            "content_overlap_at_k": 0.0,
        }
    precision = len(common) / len(ctx_tokens)
    recall = len(common) / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return {
        "question_id": getattr(qa, "question_id", None),
        "k": k,
        "content_overlap_precision": precision,
        "content_overlap_recall": recall,
        "content_overlap_at_k": f1,
    }


def evaluate_content_overlap(qa_pairs, search_results, k: int = 5) -> dict:
    """Batch wrapper. Returns per-question rows plus mean aggregates."""
    from evaluation.src.metrics.pairing import pair_by_question_id

    by_id, _ = pair_by_question_id(qa_pairs, search_results)

    per_question: list[dict] = []
    unresolved: list[str] = []
    for qa in qa_pairs:
        sr = by_id.get(qa.question_id)
        if sr is None:
            unresolved.append(qa.question_id)
            continue
        per_question.append(compute_content_overlap_at_k(qa, sr, k=k))

    if not per_question:
        return {
            "k": k,
            "per_question": [],
            "content_overlap_at_k_mean": 0.0,
            "content_overlap_precision_mean": 0.0,
            "content_overlap_recall_mean": 0.0,
            "unresolved_question_ids": unresolved,
        }

    n = len(per_question)
    return {
        "k": k,
        "per_question": per_question,
        "content_overlap_at_k_mean": sum(
            p["content_overlap_at_k"] for p in per_question
        ) / n,
        "content_overlap_precision_mean": sum(
            p["content_overlap_precision"] for p in per_question
        ) / n,
        "content_overlap_recall_mean": sum(
            p["content_overlap_recall"] for p in per_question
        ) / n,
        "unresolved_question_ids": unresolved,
    }
