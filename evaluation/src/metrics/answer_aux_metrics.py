"""
Answer-level auxiliary metrics: mem0-compatible F1 / BLEU-1 aggregates.

Consumed by the pipeline to emit ``answer_aux_metrics.json``. Per-question
breakdown is kept so downstream analysis (e.g. per-category drill-down) can
go deeper without re-running the evaluator.
"""
from __future__ import annotations

from typing import Iterable

from evaluation.src.metrics.text_overlap import compute_bleu1, compute_f1


def build_answer_aux_metrics(answer_results: Iterable) -> dict:
    per_question = []
    f1_scores = []
    bleu_scores = []
    for ar in answer_results:
        pred = getattr(ar, "answer", "") or ""
        gold = getattr(ar, "golden_answer", "") or ""
        f1 = compute_f1(pred, gold)
        bleu = compute_bleu1(pred, gold)
        per_question.append(
            {
                "question_id": getattr(ar, "question_id", None),
                "f1": f1,
                "bleu1": bleu,
            }
        )
        f1_scores.append(f1)
        bleu_scores.append(bleu)

    n = len(per_question) or 1
    return {
        "per_question": per_question,
        "f1_mean": sum(f1_scores) / n,
        "bleu1_mean": sum(bleu_scores) / n,
        "count": len(per_question),
    }
