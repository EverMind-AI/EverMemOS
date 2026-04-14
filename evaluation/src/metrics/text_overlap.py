"""
Canonical (mem0-compatible) answer-level text overlap metrics.

F1 is set-based (unique-token overlap) and BLEU is BLEU-1 with method-1
smoothing. These match the evaluation script bundled with the mem0 research
release so openclaw answers can be compared with their published numbers
without re-running mem0 on our data.

bench/compute_metrics.py ships a different F1/BLEU flavour (multiset F1 and
BLEU-4). That file is tagged legacy; new reports should consume this module
instead.
"""
from __future__ import annotations

from typing import Sequence


def simple_tokenize(text: str) -> list[str]:
    """Lowercase + punctuation stripping. Matches mem0's pre-processing."""
    normalized = str(text).lower()
    for ch in (".", ",", "!", "?", ";", ":", "\n", "\t"):
        normalized = normalized.replace(ch, " ")
    return normalized.split()


def compute_f1(prediction: str, reference: str) -> float:
    """Set-based F1: intersection of unique tokens / precision + recall."""
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_bleu1(prediction: str, reference: str) -> float:
    """BLEU-1 with method-1 smoothing (nltk)."""
    try:
        import nltk
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    except ImportError:  # pragma: no cover - nltk is a hard dep in practice
        return _fallback_bleu1(prediction, reference)

    pred_tokens = _ensure_nltk_tokenize(nltk, prediction.lower())
    ref_tokens = _ensure_nltk_tokenize(nltk, reference.lower())
    if not pred_tokens or not ref_tokens:
        return 0.0
    smooth = SmoothingFunction().method1
    return sentence_bleu(
        [ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth
    )


def _ensure_nltk_tokenize(nltk_mod, text: str) -> list[str]:
    try:
        return nltk_mod.word_tokenize(text)
    except LookupError:
        # punkt not downloaded - fall back to whitespace split so metrics keep
        # running in air-gapped environments.
        return text.split()


def _fallback_bleu1(prediction: str, reference: str) -> float:
    pred_tokens = simple_tokenize(prediction)
    ref_tokens = set(simple_tokenize(reference))
    if not pred_tokens or not ref_tokens:
        return 0.0
    matches = sum(1 for t in pred_tokens if t in ref_tokens)
    return matches / len(pred_tokens)
