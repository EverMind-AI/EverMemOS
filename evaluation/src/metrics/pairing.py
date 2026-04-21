"""Shared helper: pair QAPair with SearchResult by question_id.

search_stage stashes ``question_id`` inside each SearchResult's
``retrieval_metadata`` so downstream consumers can pair by id instead
of relying on list position. A checkpoint-based resume or a subset
filter upstream will silently re-order things; positional zip would
misalign which search result scores which qa and corrupt every
retrieval metric downstream.

This module is the single source of truth for the pairing logic, used
by retrieval_metrics, content_overlap, and answer_stage.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple


def pair_by_question_id(
    qa_pairs: Sequence,
    search_results: Sequence,
) -> Tuple[Dict[str, object], int]:
    """Build a {question_id: SearchResult} map.

    Prefers ``retrieval_metadata.question_id``. For SearchResult
    objects that never saw search_stage (older checkpoints, hand-built
    fixtures), falls back to positional pairing against the qa_pairs
    that remain unmatched.

    Returns ``(by_id, missing_meta_count)`` — callers that need to
    surface "N results were paired positionally" (e.g. retrieval_metrics)
    use the second value, others can ignore it.
    """
    by_id: Dict[str, object] = {}
    missing_meta: List = []
    for sr in search_results:
        qid = (getattr(sr, "retrieval_metadata", {}) or {}).get("question_id")
        if qid is None:
            missing_meta.append(sr)
        else:
            by_id[qid] = sr

    if missing_meta:
        unpaired = [qa for qa in qa_pairs if qa.question_id not in by_id]
        for qa, sr in zip(unpaired, missing_meta):
            by_id[qa.question_id] = sr

    return by_id, len(missing_meta)
