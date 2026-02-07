"""Unit tests for retrieval cosine similarity safety helpers."""

import numpy as np

from agentic_layer import retrieval_utils


class Candidate:
    def __init__(self, embedding):
        self.extend = {"embedding": embedding}


def test_safe_cosine_similarity_returns_none_for_missing_embedding():
    candidate = Candidate([])
    score = retrieval_utils._safe_cosine_similarity(
        np.array([1.0, 2.0]), 1.0, candidate
    )
    assert score is None


def test_safe_cosine_similarity_returns_none_for_dimension_mismatch():
    candidate = Candidate([1.0])
    score = retrieval_utils._safe_cosine_similarity(
        np.array([1.0, 2.0]), 1.0, candidate
    )
    assert score is None


def test_safe_cosine_similarity_returns_score_for_valid_vectors():
    candidate = Candidate([1.0, 0.0])
    score = retrieval_utils._safe_cosine_similarity(
        np.array([1.0, 0.0]), 1.0, candidate
    )
    assert score == 1.0
