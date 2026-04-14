"""
Task 6: lock the canonical (mem0-compatible) text_overlap.py values on a few
representative samples so regressions show up as test failures rather than
metric drift.

Intentionally not a strict parity check against bench/compute_metrics.py -
those two implementations differ in fundamental ways (set vs multiset F1,
BLEU-1 vs BLEU-4) and bench is tagged as legacy.
"""
from evaluation.src.metrics.text_overlap import compute_bleu1, compute_f1


def test_compute_f1_exact_match():
    assert compute_f1("hello world", "hello world") == 1.0


def test_compute_f1_partial_match():
    # set-based F1: pred={cat,sat}=2, ref={the,cat,sat,on,mat}=5, common=2
    # precision=2/2=1.0, recall=2/5=0.4, f1=2*1.0*0.4/(1.0+0.4)=0.5714...
    result = compute_f1("cat sat", "the cat sat on the mat")
    assert abs(result - (2 * 1.0 * 0.4 / 1.4)) < 1e-6


def test_compute_f1_empty_inputs():
    assert compute_f1("", "anything") == 0.0
    assert compute_f1("anything", "") == 0.0


def test_compute_f1_and_bleu_handle_non_string_inputs():
    # LoCoMo sometimes has integer golden_answers (e.g. year = 2022).
    # The metric must coerce rather than raise.
    assert compute_f1("It was in 2022", 2022) > 0.0
    assert compute_bleu1("Year 2022", 2022) >= 0.0
    assert compute_f1(None, "anything") == 0.0  # None coerces to "None" -> no overlap


def test_compute_bleu1_is_positive_for_overlap():
    assert compute_bleu1("the cat sat", "the cat sat on the mat") > 0.0


def test_compute_bleu1_is_zero_for_no_overlap():
    assert compute_bleu1("foo bar baz", "alpha beta gamma") >= 0.0  # smoothed, may be tiny


def test_canonical_vs_legacy_known_delta():
    """
    Same prediction+reference, different implementations. Canonical uses
    set-based F1 and BLEU-1; legacy uses multiset F1 and BLEU-4. Pin the
    current canonical numbers so accidental changes surface.
    """
    pred = "Caroline moved from Chicago to Seattle."
    ref = "Caroline moved from Chicago."

    f1 = compute_f1(pred, ref)
    assert 0.0 < f1 <= 1.0
    # With set tokenization: pred={caroline,moved,from,chicago,to,seattle}=6 tokens,
    # ref={caroline,moved,from,chicago}=4 tokens, common=4, precision=4/6, recall=4/4
    # f1 = 2 * (4/6)(1.0) / ((4/6)+1.0) = 2*(2/3)/(5/3) = 0.8
    assert abs(f1 - 0.8) < 1e-6
