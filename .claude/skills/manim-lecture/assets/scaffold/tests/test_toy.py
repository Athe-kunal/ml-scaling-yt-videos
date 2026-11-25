"""Every number shown in the lecture, checked against the source (notebook comments, paper table…)."""
import toy


def test_result_matches_source():
    assert toy.algorithm(toy.EXAMPLE) == 6


def test_trace_ends_at_result():
    t = toy.trace(toy.EXAMPLE)
    assert [s.after for s in t] == [1, 3, 6]
    assert t[-1].after == toy.EXAMPLE_RESULT
