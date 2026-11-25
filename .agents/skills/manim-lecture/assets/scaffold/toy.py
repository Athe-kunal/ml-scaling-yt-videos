"""Single source of truth for every number on screen.

Put the algorithm here (copied from the user's notebook/paper, framework-free if possible) plus
a *trace*: a list of dataclasses, one per step the lecture shows. Lessons only draw the trace;
tests/test_toy.py checks it against the source's own numbers.
"""
from dataclasses import dataclass

EXAMPLE = [1, 2, 3]


def algorithm(xs):
    return sum(xs)


@dataclass
class Step:
    k: int
    before: int
    after: int


def trace(xs):
    out, acc = [], 0
    for k, x in enumerate(xs):
        out.append(Step(k, acc, acc + x))
        acc += x
    return out


EXAMPLE_RESULT = algorithm(EXAMPLE)
