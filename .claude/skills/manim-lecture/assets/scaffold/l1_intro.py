"""Lesson 1 · <title>. Template: copy to lN_<slug>.py, rename the class, add to render.LESSONS."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E401
from style import *  # noqa: F401,F403
import toy


class L1_Intro(Lesson):
    NUM, TITLE, SUB = 1, "Title here", "one-line subtitle"
    # one method per section; each section renders as its own scene (cached, parallel)
    SECTIONS = ["title", "idea", "example"]
    # sections listed here get a roadmap slide first ("what's coming"); omit sections that
    # continue the previous one's picture (e.g. iteration 2 of a loop) so they join seamlessly
    TOPICS = {"idea": "the idea in one picture", "example": "a worked example"}

    def idea(self):
        a = T("before", 40, SOFT).move_to(LEFT * 3)
        b = T("after", 40, ACCENT).move_to(RIGHT * 3)
        self.play(FadeIn(a))
        self.say("start from the before state")     # bottom caption, ≤ 6 words ideally
        self.beat()                                  # = one → press
        self.play(TransformFromCopy(a, b), GrowArrow(arrow(a.get_right(), b.get_left(), ACCENT)))
        self.say("show what changes", ACCENT)
        self.beat()
        self.clear_all()                             # fade out when the next animation starts

    def example(self):
        m = M(r"y = " + str(toy.EXAMPLE_RESULT), 60)   # every number comes from toy.py
        self.play(Write(m))
        self.say("real numbers, from toy.py")
        self.beat()
        self.clear_all()


split_sections(L1_Intro, globals())
