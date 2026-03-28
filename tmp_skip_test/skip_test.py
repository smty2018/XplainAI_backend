from manim import *

class SkipDemo(Scene):
    def construct(self):
        self.next_section("intro", skip_animations=True)
        t = Text("Intro")
        self.play(Write(t))
        self.wait(0.1)
        self.next_section("middle")
        s = Square()
        self.play(ReplacementTransform(t, s))
        self.wait(0.1)
