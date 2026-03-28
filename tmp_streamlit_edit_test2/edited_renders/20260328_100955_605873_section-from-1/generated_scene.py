from manim import *

class SectionDemo(Scene):
    def construct(self):
        self.next_section("intro")
        t = Text("Intro")
        self.play(Write(t))
        self.wait(0.1)
        self.next_section("middle")
        s = Square(color=RED)
        self.play(ReplacementTransform(t, s))
        self.wait(0.1)
        self.next_section("end")
        c = Circle()
        self.play(ReplacementTransform(s, c))
        self.wait(0.1)
