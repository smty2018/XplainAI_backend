from manim.scene.scene import Scene as _XplainAIScene
_XPLAINAI_SKIP_BEFORE_SECTION_INDEX = 1
_xplainai_original_next_section = _XplainAIScene.next_section
def _xplainai_patched_next_section(self, *args, **kwargs):
    section_counter = getattr(self, '_xplainai_section_counter', 0)
    self._xplainai_section_counter = section_counter + 1
    if section_counter < _XPLAINAI_SKIP_BEFORE_SECTION_INDEX:
        kwargs['skip_animations'] = True
    return _xplainai_original_next_section(self, *args, **kwargs)
_XplainAIScene.next_section = _xplainai_patched_next_section

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
