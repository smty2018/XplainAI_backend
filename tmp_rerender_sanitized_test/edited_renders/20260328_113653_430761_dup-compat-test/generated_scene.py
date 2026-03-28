from manim import *
import numpy as np

# XplainAI Manim runtime compatibility aliases
try:
    from pydub import AudioSegment as _XplainAIAudioSegment
    import imageio_ffmpeg as _xplainai_imageio_ffmpeg
    _XplainAIAudioSegment.converter = _xplainai_imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass
try:
    _XplainAIOriginalAxes = Axes
    def Axes(*args, **kwargs):
        if 'width' in kwargs and 'x_length' not in kwargs:
            kwargs['x_length'] = kwargs.pop('width')
        if 'height' in kwargs and 'y_length' not in kwargs:
            kwargs['y_length'] = kwargs.pop('height')
        return _XplainAIOriginalAxes(*args, **kwargs)
except Exception:
    pass
try:
    _XplainAIOriginalNumberPlane = NumberPlane
    def NumberPlane(*args, **kwargs):
        if 'width' in kwargs and 'x_length' not in kwargs:
            kwargs['x_length'] = kwargs.pop('width')
        if 'height' in kwargs and 'y_length' not in kwargs:
            kwargs['y_length'] = kwargs.pop('height')
        return _XplainAIOriginalNumberPlane(*args, **kwargs)
except Exception:
    pass
try:
    import inspect as _xplainai_inspect
    _XplainAIOriginalScenePlay = Scene.play
    _XPLAINAI_PLAY_KWARGS = {'run_time', 'rate_func', 'lag_ratio', 'subcaption', 'subcaption_duration', 'subcaption_offset'}
    def _xplainai_scene_play_compat(self, *args, **kwargs):
        if args and _xplainai_inspect.ismethod(args[0]):
            _method = args[0]
            _target = getattr(_method, '__self__', None)
            _name = getattr(_method, '__name__', '')
            if _target is not None and _name and hasattr(_target, 'animate'):
                _play_kwargs = {k: v for k, v in kwargs.items() if k in _XPLAINAI_PLAY_KWARGS}
                _method_kwargs = {k: v for k, v in kwargs.items() if k not in _XPLAINAI_PLAY_KWARGS}
                _builder = getattr(_target.animate, _name)(*args[1:], **_method_kwargs)
                return _XplainAIOriginalScenePlay(self, _builder, **_play_kwargs)
        return _XplainAIOriginalScenePlay(self, *args, **kwargs)
    Scene.play = _xplainai_scene_play_compat
except Exception:
    pass
try:
    CYAN
except NameError:
    CYAN = '#00BCD4'
try:
    AQUA
except NameError:
    AQUA = '#00BCD4'
try:
    FUCHSIA
except NameError:
    FUCHSIA = '#D147BD'

# Color scheme as specified
A_COLOR = "#1E88E5"  # Blue
B_COLOR = "#E53935"  # Red
COEFF_COLOR = WHITE
HIGHLIGHT_COLOR = "#FFD600"  # Yellow
FINAL_BOX_COLOR = "#4CAF50"  # Green

def layout_box(x, y, width, height):
    return {
        "center": np.array([x, y, 0.0]),
        "width": width,
        "height": height,
    }

# Scene boxes with careful spacing to prevent overlap
SCENE_BOXES = {
    # Title takes top 15%
    "title": layout_box(0.0, 3.5, 14.0, 1.5),
    
    # Main derivation uses 60% of vertical space with sub-boxes
    "main_top": layout_box(0.0, 2.0, 14.0, 1.8),
    "main_mid": layout_box(0.0, 0.5, 14.0, 1.8),
    "main_bottom": layout_box(0.0, -1.0, 14.0, 1.8),
    
    # Extension at bottom 25%
    "extension": layout_box(0.0, -3.0, 14.0, 1.5),
}

class BinomialCubeExpansion(Scene):
    def fit_to_box(self, mob, box, pad_x=0.15, pad_y=0.10, allow_upscale=False):
        avail_width = max(0.2, box["width"] - 2 * pad_x)
        avail_height = max(0.2, box["height"] - 2 * pad_y)
        scales = []
        if mob.width > 0:
            scales.append(avail_width / mob.width)
        if mob.height > 0:
            scales.append(avail_height / mob.height)
        if not scales:
            return mob
        scale = min(scales)
        if not allow_upscale:
            scale = min(scale, 1.0)
        mob.scale(scale)
        return mob

    def keep_inside_box(self, mob, box, pad_x=0.15, pad_y=0.10):
        left_limit = box["center"][0] - box["width"] / 2 + pad_x
        right_limit = box["center"][0] + box["width"] / 2 - pad_x
        bottom_limit = box["center"][1] - box["height"] / 2 + pad_y
        top_limit = box["center"][1] + box["height"] / 2 - pad_y

        dx = 0.0
        dy = 0.0
        if mob.get_left()[0] < left_limit:
            dx = left_limit - mob.get_left()[0]
        elif mob.get_right()[0] > right_limit:
            dx = right_limit - mob.get_right()[0]

        if mob.get_bottom()[1] < bottom_limit:
            dy = bottom_limit - mob.get_bottom()[1]
        elif mob.get_top()[1] > top_limit:
            dy = top_limit - mob.get_top()[1]

        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            mob.shift(np.array([dx, dy, 0.0]))
        return mob

    def place_in_box(self, mob, box, pad_x=0.15, pad_y=0.10, allow_upscale=False):
        self.fit_to_box(mob, box, pad_x=pad_x, pad_y=pad_y, allow_upscale=allow_upscale)
        mob.move_to(box["center"])
        return self.keep_inside_box(mob, box, pad_x=pad_x, pad_y=pad_y)

    def stack_in_box(self, mobs, box, gap=0.3, pad_x=0.15, pad_y=0.12):
        group = VGroup(*mobs).arrange(DOWN, buff=gap, aligned_edge=LEFT)
        self.fit_to_box(group, box, pad_x=pad_x, pad_y=pad_y)
        group.move_to(box["center"])
        self.keep_inside_box(group, box, pad_x=pad_x, pad_y=pad_y)
        return group

    def replace_in_box(self, old_mob, new_mob, box, pad_x=0.15, pad_y=0.12):
        self.place_in_box(new_mob, box, pad_x=pad_x, pad_y=pad_y)
        return AnimationGroup(
            FadeOut(old_mob, shift=0.1 * UP),
            FadeIn(new_mob, shift=0.1 * UP),
            lag_ratio=0.0,
        )

    def fade_swap(self, old_mob, new_mob, shift=0.1 * UP):
        return AnimationGroup(
            FadeOut(old_mob, shift=shift),
            FadeIn(new_mob, shift=shift),
            lag_ratio=0.0,
        )

    def construct(self):
        # Scene 1: Introduction and Problem Statement
        title = Text("Expansion of  (a+b)³", font_size=48, weight=BOLD, color=WHITE)
        subtitle = Text("Cube of a Binomial", font_size=32, color=GRAY)
        self.place_in_box(title, SCENE_BOXES["title"])
        subtitle.next_to(title, DOWN, buff=0.2)
        self.keep_inside_box(subtitle, SCENE_BOXES["title"])

        problem_expr = MathTex(r"(a+b)^3", substrings_to_isolate=["a", "b"])
        problem_expr.set_color_by_tex("a", A_COLOR)
        problem_expr.set_color_by_tex("b", B_COLOR)
        self.place_in_box(problem_expr, SCENE_BOXES["main_top"])

        self.play(Write(title), FadeIn(subtitle, shift=DOWN))
        self.wait(0.5)
        self.play(Create(problem_expr))
        self.wait(1)

        # Scene 2: Representing as Repeated Multiplication
        repeated_expr = MathTex(
            r"(a+b) \times (a+b) \times (a+b)",
            substrings_to_isolate=["a", "b"]
        )
        repeated_expr.set_color_by_tex("a", A_COLOR)
        repeated_expr.set_color_by_tex("b", B_COLOR)
        
        self.play(self.replace_in_box(problem_expr, repeated_expr, SCENE_BOXES["main_top"]))
        self.wait(1)

        # Scene 3: Multiplying the First Two Binomials
        # Clear bottom boxes first
        self.play(FadeOut(repeated_expr))
        
        # Step 3.1: Show (a+b)(a+b)
        first_two = MathTex(
            r"(a+b)(a+b) = ?",
            substrings_to_isolate=["a", "b"]
        )
        first_two.set_color_by_tex("a", A_COLOR)
        first_two.set_color_by_tex("b", B_COLOR)
        self.place_in_box(first_two, SCENE_BOXES["main_top"])
        self.play(FadeIn(first_two))
        self.wait(0.5)

        # Step 3.2: Show distributive form in middle box
        distributive = MathTex(
            r"= a(a+b) + b(a+b)",
            substrings_to_isolate=["a", "b"]
        )
        distributive.set_color_by_tex("a", A_COLOR)
        distributive.set_color_by_tex("b", B_COLOR)
        self.place_in_box(distributive, SCENE_BOXES["main_mid"])
        self.play(Write(distributive))
        self.wait(0.5)

        # Step 3.3: Show expanded form in bottom box
        expanded = MathTex(
            r"= a^2 + ab + ab + b^2",
            substrings_to_isolate=["a", "b"]
        )
        expanded.set_color_by_tex("a", A_COLOR)
        expanded.set_color_by_tex("b", B_COLOR)
        self.place_in_box(expanded, SCENE_BOXES["main_bottom"])
        self.play(Write(expanded))
        self.wait(0.5)

        # Step 3.4: Combine terms (replace the expanded form)
        combined = MathTex(
            r"= a^2 + 2ab + b^2",
            substrings_to_isolate=["a", "b", "2"]
        )
        combined.set_color_by_tex("a", A_COLOR)
        combined.set_color_by_tex("b", B_COLOR)
        combined.set_color_by_tex("2", COEFF_COLOR)
        self.place_in_box(combined, SCENE_BOXES["main_bottom"])
        
        # Highlight the ab terms before combining
        ab_terms = VGroup(expanded[0][8:10], expanded[0][12:14])
        highlight_box = SurroundingRectangle(ab_terms, color=HIGHLIGHT_COLOR, buff=0.05)
        
        self.play(Create(highlight_box))
        self.wait(0.3)
        self.play(
            Transform(expanded, combined),
            FadeOut(highlight_box)
        )
        self.wait(1)

        # Scene 4: Preparing the Final Multiplication
        # Clear all boxes except the final result from step 3
        final_setup = MathTex(
            r"(a^2 + 2ab + b^2)(a+b)",
            substrings_to_isolate=["a", "b", "2"]
        )
        final_setup.set_color_by_tex("a", A_COLOR)
        final_setup.set_color_by_tex("b", B_COLOR)
        final_setup.set_color_by_tex("2", COEFF_COLOR)
        self.place_in_box(final_setup, SCENE_BOXES["main_top"])
        
        self.play(
            FadeOut(first_two),
            FadeOut(distributive),
            expanded.animate.move_to(SCENE_BOXES["main_top"]["center"]),
        )
        self.play(Transform(expanded, final_setup))
        self.wait(1)

        # Scene 5: Distributing Each Term
        distributed = MathTex(
            r"= a^2(a+b) + 2ab(a+b) + b^2(a+b)",
            substrings_to_isolate=["a", "b", "2"]
        )
        distributed.set_color_by_tex("a", A_COLOR)
        distributed.set_color_by_tex("b", B_COLOR)
        distributed.set_color_by_tex("2", COEFF_COLOR)
        self.place_in_box(distributed, SCENE_BOXES["main_mid"])
        
        self.play(self.fade_swap(expanded, distributed, shift=DOWN*0.5))
        self.wait(1)

        # Scene 6: Expanding the Distributed Terms
        # Clear previous and show expanded terms in three boxes
        line1 = MathTex(r"= a^3 + a^2b", substrings_to_isolate=["a", "b", "3"])
        line2 = MathTex(r"+ 2a^2b + 2ab^2", substrings_to_isolate=["a", "b", "2"])
        line3 = MathTex(r"+ ab^2 + b^3", substrings_to_isolate=["a", "b", "3"])
        
        for line in [line1, line2, line3]:
            line.set_color_by_tex("a", A_COLOR)
            line.set_color_by_tex("b", B_COLOR)
            line.set_color_by_tex("2", COEFF_COLOR)
            line.set_color_by_tex("3", COEFF_COLOR)
        
        expanded_group = VGroup(line1, line2, line3).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        self.place_in_box(expanded_group, SCENE_BOXES["main_top"])
        
        self.play(self.fade_swap(distributed, expanded_group))
        self.wait(1)

        # Scene 7: Combining Like Terms
        # Show grouping animation in middle and bottom boxes
        a2b_group = MathTex(r"a^2b + 2a^2b = 3a^2b", substrings_to_isolate=["a", "b", "2", "3"])
        a2b_group.set_color_by_tex("a", A_COLOR)
        a2b_group.set_color_by_tex("b", B_COLOR)
        a2b_group.set_color_by_tex("2", COEFF_COLOR)
        a2b_group.set_color_by_tex("3", COEFF_COLOR)
        self.place_in_box(a2b_group, SCENE_BOXES["main_mid"])
        
        ab2_group = MathTex(r"2ab^2 + ab^2 = 3ab^2", substrings_to_isolate=["a", "b", "2", "3"])
        ab2_group.set_color_by_tex("a", A_COLOR)
        ab2_group.set_color_by_tex("b", B_COLOR)
        ab2_group.set_color_by_tex("2", COEFF_COLOR)
        ab2_group.set_color_by_tex("3", COEFF_COLOR)
        self.place_in_box(ab2_group, SCENE_BOXES["main_bottom"])
        
        # Highlight the terms being combined
        a2b_highlight = SurroundingRectangle(VGroup(line1[0][-3:], line2[0][:3]), color=HIGHLIGHT_COLOR, buff=0.08)
        ab2_highlight = SurroundingRectangle(VGroup(line2[0][-3:], line3[0][:3]), color=HIGHLIGHT_COLOR, buff=0.08)
        
        self.play(Create(a2b_highlight))
        self.wait(0.3)
        self.play(FadeIn(a2b_group), FadeOut(a2b_highlight))
        
        self.play(Create(ab2_highlight))
        self.wait(0.3)
        self.play(FadeIn(ab2_group), FadeOut(ab2_highlight))
        self.wait(1)

        # Scene 8: Final Result and Boxing
        # Clear all and show final result in top box
        final_equation = MathTex(
            r"(a+b)^3 = a^3 + 3a^2b + 3ab^2 + b^3",
            substrings_to_isolate=["a", "b", "3"]
        )
        final_equation.set_color_by_tex("a", A_COLOR)
        final_equation.set_color_by_tex("b", B_COLOR)
        final_equation.set_color_by_tex("3", COEFF_COLOR)
        self.place_in_box(final_equation, SCENE_BOXES["main_top"])
        
        final_box = SurroundingRectangle(final_equation, color=FINAL_BOX_COLOR, buff=0.2, corner_radius=0.1)
        
        self.play(
            FadeOut(expanded_group),
            FadeOut(a2b_group),
            FadeOut(ab2_group),
            FadeIn(final_equation)
        )
        self.wait(0.5)
        self.play(Create(final_box))
        self.wait(2)

        # Scene 9: Connection and Pattern (Optional)
        extension_text1 = Text("Coefficients: 1, 3, 3, 1", font_size=28, color=WHITE)
        extension_text2 = Text("Third row of Pascal's Triangle", font_size=24, color=GRAY)
        extension_text3 = Text("In quantum mechanics, such expansions appear", 
                              font_size=20, color=LIGHT_GRAY)
        extension_text4 = Text("in operators like the Hamiltonian.", 
                              font_size=20, color=LIGHT_GRAY)
        
        extension_group = VGroup(extension_text1, extension_text2, 
                                extension_text3, extension_text4).arrange(DOWN, buff=0.1)
        self.place_in_box(extension_group, SCENE_BOXES["extension"])
        
        self.play(FadeIn(extension_group, shift=UP))
        self.wait(3)
