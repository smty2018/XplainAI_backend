from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.coqui import CoquiService
from pydub import AudioSegment
import imageio_ffmpeg
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
    BoxLayoutScene
except NameError:
    try:
        _XplainAIBaseLayoutScene = VoiceoverScene
    except NameError:
        _XplainAIBaseLayoutScene = Scene
    class BoxLayoutScene(_XplainAIBaseLayoutScene):
        def p(self, x, y):
            return np.array([x, y, 0.0])
        def fit_to_box(self, mob, box, pad_x=0.10, pad_y=0.08, allow_upscale=False):
            avail_width = max(0.2, box['width'] - 2 * pad_x)
            avail_height = max(0.2, box['height'] - 2 * pad_y)
            scales = []
            if getattr(mob, 'width', 0) > 0:
                scales.append(avail_width / mob.width)
            if getattr(mob, 'height', 0) > 0:
                scales.append(avail_height / mob.height)
            if not scales:
                return mob
            scale = min(scales)
            if not allow_upscale:
                scale = min(scale, 1.0)
            mob.scale(scale)
            return mob
        def keep_inside_box(self, mob, box, pad_x=0.10, pad_y=0.08):
            left_limit = box['center'][0] - box['width'] / 2 + pad_x
            right_limit = box['center'][0] + box['width'] / 2 - pad_x
            bottom_limit = box['center'][1] - box['height'] / 2 + pad_y
            top_limit = box['center'][1] + box['height'] / 2 - pad_y
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
        def place_in_box(self, mob, box, pad_x=0.10, pad_y=0.08, allow_upscale=False):
            self.fit_to_box(mob, box, pad_x=pad_x, pad_y=pad_y, allow_upscale=allow_upscale)
            mob.move_to(box['center'])
            return self.keep_inside_box(mob, box, pad_x=pad_x, pad_y=pad_y)
        def mobjects_overlap(self, mob_a, mob_b, gap=0.06):
            x_overlap = min(mob_a.get_right()[0], mob_b.get_right()[0]) - max(mob_a.get_left()[0], mob_b.get_left()[0])
            y_overlap = min(mob_a.get_top()[1], mob_b.get_top()[1]) - max(mob_a.get_bottom()[1], mob_b.get_bottom()[1])
            return x_overlap > -gap and y_overlap > -gap
        def resolve_overlap(self, mob, blockers, box, gap=0.08, step=0.08):
            mob = self.keep_inside_box(mob, box, pad_x=gap, pad_y=gap)
            directions = [DOWN, UP, RIGHT, LEFT]
            for _ in range(20):
                active = [other for other in blockers if self.mobjects_overlap(mob, other, gap=gap)]
                if not active:
                    break
                moved = False
                for direction in directions:
                    trial = mob.copy().shift(direction * step)
                    self.keep_inside_box(trial, box, pad_x=gap, pad_y=gap)
                    if not any(self.mobjects_overlap(trial, other, gap=gap) for other in blockers):
                        mob.move_to(trial.get_center())
                        moved = True
                        break
                if not moved:
                    break
            return mob
        def stack_in_box(self, mobs, box, gap=0.18, pad_x=0.12, pad_y=0.10):
            group = VGroup(*mobs).arrange(DOWN, buff=gap, aligned_edge=LEFT)
            self.fit_to_box(group, box, pad_x=pad_x, pad_y=pad_y)
            group.move_to(box['center'])
            self.keep_inside_box(group, box, pad_x=pad_x, pad_y=pad_y)
            return group
        def replace_in_box(self, old_mob, new_mob, box, pad_x=0.12, pad_y=0.10):
            self.place_in_box(new_mob, box, pad_x=pad_x, pad_y=pad_y)
            return AnimationGroup(FadeOut(old_mob, shift=0.08 * UP), FadeIn(new_mob, shift=0.08 * UP), lag_ratio=0.0)
        def fade_swap(self, old_mob, new_mob, shift=0.08 * UP):
            return AnimationGroup(FadeOut(old_mob, shift=shift), FadeIn(new_mob, shift=shift), lag_ratio=0.0)
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

AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

# Color scheme
EQUATION_COLOR = WHITE
OPERATION_COLOR = GREEN
CANCEL_COLOR = GRAY
GRAPH_LINE_COLOR = BLUE
INTERCEPT_COLOR = RED
FINAL_BOX_COLOR = YELLOW

# Box layout helper
def layout_box(x, y, width, height):
    return {
        "center": np.array([x, y, 0.0]),
        "width": width,
        "height": height,
    }

# Scene-specific box dictionaries
SCENE1_BOXES = {
    "title": layout_box(0.0, 3.2, 12.0, 0.5),
    "main_eq": layout_box(0.0, 1.0, 10.0, 1.0),
}

SCENE2_BOXES = {
    "title": layout_box(0.0, 3.2, 12.0, 0.5),
    "main_eq": layout_box(0.0, 1.0, 10.0, 2.0),
}

SCENE3_BOXES = {
    "title": layout_box(0.0, 3.2, 12.0, 0.5),
    "graph": layout_box(0.0, -0.5, 11.0, 5.0),
    "equation_label": layout_box(-4.0, 2.5, 4.0, 0.5),
}

SCENE4_BOXES = {
    "title": layout_box(0.0, 3.2, 12.0, 0.5),
    "main_eq": layout_box(0.0, 1.0, 10.0, 1.0),
    "final_answer": layout_box(0.0, -1.5, 8.0, 1.0),
}

class SolvingLinearEquation(BoxLayoutScene):
    def fit_to_box(self, mob, box, pad_x=0.10, pad_y=0.08, allow_upscale=False):
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

    def keep_inside_box(self, mob, box, pad_x=0.10, pad_y=0.08):
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

    def place_in_box(self, mob, box, pad_x=0.10, pad_y=0.08, allow_upscale=False):
        self.fit_to_box(mob, box, pad_x=pad_x, pad_y=pad_y, allow_upscale=allow_upscale)
        mob.move_to(box["center"])
        return self.keep_inside_box(mob, box, pad_x=pad_x, pad_y=pad_y)

    def stack_in_box(self, mobs, box, gap=0.18, pad_x=0.12, pad_y=0.10):
        group = VGroup(*mobs).arrange(DOWN, buff=gap, aligned_edge=LEFT)
        self.fit_to_box(group, box, pad_x=pad_x, pad_y=pad_y)
        group.move_to(box["center"])
        self.keep_inside_box(group, box, pad_x=pad_x, pad_y=pad_y)
        return group

    def replace_in_box(self, old_mob, new_mob, box, pad_x=0.12, pad_y=0.10):
        self.place_in_box(new_mob, box, pad_x=pad_x, pad_y=pad_y)
        return AnimationGroup(
            FadeOut(old_mob, shift=0.08 * UP),
            FadeIn(new_mob, shift=0.08 * UP),
            lag_ratio=0.0,
        )

    def mobjects_overlap(self, mob_a, mob_b, gap=0.06):
        x_overlap = min(mob_a.get_right()[0], mob_b.get_right()[0]) - max(
            mob_a.get_left()[0], mob_b.get_left()[0]
        )
        y_overlap = min(mob_a.get_top()[1], mob_b.get_top()[1]) - max(
            mob_a.get_bottom()[1], mob_b.get_bottom()[1]
        )
        return x_overlap > -gap and y_overlap > -gap

    def resolve_overlap(self, mob, blockers, box, gap=0.08, step=0.08):
        mob = self.keep_inside_box(mob, box, pad_x=gap, pad_y=gap)
        for direction in [DOWN, UP, RIGHT, LEFT]:
            trial = mob.copy().shift(direction * step)
            self.keep_inside_box(trial, box, pad_x=gap, pad_y=gap)
            if not any(self.mobjects_overlap(trial, other, gap=gap) for other in blockers):
                mob.move_to(trial.get_center())
                break
        return mob

    def fade_swap(self, old_mob, new_mob, shift=0.08 * UP):
        return AnimationGroup(
            FadeOut(old_mob, shift=shift),
            FadeIn(new_mob, shift=shift),
            lag_ratio=0.0,
        )

    def construct(self):
        self.set_speech_service(
            CoquiService(model_name="tts_models/en/vctk/vits", speaker_idx=7)
        )

        # Scene 1: Introduction and Problem Statement
        self.next_section("Scene 1: Introduction", skip_animations=False)
        
        title = Text("Solving a Linear Equation", font_size=40, weight=BOLD, color=EQUATION_COLOR)
        equation = MathTex("x - 2 = 0", color=EQUATION_COLOR)
        
        self.place_in_box(title, SCENE1_BOXES["title"])
        self.place_in_box(equation, SCENE1_BOXES["main_eq"])
        
        with self.voiceover(text="We need to solve for x in the equation: x minus two equals zero.") as tracker:
            self.play(FadeIn(title))
            self.play(Write(equation))
        
        self.wait(1)
        
        # Scene 2: Algebraic Solution - The Balance Principle
        self.next_section("Scene 2: Algebraic Solution", skip_animations=False)
        
        equation1 = MathTex("x - 2 = 0", color=EQUATION_COLOR)
        equation2 = MathTex("x - 2 + 2 = 0 + 2", color=EQUATION_COLOR)
        equation3 = MathTex("x = 2", color=EQUATION_COLOR)
        
        plus2 = MathTex("+ 2", color=OPERATION_COLOR).scale(0.8)
        cancel_group1 = MathTex("-2 + 2", color=CANCEL_COLOR)
        cancel_group2 = MathTex("0 + 2", color=CANCEL_COLOR)
        
        self.place_in_box(equation1, SCENE2_BOXES["main_eq"])
        self.place_in_box(equation2, SCENE2_BOXES["main_eq"])
        self.place_in_box(equation3, SCENE2_BOXES["main_eq"])
        
        with self.voiceover(text="To isolate x, we perform the inverse operation. Adding 2 to both sides keeps the equation balanced.") as tracker:
            self.play(self.replace_in_box(equation, equation1, SCENE2_BOXES["main_eq"]))
            self.wait(0.5)
            
            # Highlight -2 term
            minus2_part = equation1.get_part_by_tex("-2")
            self.play(minus2_part.animate.set_color(YELLOW))
            self.wait(0.5)
            
            # Show adding 2 to both sides
            plus2_left = plus2.copy().next_to(equation1[0][0], DOWN, buff=0.2)
            plus2_right = plus2.copy().next_to(equation1[0][-1], DOWN, buff=0.2)
            
            self.play(FadeIn(plus2_left), FadeIn(plus2_right))
            self.wait(0.5)
            
            # Transform to equation2
            self.play(
                Transform(equation1, equation2),
                FadeOut(plus2_left),
                FadeOut(plus2_right)
            )
        
        with self.voiceover(text="Simplifying gives us the solution.") as tracker:
            self.wait(0.5)
            
            # Show cancellation
            cancel1 = cancel_group1.copy().move_to(equation2.get_part_by_tex("-2 + 2").get_center())
            cancel2 = cancel_group2.copy().move_to(equation2.get_part_by_tex("0 + 2").get_center())
            
            self.play(FadeIn(cancel1), FadeIn(cancel2))
            self.wait(0.5)
            
            # Fade out cancelled terms and transform to final solution
            self.play(
                FadeOut(cancel1),
                FadeOut(cancel2),
                equation2[0][1:4].animate.set_opacity(0),  # Hide -2 + 2
                equation2[0][-2:].animate.set_opacity(0),   # Hide + 2
            )
            
            self.play(self.replace_in_box(equation2, equation3, SCENE2_BOXES["main_eq"]))
        
        self.wait(1)
        
        # Scene 3: Graphical Confirmation
        self.next_section("Scene 3: Graphical Confirmation", skip_animations=False)
        
        graph_title = Text("Graphical Confirmation", font_size=34, weight=BOLD, color=EQUATION_COLOR)
        self.place_in_box(graph_title, SCENE3_BOXES["title"])
        
        axes = Axes(
            x_range=[-1, 4, 1],
            y_range=[-3, 2, 1],
            x_length=SCENE3_BOXES["graph"]["width"] * 0.9,
            y_length=SCENE3_BOXES["graph"]["height"] * 0.9,
            axis_config={"color": WHITE},
            tips=False,
        )
        self.place_in_box(axes, SCENE3_BOXES["graph"])
        
        line = axes.plot(lambda x: x - 2, color=GRAPH_LINE_COLOR)
        line_label = MathTex("y = x - 2", color=GRAPH_LINE_COLOR).scale(0.8)
        self.place_in_box(line_label, SCENE3_BOXES["equation_label"])
        
        x_axis_line = DashedLine(
            axes.c2p(-1, 0),
            axes.c2p(4, 0),
            color=WHITE,
            stroke_width=1
        )
        
        intercept_point = Dot(axes.c2p(2, 0), color=INTERCEPT_COLOR, radius=0.08)
        intercept_label = MathTex("(2, 0)", color=INTERCEPT_COLOR).scale(0.7)
        intercept_label.next_to(intercept_point, DOWN)
        
        with self.voiceover(text="We can confirm this solution graphically. The equation x minus two equals zero is equivalent to setting y equals zero in the line y equals x minus two.") as tracker:
            self.play(
                self.fade_swap(equation3, graph_title, shift=UP),
                Create(axes),
            )
            self.play(FadeIn(line_label))
            self.play(Create(line))
        
        with self.voiceover(text="The solution x equals two is the x-intercept where the line crosses the axis.") as tracker:
            self.play(Create(x_axis_line))
            self.play(FadeIn(intercept_point))
            self.play(Write(intercept_label))
            
            # Draw vertical line to emphasize x=2
            vertical_line = DashedLine(
                axes.c2p(2, 0),
                axes.c2p(2, -2),
                color=INTERCEPT_COLOR,
                stroke_width=1.5
            )
            self.play(Create(vertical_line))
        
        self.wait(1)
        
        # Scene 4: Verification and Final Answer
        self.next_section("Scene 4: Verification", skip_animations=False)
        
        verification_title = Text("Verification", font_size=34, weight=BOLD, color=EQUATION_COLOR)
        self.place_in_box(verification_title, SCENE4_BOXES["title"])
        
        original_eq = MathTex("x - 2 = 0", color=EQUATION_COLOR)
        substitution_eq = MathTex("2 - 2 = 0", color=EQUATION_COLOR)
        simplified_eq = MathTex("0 = 0", color=EQUATION_COLOR)
        final_eq = MathTex("x = 2", color=EQUATION_COLOR)
        
        self.place_in_box(original_eq, SCENE4_BOXES["main_eq"])
        
        with self.voiceover(text="Let's verify. Substituting 2 back into the original equation gives zero equals zero, which is true.") as tracker:
            self.play(
                self.fade_swap(graph_title, verification_title, shift=UP),
                FadeOut(axes),
                FadeOut(line),
                FadeOut(line_label),
                FadeOut(x_axis_line),
                FadeOut(intercept_point),
                FadeOut(intercept_label),
                FadeOut(vertical_line),
            )
            
            self.play(Write(original_eq))
            self.wait(0.5)
            
            # Transform x to 2
            self.play(ReplacementTransform(original_eq[0][0], substitution_eq[0][0]))
            self.play(
                original_eq[0][1:].animate.move_to(substitution_eq[0][1:].get_center())
            )
            
            self.remove(original_eq)
            self.add(substitution_eq)
            self.wait(0.5)
            
            # Simplify
            self.play(Transform(substitution_eq, simplified_eq))
            self.wait(0.5)
        
        with self.voiceover(text="Therefore, our solution is correct.") as tracker:
            # Show final answer
            self.place_in_box(final_eq, SCENE4_BOXES["final_answer"])
            self.play(
                FadeOut(substitution_eq),
                FadeIn(final_eq)
            )
            
            # Box the final answer
            box = SurroundingRectangle(final_eq, color=FINAL_BOX_COLOR, buff=0.2)
            self.play(Create(box))
            
            # Emphasize with scale
            final_group = VGroup(final_eq, box)
            self.play(
                final_group.animate.scale(1.2),
                run_time=0.5
            )
            self.play(
                final_group.animate.scale(1/1.2),
                run_time=0.5
            )
        
        self.wait(2)
