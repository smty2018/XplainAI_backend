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

# Color assignments
VARIABLE_COLOR = BLUE
CONSTANT_COLOR = WHITE
OPERATION_COLOR = YELLOW
HIGHLIGHT_COLOR = GREEN
FINAL_COLOR = GOLD

def layout_box(x, y, width, height):
    """Create a layout box dictionary."""
    return {
        "center": np.array([x, y, 0.0]),
        "width": width,
        "height": height,
    }

# Scene-specific box definitions
SCENE1_BOXES = {
    "title": layout_box(0.0, 3.5, 14.0, 1.0),
    "subtitle": layout_box(0.0, 2.5, 14.0, 0.8),
    "equation": layout_box(0.0, 0.0, 14.0, 1.5),
}

SCENE2_BOXES = {
    "equation": layout_box(0.0, 0.0, 14.0, 1.5),
    "annotation": layout_box(0.0, -1.5, 14.0, 0.8),
}

SCENE3_BOXES = {
    "equation": layout_box(0.0, 0.0, 14.0, 1.5),
}

SCENE4_BOXES = {
    "equation": layout_box(0.0, 0.0, 14.0, 1.5),
    "solution_box": layout_box(0.0, -1.5, 14.0, 1.5),
}

SCENE5_BOXES = {
    "equation": layout_box(0.0, 1.5, 14.0, 1.5),
    "verification": layout_box(0.0, -0.5, 14.0, 1.5),
    "checkmark": layout_box(4.0, -1.5, 2.0, 1.0),
}

SCENE6_BOXES = {
    "final_answer": layout_box(0.0, 0.0, 14.0, 1.5),
}

class BoxLayoutScene(Scene):
    """Base scene with box layout helpers."""
    
    def fit_to_box(self, mob, box, pad_x=0.10, pad_y=0.08, allow_upscale=False):
        """Scale mobject to fit inside box with padding."""
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
        """Shift mobject to stay inside box boundaries."""
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
        """Position and scale mobject inside box."""
        self.fit_to_box(mob, box, pad_x=pad_x, pad_y=pad_y, allow_upscale=allow_upscale)
        mob.move_to(box["center"])
        return self.keep_inside_box(mob, box, pad_x=pad_x, pad_y=pad_y)
    
    def mobjects_overlap(self, mob_a, mob_b, gap=0.06):
        """Check if two mobjects overlap."""
        x_overlap = min(mob_a.get_right()[0], mob_b.get_right()[0]) - max(
            mob_a.get_left()[0], mob_b.get_left()[0]
        )
        y_overlap = min(mob_a.get_top()[1], mob_b.get_top()[1]) - max(
            mob_a.get_bottom()[1], mob_b.get_bottom()[1]
        )
        return x_overlap > -gap and y_overlap > -gap
    
    def resolve_overlap(self, mob, blockers, box, gap=0.08, step=0.08):
        """Resolve overlap by shifting mobject within box."""
        mob = self.keep_inside_box(mob, box, pad_x=gap, pad_y=gap)
        for direction in [DOWN, UP, RIGHT, LEFT]:
            trial = mob.copy().shift(direction * step)
            self.keep_inside_box(trial, box, pad_x=gap, pad_y=gap)
            if not any(self.mobjects_overlap(trial, other, gap=gap) for other in blockers):
                mob.move_to(trial.get_center())
                break
        return mob
    
    def stack_in_box(self, mobs, box, gap=0.18, pad_x=0.12, pad_y=0.10):
        """Arrange mobjects vertically in box."""
        group = VGroup(*mobs).arrange(DOWN, buff=gap, aligned_edge=LEFT)
        self.fit_to_box(group, box, pad_x=pad_x, pad_y=pad_y)
        group.move_to(box["center"])
        self.keep_inside_box(group, box, pad_x=pad_x, pad_y=pad_y)
        return group

class SolvingLinearEquation(BoxLayoutScene):
    def construct(self):
        # === SCENE 1: Title and Problem Introduction ===
        title_text = Text("Solving a Linear Equation", font_size=48, weight=BOLD)
        self.place_in_box(title_text, SCENE1_BOXES["title"])
        
        subtitle_text = Text("x + 3 = 7", font_size=36, color=CONSTANT_COLOR)
        self.place_in_box(subtitle_text, SCENE1_BOXES["subtitle"])
        
        equation_initial = MathTex("x", "+", "3", "=", "7", font_size=72)
        equation_initial[0].set_color(VARIABLE_COLOR)
        equation_initial[1].set_color(OPERATION_COLOR)
        equation_initial[2].set_color(CONSTANT_COLOR)
        equation_initial[3].set_color(OPERATION_COLOR)
        equation_initial[4].set_color(CONSTANT_COLOR)
        self.place_in_box(equation_initial, SCENE1_BOXES["equation"])
        
        self.play(FadeIn(title_text), FadeIn(subtitle_text))
        self.play(Write(equation_initial))
        self.wait(1)
        
        # === SCENE 2: Identifying the Operation ===
        self.play(FadeOut(title_text), FadeOut(subtitle_text))
        
        # Create highlight for +3
        plus_three_highlight = SurroundingRectangle(
            VGroup(equation_initial[1], equation_initial[2]),
            color=HIGHLIGHT_COLOR,
            buff=0.1,
            stroke_width=3
        )
        annotation_text = Text("Add 3", font_size=28, color=HIGHLIGHT_COLOR)
        self.place_in_box(annotation_text, SCENE2_BOXES["annotation"])
        
        # Resolve potential overlap
        blockers = [equation_initial]
        annotation_text = self.resolve_overlap(annotation_text, blockers, SCENE2_BOXES["annotation"])
        
        self.play(Create(plus_three_highlight))
        self.play(Write(annotation_text))
        self.wait(1)
        self.play(FadeOut(plus_three_highlight), FadeOut(annotation_text))
        
        # === SCENE 3: Applying Inverse Operation ===
        equation_transformed = MathTex("x", "+", "3", "-", "3", "=", "7", "-", "3", font_size=72)
        for i, part in enumerate(equation_transformed):
            if i in [0]:  # x
                part.set_color(VARIABLE_COLOR)
            elif i in [1, 3, 5, 7]:  # +, -, =, -
                part.set_color(OPERATION_COLOR)
            else:  # 3, 3, 7, 3
                part.set_color(CONSTANT_COLOR)
        
        self.place_in_box(equation_transformed, SCENE3_BOXES["equation"])
        
        self.play(
            ReplacementTransform(equation_initial[0], equation_transformed[0]),
            ReplacementTransform(equation_initial[1], equation_transformed[1]),
            ReplacementTransform(equation_initial[2], equation_transformed[2]),
            Write(equation_transformed[3]),
            Write(equation_transformed[4]),
            ReplacementTransform(equation_initial[3], equation_transformed[5]),
            ReplacementTransform(equation_initial[4], equation_transformed[6]),
            Write(equation_transformed[7]),
            Write(equation_transformed[8]),
            lag_ratio=0.1
        )
        self.wait(1)
        
        # === SCENE 4: Simplifying to Solution ===
        equation_solution = MathTex("x", "=", "4", font_size=72)
        equation_solution[0].set_color(VARIABLE_COLOR)
        equation_solution[1].set_color(OPERATION_COLOR)
        equation_solution[2].set_color(CONSTANT_COLOR)
        
        solution_box = SurroundingRectangle(equation_solution, color=FINAL_COLOR, buff=0.3, stroke_width=4)
        solution_group = VGroup(equation_solution, solution_box)
        self.place_in_box(solution_group, SCENE4_BOXES["solution_box"])
        
        # Ensure no overlap with transformed equation
        blockers = [equation_transformed]
        solution_group = self.resolve_overlap(solution_group, blockers, SCENE4_BOXES["solution_box"])
        
        self.play(
            FadeOut(VGroup(equation_transformed[1], equation_transformed[2], 
                         equation_transformed[3], equation_transformed[4],
                         equation_transformed[7], equation_transformed[8])),
            ReplacementTransform(equation_transformed[0], equation_solution[0]),
            ReplacementTransform(equation_transformed[5], equation_solution[1]),
            ReplacementTransform(equation_transformed[6], equation_solution[2]),
        )
        self.play(Create(solution_box))
        self.wait(1)
        
        # === SCENE 5: Verification ===
        # Keep solution visible
        self.play(
            solution_group.animate.move_to(SCENE5_BOXES["equation"]["center"])
        )
        
        # Show original equation for verification
        verify_original = MathTex("x", "+", "3", "=", "7", font_size=48)
        verify_original[0].set_color(VARIABLE_COLOR)
        verify_original[1].set_color(OPERATION_COLOR)
        verify_original[2].set_color(CONSTANT_COLOR)
        verify_original[3].set_color(OPERATION_COLOR)
        verify_original[4].set_color(CONSTANT_COLOR)
        
        # Show substitution
        verify_substitution = MathTex("4", "+", "3", "=", "7", font_size=48)
        for i in range(5):
            if i in [0, 2, 4]:  # numbers
                verify_substitution[i].set_color(CONSTANT_COLOR)
            else:  # operations
                verify_substitution[i].set_color(OPERATION_COLOR)
        
        verify_final = MathTex("7", "=", "7", font_size=48, color=HIGHLIGHT_COLOR)
        checkmark = Text("✓", font_size=72, color=HIGHLIGHT_COLOR)
        
        verification_group = VGroup(verify_original, verify_substitution, verify_final).arrange(DOWN, buff=0.5)
        self.place_in_box(verification_group, SCENE5_BOXES["verification"])
        self.place_in_box(checkmark, SCENE5_BOXES["checkmark"])
        
        # Resolve any overlap between verification and solution
        blockers = [solution_group]
        verification_group = self.resolve_overlap(verification_group, blockers, SCENE5_BOXES["verification"])
        
        self.play(FadeIn(verify_original, shift=UP*0.3))
        self.wait(0.5)
        self.play(
            ReplacementTransform(verify_original[0].copy(), verify_substitution[0]),
            ReplacementTransform(verify_original[1], verify_substitution[1]),
            ReplacementTransform(verify_original[2], verify_substitution[2]),
            ReplacementTransform(verify_original[3], verify_substitution[3]),
            ReplacementTransform(verify_original[4], verify_substitution[4]),
        )
        self.wait(0.5)
        self.play(
            ReplacementTransform(verify_substitution[0:3].copy(), verify_final[0]),
            ReplacementTransform(verify_substitution[3], verify_final[1]),
            ReplacementTransform(verify_substitution[4], verify_final[2]),
        )
        self.play(FadeIn(checkmark, scale=1.5))
        self.wait(1)
        
        # === SCENE 6: Final Result Display ===
        final_answer = MathTex("x = 4", font_size=96, color=FINAL_COLOR)
        solution_label = Text("Solution:", font_size=48, color=HIGHLIGHT_COLOR)
        
        final_group = VGroup(solution_label, final_answer).arrange(RIGHT, buff=0.5)
        self.place_in_box(final_group, SCENE6_BOXES["final_answer"])
        
        self.play(
            FadeOut(solution_group),
            FadeOut(verification_group),
            FadeOut(checkmark),
        )
        self.play(FadeIn(solution_label), Write(final_answer))
        self.wait(2)
