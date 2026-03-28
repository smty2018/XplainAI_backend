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

# Color scheme
DEFAULT_COLOR = WHITE
OPERATION_COLOR = BLUE
CANCEL_COLOR = GRAY
FINAL_BOX_COLOR = YELLOW
VERIFY_COLOR = GREEN

def layout_box(x, y, width, height):
    """Create a layout box dictionary."""
    return {
        "center": np.array([x, y, 0.0]),
        "width": width,
        "height": height,
    }

# Scene boxes - carefully defined to prevent overlap
SCENE_BOXES = {
    # Title at top (15% of screen)
    "title": layout_box(0.0, 3.5, 14.0, 1.0),
    
    # Main equation area (center 50% of screen)
    "equation_main": layout_box(0.0, 1.5, 12.0, 1.2),
    "equation_step": layout_box(0.0, 0.0, 12.0, 1.2),
    
    # Balance scale area (bottom-right)
    "balance": layout_box(5.0, -2.0, 5.0, 2.0),
    
    # Final answer area (bottom-center)
    "final_answer": layout_box(0.0, -2.0, 8.0, 1.0),
    
    # Verification area (center)
    "verify": layout_box(0.0, 1.5, 12.0, 1.2),
}


class SolvingLinearEquationScene(Scene):
    """Scene for solving x - 2 = 9 step by step with no overlap."""
    
    def fit_to_box(self, mob, box, pad_x=0.12, pad_y=0.10, allow_upscale=False):
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
    
    def keep_inside_box(self, mob, box, pad_x=0.12, pad_y=0.10):
        """Ensure mobject stays within box boundaries."""
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
    
    def place_in_box(self, mob, box, pad_x=0.12, pad_y=0.10, allow_upscale=False):
        """Place and fit mobject in box."""
        self.fit_to_box(mob, box, pad_x=pad_x, pad_y=pad_y, allow_upscale=allow_upscale)
        mob.move_to(box["center"])
        return self.keep_inside_box(mob, box, pad_x=pad_x, pad_y=pad_y)
    
    def mobjects_overlap(self, mob_a, mob_b, gap=0.08):
        """Check if two mobjects overlap with safety gap."""
        x_overlap = min(mob_a.get_right()[0], mob_b.get_right()[0]) - max(
            mob_a.get_left()[0], mob_b.get_left()[0]
        )
        y_overlap = min(mob_a.get_top()[1], mob_b.get_top()[1]) - max(
            mob_a.get_bottom()[1], mob_b.get_bottom()[1]
        )
        return x_overlap > -gap and y_overlap > -gap
    
    def resolve_overlap(self, mob, blockers, box, gap=0.08, step=0.1):
        """Resolve overlap by shifting mobject within box."""
        mob = self.keep_inside_box(mob, box, pad_x=gap, pad_y=gap)
        
        for direction in [DOWN, UP, RIGHT, LEFT]:
            trial = mob.copy().shift(direction * step)
            self.keep_inside_box(trial, box, pad_x=gap, pad_y=gap)
            
            overlap_found = False
            for other in blockers:
                if self.mobjects_overlap(trial, other, gap=gap):
                    overlap_found = True
                    break
            
            if not overlap_found:
                mob.move_to(trial.get_center())
                break
        
        return mob
    
    def create_balance_scale(self):
        """Create a simple balance scale graphic."""
        # Post
        post = Line(
            start=np.array([0, -0.5, 0]),
            end=np.array([0, 0.5, 0]),
            color=DEFAULT_COLOR,
            stroke_width=8
        )
        
        # Beam
        beam = Line(
            start=np.array([-1.2, 0.5, 0]),
            end=np.array([1.2, 0.5, 0]),
            color=DEFAULT_COLOR,
            stroke_width=6
        )
        
        # Pans
        left_pan = Polygon(
            np.array([-1.2, 0.5, 0]),
            np.array([-0.7, 0.5, 0]),
            np.array([-0.95, 0.8, 0]),
            color=DEFAULT_COLOR,
            stroke_width=4,
            fill_opacity=0.1
        )
        
        right_pan = Polygon(
            np.array([1.2, 0.5, 0]),
            np.array([0.7, 0.5, 0]),
            np.array([0.95, 0.8, 0]),
            color=DEFAULT_COLOR,
            stroke_width=4,
            fill_opacity=0.1
        )
        
        # Support lines
        left_support = Line(
            start=np.array([-0.95, 0.5, 0]),
            end=np.array([-0.95, 0.8, 0]),
            color=DEFAULT_COLOR,
            stroke_width=3
        )
        
        right_support = Line(
            start=np.array([0.95, 0.5, 0]),
            end=np.array([0.95, 0.8, 0]),
            color=DEFAULT_COLOR,
            stroke_width=3
        )
        
        # Group everything
        scale = VGroup(post, beam, left_pan, right_pan, left_support, right_support)
        return scale
    
    def construct(self):
        """Main animation sequence."""
        
        # Scene 1: Introduce the Problem
        self.next_section("Scene 1: Introduce Problem", skip_animations=False)
        
        # Title
        title = Text("Solving a Linear Equation", font_size=40, weight=BOLD, color=DEFAULT_COLOR)
        self.place_in_box(title, SCENE_BOXES["title"])
        
        # Initial equation
        equation = MathTex(r"x - 2 = 9", color=DEFAULT_COLOR)
        self.place_in_box(equation, SCENE_BOXES["equation_main"])
        
        # Balance scale
        balance_scale = self.create_balance_scale()
        self.place_in_box(balance_scale, SCENE_BOXES["balance"], allow_upscale=True)
        
        # Final safety check for Scene 1
        blockers = [title]
        equation = self.resolve_overlap(equation, blockers, SCENE_BOXES["equation_main"])
        
        self.play(Write(title), run_time=0.8)
        self.wait(0.2)
        self.play(Write(equation), run_time=1.0)
        self.wait(0.2)
        self.play(FadeIn(balance_scale), run_time=1.0)
        self.wait(0.5)
        
        # Scene 2: Perform the Inverse Operation
        self.next_section("Scene 2: Perform Inverse Operation", skip_animations=False)
        
        # Create operation annotations
        plus_two_left = MathTex(r"+2", color=OPERATION_COLOR)
        plus_two_right = MathTex(r"+2", color=OPERATION_COLOR)
        
        # Position annotations below equation
        plus_two_left.next_to(equation.get_left(), DOWN, buff=0.2).shift(RIGHT * 0.5)
        plus_two_right.next_to(equation.get_right(), DOWN, buff=0.2).shift(LEFT * 0.5)
        
        # Create expanded equation in separate box
        expanded_eq = MathTex(r"x - 2 + 2 = 9 + 2", color=DEFAULT_COLOR)
        self.place_in_box(expanded_eq, SCENE_BOXES["equation_step"])
        
        # Arrow between steps
        arrow = MathTex(r"\Rightarrow", color=DEFAULT_COLOR, font_size=36)
        arrow.move_to([0, 0.7, 0])
        
        # Show operation annotations
        self.play(
            FadeIn(plus_two_left, shift=UP*0.1),
            FadeIn(plus_two_right, shift=UP*0.1),
            run_time=0.8
        )
        self.wait(0.3)
        
        # Animate balance scale
        self.play(
            balance_scale.rotate, 0.08, about_point=balance_scale.get_center(),
            run_time=0.3
        )
        self.play(
            balance_scale.rotate, -0.08, about_point=balance_scale.get_center(),
            run_time=0.3
        )
        
        # Show transition
        self.play(Write(arrow), run_time=0.5)
        self.play(
            ReplacementTransform(equation.copy(), expanded_eq),
            FadeOut(plus_two_left),
            FadeOut(plus_two_right),
            run_time=1.2
        )
        self.play(FadeOut(arrow), run_time=0.3)
        self.wait(0.5)
        
        # Scene 3: Simplify and Present Solution
        self.next_section("Scene 3: Simplify and Present Solution", skip_animations=False)
        
        # Create simplified equation
        simplified_eq = MathTex(r"x = 11", color=DEFAULT_COLOR)
        self.place_in_box(simplified_eq, SCENE_BOXES["equation_main"])
        
        # Create box around final answer
        answer_box = SurroundingRectangle(
            simplified_eq, 
            color=FINAL_BOX_COLOR, 
            buff=0.2, 
            stroke_width=3,
            corner_radius=0.1
        )
        
        # Move answer to final answer box
        final_answer_group = VGroup(simplified_eq, answer_box)
        self.place_in_box(final_answer_group, SCENE_BOXES["final_answer"])
        
        # Animate simplification
        self.play(
            FadeOut(expanded_eq, shift=UP*0.1),
            FadeIn(simplified_eq, shift=UP*0.1),
            run_time=1.0
        )
        self.wait(0.3)
        self.play(Create(answer_box), run_time=0.8)
        self.wait(0.5)
        
        # Scene 4: Verify the Solution
        self.next_section("Scene 4: Verify the Solution", skip_animations=False)
        
        # Bring back original equation
        original_eq = MathTex(r"x - 2 = 9", color=DEFAULT_COLOR)
        self.place_in_box(original_eq, SCENE_BOXES["verify"])
        
        # Create substitution equation
        substitution_eq = MathTex(r"11 - 2 = 9", color=DEFAULT_COLOR)
        self.place_in_box(substitution_eq, SCENE_BOXES["verify"])
        
        # Create checkmark
        checkmark = Text("✓", color=VERIFY_COLOR, font_size=48)
        checkmark.next_to(substitution_eq, RIGHT, buff=0.3)
        
        # Final safety check for verification scene
        verify_blockers = []
        checkmark = self.resolve_overlap(checkmark, verify_blockers, SCENE_BOXES["verify"])
        
        # Show verification
        self.play(FadeIn(original_eq, shift=UP*0.1), run_time=0.8)
        self.wait(0.3)
        
        self.play(
            ReplacementTransform(original_eq, substitution_eq),
            run_time=1.2
        )
        self.wait(0.3)
        
        self.play(GrowFromCenter(checkmark), run_time=0.8)
        self.wait(0.5)
        
        # Final hold with all elements
        self.play(
            final_answer_group.animate.set_opacity(0.7),
            balance_scale.animate.set_opacity(0.7),
            run_time=0.5
        )
        self.wait(2.0)
