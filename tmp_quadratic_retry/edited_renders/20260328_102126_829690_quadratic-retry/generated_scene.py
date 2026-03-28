# XplainAI Manim runtime compatibility aliases
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

from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.coqui import CoquiService
import numpy as np

# Color scheme as per Scene Planner
EQUATION_COLOR = WHITE
COEFF_COLOR = GREEN
DISCRIMINANT_COLOR = RED
QUADRATIC_FORMULA_COLOR = YELLOW
ROOTS_COLOR = CYAN
PARABOLA_COLOR = BLUE
DOT_COLOR = CYAN

def layout_box(x, y, width, height):
    return {
        "center": np.array([x, y, 0.0]),
        "width": width,
        "height": height,
    }

# Scene 1 boxes
SCENE1_BOXES = {
    "title": layout_box(0.0, 3.4, 12.0, 0.6),
    "subtitle": layout_box(0.0, 2.6, 12.0, 0.5),
    "formula_left": layout_box(-2.5, 0.5, 5.0, 2.5),
    "graph_right": layout_box(3.5, 0.0, 5.0, 4.0),
    "coefficient_labels": layout_box(-2.5, -1.5, 5.0, 0.8),
}

# Scene 2 boxes
SCENE2_BOXES = {
    "title": layout_box(0.0, 3.4, 12.0, 0.6),
    "standard_form": layout_box(-3.0, 1.8, 6.0, 0.8),
    "discriminant_formula": layout_box(-3.0, 0.5, 6.0, 1.0),
    "callout_right": layout_box(3.0, 0.5, 6.0, 3.0),
}

# Scene 3 boxes
SCENE3_BOXES = {
    "title": layout_box(0.0, 3.4, 12.0, 0.6),
    "standard_form_top": layout_box(0.0, 2.0, 10.0, 0.8),
    "quadratic_formula_main": layout_box(0.0, 0.0, 10.0, 1.5),
    "compact_form": layout_box(0.0, -1.5, 10.0, 1.0),
}

# Scene 4 boxes
SCENE4_BOXES = {
    "title": layout_box(0.0, 3.4, 12.0, 0.6),
    "quadratic_formula": layout_box(-2.5, 1.0, 5.0, 1.2),
    "roots_statement": layout_box(-2.5, -0.5, 5.0, 0.8),
    "graph_right": layout_box(3.5, 0.0, 5.0, 4.0),
}

class QuadraticEquationAnimation(VoiceoverScene):
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

    def create_parabola_with_roots(self, axes, a=1, b=0, c=-1):
        # Generic upward-opening parabola with two real roots
        parabola = axes.plot(
            lambda x: a * x**2 + b * x + c,
            color=PARABOLA_COLOR,
            stroke_width=3
        )
        root1_dot = Dot(axes.c2p(-1, 0), color=DOT_COLOR, radius=0.08)
        root2_dot = Dot(axes.c2p(1, 0), color=DOT_COLOR, radius=0.08)
        root1_label = MathTex("x_1", color=ROOTS_COLOR, font_size=24).next_to(root1_dot, DOWN)
        root2_label = MathTex("x_2", color=ROOTS_COLOR, font_size=24).next_to(root2_dot, DOWN)
        roots_label = Text("Roots", color=ROOTS_COLOR, font_size=28).next_to(
            VGroup(root1_label, root2_label), UP, buff=0.3
        )
        return VGroup(parabola, root1_dot, root2_dot, root1_label, root2_label, roots_label)

    def construct(self):
        self.set_speech_service(
            CoquiService(model_name="tts_models/en/vctk/vits", speaker_idx=7)
        )

        # Scene 1: Introduction to Quadratic Equation
        self.next_section("Scene 1: Introduction", skip_animations=False)
        
        title1 = Text("Quadratic Equations", font_size=48, weight=BOLD, color=EQUATION_COLOR)
        subtitle1 = Text("Standard Form", font_size=32, color=EQUATION_COLOR)
        self.place_in_box(title1, SCENE1_BOXES["title"])
        self.place_in_box(subtitle1, SCENE1_BOXES["subtitle"])
        
        # Main equation with colored coefficients
        standard_form = MathTex("a", "x^2", "+", "b", "x", "+", "c", "=", "0")
        standard_form.set_color_by_tex("a", COEFF_COLOR)
        standard_form.set_color_by_tex("b", COEFF_COLOR)
        standard_form.set_color_by_tex("c", COEFF_COLOR)
        self.place_in_box(standard_form, SCENE1_BOXES["formula_left"])
        
        # Coefficient labels
        a_label = MathTex(r"a \ (\neq 0)", color=COEFF_COLOR, font_size=28)
        b_label = MathTex(r"b", color=COEFF_COLOR, font_size=28)
        c_label = MathTex(r"c", color=COEFF_COLOR, font_size=28)
        labels = VGroup(a_label, b_label, c_label).arrange(RIGHT, buff=1.5)
        labels.move_to(SCENE1_BOXES["coefficient_labels"]["center"])
        
        # Graph setup
        axes = Axes(
            x_range=[-2.5, 2.5, 1],
            y_range=[-2, 4, 1],
            x_length=SCENE1_BOXES["graph_right"]["width"] * 0.9,
            y_length=SCENE1_BOXES["graph_right"]["height"] * 0.9,
            tips=False,
            axis_config={"color": WHITE}
        )
        self.place_in_box(axes, SCENE1_BOXES["graph_right"])
        graph_group = self.create_parabola_with_roots(axes)
        
        # Animation for Scene 1
        with self.voiceover(text="A quadratic equation is a second-degree polynomial set to zero.") as tracker:
            self.play(FadeIn(title1), FadeIn(subtitle1))
        with self.voiceover(text="Its standard form is a x squared plus b x plus c equals zero, where a, b, and c are coefficients, and a cannot be zero.") as tracker:
            self.play(Write(standard_form))
            self.play(FadeIn(labels))
            a_highlight = SurroundingRectangle(standard_form[0], color=COEFF_COLOR, buff=0.05)
            self.play(Create(a_highlight))
            self.play(FadeOut(a_highlight))
        with self.voiceover(text="The graph of a quadratic is a parabola. Its x-intercepts are called roots.") as tracker:
            self.play(Create(axes))
            self.play(Create(graph_group[0]))  # parabola
            self.play(LaggedStart(
                FadeIn(graph_group[1]), FadeIn(graph_group[2]),
                FadeIn(graph_group[3]), FadeIn(graph_group[4]),
                FadeIn(graph_group[5]),
                lag_ratio=0.3
            ))
        
        self.wait(0.5)

        # Scene 2: The Discriminant
        self.next_section("Scene 2: Discriminant", skip_animations=False)
        
        # Fade out graph and prepare for Scene 2
        self.play(FadeOut(graph_group), FadeOut(axes))
        
        title2 = Text("The Discriminant", font_size=48, weight=BOLD, color=EQUATION_COLOR)
        self.place_in_box(title2, SCENE2_BOXES["title"])
        
        # Keep standard form from Scene 1
        self.play(self.fade_swap(title1, title2))
        self.play(self.fade_swap(subtitle1, VGroup()))  # Remove subtitle
        
        # Move standard form to its new position
        self.play(standard_form.animate.move_to(SCENE2_BOXES["standard_form"]["center"]))
        
        # Discriminant formula
        discriminant_formula = MathTex(r"\Delta", "=", "b^2", "-", "4ac")
        discriminant_formula.set_color_by_tex(r"\Delta", DISCRIMINANT_COLOR)
        discriminant_formula.set_color_by_tex("b^2", DISCRIMINANT_COLOR)
        discriminant_formula.set_color_by_tex("4ac", DISCRIMINANT_COLOR)
        discriminant_label = Text("Discriminant", color=DISCRIMINANT_COLOR, font_size=32)
        discriminant_group = VGroup(discriminant_label, discriminant_formula).arrange(DOWN, buff=0.3)
        self.place_in_box(discriminant_group, SCENE2_BOXES["discriminant_formula"])
        
        # Callout for discriminant outcomes
        case1 = MathTex(r"\Delta > 0 : \text{Two distinct real roots}", color=WHITE, font_size=24)
        case2 = MathTex(r"\Delta = 0 : \text{One repeated real root}", color=WHITE, font_size=24)
        case3 = MathTex(r"\Delta < 0 : \text{Two complex roots}", color=WHITE, font_size=24)
        callout_group = VGroup(case1, case2, case3).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        self.place_in_box(callout_group, SCENE2_BOXES["callout_right"])
        
        # Mini graphs for each case
        mini_axes_config = {
            "x_range": [-2, 2, 1], "y_range": [-1, 3, 1], 
            "width": 1.5, "height": 1.0, "tips": False,
            "axis_config": {"stroke_width": 2, "color": WHITE}
        }
        
        # Case 1: Δ > 0
        ax1 = Axes(**mini_axes_config)
        ax1.move_to(case1.get_right() + RIGHT * 1.2)
        parabola1 = ax1.plot(lambda x: x**2 - 1, color=PARABOLA_COLOR, stroke_width=2)
        dot1a = Dot(ax1.c2p(-1, 0), color=DOT_COLOR, radius=0.05)
        dot1b = Dot(ax1.c2p(1, 0), color=DOT_COLOR, radius=0.05)
        
        # Case 2: Δ = 0
        ax2 = Axes(**mini_axes_config)
        ax2.move_to(case2.get_right() + RIGHT * 1.2)
        parabola2 = ax2.plot(lambda x: x**2, color=PARABOLA_COLOR, stroke_width=2)
        dot2 = Dot(ax2.c2p(0, 0), color=DOT_COLOR, radius=0.05)
        
        # Case 3: Δ < 0
        ax3 = Axes(**mini_axes_config)
        ax3.move_to(case3.get_right() + RIGHT * 1.2)
        parabola3 = ax3.plot(lambda x: x**2 + 1, color=PARABOLA_COLOR, stroke_width=2)
        
        # Animation for Scene 2
        with self.voiceover(text="The discriminant, Delta, equals b squared minus 4 a c.") as tracker:
            self.play(FadeIn(discriminant_group))
            b2_highlight = SurroundingRectangle(discriminant_formula[2], color=DISCRIMINANT_COLOR, buff=0.05)
            ac_highlight = SurroundingRectangle(discriminant_formula[4], color=DISCRIMINANT_COLOR, buff=0.05)
            self.play(Create(b2_highlight), Create(ac_highlight))
            self.play(FadeOut(b2_highlight), FadeOut(ac_highlight))
        
        with self.voiceover(text="Its value determines the nature of the roots.") as tracker:
            self.play(FadeIn(callout_group))
            self.play(Create(ax1), Create(parabola1), FadeIn(dot1a), FadeIn(dot1b))
            self.wait(0.3)
            self.play(Create(ax2), Create(parabola2), FadeIn(dot2))
            self.wait(0.3)
            self.play(Create(ax3), Create(parabola3))
        
        self.wait(0.5)
        
        # Clean up mini graphs
        self.play(FadeOut(ax1), FadeOut(parabola1), FadeOut(dot1a), FadeOut(dot1b),
                  FadeOut(ax2), FadeOut(parabola2), FadeOut(dot2),
                  FadeOut(ax3), FadeOut(parabola3))

        # Scene 3: Quadratic Formula
        self.next_section("Scene 3: Quadratic Formula", skip_animations=False)
        
        title3 = Text("Quadratic Formula", font_size=48, weight=BOLD, color=EQUATION_COLOR)
        self.place_in_box(title3, SCENE3_BOXES["title"])
        
        # Move standard form to top
        self.play(self.fade_swap(title2, title3))
        self.play(standard_form.animate.move_to(SCENE3_BOXES["standard_form_top"]["center"]))
        self.play(FadeOut(discriminant_group), FadeOut(callout_group))
        
        # Quadratic formula
        quadratic_formula_full = MathTex(
            "x", "=", r"{-b", r"\pm", r"\sqrt{", "b^2", "-", "4ac", "}", r"\over", "2a}"
        )
        quadratic_formula_full.set_color_by_tex(r"\pm", QUADRATIC_FORMULA_COLOR)
        quadratic_formula_full.set_color_by_tex("b^2", QUADRATIC_FORMULA_COLOR)
        quadratic_formula_full.set_color_by_tex("4ac", QUADRATIC_FORMULA_COLOR)
        quadratic_formula_full.set_color_by_tex("2a}", QUADRATIC_FORMULA_COLOR)
        self.place_in_box(quadratic_formula_full, SCENE3_BOXES["quadratic_formula_main"])
        
        # Compact form with Δ
        quadratic_formula_compact = MathTex(
            "x", "=", r"{-b", r"\pm", r"\sqrt{", r"\Delta", "}", r"\over", "2a}"
        )
        quadratic_formula_compact.set_color_by_tex(r"\pm", QUADRATIC_FORMULA_COLOR)
        quadratic_formula_compact.set_color_by_tex(r"\Delta", DISCRIMINANT_COLOR)
        quadratic_formula_compact.set_color_by_tex("2a}", QUADRATIC_FORMULA_COLOR)
        self.place_in_box(quadratic_formula_compact, SCENE3_BOXES["compact_form"])
        
        # Animation for Scene 3
        with self.voiceover(text="The quadratic formula gives the roots directly.") as tracker:
            self.play(Write(quadratic_formula_full))
            highlight_box = SurroundingRectangle(standard_form, color=YELLOW, buff=0.1)
            self.play(Create(highlight_box))
            self.play(FadeOut(highlight_box))
        
        with self.voiceover(text="The plus-minus symbol accounts for both solutions, and the discriminant sits inside the square root.") as tracker:
            self.play(quadratic_formula_full[5:9].animate.set_color(DISCRIMINANT_COLOR))
            self.wait(0.3)
            delta_symbol = MathTex(r"\Delta", color=DISCRIMINANT_COLOR, font_size=36)
            delta_symbol.move_to(quadratic_formula_full[5:9].get_center())
            self.play(ReplacementTransform(quadratic_formula_full[5:9], delta_symbol))
            self.play(FadeIn(quadratic_formula_compact))
        
        self.wait(0.5)
        
        # Scene 4: Conclusion
        self.next_section("Scene 4: Conclusion", skip_animations=False)
        
        title4 = Text("Roots of a Quadratic", font_size=48, weight=BOLD, color=EQUATION_COLOR)
        self.place_in_box(title4, SCENE4_BOXES["title"])
        
        # Move quadratic formula to left
        self.play(self.fade_swap(title3, title4))
        self.play(FadeOut(standard_form))
        
        # Keep compact form
        self.play(quadratic_formula_compact.animate.move_to(SCENE4_BOXES["quadratic_formula"]["center"]))
        self.play(FadeOut(quadratic_formula_full), FadeOut(delta_symbol))
        
        # Roots statement
        roots_statement = MathTex(r"\text{Roots: }", "x_1", r"\text{ and }", "x_2", color=ROOTS_COLOR)
        roots_statement.set_color_by_tex("x_1", ROOTS_COLOR)
        roots_statement.set_color_by_tex("x_2", ROOTS_COLOR)
        self.place_in_box(roots_statement, SCENE4_BOXES["roots_statement"])
        
        # Bring back graph
        self.place_in_box(axes, SCENE4_BOXES["graph_right"])
        graph_group = self.create_parabola_with_roots(axes)
        
        # Animation for Scene 4
        with self.voiceover(text="These are the roots. They solve the equation and are precisely the x-intercepts of the parabola.") as tracker:
            self.play(FadeIn(roots_statement))
            self.play(FadeIn(axes), FadeIn(graph_group))
            
            # Create arrows from roots statement to graph dots
            arrow1 = Arrow(
                roots_statement[1].get_right(),
                graph_group[1].get_left(),
                color=ROOTS_COLOR,
                stroke_width=3,
                buff=0.1
            )
            arrow2 = Arrow(
                roots_statement[3].get_right(),
                graph_group[2].get_left(),
                color=ROOTS_COLOR,
                stroke_width=3,
                buff=0.1
            )
            self.play(Create(arrow1), Create(arrow2))
            
            # Highlight final answer
            final_highlight = SurroundingRectangle(
                VGroup(quadratic_formula_compact, roots_statement),
                color=ROOTS_COLOR,
                buff=0.2,
                stroke_width=3
            )
            self.play(Create(final_highlight))
        
        self.wait(1.0)
        
        # Final fade out
        self.play(*[FadeOut(mob) for mob in self.mobjects])
