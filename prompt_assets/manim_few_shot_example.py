from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.coqui import CoquiService
from pydub import AudioSegment
import imageio_ffmpeg
import numpy as np

AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

SIGNAL_COLOR = BLUE
STEP_COLOR = GREEN
SHIFT_COLOR = YELLOW
SYSTEM_COLOR = WHITE
CONV_COLOR = RED


def layout_box(x, y, width, height):
    return {
        "center": np.array([x, y, 0.0]),
        "width": width,
        "height": height,
    }


SCENE1_BOXES = {
    "title": layout_box(0.0, 3.1, 11.5, 0.45),
    "subtitle": layout_box(0.0, 2.4, 11.5, 0.40),
    "formula": layout_box(0.0, 1.7, 11.0, 0.95),
    "graph": layout_box(0.0, -0.6, 10.4, 3.6),
    "callout_right": layout_box(4.1, 0.8, 2.0, 0.9),
    "final_answer": layout_box(0.0, -2.8, 11.0, 0.7),
}


class BoxLayoutScene(VoiceoverScene):
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

    def place_in_box(
        self,
        mob,
        box,
        pad_x=0.10,
        pad_y=0.08,
        allow_upscale=False,
    ):
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

        title = Text("Example Title", font_size=40, weight=BOLD)
        subtitle = Text("Subtitle", font_size=26, color=SYSTEM_COLOR)
        formula = MathTex(r"y(t)=2u(t+2)-3u(t+1)+u(t-2)")
        formula_next = MathTex(r"y^2(t)=4")

        self.place_in_box(title, SCENE1_BOXES["title"])
        self.place_in_box(subtitle, SCENE1_BOXES["subtitle"])
        self.place_in_box(formula, SCENE1_BOXES["formula"])

        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 4, 1],
            x_length=SCENE1_BOXES["graph"]["width"],
            y_length=SCENE1_BOXES["graph"]["height"],
            tips=False,
        ).move_to(SCENE1_BOXES["graph"]["center"])

        callout = Text("Important note", font_size=20, color=YELLOW)
        self.place_in_box(callout, SCENE1_BOXES["callout_right"])

        result = Text("Energy = 7", font_size=34, color=WHITE)
        self.place_in_box(result, SCENE1_BOXES["final_answer"])

        with self.voiceover(text="Demonstrate the box-based layout system.") as tracker:
            self.play(Write(title), FadeIn(subtitle), FadeIn(formula))
            self.play(self.replace_in_box(formula, formula_next, SCENE1_BOXES["formula"]))
            self.play(Create(axes), FadeIn(callout), FadeIn(result))
