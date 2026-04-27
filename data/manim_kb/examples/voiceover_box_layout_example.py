from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.coqui import CoquiService
from pydub import AudioSegment
import imageio_ffmpeg
import numpy as np

AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()


def layout_box(x, y, width, height):
    return {
        "center": np.array([x, y, 0.0]),
        "width": width,
        "height": height,
    }


TITLE_BOX = layout_box(0.0, 3.1, 11.5, 0.5)
FORMULA_BOX = layout_box(0.0, 1.5, 11.0, 0.9)
GRAPH_BOX = layout_box(0.0, -0.8, 10.5, 3.5)


class ExampleBoxLayoutScene(VoiceoverScene):
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

    def place_in_box(self, mob, box, pad_x=0.10, pad_y=0.08):
        self.fit_to_box(mob, box, pad_x=pad_x, pad_y=pad_y)
        mob.move_to(box["center"])
        return self.keep_inside_box(mob, box, pad_x=pad_x, pad_y=pad_y)

    def construct(self):
        self.set_speech_service(CoquiService(model_name="tts_models/en/vctk/vits", speaker_idx=7))
        self.next_section("intro")

        title = Text("Example", font_size=38, weight=BOLD)
        formula = MathTex(r"y=x^2+1")
        axes = Axes(x_range=[-3, 3, 1], y_range=[0, 6, 1], x_length=GRAPH_BOX["width"], y_length=GRAPH_BOX["height"], tips=False)
        graph = axes.plot(lambda x: x**2 + 1, x_range=[-2.5, 2.5], color=YELLOW)

        self.place_in_box(title, TITLE_BOX)
        self.place_in_box(formula, FORMULA_BOX)
        axes.move_to(GRAPH_BOX["center"])

        with self.voiceover(text="Demonstrate a narrated scene with fixed layout boxes and a graph.") as tracker:
            self.play(Write(title), FadeIn(formula))
            self.play(Create(axes), Create(graph))
