from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from hashlib import md5
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import fitz
import imageio_ffmpeg
import streamlit as st
from PIL import Image

from src.parser_replicate_vl2 import ReplicateDeepSeekVL2Parser
from src.reasoner import SolutionOrchestrator


PROJECT_ROOT = Path(__file__).resolve().parent
STREAMLIT_RUNS_DIR = PROJECT_ROOT / "outputs" / "streamlit_runs"
STREAMLIT_RUNS_DIR.mkdir(parents=True, exist_ok=True)

PARSER_BACKENDS = {
    "Replicate API": "replicate",
    "Local DeepSeek-VL2 Tiny": "local",
}

RENDER_QUALITIES = {
    "Low (fastest)": "l",
    "Medium": "m",
    "High": "h",
}


@st.cache_resource(show_spinner=False)
def get_orchestrator() -> SolutionOrchestrator:
    return SolutionOrchestrator("config/config.yaml")


@st.cache_resource(show_spinner=False)
def get_replicate_parser() -> ReplicateDeepSeekVL2Parser:
    return ReplicateDeepSeekVL2Parser("config/config.yaml")


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return cleaned.strip("-") or "run"


def make_run_dir(label: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = md5(f"{stamp}:{label}".encode("utf-8")).hexdigest()[:8]
    run_dir = STREAMLIT_RUNS_DIR / f"{stamp}_{slugify(label)}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def make_render_dir(run_dir: Path, label: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    render_dir = run_dir / "edited_renders" / f"{stamp}_{slugify(label)}"
    render_dir.mkdir(parents=True, exist_ok=True)
    return render_dir


def save_uploaded_file(uploaded_file: Any, run_dir: Path) -> Path:
    target = run_dir / uploaded_file.name
    target.write_bytes(uploaded_file.getbuffer())
    return target


def persist_bundle(output: Dict[str, Any]) -> None:
    bundle_path = Path(output["bundle_path"])
    bundle_path.write_text(
        json.dumps(output["pipeline_result"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def persist_current_code(output: Dict[str, Any], code: str) -> Path:
    code = strip_manim_runtime_compatibility(code)
    current_code_path = Path(output["current_code_path"])
    current_code_path.write_text(code.rstrip() + "\n", encoding="utf-8")
    output["pipeline_result"].setdefault("manim_code", {})["text"] = code
    persist_bundle(output)
    return current_code_path


def render_pdf_preview(uploaded_file: Any) -> Optional[Image.Image]:
    try:
        document = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
        if document.page_count == 0:
            return None
        page = document.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    except Exception:
        return None


def detect_scene_classes(code: str) -> List[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        pattern = re.compile(r"^class\s+(\w+)\(([^)]*Scene[^)]*)\)\s*:", flags=re.MULTILINE)
        return [match.group(1) for match in pattern.finditer(code)]

    classes: List[tuple[str, bool]] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        base_names: List[str] = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_names.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_names.append(base.attr)

        if not any(name.endswith("Scene") for name in base_names):
            continue

        has_construct = any(
            isinstance(item, ast.FunctionDef) and item.name == "construct"
            for item in node.body
        )
        classes.append((node.name, has_construct))

    if not classes:
        return []

    concrete = [name for name, has_construct in classes if has_construct]
    return concrete or [name for name, _ in classes]


def compile_python_script(script_path: Path) -> None:
    compile_cmd = [sys.executable, "-m", "py_compile", str(script_path)]
    compile_result = subprocess.run(
        compile_cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    if compile_result.returncode != 0:
        raise RuntimeError(
            "Generated Manim code did not compile.\n"
            + (compile_result.stderr or compile_result.stdout or "Unknown py_compile error.")
        )


def build_manim_compatibility_preamble() -> str:
    color_aliases = {
        "CYAN": "#00BCD4",
        "AQUA": "#00BCD4",
        "FUCHSIA": "#D147BD",
    }
    lines = [
        "# XplainAI Manim runtime compatibility aliases",
        "try:",
        "    from pydub import AudioSegment as _XplainAIAudioSegment",
        "    import imageio_ffmpeg as _xplainai_imageio_ffmpeg",
        "    _XplainAIAudioSegment.converter = _xplainai_imageio_ffmpeg.get_ffmpeg_exe()",
        "except Exception:",
        "    pass",
        "try:",
        "    _XplainAIOriginalAxes = Axes",
        "    def Axes(*args, **kwargs):",
        "        if 'width' in kwargs and 'x_length' not in kwargs:",
        "            kwargs['x_length'] = kwargs.pop('width')",
        "        if 'height' in kwargs and 'y_length' not in kwargs:",
        "            kwargs['y_length'] = kwargs.pop('height')",
        "        return _XplainAIOriginalAxes(*args, **kwargs)",
        "except Exception:",
        "    pass",
        "try:",
        "    _XplainAIOriginalNumberPlane = NumberPlane",
        "    def NumberPlane(*args, **kwargs):",
        "        if 'width' in kwargs and 'x_length' not in kwargs:",
        "            kwargs['x_length'] = kwargs.pop('width')",
        "        if 'height' in kwargs and 'y_length' not in kwargs:",
        "            kwargs['y_length'] = kwargs.pop('height')",
        "        return _XplainAIOriginalNumberPlane(*args, **kwargs)",
        "except Exception:",
        "    pass",
        "try:",
        "    import inspect as _xplainai_inspect",
        "    _XplainAIOriginalScenePlay = Scene.play",
        "    _XPLAINAI_PLAY_KWARGS = {'run_time', 'rate_func', 'lag_ratio', 'subcaption', 'subcaption_duration', 'subcaption_offset'}",
        "    def _xplainai_scene_play_compat(self, *args, **kwargs):",
        "        if args and _xplainai_inspect.ismethod(args[0]):",
        "            _method = args[0]",
        "            _target = getattr(_method, '__self__', None)",
        "            _name = getattr(_method, '__name__', '')",
        "            if _target is not None and _name and hasattr(_target, 'animate'):",
        "                _play_kwargs = {k: v for k, v in kwargs.items() if k in _XPLAINAI_PLAY_KWARGS}",
        "                _method_kwargs = {k: v for k, v in kwargs.items() if k not in _XPLAINAI_PLAY_KWARGS}",
        "                _builder = getattr(_target.animate, _name)(*args[1:], **_method_kwargs)",
        "                return _XplainAIOriginalScenePlay(self, _builder, **_play_kwargs)",
        "        return _XplainAIOriginalScenePlay(self, *args, **kwargs)",
        "    Scene.play = _xplainai_scene_play_compat",
        "except Exception:",
        "    pass",
        "try:",
        "    import re as _xplainai_re",
        "    _XplainAIOriginalSceneNextSection = Scene.next_section",
        "    def _xplainai_sanitize_section_name(name):",
        "        if not isinstance(name, str):",
        "            return name",
        "        cleaned = _xplainai_re.sub(r'[<>:\"/\\\\|?*]+', ' - ', name)",
        "        cleaned = _xplainai_re.sub(r'\\s+', ' ', cleaned).strip()",
        "        return cleaned or 'section'",
        "    def _xplainai_scene_next_section_compat(self, name='unnamed', *args, **kwargs):",
        "        return _XplainAIOriginalSceneNextSection(self, _xplainai_sanitize_section_name(name), *args, **kwargs)",
        "    Scene.next_section = _xplainai_scene_next_section_compat",
        "except Exception:",
        "    pass",
        "try:",
        "    _XplainAIOriginalMathTexGetPartByTex = MathTex.get_part_by_tex",
        "    def _xplainai_safe_get_part_by_tex(self, *args, **kwargs):",
        "        _part = _XplainAIOriginalMathTexGetPartByTex(self, *args, **kwargs)",
        "        return _part if _part is not None else self",
        "    MathTex.get_part_by_tex = _xplainai_safe_get_part_by_tex",
        "except Exception:",
        "    pass",
        "try:",
        "    _XplainAIOriginalTexGetPartByTex = Tex.get_part_by_tex",
        "    def _xplainai_safe_tex_get_part_by_tex(self, *args, **kwargs):",
        "        _part = _XplainAIOriginalTexGetPartByTex(self, *args, **kwargs)",
        "        return _part if _part is not None else self",
        "    Tex.get_part_by_tex = _xplainai_safe_tex_get_part_by_tex",
        "except Exception:",
        "    pass",
        "try:",
        "    BoxLayoutScene",
        "except NameError:",
        "    try:",
        "        _XplainAIBaseLayoutScene = VoiceoverScene",
        "    except NameError:",
        "        _XplainAIBaseLayoutScene = Scene",
        "    class BoxLayoutScene(_XplainAIBaseLayoutScene):",
        "        def p(self, x, y):",
        "            return np.array([x, y, 0.0])",
        "        def fit_to_box(self, mob, box, pad_x=0.10, pad_y=0.08, allow_upscale=False):",
        "            avail_width = max(0.2, box['width'] - 2 * pad_x)",
        "            avail_height = max(0.2, box['height'] - 2 * pad_y)",
        "            scales = []",
        "            if getattr(mob, 'width', 0) > 0:",
        "                scales.append(avail_width / mob.width)",
        "            if getattr(mob, 'height', 0) > 0:",
        "                scales.append(avail_height / mob.height)",
        "            if not scales:",
        "                return mob",
        "            scale = min(scales)",
        "            if not allow_upscale:",
        "                scale = min(scale, 1.0)",
        "            mob.scale(scale)",
        "            return mob",
        "        def keep_inside_box(self, mob, box, pad_x=0.10, pad_y=0.08):",
        "            left_limit = box['center'][0] - box['width'] / 2 + pad_x",
        "            right_limit = box['center'][0] + box['width'] / 2 - pad_x",
        "            bottom_limit = box['center'][1] - box['height'] / 2 + pad_y",
        "            top_limit = box['center'][1] + box['height'] / 2 - pad_y",
        "            dx = 0.0",
        "            dy = 0.0",
        "            if mob.get_left()[0] < left_limit:",
        "                dx = left_limit - mob.get_left()[0]",
        "            elif mob.get_right()[0] > right_limit:",
        "                dx = right_limit - mob.get_right()[0]",
        "            if mob.get_bottom()[1] < bottom_limit:",
        "                dy = bottom_limit - mob.get_bottom()[1]",
        "            elif mob.get_top()[1] > top_limit:",
        "                dy = top_limit - mob.get_top()[1]",
        "            if abs(dx) > 1e-6 or abs(dy) > 1e-6:",
        "                mob.shift(np.array([dx, dy, 0.0]))",
        "            return mob",
        "        def place_in_box(self, mob, box, pad_x=0.10, pad_y=0.08, allow_upscale=False):",
        "            self.fit_to_box(mob, box, pad_x=pad_x, pad_y=pad_y, allow_upscale=allow_upscale)",
        "            mob.move_to(box['center'])",
        "            return self.keep_inside_box(mob, box, pad_x=pad_x, pad_y=pad_y)",
        "        def mobjects_overlap(self, mob_a, mob_b, gap=0.06):",
        "            x_overlap = min(mob_a.get_right()[0], mob_b.get_right()[0]) - max(mob_a.get_left()[0], mob_b.get_left()[0])",
        "            y_overlap = min(mob_a.get_top()[1], mob_b.get_top()[1]) - max(mob_a.get_bottom()[1], mob_b.get_bottom()[1])",
        "            return x_overlap > -gap and y_overlap > -gap",
        "        def resolve_overlap(self, mob, blockers, box, gap=0.08, step=0.08):",
        "            mob = self.keep_inside_box(mob, box, pad_x=gap, pad_y=gap)",
        "            directions = [DOWN, UP, RIGHT, LEFT]",
        "            for _ in range(20):",
        "                active = [other for other in blockers if self.mobjects_overlap(mob, other, gap=gap)]",
        "                if not active:",
        "                    break",
        "                moved = False",
        "                for direction in directions:",
        "                    trial = mob.copy().shift(direction * step)",
        "                    self.keep_inside_box(trial, box, pad_x=gap, pad_y=gap)",
        "                    if not any(self.mobjects_overlap(trial, other, gap=gap) for other in blockers):",
        "                        mob.move_to(trial.get_center())",
        "                        moved = True",
        "                        break",
        "                if not moved:",
        "                    break",
        "            return mob",
        "        def stack_in_box(self, mobs, box, gap=0.18, pad_x=0.12, pad_y=0.10):",
        "            group = VGroup(*mobs).arrange(DOWN, buff=gap, aligned_edge=LEFT)",
        "            self.fit_to_box(group, box, pad_x=pad_x, pad_y=pad_y)",
        "            group.move_to(box['center'])",
        "            self.keep_inside_box(group, box, pad_x=pad_x, pad_y=pad_y)",
        "            return group",
        "        def replace_in_box(self, old_mob, new_mob, box, pad_x=0.12, pad_y=0.10):",
        "            self.place_in_box(new_mob, box, pad_x=pad_x, pad_y=pad_y)",
        "            return AnimationGroup(FadeOut(old_mob, shift=0.08 * UP), FadeIn(new_mob, shift=0.08 * UP), lag_ratio=0.0)",
        "        def fade_swap(self, old_mob, new_mob, shift=0.08 * UP):",
        "            return AnimationGroup(FadeOut(old_mob, shift=shift), FadeIn(new_mob, shift=shift), lag_ratio=0.0)",
    ]
    for name, value in color_aliases.items():
        lines.extend(
            [
                "try:",
                f"    {name}",
                "except NameError:",
                f"    {name} = {value!r}",
            ]
        )
    return "\n".join(lines) + "\n\n"


def strip_manim_runtime_compatibility(code: str) -> str:
    sanitized = code
    compat_preamble = build_manim_compatibility_preamble()
    while compat_preamble in sanitized:
        sanitized = sanitized.replace(compat_preamble, "", 1)
    return sanitized.lstrip("\ufeff")


def apply_manim_runtime_compatibility(code: str) -> str:
    content = strip_manim_runtime_compatibility(code).rstrip() + "\n"
    preamble = build_manim_compatibility_preamble()

    lines = content.splitlines(keepends=True)
    insert_at = 0

    while insert_at < len(lines) and lines[insert_at].startswith("from __future__ import "):
        insert_at += 1

    while insert_at < len(lines):
        stripped = lines[insert_at].strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            insert_at += 1
            continue
        if stripped == "":
            insert_at += 1
            continue
        break

    return "".join(lines[:insert_at]) + preamble + "".join(lines[insert_at:])


def build_skip_sections_preamble(skip_before_section_index: int) -> str:
    return (
        "import re as _xplainai_re\n"
        "from manim.scene.scene import Scene as _XplainAIScene\n"
        f"_XPLAINAI_SKIP_BEFORE_SECTION_INDEX = {int(skip_before_section_index)}\n"
        "_xplainai_original_next_section = _XplainAIScene.next_section\n"
        "def _xplainai_sanitize_section_name(name):\n"
        "    if not isinstance(name, str):\n"
        "        return name\n"
        "    cleaned = _xplainai_re.sub(r'[<>:\"/\\\\|?*]+', ' - ', name)\n"
        "    cleaned = _xplainai_re.sub(r'\\s+', ' ', cleaned).strip()\n"
        "    return cleaned or 'section'\n"
        "def _xplainai_patched_next_section(self, *args, **kwargs):\n"
        "    section_counter = getattr(self, '_xplainai_section_counter', 0)\n"
        "    self._xplainai_section_counter = section_counter + 1\n"
        "    if args:\n"
        "        args = (_xplainai_sanitize_section_name(args[0]), *args[1:])\n"
        "    elif 'name' in kwargs:\n"
        "        kwargs['name'] = _xplainai_sanitize_section_name(kwargs['name'])\n"
        "    if section_counter < _XPLAINAI_SKIP_BEFORE_SECTION_INDEX:\n"
        "        kwargs['skip_animations'] = True\n"
        "    return _xplainai_original_next_section(self, *args, **kwargs)\n"
        "_XplainAIScene.next_section = _xplainai_patched_next_section\n\n"
    )


def is_valid_video_clip(video_path: Path) -> bool:
    return (
        video_path.exists()
        and video_path.is_file()
        and video_path.suffix.lower() == ".mp4"
        and video_path.stat().st_size > 0
    )


def load_section_assets(
    video_path: Path,
    scene_name: str,
    index_offset: int = 0,
) -> List[Dict[str, Any]]:
    sections_dir = video_path.parent / "sections"
    candidate_paths = [
        sections_dir / f"{video_path.stem}.json",
        sections_dir / f"{scene_name}.json",
    ]
    section_json_path = next((path for path in candidate_paths if path.exists()), None)
    if section_json_path is None:
        return []

    payload = json.loads(section_json_path.read_text(encoding="utf-8"))
    sections: List[Dict[str, Any]] = []
    for local_index, item in enumerate(payload):
        section_video_path = sections_dir / str(item.get("video", "")).strip()
        if not is_valid_video_clip(section_video_path):
            continue
        sections.append(
            {
                "index": index_offset + local_index,
                "name": str(item.get("name") or f"section_{index_offset + local_index}"),
                "duration": str(item.get("duration") or ""),
                "video_path": str(section_video_path),
                "metadata_path": str(section_json_path),
            }
        )
    return sections


def render_manim_video(
    code: str,
    run_dir: Path,
    quality: str,
    *,
    render_label: str = "full-render",
    save_sections: bool = True,
    skip_before_section_index: Optional[int] = None,
) -> Dict[str, Any]:
    render_dir = make_render_dir(run_dir, render_label)
    media_dir = render_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    base_script_path = render_dir / "generated_scene.py"
    compatible_code = apply_manim_runtime_compatibility(code)
    base_script_path.write_text(compatible_code, encoding="utf-8")

    render_script_path = base_script_path
    if skip_before_section_index is not None and skip_before_section_index > 0:
        render_script_path = render_dir / "generated_scene_partial.py"
        render_script_path.write_text(
            build_skip_sections_preamble(skip_before_section_index) + compatible_code,
            encoding="utf-8",
        )

    compile_python_script(render_script_path)

    scene_classes = detect_scene_classes(code)
    if not scene_classes:
        raise RuntimeError("Could not find a Manim scene class in the generated code.")

    scene_name = scene_classes[0]
    render_cmd = [
        sys.executable,
        "-m",
        "manim",
        "render",
        f"-q{quality}",
        "--progress_bar",
        "none",
        "--media_dir",
        str(media_dir),
        "--output_file",
        "visualization",
    ]
    if save_sections:
        render_cmd.append("--save_sections")
    render_cmd.extend([str(render_script_path), scene_name])

    render_result = subprocess.run(
        render_cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=3600,
    )
    if render_result.returncode != 0:
        raise RuntimeError(
            "Manim render failed.\n"
            + (render_result.stderr or render_result.stdout or "Unknown Manim error.")
        )

    videos = sorted(media_dir.rglob("visualization.mp4"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not videos:
        videos = sorted(media_dir.rglob("*.mp4"), key=lambda path: path.stat().st_mtime, reverse=True)
    still_images = sorted(media_dir.rglob("*.png"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not videos and not still_images:
        raise RuntimeError("Manim finished without producing a video or image output.")

    video_path = videos[0] if videos else None
    image_path = still_images[0] if still_images else None
    has_audio = video_has_audio_stream(video_path) if video_path else False
    sections = (
        load_section_assets(
            video_path,
            scene_name,
            index_offset=int(skip_before_section_index or 0),
        )
        if (save_sections and video_path is not None)
        else []
    )

    return {
        "video_path": str(video_path) if video_path else "",
        "image_path": str(image_path) if image_path else "",
        "output_kind": "video" if video_path else "image",
        "has_audio": has_audio,
        "scene_name": scene_name,
        "stdout": render_result.stdout,
        "stderr": render_result.stderr,
        "script_path": str(base_script_path),
        "render_script_path": str(render_script_path),
        "media_dir": str(media_dir),
        "sections": sections,
        "stitched": False,
        "render_label": render_label,
    }


def stitch_videos(video_paths: List[Path], output_path: Path) -> Path:
    valid_paths = [path for path in video_paths if is_valid_video_clip(path)]
    if not valid_paths:
        raise RuntimeError("No valid video clips were available for stitching.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(valid_paths) == 1:
        shutil.copyfile(valid_paths[0], output_path)
        if video_has_audio_stream(valid_paths[0]) and not video_has_audio_stream(output_path):
            raise RuntimeError("Copied stitched output lost its audio track unexpectedly.")
        return output_path

    concat_list_path = output_path.with_suffix(".txt")
    concat_lines = []
    for video_path in valid_paths:
        normalized = video_path.resolve().as_posix().replace("'", "'\\''")
        concat_lines.append(f"file '{normalized}'")
    concat_list_path.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    expected_audio = any(video_has_audio_stream(path) for path in valid_paths)
    concat_cmd = [
        ffmpeg_exe,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list_path),
        "-c",
        "copy",
        str(output_path),
    ]
    concat_result = subprocess.run(
        concat_cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=3600,
    )
    if concat_result.returncode == 0 and (not expected_audio or video_has_audio_stream(output_path)):
        return output_path

    if concat_result.returncode != 0 or (expected_audio and not video_has_audio_stream(output_path)):
        reencode_cmd = [
            ffmpeg_exe,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_path),
        ]
        reencode_result = subprocess.run(
            reencode_cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3600,
        )
        if reencode_result.returncode != 0:
            raise RuntimeError(
                "Could not stitch the rerendered video clips.\n"
                + (reencode_result.stderr or reencode_result.stdout or "Unknown ffmpeg error.")
            )
        if expected_audio and not video_has_audio_stream(output_path):
            raise RuntimeError("Stitched video was created, but the audio track is missing.")

    return output_path


def video_has_audio_stream(video_path: Path) -> bool:
    if not video_path.exists():
        return False
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    probe_result = subprocess.run(
        [ffmpeg, "-i", str(video_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    probe_text = f"{probe_result.stdout}\n{probe_result.stderr}"
    return " Audio:" in probe_text


def merge_section_entries(
    existing_sections: List[Dict[str, Any]],
    updated_sections: List[Dict[str, Any]],
    start_index: int,
) -> List[Dict[str, Any]]:
    prefix = [dict(item) for item in existing_sections if int(item.get("index", -1)) < start_index]
    suffix = [dict(item) for item in updated_sections]
    merged = prefix + suffix
    merged.sort(key=lambda item: int(item.get("index", 0)))
    return merged


def rerender_edited_video(
    output: Dict[str, Any],
    code: str,
    quality: str,
    rerender_mode: str,
    section_index: Optional[int] = None,
) -> Dict[str, Any]:
    run_dir = Path(output["run_dir"])
    current_render = output.get("render_result") or {}
    existing_sections = list(current_render.get("sections") or [])

    if rerender_mode == "section" and section_index is None:
        raise ValueError("Please choose a section before rerendering from section onward.")

    if rerender_mode == "section" and int(section_index or 0) <= 0:
        return render_manim_video(
            code,
            run_dir,
            quality,
            render_label="full-rerender-from-section-zero",
            save_sections=True,
        )

    if rerender_mode == "section" and not existing_sections:
        raise ValueError(
            "This video does not have saved Manim sections yet. Use a full rerender first."
        )

    if rerender_mode == "section":
        start_index = int(section_index or 0)
        available_prefix_indexes = {
            int(item.get("index", -1))
            for item in existing_sections
            if is_valid_video_clip(Path(str(item.get("video_path", ""))))
        }
        missing_prefix_indexes = [idx for idx in range(start_index) if idx not in available_prefix_indexes]
        if missing_prefix_indexes:
            return render_manim_video(
                code,
                run_dir,
                quality,
                render_label="full-rerender-missing-prefix-sections",
                save_sections=True,
            )

        partial_render = render_manim_video(
            code,
            run_dir,
            quality,
            render_label=f"section-from-{start_index}",
            save_sections=True,
            skip_before_section_index=start_index,
        )
        updated_sections = list(partial_render.get("sections") or [])
        if not updated_sections:
            return render_manim_video(
                code,
                run_dir,
                quality,
                render_label="full-rerender-no-updated-sections",
                save_sections=True,
            )

        merged_sections = merge_section_entries(existing_sections, updated_sections, start_index)
        stitched_video_path = make_render_dir(run_dir, "stitched-video") / "visualization_stitched.mp4"
        stitched_video_path = stitch_videos(
            [Path(item["video_path"]) for item in merged_sections],
            stitched_video_path,
        )
        stitched_has_audio = video_has_audio_stream(stitched_video_path)
        if partial_render.get("has_audio") and not stitched_has_audio:
            return render_manim_video(
                code,
                run_dir,
                quality,
                render_label="full-rerender-audio-fallback",
                save_sections=True,
            )

        rerender_result = dict(partial_render)
        rerender_result["video_path"] = str(stitched_video_path)
        rerender_result["has_audio"] = stitched_has_audio
        rerender_result["sections"] = merged_sections
        rerender_result["stitched"] = True
        rerender_result["stitch_mode"] = "from_section"
        rerender_result["stitch_start_index"] = start_index
        return rerender_result

    return render_manim_video(
        code,
        run_dir,
        quality,
        render_label="full-rerender",
        save_sections=True,
    )


def run_pipeline(
    *,
    input_mode: str,
    backend: str,
    prompt_text: str,
    include_reasoning_trace: bool,
    render_video: bool,
    render_quality: str,
    text_input: str = "",
    uploaded_file: Any = None,
    progress_callback: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    orchestrator = get_orchestrator()
    run_label = uploaded_file.name if uploaded_file is not None else (text_input[:32] or "text")
    run_dir = make_run_dir(run_label)

    source_path: Optional[Path] = None
    parsed_input: Optional[Dict[str, Any]] = None

    if input_mode == "Text":
        if not text_input.strip():
            raise ValueError("Please enter some text before running the pipeline.")
        source_path = run_dir / "input.txt"
        source_path.write_text(text_input, encoding="utf-8")
        if progress_callback is not None:
            progress_callback("input", "Saved text input for this run.")
    else:
        if uploaded_file is None:
            raise ValueError(f"Please upload a {input_mode.lower()} file before running the pipeline.")
        source_path = save_uploaded_file(uploaded_file, run_dir)
        if progress_callback is not None:
            progress_callback("input", f"Saved uploaded {input_mode.lower()} file: {source_path.name}")

    if backend == "replicate":
        parser = get_replicate_parser()
        if input_mode == "Text":
            if progress_callback is not None:
                progress_callback("parse", "Calling Replicate DeepSeek-VL2 parser for text input...")
            parsed_input = parser.parse_text(text_input.strip())
        elif input_mode == "Image":
            if progress_callback is not None:
                progress_callback("parse", "Calling Replicate DeepSeek-VL2 parser for image input...")
            with Image.open(source_path) as image:
                parsed_input = parser.parse_image(image.convert("RGB"), prompt_text=prompt_text or None)
        else:
            if progress_callback is not None:
                progress_callback("parse", "Calling Replicate DeepSeek-VL2 parser for PDF input...")
            parsed_input = parser.parse_pdf(str(source_path), prompt_text=prompt_text or None)
        parsed_input["_uploaded_filename"] = source_path.name
        if progress_callback is not None:
            timing = parsed_input.get("_timing", {})
            model_name = parsed_input.get("_model") or "Replicate VL2"
            seconds = timing.get("total_parse_seconds")
            suffix = f" in {seconds:.2f}s" if isinstance(seconds, (int, float)) else ""
            progress_callback("parse", f"Replicate parser complete using {model_name}{suffix}.")

    if parsed_input is not None:
        result = orchestrator.process(
            parsed_input,
            input_type="json",
            prompt_text=prompt_text,
            include_reasoning_trace=include_reasoning_trace,
            generate_scene_planner=True,
            generate_manim_code=True,
            progress_callback=progress_callback,
        )
    else:
        input_type = {"Text": "text", "Image": "image", "PDF": "pdf"}[input_mode]
        result = orchestrator.process(
            str(source_path) if input_mode != "Text" else text_input.strip(),
            input_type=input_type,
            prompt_text=prompt_text,
            include_reasoning_trace=include_reasoning_trace,
            generate_scene_planner=True,
            generate_manim_code=True,
            progress_callback=progress_callback,
        )

    current_code = str((result.get("manim_code") or {}).get("text", "")).strip()
    current_code_path = run_dir / "current_generated_scene.py"
    current_code_path.write_text(current_code + ("\n" if current_code else ""), encoding="utf-8")
    if progress_callback is not None:
        progress_callback("save", f"Saved current generated Manim code to {current_code_path.name}.")

    render_result = None
    if render_video:
        if not current_code:
            raise RuntimeError("The pipeline finished without generating Manim code.")
        if progress_callback is not None:
            progress_callback("render", f"Starting local Manim render at quality `{render_quality}`...")
        render_result = render_manim_video(current_code, run_dir, render_quality, render_label="initial-render")
        if progress_callback is not None:
            kind = render_result.get("output_kind", "video")
            progress_callback("render", f"Manim render complete. Produced {kind}.")

    bundle_path = run_dir / "streamlit_result.json"
    output = {
        "run_dir": str(run_dir),
        "input_path": str(source_path) if source_path else "",
        "pipeline_result": result,
        "render_result": render_result,
        "bundle_path": str(bundle_path),
        "current_code_path": str(current_code_path),
    }
    persist_bundle(output)
    if progress_callback is not None:
        progress_callback("done", "Pipeline finished and Streamlit bundle saved.")
    return output


def render_preview(input_mode: str, text_input: str = "", uploaded_file: Any = None) -> None:
    if input_mode == "Text":
        st.text_area("Input preview", value=text_input, height=180, disabled=True)
        return

    if uploaded_file is None:
        return

    if input_mode == "Image":
        st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
        return

    preview = render_pdf_preview(uploaded_file)
    if preview is not None:
        st.image(preview, caption=f"{uploaded_file.name} (first page)", use_container_width=True)
    else:
        st.info(uploaded_file.name)


def render_saved_artifact_buttons(saved_files: Dict[str, str], render_result: Optional[Dict[str, Any]]) -> None:
    for label, path_str in saved_files.items():
        path = Path(path_str)
        if path.exists():
            st.download_button(
                f"Download {label.replace('_', ' ')}",
                data=path.read_bytes(),
                file_name=path.name,
                mime="application/octet-stream",
                use_container_width=True,
            )

    if render_result:
        video_path = Path(render_result.get("video_path", "")) if render_result.get("video_path") else None
        image_path = Path(render_result.get("image_path", "")) if render_result.get("image_path") else None
        if video_path and video_path.exists():
            st.download_button(
                "Download visualization video",
                data=video_path.read_bytes(),
                file_name=video_path.name,
                mime="video/mp4",
                use_container_width=True,
            )
        if image_path and image_path.exists():
            st.download_button(
                "Download visualization image",
                data=image_path.read_bytes(),
                file_name=image_path.name,
                mime="image/png",
                use_container_width=True,
            )


def _preferred_render_video(run_dir: Path) -> Optional[Path]:
    candidates = [
        path for path in run_dir.rglob("*.mp4")
        if "partial_movie_files" not in path.parts and "sections" not in path.parts
    ]
    if not candidates:
        return None

    def sort_key(path: Path) -> tuple[int, float]:
        priority = 0
        if path.name == "visualization_stitched.mp4":
            priority = 3
        elif path.name == "visualization.mp4":
            priority = 2
        elif path.parent.name in {"480p15", "720p30", "1080p60"}:
            priority = 1
        return (priority, path.stat().st_mtime)

    return sorted(candidates, key=sort_key, reverse=True)[0]


def _preferred_render_image(run_dir: Path) -> Optional[Path]:
    candidates = [
        path for path in run_dir.rglob("*.png")
        if "texts" not in path.parts
    ]
    if not candidates:
        return None

    def sort_key(path: Path) -> tuple[int, float]:
        priority = 0
        if path.name == "visualization.png":
            priority = 2
        elif "images" in path.parts:
            priority = 1
        return (priority, path.stat().st_mtime)

    return sorted(candidates, key=sort_key, reverse=True)[0]


def _preferred_code_path(run_dir: Path) -> Optional[Path]:
    current_path = run_dir / "current_generated_scene.py"
    if current_path.exists():
        return current_path

    candidates = [
        path for path in run_dir.rglob("generated_scene.py")
        if "__pycache__" not in path.parts
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def discover_demo_runs(limit: int = 8) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for run_dir in STREAMLIT_RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        code_path = _preferred_code_path(run_dir)
        video_path = _preferred_render_video(run_dir)
        image_path = _preferred_render_image(run_dir)
        if code_path is None or (video_path is None and image_path is None):
            continue

        label = run_dir.name
        if (run_dir / "input.txt").exists():
            preview = (run_dir / "input.txt").read_text(encoding="utf-8", errors="replace").strip()
            if preview:
                label = f"{preview[:42]}{'...' if len(preview) > 42 else ''} [{run_dir.name}]"

        candidates.append(
            {
                "run_dir": str(run_dir),
                "label": label,
                "code_path": str(code_path),
                "video_path": str(video_path) if video_path else "",
                "image_path": str(image_path) if image_path else "",
                "has_video": bool(video_path),
                "updated_at": run_dir.stat().st_mtime,
            }
        )
    candidates.sort(key=lambda item: (int(item["has_video"]), float(item["updated_at"])), reverse=True)
    return candidates[:limit]


def build_demo_output(run_dir: Path | str) -> Optional[Dict[str, Any]]:
    run_dir = Path(run_dir)
    code_path = _preferred_code_path(run_dir)
    video_path = _preferred_render_video(run_dir)
    image_path = _preferred_render_image(run_dir)
    if code_path is None or (video_path is None and image_path is None):
        return None

    code_text = strip_manim_runtime_compatibility(
        code_path.read_text(encoding="utf-8", errors="replace")
    )
    bundle_path = run_dir / "streamlit_result.json"
    pipeline_result: Dict[str, Any]
    if bundle_path.exists():
        try:
            pipeline_result = json.loads(bundle_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pipeline_result = {}
    else:
        pipeline_result = {}

    if not pipeline_result:
        pipeline_result = {
            "parsed_input": {
                "topic": "Demo visualization",
                "domain": "mathematics",
                "complexity": "basic",
                "intent": "concept_explanation",
            },
            "solution": {
                "full_text": "Loaded a saved demo render so you can test the code editor and rerender workflow without running the full pipeline.",
            },
            "scene_planner": {
                "text": "Loaded from a saved demo render.",
            },
            "pipeline_metadata": {
                "parse_timing": {},
                "reasoning_timing": {},
                "scene_planner_timing": {},
                "manim_code_timing": {},
                "total_pipeline_seconds": 0.0,
                "saved_files": {},
            },
        }

    pipeline_result.setdefault("manim_code", {})["text"] = code_text
    pipeline_result.setdefault("pipeline_metadata", {})
    pipeline_result["pipeline_metadata"].setdefault("saved_files", {})

    scene_classes = detect_scene_classes(code_text)
    scene_name = scene_classes[0] if scene_classes else "Scene"
    render_result = {
        "video_path": str(video_path) if video_path else "",
        "image_path": str(image_path) if image_path else "",
        "output_kind": "video" if video_path else "image",
        "has_audio": video_has_audio_stream(video_path) if video_path is not None else False,
        "scene_name": scene_name,
        "stdout": "Loaded from saved demo render.",
        "stderr": "",
        "script_path": str(code_path),
        "render_script_path": str(code_path),
        "media_dir": str((video_path or image_path).parent.parent if (video_path or image_path) else run_dir),
        "sections": (
            load_section_assets(video_path, scene_name, index_offset=0)
            if video_path is not None
            else []
        ),
        "stitched": False,
        "render_label": "demo-loaded",
    }

    input_path = run_dir / "input.txt"
    return {
        "run_dir": str(run_dir),
        "input_path": str(input_path) if input_path.exists() else "",
        "pipeline_result": pipeline_result,
        "render_result": render_result,
        "bundle_path": str(bundle_path),
        "current_code_path": str(code_path),
    }


def load_demo_into_session(run_dir: Path) -> None:
    demo_output = build_demo_output(run_dir)
    if demo_output is None:
        raise RuntimeError(f"Could not load a usable demo from {run_dir}.")
    st.session_state["xplainai_frontend_output"] = demo_output
    st.session_state["xplainai_demo_loaded"] = True
    st.session_state.pop("xplainai_editor_run_dir", None)
    st.session_state.pop("xplainai_code_editor", None)
    st.session_state.pop("xplainai_code_edit_mode", None)
    st.session_state.pop("xplainai_rerender_scope", None)


def ensure_editor_state(output: Dict[str, Any]) -> None:
    run_dir = output["run_dir"]
    current_code = str((output["pipeline_result"].get("manim_code") or {}).get("text", ""))
    current_editor = str(st.session_state.get("xplainai_code_editor", ""))
    if st.session_state.get("xplainai_editor_run_dir") != run_dir:
        st.session_state["xplainai_editor_run_dir"] = run_dir
        st.session_state["xplainai_code_editor"] = current_code
        st.session_state["xplainai_code_edit_mode"] = False
        st.session_state["xplainai_rerender_scope"] = "full"
    elif current_code.strip() and not current_editor.strip():
        st.session_state["xplainai_code_editor"] = current_code


def render_sections_summary(render_result: Optional[Dict[str, Any]]) -> None:
    sections = list((render_result or {}).get("sections") or [])
    if not sections:
        st.info("This render does not currently expose section clips. Full rerender will still work.")
        return

    st.caption("Saved Manim sections")
    rows = [
        {
            "index": int(item.get("index", 0)),
            "name": item.get("name", ""),
            "duration": item.get("duration", ""),
            "video_path": item.get("video_path", ""),
        }
        for item in sections
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="XplainAI Visualizer", layout="wide")
    st.title("XplainAI Visualizer")
    st.caption(
        "Upload a text problem, image, or PDF and run the full parser -> reasoning -> scene planner -> Manim video pipeline."
    )

    with st.sidebar:
        st.header("Pipeline")
        input_mode = st.radio("Input type", ["Image", "PDF", "Text"], index=0)
        backend_label = st.selectbox("Parser backend", list(PARSER_BACKENDS.keys()), index=0)
        backend = PARSER_BACKENDS[backend_label]
        prompt_text = st.text_area(
            "Optional instruction",
            value="",
            help="Extra instruction for image or PDF parsing, or for steering the overall explanation.",
        )
        include_reasoning_trace = st.checkbox("Include reasoning trace summary", value=False)
        render_video = st.checkbox("Render visualization video with Manim", value=True)
        render_quality_label = st.selectbox("Render quality", list(RENDER_QUALITIES.keys()), index=0)
        render_quality = RENDER_QUALITIES[render_quality_label]
        st.markdown(
            "The full run can take a few minutes because it includes parsing, reasoning, scene planning, code generation, and video rendering."
        )
        st.divider()
        st.subheader("Editing Demo")
        demo_candidates = discover_demo_runs()
        if demo_candidates:
            demo_labels = {item["label"]: item["run_dir"] for item in demo_candidates}
            selected_demo_label = st.selectbox(
                "Saved demo run",
                list(demo_labels.keys()),
                help="Load a recent saved render into the editor so you can test code edits and rerendering without running the full pipeline.",
            )
            if st.button("Load Demo", use_container_width=True):
                load_demo_into_session(Path(demo_labels[selected_demo_label]))
                st.rerun()
        else:
            st.caption("No saved demo renders found yet.")

    if input_mode == "Text":
        text_input = st.text_area("Problem text", height=220, placeholder="Paste the problem statement here...")
        uploaded_file = None
    elif input_mode == "Image":
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        text_input = ""
    else:
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
        text_input = ""

    preview_col, action_col = st.columns([3, 2], gap="large")
    with preview_col:
        st.subheader("Input Preview")
        render_preview(input_mode, text_input=text_input, uploaded_file=uploaded_file)
    with action_col:
        st.subheader("Run")
        st.write(f"Parser backend: `{backend_label}`")
        st.write(
            "The app will always generate reasoning, a Scene Planner, Manim code, and optionally render the video."
        )
        run_clicked = st.button("Run Full Pipeline", type="primary", use_container_width=True)

    progress_mount = st.empty()
    results_mount = st.empty()

    if run_clicked:
        try:
            st.session_state.pop("xplainai_frontend_output", None)
            progress_lines: List[str] = []

            def render_progress_panel() -> None:
                with progress_mount.container():
                    st.markdown("**Pipeline Progress**")
                    if progress_lines:
                        st.info(f"Current step: {progress_lines[-1]}")
                        with st.expander("Step log", expanded=True):
                            st.code("\n".join(progress_lines), language="text")
                    else:
                        st.info("Current step: Preparing pipeline...")

            with results_mount.container():
                st.info("Running a new pipeline. Previous visualization is hidden until the new result is ready.")
            render_progress_panel()
            with st.status("Running the full pipeline...", expanded=True) as status:
                def progress_update(stage: str, message: str) -> None:
                    stage_labels = {
                        "input": "Input",
                        "cache": "Cache",
                        "parse": "Parser",
                        "reason": "Reasoner",
                        "scene_planner": "Scene Planner",
                        "manim_code": "Manim Code",
                        "save": "Save",
                        "render": "Render",
                        "done": "Done",
                    }
                    entry = f"[{stage_labels.get(stage, stage.title())}] {message}"
                    progress_lines.append(entry)
                    render_progress_panel()
                    status.write(entry)

                initial_entry = "[Input] Preparing input and parser backend..."
                progress_lines.append(initial_entry)
                render_progress_panel()
                status.write(initial_entry)
                output = run_pipeline(
                    input_mode=input_mode,
                    backend=backend,
                    prompt_text=prompt_text,
                    include_reasoning_trace=include_reasoning_trace,
                    render_video=render_video,
                    render_quality=render_quality,
                    text_input=text_input,
                    uploaded_file=uploaded_file,
                    progress_callback=progress_update,
                )
                status.update(label="Pipeline complete", state="complete")
            with progress_mount.container():
                st.markdown("**Pipeline Progress**")
                if progress_lines:
                    st.success(f"Current step: {progress_lines[-1]}")
                    with st.expander("Step log", expanded=False):
                        st.code("\n".join(progress_lines), language="text")
            st.session_state["xplainai_frontend_output"] = output
        except Exception as exc:
            st.exception(exc)

    with results_mount.container():
        output = st.session_state.get("xplainai_frontend_output")
        if not output:
            demo_candidates = discover_demo_runs(limit=1)
            if demo_candidates:
                load_demo_into_session(Path(demo_candidates[0]["run_dir"]))
                output = st.session_state.get("xplainai_frontend_output")
                st.info("Loaded a recent saved demo so you can test editing immediately.")

        if not output:
            st.info(
                "Run the pipeline to see the generated parser output, reasoning, Scene Planner, Manim code, and rendered video."
            )
            return

        ensure_editor_state(output)

        rerender_notice = st.session_state.pop("xplainai_rerender_notice", "")
        if rerender_notice:
            st.success(rerender_notice)

        result = output["pipeline_result"]
        render_result = output.get("render_result")
        saved_files = (result.get("pipeline_metadata") or {}).get("saved_files", {})

        timings = result.get("pipeline_metadata", {})
        metrics = st.columns(5)
        metrics[0].metric("Parse", f"{(timings.get('parse_timing') or {}).get('total_parse_seconds', 0):.2f}s")
        metrics[1].metric("Reason", f"{(timings.get('reasoning_timing') or {}).get('generation_seconds', 0):.2f}s")
        metrics[2].metric("Planner", f"{(timings.get('scene_planner_timing') or {}).get('generation_seconds', 0):.2f}s")
        metrics[3].metric("Code", f"{(timings.get('manim_code_timing') or {}).get('generation_seconds', 0):.2f}s")
        metrics[4].metric("Total", f"{timings.get('total_pipeline_seconds', 0):.2f}s")

        tabs = st.tabs(["Visualization", "Parsed JSON", "Solution", "Scene Planner", "Manim Code", "Artifacts"])

        with tabs[0]:
            st.subheader("Visualization")
            if render_result and render_result.get("video_path") and Path(render_result["video_path"]).exists():
                st.video(render_result["video_path"])
                caption = f"Scene rendered from `{render_result['scene_name']}`"
                if render_result.get("stitched"):
                    caption += " using stitched section clips"
                st.caption(caption)
                if not bool(render_result.get("has_audio", False)):
                    st.warning("This rendered MP4 does not contain an audio track.")
            elif render_result and render_result.get("image_path") and Path(render_result["image_path"]).exists():
                st.image(render_result["image_path"], use_container_width=True)
                st.caption(f"Scene rendered as a still image from `{render_result['scene_name']}`")
            else:
                st.warning("No rendered video or image is available for this run.")

            render_sections_summary(render_result)

            st.subheader("Run Details")
            st.json(
                {
                    "run_dir": output["run_dir"],
                    "input_path": output["input_path"],
                    "bundle_path": output["bundle_path"],
                    "current_code_path": output["current_code_path"],
                    "video_path": (render_result or {}).get("video_path"),
                    "image_path": (render_result or {}).get("image_path"),
                }
            )

        with tabs[1]:
            st.subheader("Parsed JSON")
            st.json(result.get("parsed_input", {}))

        with tabs[2]:
            st.subheader("Reasoning Output")
            solution = result.get("solution", {})
            st.markdown(solution.get("full_text", "_No solution text generated._"))

        with tabs[3]:
            st.subheader("Scene Planner")
            st.text((result.get("scene_planner") or {}).get("text", ""))

        with tabs[4]:
            original_code = str((output["pipeline_result"].get("manim_code") or {}).get("text", ""))
            if original_code.strip() and not str(st.session_state.get("xplainai_code_editor", "")).strip():
                st.session_state["xplainai_code_editor"] = original_code

            header_left, header_edit, header_toggle = st.columns([6, 1, 1])
            with header_left:
                st.subheader("Generated Manim Code")
            with header_edit:
                if st.button("✏️ Edit", use_container_width=True):
                    st.session_state["xplainai_code_edit_mode"] = not st.session_state.get(
                        "xplainai_code_edit_mode",
                        False,
                    )
            with header_toggle:
                if st.button("↺ Reset", use_container_width=True):
                    st.session_state["xplainai_code_editor"] = original_code

            edit_mode = st.session_state.get("xplainai_code_edit_mode", False)
            if edit_mode:
                st.caption(
                    "Edit the generated code here. Then rerender the full video or rerender from the section where your change starts."
                )
                st.text_area(
                    "Editable Manim code",
                    key="xplainai_code_editor",
                    height=560,
                )

                current_sections = list((output.get("render_result") or {}).get("sections") or [])
                rerender_mode_label = st.radio(
                    "Rerender mode",
                    ["Full rerender", "Rerender from section onward"],
                    horizontal=True,
                )
                rerender_mode = "full" if rerender_mode_label == "Full rerender" else "section"

                selected_section_index: Optional[int] = None
                if rerender_mode == "section":
                    if current_sections:
                        section_options = {
                            f"{int(item.get('index', 0))}. {item.get('name', '')}": int(item.get("index", 0))
                            for item in current_sections
                        }
                        selected_label = st.selectbox(
                            "Changed section",
                            list(section_options.keys()),
                            help="Choose the earliest section affected by your edit. The app will rerender this section and all later ones, then stitch the final video.",
                        )
                        selected_section_index = section_options[selected_label]
                    else:
                        st.warning(
                            "No saved section clips are available for this run yet, so section-only rerender is disabled."
                        )
                        rerender_mode = "full"

                action_cols = st.columns([1, 1, 2])
                with action_cols[0]:
                    save_clicked = st.button("💾 Save Code", use_container_width=True)
                with action_cols[1]:
                    rerender_clicked = st.button("🔁 Re-render", use_container_width=True)

                current_code = st.session_state.get("xplainai_code_editor", "")
                if save_clicked:
                    persist_current_code(output, current_code)
                    st.success("Saved edited Manim code.")

                if rerender_clicked:
                    try:
                        persist_current_code(output, current_code)
                        with st.status("Rendering edited code...", expanded=True) as status:
                            status.write("Saved the latest code edit.")
                            if rerender_mode == "section" and selected_section_index is not None:
                                status.write(
                                    f"Rerendering from section index {selected_section_index} onward and preserving earlier section clips..."
                                )
                            else:
                                status.write("Running a full rerender from the edited code...")

                            new_render_result = rerender_edited_video(
                                output,
                                current_code,
                                render_quality,
                                rerender_mode,
                                section_index=selected_section_index,
                            )
                            output["render_result"] = new_render_result
                            persist_bundle(output)

                            if new_render_result.get("stitched"):
                                status.write("Stitched preserved prefix clips with the updated suffix clips.")
                            status.update(label="Edited render complete", state="complete")

                        st.session_state["xplainai_frontend_output"] = output
                        st.session_state["xplainai_rerender_notice"] = "Video rerendered from the edited code."
                        st.rerun()
                    except Exception as exc:
                        st.exception(exc)
            else:
                st.code(st.session_state.get("xplainai_code_editor", ""), language="python")

        with tabs[5]:
            st.subheader("Downloads")
            render_saved_artifact_buttons(saved_files, render_result)
            current_code_path = Path(output["current_code_path"])
            if current_code_path.exists():
                st.download_button(
                    "Download current editable code",
                    data=current_code_path.read_bytes(),
                    file_name=current_code_path.name,
                    mime="text/x-python",
                    use_container_width=True,
                )
            if render_result:
                with st.expander("Manim render logs"):
                    st.text(render_result.get("stdout", ""))
                    stderr = render_result.get("stderr", "")
                    if stderr.strip():
                        st.text(stderr)


if __name__ == "__main__":
    main()
