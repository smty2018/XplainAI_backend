"""Replicate-hosted parser using deepseek-ai/deepseek-vl2."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import textwrap
import time
from contextlib import redirect_stdout
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

from .parser import LocalParser


class ReplicateAPIError(RuntimeError):
    """Raised when a Replicate API request fails."""


class ReplicatePredictionError(ReplicateAPIError):
    """Raised when a Replicate prediction fails."""


class ReplicateDeepSeekVL2Parser(LocalParser):
    """Parser that preserves the current schema but runs DeepSeek-VL2 on Replicate."""

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        api_token: Optional[str] = None,
    ):
        with redirect_stdout(io.StringIO()):
            super().__init__(config_path)

        load_dotenv(self.project_root / ".env")

        self.model_name = self.config.get("replicate_model", "deepseek-ai/deepseek-vl2")
        self.model_version = self.config.get(
            "replicate_model_version",
            "e5caf557dd9e5dcee46442e1315291ef1867f027991ede8ff95e304d4f734200",
        )
        self.api_base_url = self.config.get("replicate_api_base_url", "https://api.replicate.com/v1").rstrip("/")
        self.api_timeout_seconds = int(self.config.get("replicate_api_timeout_seconds", 180))
        self.api_wait_seconds = min(
            max(int(self.config.get("replicate_api_wait_seconds", 60)), 1),
            60,
        )
        self.poll_interval_seconds = float(self.config.get("replicate_poll_interval_seconds", 2.0))
        self.max_inline_image_bytes = int(self.config.get("replicate_max_inline_image_bytes", 900000))
        self.max_image_side = int(self.config.get("replicate_max_image_side", 1400))
        self.api_token = self._resolve_api_token(api_token)
        self.domain_terms["mathematics"] = self._merge_unique_strings(
            self.domain_terms.get("mathematics", []),
            [
                "fourier series",
                "periodic signal",
                "sine",
                "cosine",
                "complex exponential",
                "fundamental frequency",
                "harmonic",
            ],
        )

        try:
            self.model_owner, self.model_slug = self.model_name.split("/", 1)
        except ValueError as exc:
            raise ValueError(
                "replicate_model must be in the form 'owner/model', "
                f"got {self.model_name!r}."
            ) from exc

        self.predictions_url = f"{self.api_base_url}/predictions"

        print(f"Project root: {self.project_root}")
        print(f"Data folder: {self.data_dir}")
        print(f"Output folder: {self.output_dir}")
        print(f"Using model: {self.model_name}")
        print(f"Using Replicate version: {self.model_version}")
        print("Replicate-hosted DeepSeek-VL2 will be called on demand.")

    def _resolve_api_token(self, explicit_token: Optional[str]) -> str:
        token = (
            explicit_token
            or os.getenv("REPLICATE_API_TOKEN")
            or os.getenv("REPLICATE_TOKEN")
            or os.getenv("tokenreplicate")
            or os.getenv("replicate")
        )
        if not token:
            raise RuntimeError(
                "Replicate API token not found. Set REPLICATE_API_TOKEN, REPLICATE_TOKEN, tokenreplicate, or replicate in the environment or .env."
            )
        return token.strip().strip("'").strip('"')

    def _resample_filter(self):
        if hasattr(Image, "Resampling"):
            return Image.Resampling.LANCZOS
        return Image.LANCZOS

    def _load_font(self, size: int) -> ImageFont.ImageFont:
        candidates = [
            Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf",
            Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf",
            Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "calibri.ttf",
            Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "consola.ttf",
        ]
        for candidate in candidates:
            if candidate.exists():
                try:
                    return ImageFont.truetype(str(candidate), size=size)
                except OSError:
                    continue
        return ImageFont.load_default()

    def _resize_longest_side(self, image: Image.Image, max_side: int) -> Image.Image:
        width, height = image.size
        current_max = max(width, height)
        if current_max <= max_side:
            return image
        scale = max_side / float(current_max)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return image.resize(new_size, self._resample_filter())

    def _text_to_image(self, text: str) -> Image.Image:
        content = self._coerce_string(text)
        font = self._load_font(28)
        margin = 40
        line_gap = 10
        wrapped_lines: List[str] = []
        for paragraph in content.splitlines() or [""]:
            paragraph = paragraph.rstrip()
            if not paragraph:
                wrapped_lines.append("")
                continue
            wrapped_lines.extend(textwrap.wrap(paragraph, width=72) or [""])

        if not wrapped_lines:
            wrapped_lines = [""]

        probe = Image.new("RGB", (1, 1), "white")
        draw = ImageDraw.Draw(probe)
        line_heights: List[int] = []
        max_width = 0
        for line in wrapped_lines:
            bbox = draw.textbbox((0, 0), line or " ", font=font)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            max_width = max(max_width, width)
            line_heights.append(height or 28)

        canvas_width = min(max(max_width + margin * 2, 900), 1600)
        canvas_height = margin * 2 + sum(line_heights) + line_gap * max(0, len(wrapped_lines) - 1)
        image = Image.new("RGB", (canvas_width, max(canvas_height, 220)), "white")
        draw = ImageDraw.Draw(image)

        y = margin
        for line, line_height in zip(wrapped_lines, line_heights):
            draw.text((margin, y), line, fill="black", font=font)
            y += line_height + line_gap

        return image

    def _compose_images(self, images: List[Image.Image], label_prefix: str = "Page") -> Image.Image:
        if not images:
            raise ValueError("At least one image is required.")
        if len(images) == 1:
            return images[0].convert("RGB")

        font = self._load_font(24)
        margin = 24
        label_gap = 12
        gutter = 28
        prepared: List[Tuple[str, Image.Image]] = []
        max_width = 0
        total_height = margin

        probe = Image.new("RGB", (1, 1), "white")
        draw = ImageDraw.Draw(probe)

        for index, image in enumerate(images, start=1):
            prepared_image = self._resize_longest_side(image.convert("RGB"), self.max_image_side)
            label = f"{label_prefix} {index}"
            label_bbox = draw.textbbox((0, 0), label, font=font)
            label_height = label_bbox[3] - label_bbox[1]
            prepared.append((label, prepared_image))
            max_width = max(max_width, prepared_image.width)
            total_height += label_height + label_gap + prepared_image.height + gutter

        total_height += margin - gutter
        canvas = Image.new("RGB", (max_width + margin * 2, total_height), "white")
        draw = ImageDraw.Draw(canvas)

        y = margin
        for label, image in prepared:
            draw.text((margin, y), label, fill="black", font=font)
            label_bbox = draw.textbbox((0, 0), label, font=font)
            label_height = label_bbox[3] - label_bbox[1]
            y += label_height + label_gap
            x = (canvas.width - image.width) // 2
            canvas.paste(image, (x, y))
            y += image.height + gutter

        return canvas

    def _image_to_data_uri(self, image: Image.Image) -> Tuple[str, Dict[str, Any]]:
        prepared = self._resize_longest_side(image.convert("RGB"), self.max_image_side)
        last_payload = b""
        last_quality = 0

        for _ in range(6):
            for quality in (80, 70, 60, 50, 40, 30):
                buffer = io.BytesIO()
                prepared.save(buffer, format="JPEG", quality=quality, optimize=True)
                payload = buffer.getvalue()
                last_payload = payload
                last_quality = quality
                if len(payload) <= self.max_inline_image_bytes:
                    data_uri = "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")
                    return data_uri, {
                        "bytes": len(payload),
                        "width": prepared.width,
                        "height": prepared.height,
                        "jpeg_quality": quality,
                    }

            next_width = max(320, int(prepared.width * 0.85))
            next_height = max(320, int(prepared.height * 0.85))
            if (next_width, next_height) == prepared.size:
                break
            prepared = prepared.resize((next_width, next_height), self._resample_filter())

        data_uri = "data:image/jpeg;base64," + base64.b64encode(last_payload).decode("ascii")
        return data_uri, {
            "bytes": len(last_payload),
            "width": prepared.width,
            "height": prepared.height,
            "jpeg_quality": last_quality,
        }

    def _extract_error_message(self, payload: str) -> str:
        try:
            body = json.loads(payload)
        except json.JSONDecodeError:
            return payload.strip() or "Unknown Replicate error."

        if isinstance(body, dict):
            if isinstance(body.get("detail"), str):
                return body["detail"]
            if isinstance(body.get("error"), str):
                return body["error"]
            if isinstance(body.get("title"), str):
                return body["title"]
        return payload.strip() or "Unknown Replicate error."

    def _request_json(
        self,
        url: str,
        method: str = "GET",
        payload: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)

        last_error: Optional[Exception] = None
        for attempt in range(4):
            req = urlrequest.Request(url, data=body, headers=headers, method=method)
            try:
                with urlrequest.urlopen(req, timeout=self.api_timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urlerror.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                message = self._extract_error_message(error_body)
                if exc.code == 429 and attempt < 3:
                    reset_match = re.search(r"resets in ~(\d+)s", message, flags=re.IGNORECASE)
                    delay_seconds = int(reset_match.group(1)) + 1 if reset_match else 3
                    time.sleep(max(delay_seconds, 1))
                    last_error = exc
                    continue
                raise ReplicateAPIError(
                    f"Replicate API request failed ({exc.code}): {message}"
                ) from exc
            except urlerror.URLError as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(2)
                    continue
                raise ReplicateAPIError(f"Replicate API network error: {exc}") from exc

        if last_error:
            raise ReplicateAPIError(f"Replicate API request failed: {last_error}") from last_error
        raise ReplicateAPIError("Replicate API request failed for an unknown reason.")

    def _wait_for_prediction(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        status = prediction.get("status")
        if status in {"succeeded", "failed", "canceled"}:
            return prediction

        poll_url = ((prediction.get("urls") or {}).get("get") or "").strip()
        if not poll_url:
            raise ReplicatePredictionError("Replicate prediction did not include a polling URL.")

        deadline = perf_counter() + self.api_timeout_seconds
        current = prediction
        while perf_counter() < deadline:
            time.sleep(self.poll_interval_seconds)
            current = self._request_json(poll_url)
            status = current.get("status")
            if status in {"succeeded", "failed", "canceled"}:
                return current

        raise ReplicatePredictionError(
            f"Replicate prediction timed out after {self.api_timeout_seconds} seconds."
        )

    def _prediction_output_to_text(self, output: Any) -> str:
        if isinstance(output, str):
            return output
        if isinstance(output, list):
            return "".join(str(item) for item in output)
        if output is None:
            return ""
        return json.dumps(output, ensure_ascii=False)

    def _ensure_image_placeholder(self, prompt: str, images: Optional[List[Image.Image]]) -> str:
        if not images:
            return prompt
        if "<image>" in prompt:
            return prompt
        return f"Use the provided <image> as the primary input.\n{prompt}"

    def _count_marker_hits(self, text: str, markers: List[str]) -> int:
        return sum(1 for marker in markers if marker in text)

    def _extract_clean_source_equations(self, text: str) -> List[str]:
        equations: List[str] = []
        seen: set[str] = set()
        label_prefixes = {"vector form", "cartesian form", "general form", "normal form"}

        for raw_line in text.splitlines():
            line = self._coerce_string(raw_line)
            if not line or "=" not in line:
                continue

            lower_line = line.casefold()
            if lower_line.startswith(("find ", "given ", "if ", "key tip", "practice problems")):
                continue

            if ":" in line:
                prefix, suffix = line.split(":", 1)
                if prefix.strip().casefold() in label_prefixes and "=" in suffix:
                    line = suffix.strip()

            if len(line) > 140:
                continue

            sanitized = self._sanitize_equation_candidate(line)
            if not sanitized:
                continue
            if sanitized.endswith("=") and line.count("=") == 1:
                sanitized = re.sub(r"\s+", " ", line).strip(" \t\n\r.,;:")

            normalized_key = sanitized.casefold()
            if normalized_key in seen:
                continue

            seen.add(normalized_key)
            equations.append(sanitized)

        return equations

    def _extract_pdf_headings(self, text: str) -> List[str]:
        headings: List[str] = []
        seen: set[str] = set()

        for raw_line in text.splitlines():
            line = self._coerce_string(raw_line)
            if not line or "=" in line:
                continue

            lower_line = line.casefold()
            if lower_line.startswith(("find ", "given ", "if ", "key tip", "practice problems")):
                continue

            heading = line
            numbered_match = re.match(r"^\d+\.\s+(.+)$", line)
            if numbered_match:
                heading = numbered_match.group(1).strip()
            elif len(line.split()) > 8:
                continue

            normalized_key = heading.casefold()
            if normalized_key in seen:
                continue

            seen.add(normalized_key)
            headings.append(heading)

        return headings

    def _remove_contaminated_concepts(
        self,
        result: Dict[str, Any],
        banned_concepts: set[str],
    ) -> Dict[str, Any]:
        result["key_concepts"] = [
            item for item in result.get("key_concepts", []) if item not in banned_concepts
        ]

        entities = result.get("entities", {})
        for key in ["concepts", "constants"]:
            values = entities.get(key, [])
            if isinstance(values, list):
                entities[key] = [item for item in values if item not in banned_concepts]
        result["entities"] = entities

        analysis = result.get("_analysis", {})
        if isinstance(analysis, dict):
            technical_terms = analysis.get("technical_terms_found", [])
            if isinstance(technical_terms, list):
                analysis["technical_terms_found"] = [
                    item for item in technical_terms if item not in banned_concepts
                ]
            result["_analysis"] = analysis

        return result

    def _condense_concept_text(self, value: str) -> List[str]:
        text = self._coerce_string(value)
        if not text:
            return []

        candidates: List[str] = []
        lowered = text.casefold()
        phrase_patterns = [
            r"\bfourier series\b",
            r"\bperiodic signal\b",
            r"\bfundamental frequency\b",
            r"\bcomplex exponentials?\b",
            r"\bsines? and cosines?\b",
            r"\bkirchhoff'?s current law\b",
            r"\bkirchhoff'?s voltage law\b",
            r"\bdirection cosines\b",
            r"\bdistance between two points\b",
            r"\bequation of a line\b",
            r"\bequation of a plane\b",
        ]
        for pattern in phrase_patterns:
            match = re.search(pattern, lowered, flags=re.IGNORECASE)
            if match:
                phrase = text[match.start():match.end()].strip(" .,:;-")
                if phrase and phrase not in candidates:
                    candidates.append(phrase)

        lead = re.split(
            r"\b(?:represents|can be decomposed|can be|is|are|refers to|denotes|states that|means)\b",
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" .,:;-")
        lead = re.sub(r"^(?:any|the|a|an)\s+", "", lead, flags=re.IGNORECASE)
        if lead and 1 <= len(lead.split()) <= 8 and lead not in candidates:
            candidates.append(lead)

        return candidates

    def _visual_text_from_result(self, result: Dict[str, Any]) -> str:
        parts: List[str] = []
        base_text = super()._visual_text_from_result(result)
        if base_text:
            parts.append(base_text)

        topic = self._coerce_string(result.get("topic"))
        if topic:
            parts.append(topic)

        for concept in self._coerce_string_list(result.get("key_concepts")):
            parts.append(concept)

        equations = result.get("equations")
        if isinstance(equations, list):
            for item in equations:
                if isinstance(item, dict):
                    for key in ["raw", "latex", "text"]:
                        value = self._coerce_string(item.get(key))
                        if value:
                            parts.append(value)
                else:
                    value = self._coerce_string(item)
                    if value:
                        parts.append(value)

        return "\n".join(part for part in parts if part).strip()

    def _visual_support_text_from_result(self, result: Dict[str, Any]) -> str:
        parts: List[str] = []
        base_text = super()._visual_text_from_result(result)
        if base_text:
            parts.append(base_text)

        topic = self._coerce_string(result.get("topic"))
        if topic:
            parts.append(topic)

        for concept in self._coerce_string_list(result.get("key_concepts")):
            parts.append(concept)

        return "\n".join(part for part in parts if part).strip()

    def _normalize_visible_math_text(self, value: str) -> str:
        text = self._coerce_string(value)
        if not text:
            return ""

        replacements = [
            ("\\\\", "\\"),
            ("\\sqrt", "sqrt"),
            ("√", "sqrt"),
            ("âˆš", "sqrt"),
            ("\\sum", "sum"),
            ("Σ", "sum"),
            ("∑", "sum"),
            ("\\int", "int"),
            ("∫", "int"),
            ("\\frac", "frac"),
            ("\\pi", "pi"),
            ("π", "pi"),
            ("Ï€", "pi"),
            ("\\omega", "omega"),
            ("\\alpha", "alpha"),
            ("\\beta", "beta"),
            ("\\gamma", "gamma"),
            ("\\theta", "theta"),
            ("\\lambda", "lambda"),
            ("\\sin", "sin"),
            ("\\cos", "cos"),
            ("\\tan", "tan"),
            ("\\log", "log"),
            ("\\ln", "ln"),
        ]
        for source, target in replacements:
            text = text.replace(source, target)

        text = text.casefold()
        text = text.replace("{", "").replace("}", "")
        text = text.replace("[", "").replace("]", "")
        text = text.replace("(", "").replace(")", "")
        text = text.replace("_", "")
        text = text.replace("^", "")
        text = text.replace("\\", "")
        text = re.sub(r"[^a-z0-9]+", "", text)
        return text

    def _has_formula_evidence(self, text: str) -> bool:
        visible_text = self._coerce_string(text)
        if not visible_text:
            return False

        formula_patterns = [
            r"[=∑Σ∫√]",
            r"\\(?:sqrt|sum|int|frac|pi|alpha|beta|gamma|omega)\b",
            r"\b(?:sqrt|sum|int|frac)\b",
            r"\b(?:sin|cos|tan|log|ln)\s*\(",
            r"\b[A-Za-z][A-Za-z0-9_]*\s*=\s*[^=\n]+",
            r"\b[xyzabcdnlmkvt]\s*[_^]?\s*\d",
        ]
        return any(
            re.search(pattern, visible_text, flags=re.IGNORECASE)
            for pattern in formula_patterns
        )

    def _has_formula_topic_cues(self, text: str) -> bool:
        visible_text = self._coerce_string(text)
        if not visible_text:
            return False

        topic_patterns = [
            r"\bformula(?:s)?\b",
            r"\bequation(?:s)?\b",
            r"\bfourier series\b",
            r"\bcoefficient(?:s)?\b",
            r"\bkirchhoff\b",
            r"\bdistance between two points\b",
            r"\bdirection cosines?\b",
            r"\btransform\b",
            r"\bidentity\b",
            r"\blaw\b",
        ]
        return any(
            re.search(pattern, visible_text, flags=re.IGNORECASE)
            for pattern in topic_patterns
        )

    def _filter_supported_image_equations(
        self,
        equations: List[Dict[str, Any]],
        support_text: str,
        allow_formula_fallback: bool = False,
    ) -> List[Dict[str, Any]]:
        if not equations:
            return []

        normalized_support = self._normalize_visible_math_text(support_text)
        if not normalized_support:
            return []

        filtered: List[Dict[str, Any]] = []
        seen: set[str] = set()

        for item in equations:
            if not isinstance(item, dict):
                continue

            forms: List[str] = []
            for key in ["raw", "latex"]:
                form = self._normalize_visible_math_text(item.get(key))
                if form and len(form) >= 4 and form not in forms:
                    forms.append(form)

            if not forms:
                continue

            raw_text = self._coerce_string(item.get("raw") or item.get("latex"))
            has_direct_support = any(form in normalized_support for form in forms)
            has_fragment_support = False
            if not has_direct_support and "=" not in raw_text:
                fragment_tokens = [
                    self._normalize_visible_math_text(token)
                    for token in re.findall(r"[A-Za-z]+\d*|\d+", raw_text)
                ]
                fragment_tokens = [
                    token for token in fragment_tokens if token and len(token) >= 2
                ]
                has_fragment_support = bool(fragment_tokens) and all(
                    token in normalized_support for token in fragment_tokens
                )

            has_formula_fallback = False
            if not has_direct_support and not has_fragment_support and allow_formula_fallback:
                advanced_markers = [
                    r"\\(?:sum|int|sqrt|frac|omega|alpha|beta|gamma)",
                    r"[Σ∑∫√]",
                    r"\b(?:a_n|b_n|c_n|x\(t\)|omega_?0|w_?0)\b",
                    r"\b(?:sin|cos|tan)\s*\(",
                    r"[A-Za-z]_[A-Za-z0-9]+",
                    r"\^[A-Za-z0-9]",
                ]
                marker_hits = sum(
                    1
                    for pattern in advanced_markers
                    if re.search(pattern, raw_text, flags=re.IGNORECASE)
                )
                operator_count = len(re.findall(r"[+\-*/=^]", raw_text))
                has_formula_fallback = (
                    "=" in raw_text
                    and (
                        marker_hits >= 1
                        or (
                            operator_count >= 3
                            and any(token in raw_text for token in ["(", ")", "_", "^"])
                        )
                    )
                )

            if not has_direct_support and not has_fragment_support and not has_formula_fallback:
                continue

            canonical = self._canonicalize_equation_text(
                self._coerce_string(item.get("latex") or item.get("raw"))
            )
            if canonical and canonical in seen:
                continue
            if canonical:
                seen.add(canonical)
            filtered.append(item)

        return filtered

    def _split_image_for_detail_pass(self, image: Image.Image) -> List[Image.Image]:
        rgb = image.convert("RGB")
        width, height = rgb.size
        if height < 420:
            return [rgb]

        overlap = max(48, int(height * 0.12))
        midpoint = height // 2
        windows = [
            (0, min(height, midpoint + overlap)),
            (max(0, midpoint - overlap), height),
        ]
        crops: List[Image.Image] = []
        for top, bottom in windows:
            if bottom - top < 160:
                continue
            crops.append(rgb.crop((0, top, width, bottom)))
        return crops or [rgb]

    def _extract_image_detail_pass(
        self,
        image: Image.Image,
        max_new_tokens: int,
    ) -> Dict[str, Any]:
        detail_prompt = (
            "Analyze this <image> crop and respond with valid JSON only.\n"
            "Return ONLY these keys: equations, key_concepts, text.\n"
            'equations must include only clearly visible formula text, coefficient definitions, and labeled math expressions from this crop as [{"raw": "...", "latex": "..."}].\n'
            "key_concepts should be short noun phrases, not full sentences.\n"
            "text should be a concise OCR-style transcription of visible labels and formulas.\n"
            "Do not infer an equation from the shape of a graph or curve. Prefer exact visible content over explanation."
        )

        start = perf_counter()
        merged_equations: List[Dict[str, Any]] = []
        merged_key_concepts: List[str] = []
        text_parts: List[str] = []
        for crop in self._split_image_for_detail_pass(image):
            raw, _ = self._run_vl2(
                detail_prompt,
                images=[crop],
                max_new_tokens=min(max_new_tokens, 220),
            )
            detail_result = self._extract_json(raw)
            equations = self._normalize_equations(detail_result.get("equations"), 0.68, default_page=1)
            merged_equations = self._merge_equation_lists(merged_equations, equations)
            merged_key_concepts = self._merge_unique_strings(
                merged_key_concepts,
                self._coerce_string_list(detail_result.get("key_concepts")),
            )
            detail_text = self._visual_support_text_from_result(detail_result)
            if detail_text:
                text_parts.append(detail_text)

        return {
            "equations": merged_equations,
            "key_concepts": merged_key_concepts,
            "text": "\n".join(text_parts).strip(),
            "detail_pass_seconds": round(perf_counter() - start, 3),
        }

    def _canonicalize_equation_text(self, value: str) -> str:
        text = self._coerce_string(value)
        if not text:
            return ""
        text = text.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        text = text.replace("\\\\", "\\")
        text = re.sub(r"\s+", "", text)
        text = text.replace("[", "").replace("]", "")
        return text.casefold()

    def _dedupe_visual_equations(self, equations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in equations:
            if not isinstance(item, dict):
                continue
            raw = self._coerce_string(item.get("raw"))
            latex = self._coerce_string(item.get("latex"))
            canonical = self._canonicalize_equation_text(latex or raw)
            if not canonical or canonical in seen:
                continue
            seen.add(canonical)
            deduped.append(item)
        return deduped

    def _extract_clean_equation_variables(self, equations: List[Dict[str, Any]]) -> List[str]:
        variables: List[str] = []
        stopwords = {"cos", "sin", "frac", "int", "sum", "infty"}
        pattern = r"\\?[A-Za-z\u03b1-\u03c9\u0391-\u03a9]+(?:_[A-Za-z0-9]+)?(?:\([A-Za-z0-9_, ]+\))?"

        for item in equations:
            if not isinstance(item, dict):
                continue
            raw = self._coerce_string(item.get("raw") or item.get("latex"))
            normalized = (
                raw.replace("\\omega", "omega")
                .replace("\\alpha", "alpha")
                .replace("\\beta", "beta")
                .replace("\\gamma", "gamma")
                .replace("\\lambda", "lambda")
            )
            for token in re.findall(pattern, normalized):
                cleaned = token.lstrip("\\")
                if cleaned.casefold() in stopwords:
                    continue
                if cleaned and cleaned not in variables:
                    variables.append(cleaned)
        return variables[:12]

    def _refine_visual_result(
        self,
        result: Dict[str, Any],
        supplemental_text: str,
        input_type: str,
    ) -> Dict[str, Any]:
        source_parts = [
            self._coerce_string(result.get("topic")),
            " ".join(result.get("asks", [])),
            supplemental_text,
        ]
        source_text = " ".join(part for part in source_parts if part).casefold()

        engineering_markers = [
            "kirchhoff",
            "current law",
            "voltage law",
            "electrical circuit",
            "circuit",
            "current",
            "voltage",
            "node",
            "loop",
        ]
        mathematics_markers = [
            "geometry",
            "distance between two points",
            "direction cosines",
            "equation of a line",
            "equation of a plane",
            "vector form",
            "cartesian form",
            "direction ratios",
            "dot product",
            "shortest distance",
            "planes",
            "axes",
            "jee",
        ]
        physics_only_markers = [
            "quantum",
            "hamiltonian",
            "schrodinger",
            "wave function",
            "relativity",
        ]
        engineering_score = self._count_marker_hits(source_text, engineering_markers)
        mathematics_score = self._count_marker_hits(source_text, mathematics_markers)
        physics_score = self._count_marker_hits(source_text, physics_only_markers)

        if engineering_score and engineering_score >= mathematics_score:
            result["domain"] = "engineering"
        elif mathematics_score and mathematics_score > engineering_score and mathematics_score >= physics_score:
            result["domain"] = "mathematics"
        elif physics_score and physics_score > max(engineering_score, mathematics_score):
            result["domain"] = "physics"

        if input_type == "pdf":
            clean_equations = self._extract_clean_source_equations(supplemental_text)
            if clean_equations:
                equation_confidence = (
                    result.get("field_confidence", {}).get("equation_confidence", 0.65)
                )
                result["equations"] = self._normalize_equations(
                    clean_equations,
                    equation_confidence,
                    default_page=1,
                )
                result["quality_flags"]["has_equations"] = bool(result["equations"])

            headings = self._extract_pdf_headings(supplemental_text)
            if headings:
                if self._coerce_string(result.get("topic")).casefold() in {"", "main concept"}:
                    topic_candidates = [heading for heading in headings if heading != "3D Geometry – Exam Ready Notes"]
                    result["topic"] = topic_candidates[0] if topic_candidates else headings[0]
                result["key_concepts"] = self._merge_unique_strings(headings[:5], result.get("key_concepts", []))[:6]
                entities = result.get("entities", {})
                concepts = entities.get("concepts", [])
                entities["concepts"] = self._merge_unique_strings(headings[:5], concepts)[:12]
                result["entities"] = entities

        if result.get("domain") != "physics" and physics_score == 0:
            banned_concepts = {
                "Hamiltonian operator",
                "quantum mechanics",
                "Ĥ",
                "reduced Planck constant",
                "wave function",
            }
            result = self._remove_contaminated_concepts(result, banned_concepts)

        result["equations"] = self._dedupe_visual_equations(result.get("equations", []))
        result["quality_flags"]["has_equations"] = bool(result["equations"])

        entities = result.get("entities", {})
        if not entities.get("concepts"):
            concept_candidates: List[str] = []
            for item in [self._coerce_string(result.get("topic")), *result.get("key_concepts", [])]:
                concept_candidates.extend(self._condense_concept_text(item))
            entities["concepts"] = self._merge_unique_strings(concept_candidates, [])[:12]

        entities["variables"] = self._extract_clean_equation_variables(result.get("equations", []))
        equation_symbol_text = " ".join(
            self._coerce_string(item.get("raw") or item.get("latex"))
            for item in result.get("equations", [])
            if isinstance(item, dict)
        )
        entities["symbols"] = self._extract_symbol_entities(f"{supplemental_text}\n{equation_symbol_text}")
        result["entities"] = entities

        condensed_key_concepts: List[str] = []
        for item in result.get("key_concepts", []):
            condensed_key_concepts.extend(self._condense_concept_text(item))
        if condensed_key_concepts:
            result["key_concepts"] = self._merge_unique_strings(
                condensed_key_concepts,
                result.get("key_concepts", []),
            )[:6]

        return result

    def _run_vl2(
        self,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
        max_new_tokens: int = 512,
    ) -> Tuple[str, Dict[str, Any]]:
        prompt = self._ensure_image_placeholder(prompt, images)
        preparation_start = perf_counter()
        if images:
            image = self._compose_images(images, label_prefix="Image")
        else:
            image = self._text_to_image(prompt)

        data_uri, image_stats = self._image_to_data_uri(image)
        input_preparation_seconds = perf_counter() - preparation_start

        payload = {
            "version": self.model_version,
            "input": {
                "image": data_uri,
                "prompt": prompt,
                "temperature": float(self.config.get("temperature", 0.1)),
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "max_length_tokens": int(max_new_tokens),
            }
        }

        request_start = perf_counter()
        prediction = self._request_json(
            self.predictions_url,
            method="POST",
            payload=payload,
            extra_headers={"Prefer": f"wait={self.api_wait_seconds}"},
        )
        prediction = self._wait_for_prediction(prediction)
        model_call_seconds = perf_counter() - request_start

        status = prediction.get("status")
        if status != "succeeded":
            raise ReplicatePredictionError(
                f"Replicate prediction ended with status={status!r}: {prediction.get('error')}"
            )

        metrics = prediction.get("metrics") or {}
        generation_seconds = metrics.get("predict_time", model_call_seconds)
        try:
            generation_seconds = float(generation_seconds)
        except (TypeError, ValueError):
            generation_seconds = model_call_seconds

        output_text = self._prediction_output_to_text(prediction.get("output"))
        return output_text, {
            "model_name": self.model_name,
            "model_loaded_now": False,
            "model_load_seconds": 0.0,
            "input_preparation_seconds": round(input_preparation_seconds, 3),
            "generation_seconds": round(generation_seconds, 3),
            "decode_seconds": 0.0,
            "model_call_seconds": round(model_call_seconds, 3),
            "prediction_id": prediction.get("id"),
            "prediction_status": status,
            "image_bytes": image_stats["bytes"],
            "image_width": image_stats["width"],
            "image_height": image_stats["height"],
            "jpeg_quality": image_stats["jpeg_quality"],
        }

    def parse_text(self, text: str) -> Dict[str, Any]:
        """Parse a text query into structured JSON using Replicate-hosted VL2."""
        parse_start = perf_counter()
        analysis = self._analyze_text(text, input_type="text")

        print(f"Detected domain: {analysis['domain']}")
        print(f"Detected complexity: {analysis['complexity']} (score: {analysis['complexity_score']})")
        print(f"Contains equations: {analysis['has_equations']}")
        if analysis["equations_found"]:
            print(f"Equations found: {analysis['equations_found']}")
        if analysis["technical_terms_found"]:
            print(f"Technical terms: {analysis['technical_terms_found'][:8]}")

        prompt = (
            "Parse this query and respond with valid JSON only.\n"
            "Return ONLY these keys to keep output short: intent, secondary_intents, asks, topic, domain, complexity, language, key_concepts, equations, entities.\n"
            "Do not return explanations or markdown.\n"
            'Intent must be one of: "concept_explanation", "equation_visualization", "step_by_step", "comparison", "application", "derivation".\n'
            'secondary_intents must be an array and can include: "implication_explanation", "application", "comparison", "equation_visualization", "step_by_step", "concept_explanation".\n'
            '"asks" must be an array of concise actionable strings.\n'
            'Domain must be one of: "mathematics", "physics", "cs", "engineering", "general".\n'
            'Complexity must be one of: "basic", "intermediate", "advanced".\n'
            'equations must be an array of objects with keys raw, latex.\n'
            'entities must include only: symbols, concepts, variables, constants.\n'
            f"Detected hints: language={analysis['language']}, domain={analysis['domain']}, "
            f"complexity={analysis['complexity']}, equations={analysis['equations_found']}, "
            f"key_concepts={analysis['key_concepts']}.\n"
            f"Query: {text}"
        )

        text_image = self._text_to_image(text)
        response, model_timing = self._run_vl2(
            prompt,
            images=[text_image],
            max_new_tokens=self.config.get("max_new_tokens_text", 320),
        )
        result = self._normalize_text_result(self._extract_json(response), analysis)
        result["_model"] = self.model_name
        result["_api_backend"] = "replicate"
        result["_timing"] = {
            **model_timing,
            "total_parse_seconds": round(perf_counter() - parse_start, 3),
        }
        return self._finalize_result(result)

    def parse_image(self, image: Image.Image, prompt_text: Optional[str] = None) -> Dict[str, Any]:
        """Parse a single image into structured JSON using Replicate-hosted VL2."""
        parse_start = perf_counter()
        user_prompt = self._coerce_string(prompt_text)
        prompt = (
            "Analyze this <image> and respond with valid JSON only.\n"
            "Return ONLY these keys to keep output short: intent, topic, domain, complexity, language, "
            "key_concepts, entities, equations, text, summary.\n"
            'intent should be "ocr_extraction".\n'
            'domain must be one of: "mathematics", "physics", "cs", "engineering", "general".\n'
            'complexity must be one of: "basic", "intermediate", "advanced".\n'
            'entities must include only: symbols, concepts, variables, constants.\n'
            'If equations are visible, return equations as [{"raw": "...", "latex": "..."}]. Include every clearly visible formula, not just the main one.\n'
            "text should be a concise OCR-style transcription of visible headings, labels, and formulas.\n"
            "Do not infer an equation from the shape of a graph or plot. Prefer exact visible text over speculation. Keep text concise and summarize the main diagram/document content."
        )
        if user_prompt:
            prompt = f"{prompt}\nUser instruction: {user_prompt}"

        response, model_timing = self._run_vl2(
            prompt,
            images=[image],
            max_new_tokens=self.config.get("max_new_tokens_image", 260),
        )
        model_result = self._extract_json(response)
        supplemental_parts = []
        if user_prompt:
            supplemental_parts.append(user_prompt)
        visual_text = self._visual_support_text_from_result(model_result)
        if visual_text:
            supplemental_parts.append(visual_text)
        supplemental_text = "\n".join(supplemental_parts)
        formula_topic_cues = self._has_formula_topic_cues(supplemental_text)
        result = self._normalize_visual_result(
            model_result,
            input_type="image",
            supplemental_text=supplemental_text,
            page_count=1,
            document_metadata={"page_count": 1, "is_scanned": True, "has_text_layer": False},
        )
        model_equations = self._normalize_equations(
            model_result.get("equations"),
            result.get("field_confidence", {}).get("equation_confidence", 0.68),
            default_page=1,
        )
        if model_equations:
            result["equations"] = self._filter_supported_image_equations(
                model_equations,
                support_text=supplemental_text,
                allow_formula_fallback=formula_topic_cues,
            )
            result["quality_flags"]["has_equations"] = bool(result["equations"])
        result = self._refine_visual_result(result, supplemental_text, input_type="image")

        detail_timing: Dict[str, Any] = {}
        formula_evidence = self._has_formula_evidence(supplemental_text)
        needs_detail_pass = (
            not result.get("entities", {}).get("concepts")
            or (
                (formula_evidence or formula_topic_cues)
                and len(result.get("equations", [])) < 3
            )
        )
        if needs_detail_pass:
            detail_pass = self._extract_image_detail_pass(
                image,
                max_new_tokens=self.config.get("max_new_tokens_image", 260),
            )
            if detail_pass["equations"]:
                result["equations"] = self._merge_equation_lists(
                    result.get("equations", []),
                    detail_pass["equations"],
                )
                result["quality_flags"]["has_equations"] = bool(result["equations"])
            if detail_pass["key_concepts"]:
                result["key_concepts"] = self._merge_unique_strings(
                    detail_pass["key_concepts"],
                    result.get("key_concepts", []),
                )[:6]
            if detail_pass["text"]:
                supplemental_text = "\n".join(
                    part for part in [supplemental_text, detail_pass["text"]] if part
                )
            if result.get("equations"):
                updated_formula_topic_cues = self._has_formula_topic_cues(supplemental_text)
                result["equations"] = self._filter_supported_image_equations(
                    result.get("equations", []),
                    support_text=supplemental_text,
                    allow_formula_fallback=updated_formula_topic_cues,
                )
                result["quality_flags"]["has_equations"] = bool(result["equations"])
            if detail_pass["text"] or detail_pass["equations"] or detail_pass["key_concepts"]:
                result = self._refine_visual_result(result, supplemental_text, input_type="image")
            detail_timing = {
                "detail_pass_seconds": detail_pass["detail_pass_seconds"],
                "detail_pass_applied": True,
            }

        result["_model"] = self.model_name
        result["_api_backend"] = "replicate"
        result["_timing"] = {
            **model_timing,
            **detail_timing,
            "total_parse_seconds": round(perf_counter() - parse_start, 3),
        }
        return self._finalize_result(result)

    def parse_pdf(
        self,
        pdf_path: str,
        pages: Optional[List[int]] = None,
        prompt_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Parse a PDF by stitching selected pages into one image for Replicate-hosted VL2."""
        parse_start = perf_counter()
        pdf_full_path = self._resolve_input_path(pdf_path)
        if not pdf_full_path.exists():
            return {"error": f"PDF not found: {pdf_full_path}"}

        print(f"Processing PDF: {pdf_full_path}")
        inspected_metadata, extracted_pdf_text = self._inspect_pdf_metadata(pdf_full_path)

        pdf_render_start = perf_counter()
        images = self._pdf_to_images(pdf_full_path)
        pdf_render_seconds = perf_counter() - pdf_render_start
        print(f"Converted {len(images)} pages to images")

        if pages:
            images = [images[index] for index in pages if index < len(images)]

        max_pages = self.config.get("max_pages", 3)
        if len(images) > max_pages:
            print(f"Limiting to first {max_pages} pages")
            images = images[:max_pages]

        if not images:
            return {"error": f"No PDF pages available for {pdf_full_path}"}

        prompt = (
            "Analyze this <image> and respond with valid JSON only.\n"
            "Return ONLY these keys: intent, topic, domain, complexity, language, key_concepts, equations, "
            "text, summary, title, tables, figures.\n"
            'intent should be "document_extraction".\n'
            'domain must be one of: "mathematics", "physics", "cs", "engineering", "general".\n'
            'complexity must be one of: "basic", "intermediate", "advanced".\n'
            "Keep the response compact. Focus on title, structure, equations, tables, and figures.\n"
            "The image input may contain multiple stitched PDF pages in reading order."
        )
        if extracted_pdf_text:
            prompt = f"{prompt}\nText layer hint:\n{extracted_pdf_text[:4000]}"
        user_prompt = self._coerce_string(prompt_text)
        if user_prompt:
            prompt = f"{prompt}\nUser instruction: {user_prompt}"

        stitched = self._compose_images(images, label_prefix="Page")
        response, model_timing = self._run_vl2(
            prompt,
            images=[stitched],
            max_new_tokens=self.config.get("max_new_tokens_pdf", 320),
        )
        model_result = self._extract_json(response)
        supplemental_parts = []
        if user_prompt:
            supplemental_parts.append(user_prompt)
        for item in [self._visual_text_from_result(model_result), extracted_pdf_text]:
            if item:
                supplemental_parts.append(item)
        supplemental_text = "\n".join(supplemental_parts)
        result = self._normalize_visual_result(
            model_result,
            input_type="pdf",
            supplemental_text=supplemental_text,
            page_count=len(images),
            document_metadata={
                **inspected_metadata,
                "page_count": len(images),
            },
        )
        result = self._refine_visual_result(result, supplemental_text, input_type="pdf")
        result["_model"] = self.model_name
        result["_api_backend"] = "replicate"
        result["_pages_processed"] = len(images)
        result["_timing"] = {
            "pdf_render_seconds": round(pdf_render_seconds, 3),
            **model_timing,
            "total_parse_seconds": round(perf_counter() - parse_start, 3),
        }
        return self._finalize_result(result)
