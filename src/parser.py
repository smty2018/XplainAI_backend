"""Parsing helpers and pipeline entrypoints using DeepSeek-VL2 Tiny only."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import unicodedata
from datetime import datetime
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple, Union

import fitz
import torch
import yaml
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalParser:
    """Local parser backed by DeepSeek-VL2 Tiny only."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.project_root = Path(__file__).parent.parent
        print(f"Project root: {self.project_root}")

        self.config = self._load_config(config_path)
        self.model_name = self.config["vl2_model"]

        self.model_cache = self.project_root / self.config["model_cache"]
        self.data_dir = self.project_root / self.config["data_dir"]
        self.output_dir = self.project_root / self.config["output_dir"]

        self.model_cache.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        os.environ["HF_HOME"] = str(self.model_cache)
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

        print(f"Models will be downloaded to: {self.model_cache}")
        print(f"Data folder: {self.data_dir}")
        print(f"Output folder: {self.output_dir}")

        self.device = self._get_device()
        self.model_dtype = self._get_model_dtype()
        print(f"Using device: {self.device}")
        print(f"Using model: {self.model_name}")
        print("DeepSeek-VL2 Tiny will be loaded on first use.")

        self.model = None
        self.processor = None
        self.tokenizer = None
        self.last_timing: Dict[str, Union[str, float, bool]] = {}
        self.domain_terms = {
            "physics": [
                "quantum mechanics",
                "quantum",
                "schrödinger",
                "schrodinger",
                "schrodinger equation",
                "hamiltonian",
                "lagrangian",
                "wave function",
                "wavefunction",
                "eigenvalue",
                "eigenstate",
                "operator",
                "uncertainty",
                "heisenberg",
                "momentum",
                "potential",
                "relativity",
                "ℏ",
                "ħ",
                "hbar",
                "ψ",
                "ϕ",
                "∂",
                "Ĥ",
            ],
            "mathematics": [
                "integral",
                "derivative",
                "fourier",
                "laplace",
                "matrix",
                "vector",
                "tensor",
                "theorem",
                "proof",
                "lemma",
                "eigenvalue",
                "∫",
                "∑",
                "∇",
            ],
            "cs": [
                "algorithm",
                "complexity",
                "neural network",
                "gradient",
                "machine learning",
                "deep learning",
                "transformer",
                "backpropagation",
                "optimization",
            ],
            "engineering": [
                "control system",
                "circuit",
                "kirchhoff",
                "current law",
                "voltage law",
                "current",
                "voltage",
                "node",
                "loop",
                "signal",
                "thermodynamics",
                "fluid",
                "stress",
                "strain",
                "transfer function",
            ],
        }
        self.topic_patterns = [
            (r"schr[öo]dinger equation", "Schrödinger equation"),
            (r"maxwell(?:'s)? equations?", "Maxwell's equations"),
            (r"navier[- ]stokes equations?", "Navier-Stokes equations"),
            (r"fourier transform", "Fourier transform"),
            (r"laplace transform", "Laplace transform"),
            (r"gradient descent", "Gradient descent"),
            (r"hamiltonian", "Hamiltonian"),
        ]

        self.language_patterns = {
            "zh": r"[\u4e00-\u9fff]",
            "ja": r"[\u3040-\u30ff]",
            "ko": r"[\uac00-\ud7af]",
            "hi": r"[\u0900-\u097f]",
            "ar": r"[\u0600-\u06ff]",
            "ru": r"[\u0400-\u04ff]",
        }
        self.valid_intents = {
            "concept_explanation",
            "equation_visualization",
            "step_by_step",
            "comparison",
            "application",
            "derivation",
            "document_extraction",
            "ocr_extraction",
        }
        self.valid_secondary_intents = {
            "implication_explanation",
            "application",
            "comparison",
            "equation_visualization",
            "step_by_step",
            "concept_explanation",
        }
        self.valid_domains = {"mathematics", "physics", "cs", "engineering", "general"}
        self.valid_complexities = {"basic", "intermediate", "advanced"}
        self.valid_layout_types = {
            "title",
            "heading",
            "paragraph",
            "list",
            "table",
            "figure",
            "caption",
            "footnote",
            "code",
            "equation",
        }
        self.valid_output_format_hints = {"json", "markdown", "html", "latex"}

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML."""
        config_file = self.project_root / config_path
        with open(config_file, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def _get_device(self) -> str:
        """Pick the best available device."""
        if self.config.get("use_gpu") and torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _get_model_dtype(self) -> torch.dtype:
        """Use a conservative dtype for the selected device."""
        if self.device == "cuda":
            return torch.bfloat16
        return torch.float32

    def _resolve_project_path(self, path_value: str) -> Path:
        """Resolve a project-relative path."""
        candidate = Path(path_value)
        if candidate.is_absolute():
            return candidate
        return self.project_root / candidate

    def _resolve_input_path(self, input_path: str) -> Path:
        """Resolve an absolute, project-relative, or data-relative input path."""
        candidate = Path(input_path)
        if candidate.is_absolute():
            return candidate

        project_relative = self.project_root / candidate
        if project_relative.exists():
            return project_relative

        return self.data_dir / candidate

    def _load_vl2_classes(self):
        """Import DeepSeek-VL2 classes from the bundled repository."""
        repo_path = self._resolve_project_path(
            self.config.get("vl2_repo_path", "./DeepSeek-VL2")
        )
        if not repo_path.exists():
            raise RuntimeError(
                f"DeepSeek-VL2 repo not found at {repo_path}. "
                "Clone it into the project root or update vl2_repo_path."
            )

        repo_str = str(repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        try:
            from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "DeepSeek-VL2 dependencies are missing. "
                "Run `pip install -r requirements.txt` inside the active venv."
            ) from exc

        return DeepseekVLV2ForCausalLM, DeepseekVLV2Processor

    def _ensure_model_loaded(self) -> Dict[str, Union[str, float, bool]]:
        """Load DeepSeek-VL2 Tiny on demand and return timing details."""
        if self.model is not None:
            return {
                "model_name": self.model_name,
                "model_loaded_now": False,
                "model_load_seconds": 0.0,
            }

        load_start = perf_counter()

        print("=" * 50)
        print("LOADING MODEL: tiny")
        print("=" * 50)
        print(f"Loading {self.model_name}...")

        DeepseekVLV2ForCausalLM, DeepseekVLV2Processor = self._load_vl2_classes()

        self.processor = DeepseekVLV2Processor.from_pretrained(
            self.model_name,
            cache_dir=str(self.model_cache),
        )
        self.tokenizer = self.processor.tokenizer

        self.model = DeepseekVLV2ForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=str(self.model_cache),
            torch_dtype=self.model_dtype,
            low_cpu_mem_usage=True,
            _attn_implementation="eager",
        )
        self.model = self.model.to(self.device).eval()

        print(f"{self.model_name} loaded successfully.")

        return {
            "model_name": self.model_name,
            "model_loaded_now": True,
            "model_load_seconds": round(perf_counter() - load_start, 3),
        }

    def _run_vl2(
        self,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Union[str, float, bool]]]:
        """Run a prompt through DeepSeek-VL2 Tiny and return raw text with timing."""
        call_start = perf_counter()
        load_timing = self._ensure_model_loaded()

        normalized_images = [image.convert("RGB") for image in (images or [])]
        image_prefix = "\n".join("<image>" for _ in normalized_images)
        user_content = prompt.strip()
        if image_prefix:
            user_content = f"{image_prefix}\n{user_content}".strip()

        conversation = [
            {"role": "<|User|>", "content": user_content},
            {"role": "<|Assistant|>", "content": ""},
        ]

        prep_start = perf_counter()
        prepare_inputs = self.processor(
            conversations=conversation,
            images=normalized_images,
            force_batchify=True,
            system_prompt="",
        ).to(self.device, dtype=self.model_dtype)
        input_preparation_seconds = perf_counter() - prep_start

        generation_max_tokens = max_new_tokens or self.config.get("max_new_tokens", 512)
        generation_start = perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=prepare_inputs.input_ids,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=generation_max_tokens,
                do_sample=False,
                use_cache=True,
            )
        generation_seconds = perf_counter() - generation_start

        decode_start = perf_counter()
        generated_tokens = outputs[0][prepare_inputs.input_ids.shape[1] :]
        decoded_text = self.tokenizer.decode(
            generated_tokens.cpu().tolist(),
            skip_special_tokens=False,
        )
        decode_seconds = perf_counter() - decode_start

        timing = {
            **load_timing,
            "input_preparation_seconds": round(input_preparation_seconds, 3),
            "generation_seconds": round(generation_seconds, 3),
            "decode_seconds": round(decode_seconds, 3),
            "model_call_seconds": round(perf_counter() - call_start, 3),
        }
        return decoded_text, timing

    def _finalize_result(self, result: Dict) -> Dict:
        """Save a result after all metadata has been attached."""
        self.last_timing = result.get("_timing", {})
        self._save_result(result)
        return result

    def _extract_equations_from_text(self, text: str) -> List[str]:
        """Extract likely equations from a text query."""
        candidates: List[str] = []

        colon_match = re.search(r":\s*([^:\n]+?=[^:\n]+)", text)
        if colon_match:
            candidates.append(colon_match.group(1))

        for match in re.findall(
            r"([A-Za-z0-9_ℏħψϕφθπσμνρτβγαδλĤ∂∇∫∑∞+\-*/^(){}\[\]\s]+[=≈≃≤≥][A-Za-z0-9_ℏħψϕφθπσμνρτβγαδλĤ∂∇∫∑∞+\-*/^(){}\[\]\s]+)",
            text,
        ):
            candidates.append(match)

        cleaned: List[str] = []
        for candidate in candidates:
            equation = re.split(
                r"\b(?:and explain|and discuss|and derive|and show|where|because|in quantum mechanics|in physics)\b",
                candidate,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            equation = equation.strip(" \t\n\r.,;:")
            equation = re.sub(r"^(?:derive|show|prove)\s+", "", equation, flags=re.IGNORECASE)
            if "=" in equation and len(equation) >= 8 and equation not in cleaned:
                cleaned.append(equation)

        filtered: List[str] = []
        for equation in sorted(cleaned, key=len, reverse=True):
            if not any(equation != other and equation in other for other in filtered):
                filtered.append(equation)

        return filtered

    def _lookup_text_variants(self, text: str) -> Tuple[str, str]:
        """Return Unicode-aware and ASCII-folded lowercase variants."""
        lowered = text.casefold()
        ascii_folded = (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("ascii")
            .casefold()
        )
        return lowered, ascii_folded

    def _detect_language(self, text: str) -> str:
        """Detect the dominant language using simple script heuristics."""
        stripped = text.strip()
        if not stripped:
            return "en"

        for language, pattern in self.language_patterns.items():
            if re.search(pattern, stripped):
                return language

        return "en"

    def _has_strong_quantum_signal(self, text: str) -> bool:
        """Require more than one quantum cue before tagging quantum-specific concepts."""
        lowered, ascii_folded = self._lookup_text_variants(text)
        explicit_terms = [
            "quantum",
            "schrodinger",
            "schrödinger",
            "hamiltonian",
            "wave function",
            "wavefunction",
        ]
        explicit_hits = sum(
            1
            for term in explicit_terms
            if term in lowered or term in ascii_folded
        )

        symbol_hits = 0
        if any(symbol in text for symbol in ["ℏ", "ħ", "â„", "Ä§"]):
            symbol_hits += 1
        if any(symbol in text for symbol in ["ψ", "Ïˆ"]):
            symbol_hits += 1
        if any(symbol in text for symbol in ["Ĥ", "Ä¤"]):
            symbol_hits += 1
        if any(symbol in text for symbol in ["∂", "âˆ‚"]):
            symbol_hits += 1

        return explicit_hits >= 1 or symbol_hits >= 2

    def _sanitize_equation_candidate(self, candidate: str) -> str:
        """Clean OCR-like equation fragments and reject sentence-like noise."""
        equation = candidate.replace("\r", "\n").strip(" \t\n\r.,;:")
        if "\n" in equation:
            lines = [line.strip() for line in equation.splitlines() if "=" in line]
            if lines:
                equation = lines[0]

        if any(char in equation for char in ["[", "]", "{", "}", "\"", "'"]):
            match = re.search(
                r"([A-Za-z0-9_Σ∑∫∆+\-*/^() ]+[=≈≃≤≥][A-Za-z0-9_Σ∑∫∆+\-*/^() ]+)",
                equation,
            )
            equation = match.group(1).strip() if match else ""

        if not equation or "=" not in equation:
            return ""

        long_words = re.findall(r"\b[A-Za-z]{4,}\b", equation)
        if len(long_words) > 2:
            return ""

        if not re.search(r"[A-Za-zΣ∑∫∆_]", equation):
            return ""

        equation = re.sub(r"\s+", " ", equation).strip()
        return equation

    def _find_technical_terms(self, text: str) -> List[str]:
        """Find notable technical terms in the text."""
        found: List[str] = []
        seen_canonical: set[str] = set()
        lowered, ascii_folded = self._lookup_text_variants(text)
        all_terms: List[str] = []
        for terms in self.domain_terms.values():
            all_terms.extend(terms)

        for term in sorted(set(all_terms), key=len, reverse=True):
                term_lower = term.casefold()
                term_ascii = (
                    unicodedata.normalize("NFKD", term)
                    .encode("ascii", "ignore")
                    .decode("ascii")
                    .casefold()
                )
                canonical = term_ascii or term_lower
                if canonical in seen_canonical:
                    continue
                if term_lower in lowered or (term_ascii and term_ascii in ascii_folded):
                    found.append(term)
                    seen_canonical.add(canonical)
        return found

    def _infer_domain(self, text: str, technical_terms: List[str]) -> Tuple[str, Dict[str, int]]:
        """Infer the most likely domain from the text."""
        lowered, ascii_folded = self._lookup_text_variants(text)
        scores: Dict[str, int] = {domain: 0 for domain in self.domain_terms}

        for domain, terms in self.domain_terms.items():
            for term in terms:
                term_lower = term.casefold()
                term_ascii = (
                    unicodedata.normalize("NFKD", term)
                    .encode("ascii", "ignore")
                    .decode("ascii")
                    .casefold()
                )
                if term_lower in lowered or (term_ascii and term_ascii in ascii_folded):
                    scores[domain] += 1

        if any(symbol in text for symbol in ["ℏ", "ħ", "ψ", "Ĥ"]):
            scores["physics"] += 2

        best_domain = max(scores, key=scores.get)
        if scores[best_domain] == 0:
            best_domain = "general"

        return best_domain, scores

    def _infer_complexity(
        self,
        text: str,
        equations: List[str],
        technical_terms: List[str],
    ) -> Tuple[str, int]:
        """Infer complexity and a numeric score."""
        score = 0
        word_count = len(text.split())

        if word_count >= 30:
            score += 20
        elif word_count >= 15:
            score += 10
        elif word_count >= 8:
            score += 5

        score += min(len(technical_terms) * 6, 30)

        if equations:
            score += 20

        if any(symbol in text for symbol in ["ℏ", "ħ", "ψ", "∂", "∫", "∑", "∇", "Ĥ"]):
            score += 15

        if any(word in text.casefold() for word in ["derive", "derivation", "hamiltonian", "implications"]):
            score += 10

        if score >= 50:
            return "advanced", score
        if score >= 25:
            return "intermediate", score
        return "basic", score

    def _infer_intent(self, text: str, input_type: str = "text") -> str:
        """Infer the user's intent from instruction keywords."""
        lower_text, ascii_folded = self._lookup_text_variants(text)
        if input_type == "pdf":
            return "document_extraction"
        if input_type == "image":
            return "ocr_extraction"
        if any(keyword in lower_text or keyword in ascii_folded for keyword in ["derive", "derivation", "show that", "prove"]):
            return "derivation"
        if any(keyword in lower_text or keyword in ascii_folded for keyword in ["compare", "difference", "versus", " vs "]):
            return "comparison"
        if any(keyword in lower_text or keyword in ascii_folded for keyword in ["visualize", "plot", "graph", "diagram"]):
            return "equation_visualization"
        if any(keyword in lower_text or keyword in ascii_folded for keyword in ["step by step", "steps", "walk through"]):
            return "step_by_step"
        if any(keyword in lower_text or keyword in ascii_folded for keyword in ["application", "use case", "implication", "implications"]):
            return "application"
        return "concept_explanation"

    def _infer_secondary_intents(self, text: str, primary_intent: str) -> List[str]:
        """Infer any secondary user intents from the prompt."""
        lower_text, ascii_folded = self._lookup_text_variants(text)
        secondary: List[str] = []

        if any(keyword in lower_text or keyword in ascii_folded for keyword in ["implication", "implications", "significance"]):
            secondary.append("implication_explanation")
        if any(keyword in lower_text or keyword in ascii_folded for keyword in ["application", "applications", "use case", "use cases"]):
            secondary.append("application")
        if any(keyword in lower_text or keyword in ascii_folded for keyword in ["compare", "difference", "versus", " vs "]):
            secondary.append("comparison")
        if any(keyword in lower_text or keyword in ascii_folded for keyword in ["visualize", "plot", "graph", "diagram"]):
            secondary.append("equation_visualization")
        if any(keyword in lower_text or keyword in ascii_folded for keyword in ["step by step", "steps", "walk through"]):
            secondary.append("step_by_step")
        if any(keyword in lower_text or keyword in ascii_folded for keyword in ["explain", "interpret", "intuition"]):
            secondary.append("concept_explanation")

        filtered: List[str] = []
        for item in secondary:
            if item != primary_intent and item in self.valid_secondary_intents and item not in filtered:
                filtered.append(item)
        return filtered

    def _extract_request_clauses(self, text: str) -> List[str]:
        """Split a multi-part prompt into imperative request clauses."""
        cleaned = re.sub(r"\s+", " ", text).strip(" .")
        if not cleaned:
            return []

        verb_pattern = (
            r"(derive|explain|solve|prove|compare|show|analyze|describe|summarize|"
            r"visualize|plot|walk through|compute|calculate)"
        )
        matches = re.findall(
            rf"({verb_pattern}.*?)(?=\s+\band\s+{verb_pattern}\b|$)",
            cleaned,
            flags=re.IGNORECASE,
        )

        clauses: List[str] = []
        for match in matches:
            clause = match[0] if isinstance(match, tuple) else match
            clause = clause.strip(" .")
            if clause:
                clauses.append(clause)

        return clauses or [cleaned]

    def _build_primary_ask(self, text: str, topic: str, intent: str) -> str:
        """Build the primary ask in a concise, orchestration-friendly form."""
        if topic and topic != "main concept":
            if intent == "derivation":
                return f"derive the {topic}"
            if intent == "comparison":
                return f"compare {topic}"
            if intent == "equation_visualization":
                return f"visualize {topic}"
            if intent == "step_by_step":
                return f"walk through {topic} step by step"
            if intent == "application":
                return f"explain applications of {topic}"
            return f"explain {topic}"

        clauses = self._extract_request_clauses(text)
        return clauses[0] if clauses else text.strip()

    def _infer_asks(
        self,
        text: str,
        topic: str,
        primary_intent: str,
        secondary_intents: List[str],
    ) -> List[str]:
        """Infer an ordered list of concrete asks from the prompt."""
        asks: List[str] = []
        primary_ask = self._build_primary_ask(text, topic, primary_intent)
        if primary_ask:
            asks.append(primary_ask)

        clauses = self._extract_request_clauses(text)
        for clause in clauses[1:]:
            normalized_clause = clause[:1].lower() + clause[1:] if clause else clause
            if normalized_clause and normalized_clause not in asks:
                asks.append(normalized_clause)

        if "implication_explanation" in secondary_intents:
            implication_match = re.search(
                r"(explain\s+(?:its|their|the)?\s*implications.*)$",
                text,
                flags=re.IGNORECASE,
            )
            implication_ask = (
                implication_match.group(1).strip(" .")
                if implication_match
                else f"explain the implications of {topic}"
            )
            implication_ask = implication_ask[:1].lower() + implication_ask[1:]
            if implication_ask not in asks:
                asks.append(implication_ask)

        return asks[:6]

    def _infer_verification_targets(
        self,
        text: str,
        domain: str,
        equations: List[str],
        intent: str,
    ) -> Dict[str, bool]:
        """Infer which verification tools are worth invoking downstream."""
        lower_text, ascii_folded = self._lookup_text_variants(text)
        has_units = bool(self._extract_units(text))
        return {
            "sympy": bool(equations) and domain in {"physics", "mathematics", "engineering"},
            "unit_check": has_units or domain == "engineering",
            "constraint_check": any(
                keyword in lower_text or keyword in ascii_folded
                for keyword in ["constraint", "subject to", "boundary condition", "given that"]
            ),
            "edge_case_check": intent in {"comparison", "step_by_step"} or any(
                keyword in lower_text or keyword in ascii_folded
                for keyword in ["edge case", "special case", "limit", "approximation"]
            ),
        }

    def _infer_retrieval_targets(
        self,
        text: str,
        intent: str,
        secondary_intents: List[str],
    ) -> Dict[str, bool]:
        """Infer what the retrieval layer should search for."""
        lower_text, ascii_folded = self._lookup_text_variants(text)
        return {
            "similar_problems": intent in {"derivation", "step_by_step", "comparison"},
            "misconceptions": "concept_explanation" in secondary_intents or "implication_explanation" in secondary_intents,
            "visualization_patterns": intent == "equation_visualization" or any(
                keyword in lower_text or keyword in ascii_folded
                for keyword in ["visualize", "diagram", "graph", "plot"]
            ),
            "explanation_style": intent in {"concept_explanation", "application"} or any(
                keyword in lower_text or keyword in ascii_folded
                for keyword in ["explain", "intuition", "implication", "implications"]
            ),
        }

    def _infer_topic(self, text: str) -> str:
        """Infer a specific topic label from the query."""
        lowered, ascii_folded = self._lookup_text_variants(text)
        for pattern, label in self.topic_patterns:
            if re.search(pattern, lowered, flags=re.IGNORECASE) or re.search(
                pattern,
                ascii_folded,
                flags=re.IGNORECASE,
            ):
                return label

        subject_match = re.search(
            r"(?:derive|explain|analyze|describe|summarize)\s+(?:the\s+)?(.+?)(?::|\band\b|\bwith\b|$)",
            text,
            flags=re.IGNORECASE,
        )
        if subject_match:
            return subject_match.group(1).strip(" .")

        return "main concept"

    def _infer_key_concepts(
        self,
        text: str,
        domain: str,
        technical_terms: List[str],
        topic: str,
    ) -> List[str]:
        """Infer a concise list of key concepts."""
        concepts: List[str] = []
        lower_text, ascii_folded = self._lookup_text_variants(text)

        if topic and topic != "main concept":
            concepts.append(topic)

        concept_map = [
            ("quantum", "quantum mechanics"),
            ("ψ", "wave function"),
            ("wave function", "wave function"),
            ("hamiltonian", "Hamiltonian operator"),
            ("Ĥ", "Hamiltonian operator"),
            ("∂", "time evolution"),
            ("eigenvalue", "eigenvalues"),
            ("operator", "operators"),
            ("fourier", "frequency-domain analysis"),
            ("gradient", "optimization"),
        ]

        for trigger, label in concept_map:
            trigger_lower = trigger.casefold()
            if trigger_lower in lower_text or trigger_lower in ascii_folded:
                concepts.append(label)

        for term in technical_terms:
            normalized = term
            if term in {"ℏ", "ħ"}:
                normalized = "reduced Planck constant"
            elif term == "∂":
                normalized = "partial derivatives"
            elif term == "ψ":
                normalized = "wave function"
            elif term == "Ĥ":
                normalized = "Hamiltonian operator"

            normalized_lower = normalized.casefold()
            if normalized_lower in {"quantum", "schrödinger", "schrodinger", "schrodinger equation"}:
                continue
            if any(normalized_lower in concept.casefold() for concept in concepts):
                continue
            if normalized not in concepts and len(concepts) < 6:
                concepts.append(normalized)

        if domain == "physics" and "quantum mechanics" not in concepts:
            concepts.append("quantum mechanics")

        return concepts[:6]

    def _extract_symbol_entities(self, text: str) -> List[str]:
        """Extract symbolic entities such as Greek letters and operators."""
        matches = re.findall(r"[\u210f\u0127\u03b1-\u03c9\u0391-\u03a9\u2202\u2207\u2211\u222b\u221e\u2248\u2264\u2265]", text)
        return self._merge_unique_strings(matches, [])

    def _extract_variable_entities(self, equations: List[str]) -> List[str]:
        """Extract likely variable tokens from equations."""
        variables: List[str] = []
        for equation in equations:
            for token in re.findall(r"[A-Za-z\u03b1-\u03c9\u0391-\u03a9_]+(?:\([A-Za-z0-9_, ]+\))?", equation):
                cleaned = token.strip()
                if cleaned and cleaned not in variables:
                    variables.append(cleaned)
        return variables[:12]

    def _extract_units(self, text: str) -> List[str]:
        """Extract common scientific and numeric units."""
        matches = re.findall(
            r"\b(?:m|cm|mm|km|kg|g|mg|s|ms|A|V|W|Hz|kHz|MHz|GHz|K|Pa|J|N|mol|rad|deg|%)\b",
            text,
        )
        return self._merge_unique_strings(matches, [])

    def _extract_dates(self, text: str) -> List[str]:
        """Extract simple date-like entities."""
        matches = re.findall(
            r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b",
            text,
            flags=re.IGNORECASE,
        )
        return self._merge_unique_strings(matches, [])

    def _extract_amounts(self, text: str) -> List[str]:
        """Extract currency-like amounts."""
        matches = re.findall(r"(?:[$\u20b9\u20ac\u00a3]\s?\d[\d,]*(?:\.\d+)?)", text)
        return self._merge_unique_strings(matches, [])

    def _extract_addresses(self, text: str) -> List[str]:
        """Extract address-like patterns conservatively."""
        matches = re.findall(
            r"\b\d{1,6}\s+[A-Za-z0-9.\- ]+\s+(?:Street|St|Road|Rd|Lane|Ln|Avenue|Ave|Drive|Dr|Boulevard|Blvd)\b",
            text,
            flags=re.IGNORECASE,
        )
        return self._merge_unique_strings(matches, [])

    def _infer_entities(
        self,
        text: str,
        equations: List[str],
        key_concepts: List[str],
        technical_terms: List[str],
    ) -> Dict[str, List[str]]:
        """Infer structured entities for parser output."""
        constants: List[str] = []
        if any(symbol in text for symbol in ["â„", "Ä§", "hbar"]):
            constants.append("reduced Planck constant")

        concepts = self._merge_unique_strings(key_concepts, [])
        for term in technical_terms:
            if term not in concepts:
                concepts.append(term)

        return {
            "symbols": self._extract_symbol_entities(text),
            "concepts": concepts[:12],
            "variables": self._extract_variable_entities(equations),
            "constants": constants,
            "units": self._extract_units(text),
            "names": [],
            "dates": self._extract_dates(text),
            "amounts": self._extract_amounts(text),
            "addresses": self._extract_addresses(text),
        }

    def _analyze_text(self, text: str, input_type: str = "text") -> Dict[str, Any]:
        """Build heuristic analysis for parser enrichment."""
        equations = self._extract_equations_from_text(text)
        technical_terms = self._find_technical_terms(text)
        domain, domain_scores = self._infer_domain(text, technical_terms)
        complexity, complexity_score = self._infer_complexity(text, equations, technical_terms)
        topic = self._infer_topic(text)
        intent = self._infer_intent(text, input_type=input_type)
        secondary_intents = self._infer_secondary_intents(text, intent)
        key_concepts = self._infer_key_concepts(text, domain, technical_terms, topic)
        language = self._detect_language(text)
        entities = self._infer_entities(text, equations, key_concepts, technical_terms)
        asks = self._infer_asks(text, topic, intent, secondary_intents)
        verification_targets = self._infer_verification_targets(text, domain, equations, intent)
        retrieval_targets = self._infer_retrieval_targets(text, intent, secondary_intents)

        return {
            "input_type": input_type,
            "has_equations": bool(equations),
            "equations_found": equations,
            "technical_terms_found": technical_terms,
            "domain": domain,
            "domain_scores": domain_scores,
            "complexity": complexity,
            "complexity_score": complexity_score,
            "intent": intent,
            "secondary_intents": secondary_intents,
            "topic": topic,
            "asks": asks,
            "key_concepts": key_concepts,
            "language": language,
            "entities": entities,
            "verification_targets": verification_targets,
            "retrieval_targets": retrieval_targets,
            "_source_text": text,
        }

    def _coerce_string(self, value: Any, default: str = "") -> str:
        """Return a normalized string value."""
        if value is None:
            return default
        text = str(value).strip()
        return text or default

    def _coerce_float(self, value: Any, default: float = 0.0) -> float:
        """Normalize a float-like value."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _coerce_int(self, value: Any, default: int = 0) -> int:
        """Normalize an int-like value."""
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _normalize_probability(self, value: Any, default: float) -> float:
        """Clamp a probability to the 0..1 range."""
        numeric = self._coerce_float(value, default)
        return round(max(0.0, min(numeric, 1.0)), 2)

    def _normalize_bbox(self, value: Any) -> Optional[List[float]]:
        """Normalize bounding boxes to [x1, y1, x2, y2]."""
        if value is None:
            return None
        if isinstance(value, dict):
            if {"x1", "y1", "x2", "y2"}.issubset(value):
                coords = [value["x1"], value["y1"], value["x2"], value["y2"]]
            else:
                return None
        elif isinstance(value, (list, tuple)) and len(value) == 4:
            coords = list(value)
        else:
            return None

        normalized: List[float] = []
        for coord in coords:
            try:
                normalized.append(round(float(coord), 3))
            except (TypeError, ValueError):
                return None
        return normalized

    def _normalize_point(self, value: Any) -> Optional[List[float]]:
        """Normalize points to [x, y]."""
        if isinstance(value, dict) and {"x", "y"}.issubset(value):
            coords = [value["x"], value["y"]]
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            coords = list(value)
        else:
            return None

        normalized: List[float] = []
        for coord in coords:
            try:
                normalized.append(round(float(coord), 3))
            except (TypeError, ValueError):
                return None
        return normalized

    def _to_latex_equation(self, equation: str) -> str:
        """Convert raw equations into lightweight LaTeX-like text."""
        latex = equation
        replacements = [
            ("â„", r"\hbar"),
            ("Ä§", r"\hbar"),
            ("Ïˆ", r"\psi"),
            ("Ï•", r"\phi"),
            ("âˆ‚", r"\partial"),
            ("âˆ‡", r"\nabla"),
            ("âˆ«", r"\int"),
            ("âˆ‘", r"\sum"),
            ("â‰ˆ", r"\approx"),
            ("â‰¤", r"\leq"),
            ("â‰¥", r"\geq"),
            ("Ä¤", r"\hat{H}"),
        ]
        for source, target in replacements:
            latex = latex.replace(source, target)
        return re.sub(r"\s+", " ", latex).strip()

    def _normalize_equation_item(
        self,
        item: Any,
        default_confidence: float,
        default_page: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """Normalize a single equation entry."""
        if isinstance(item, dict):
            raw = self._coerce_string(item.get("raw") or item.get("text"))
            latex = self._coerce_string(item.get("latex"))
            if not raw and not latex:
                return None
            if not latex and raw:
                latex = self._to_latex_equation(raw)
            return {
                "raw": raw or latex,
                "latex": latex or self._to_latex_equation(raw),
                "confidence": self._normalize_probability(item.get("confidence"), default_confidence),
                "bbox": self._normalize_bbox(item.get("bbox")),
                "page": self._coerce_int(item.get("page"), default_page),
            }

        raw = self._coerce_string(item)
        if not raw:
            return None
        return {
            "raw": raw,
            "latex": self._to_latex_equation(raw),
            "confidence": self._normalize_probability(default_confidence, default_confidence),
            "bbox": None,
            "page": default_page,
        }

    def _normalize_equations(
        self,
        value: Any,
        default_confidence: float,
        default_page: int = 1,
    ) -> List[Dict[str, Any]]:
        """Normalize equation arrays."""
        items = value if isinstance(value, list) else self._coerce_string_list(value)
        normalized: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in items:
            equation = self._normalize_equation_item(item, default_confidence, default_page)
            if not equation:
                continue
            raw_key = equation["raw"].casefold()
            if raw_key in seen:
                continue
            seen.add(raw_key)
            normalized.append(equation)
        return normalized

    def _normalize_key_value_pairs(self, value: Any) -> List[Dict[str, Any]]:
        """Normalize key-value extraction results."""
        if not isinstance(value, list):
            return []
        normalized: List[Dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            key = self._coerce_string(item.get("key"))
            field_value = self._coerce_string(item.get("value"))
            if not key and not field_value:
                continue
            normalized.append(
                {
                    "key": key,
                    "value": field_value,
                    "confidence": self._normalize_probability(item.get("confidence"), 0.65),
                    "bbox": self._normalize_bbox(item.get("bbox")),
                    "page": self._coerce_int(item.get("page"), 1),
                }
            )
        return normalized

    def _normalize_layout_elements(self, value: Any) -> List[Dict[str, Any]]:
        """Normalize layout element blocks."""
        if not isinstance(value, list):
            return []
        normalized: List[Dict[str, Any]] = []
        for index, item in enumerate(value, start=1):
            if not isinstance(item, dict):
                continue
            element_type = self._coerce_string(item.get("type"), "paragraph").casefold()
            if element_type not in self.valid_layout_types:
                element_type = "paragraph"
            normalized.append(
                {
                    "type": element_type,
                    "text": self._coerce_string(item.get("text")),
                    "bbox": self._normalize_bbox(item.get("bbox")),
                    "page": self._coerce_int(item.get("page"), 1),
                    "reading_order": self._coerce_int(item.get("reading_order"), index),
                    "confidence": self._normalize_probability(item.get("confidence"), 0.62),
                }
            )
        return normalized

    def _normalize_tables(self, value: Any) -> List[Dict[str, Any]]:
        """Normalize table outputs."""
        if not isinstance(value, list):
            return []
        normalized: List[Dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "title": self._coerce_string(item.get("title")),
                    "headers": self._coerce_string_list(item.get("headers")),
                    "rows": item.get("rows") if isinstance(item.get("rows"), list) else [],
                    "html": self._coerce_string(item.get("html")),
                    "markdown": self._coerce_string(item.get("markdown")),
                    "bbox": self._normalize_bbox(item.get("bbox")),
                    "page": self._coerce_int(item.get("page"), 1),
                    "confidence": self._normalize_probability(item.get("confidence"), 0.6),
                }
            )
        return normalized

    def _normalize_figures(self, value: Any) -> List[Dict[str, Any]]:
        """Normalize figure outputs."""
        if not isinstance(value, list):
            return []
        normalized: List[Dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "caption": self._coerce_string(item.get("caption")),
                    "description": self._coerce_string(item.get("description")),
                    "bbox": self._normalize_bbox(item.get("bbox")),
                    "page": self._coerce_int(item.get("page"), 1),
                    "chart_type": self._coerce_string(item.get("chart_type")),
                    "confidence": self._normalize_probability(item.get("confidence"), 0.58),
                }
            )
        return normalized

    def _normalize_grounding(self, value: Any) -> List[Dict[str, Any]]:
        """Normalize grounding references."""
        if not isinstance(value, list):
            return []
        normalized: List[Dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "ref_text": self._coerce_string(item.get("ref_text") or item.get("text")),
                    "bbox": self._normalize_bbox(item.get("bbox")),
                    "point": self._normalize_point(item.get("point")),
                    "page": self._coerce_int(item.get("page"), 1),
                    "confidence": self._normalize_probability(item.get("confidence"), 0.58),
                }
            )
        return normalized

    def _normalize_source_spans(self, value: Any) -> List[Dict[str, Any]]:
        """Normalize provenance spans."""
        if not isinstance(value, list):
            return []
        normalized: List[Dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "field": self._coerce_string(item.get("field")),
                    "text": self._coerce_string(item.get("text")),
                    "page": self._coerce_int(item.get("page"), 1),
                    "line": self._coerce_int(item.get("line"), 0),
                    "char_start": self._coerce_int(item.get("char_start"), 0),
                    "char_end": self._coerce_int(item.get("char_end"), 0),
                    "bbox": self._normalize_bbox(item.get("bbox")),
                }
            )
        return normalized

    def _normalize_entities(self, value: Any) -> Dict[str, List[str]]:
        """Normalize entities into a stable structure."""
        template = {
            "symbols": [],
            "concepts": [],
            "variables": [],
            "constants": [],
            "units": [],
            "names": [],
            "dates": [],
            "amounts": [],
            "addresses": [],
        }
        if not isinstance(value, dict):
            return template
        normalized = {}
        for key in template:
            normalized[key] = self._coerce_string_list(value.get(key))
        return normalized

    def _normalize_boolean_map(self, value: Any, template: Dict[str, bool]) -> Dict[str, bool]:
        """Normalize a fixed boolean map."""
        normalized = dict(template)
        if not isinstance(value, dict):
            return normalized
        for key in template:
            if key in value:
                normalized[key] = bool(value[key])
        return normalized

    def _normalize_document_metadata(
        self,
        value: Any,
        input_type: str,
        language: str,
        page_count: int = 0,
    ) -> Dict[str, Any]:
        """Normalize document-level metadata."""
        metadata = value if isinstance(value, dict) else {}
        return {
            "title": self._coerce_string(metadata.get("title")),
            "authors": self._coerce_string_list(metadata.get("authors")),
            "references": self._coerce_string_list(metadata.get("references")),
            "page_count": self._coerce_int(metadata.get("page_count"), page_count),
            "detected_language": self._coerce_string(metadata.get("detected_language"), language),
            "is_scanned": bool(metadata.get("is_scanned", input_type in {"image", "pdf"})),
            "has_text_layer": bool(metadata.get("has_text_layer", input_type == "text")),
        }

    def _normalize_quality_flags(
        self,
        value: Any,
        input_type: str,
        has_equations: bool,
        page_count: int = 0,
    ) -> Dict[str, Any]:
        """Normalize quality flags."""
        flags = value if isinstance(value, dict) else {}
        return {
            "has_equations": bool(flags.get("has_equations", has_equations)),
            "ocr_used": bool(flags.get("ocr_used", input_type in {"image", "pdf"})),
            "low_confidence_regions": flags.get("low_confidence_regions", []),
            "garbled_text_detected": bool(flags.get("garbled_text_detected", False)),
            "rotated_page": bool(flags.get("rotated_page", False)),
            "complex_layout": bool(flags.get("complex_layout", page_count > 1)),
            "handwritten_content": bool(flags.get("handwritten_content", False)),
        }

    def _estimate_confidence(
        self,
        input_type: str,
        analysis: Optional[Dict[str, Any]] = None,
        result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Estimate overall and per-field confidence heuristically."""
        base = 0.62
        if input_type == "text":
            base = 0.7
        elif input_type == "pdf":
            base = 0.66
        elif input_type == "image":
            base = 0.64

        if analysis:
            if analysis.get("technical_terms_found"):
                base += min(len(analysis["technical_terms_found"]) * 0.02, 0.12)
            if analysis.get("equations_found"):
                base += 0.08
            if analysis.get("domain") != "general":
                base += 0.04

        if result and result.get("error"):
            base = 0.2
        elif result and result.get("_parse_error"):
            base = max(0.45, base - 0.22)

        overall = round(max(0.0, min(base, 0.98)), 2)
        field_confidence = {
            "intent_confidence": overall,
            "topic_confidence": max(0.0, round(overall - 0.03, 2)),
            "domain_confidence": max(0.0, round(overall + 0.02, 2)),
            "complexity_confidence": max(0.0, round(overall - 0.01, 2)),
            "language_confidence": max(0.0, round(min(overall + 0.03, 0.99), 2)),
            "equation_confidence": max(0.0, round(min(overall + 0.05, 0.99), 2)),
            "entity_confidence": max(0.0, round(overall - 0.04, 2)),
            "layout_confidence": max(0.0, round(overall - 0.05, 2)),
            "document_confidence": max(0.0, round(overall - 0.03, 2)),
        }
        return {"confidence": overall, "field_confidence": field_confidence}

    def _score_output_format_hint(self, result: Dict[str, Any], input_type: str) -> str:
        """Choose the most useful downstream format."""
        if result.get("equations"):
            return "latex"
        if input_type == "pdf" and (result.get("tables") or result.get("layout_elements")):
            return "markdown"
        if result.get("tables"):
            return "html"
        return "json"

    def _build_base_result(self, input_type: str) -> Dict[str, Any]:
        """Return the stable result schema."""
        return {
            "intent": self._infer_intent("", input_type=input_type),
            "secondary_intents": [],
            "asks": [],
            "topic": "main concept",
            "domain": "general",
            "complexity": "basic",
            "language": "en",
            "key_concepts": [],
            "confidence": 0.0,
            "field_confidence": {
                "intent_confidence": 0.0,
                "topic_confidence": 0.0,
                "domain_confidence": 0.0,
                "complexity_confidence": 0.0,
                "language_confidence": 0.0,
                "equation_confidence": 0.0,
                "entity_confidence": 0.0,
                "layout_confidence": 0.0,
                "document_confidence": 0.0,
            },
            "equations": [],
            "entities": {
                "symbols": [],
                "concepts": [],
                "variables": [],
                "constants": [],
                "units": [],
                "names": [],
                "dates": [],
                "amounts": [],
                "addresses": [],
            },
            "key_value_pairs": [],
            "layout_elements": [],
            "tables": [],
            "figures": [],
            "grounding": [],
            "reading_order": [],
            "source_spans": [],
            "verification_targets": {
                "sympy": False,
                "unit_check": False,
                "constraint_check": False,
                "edge_case_check": False,
            },
            "retrieval_targets": {
                "similar_problems": False,
                "misconceptions": False,
                "visualization_patterns": False,
                "explanation_style": False,
            },
            "document_metadata": {
                "title": "",
                "authors": [],
                "references": [],
                "page_count": 0,
                "detected_language": "en",
                "is_scanned": input_type in {"image", "pdf"},
                "has_text_layer": input_type == "text",
            },
            "quality_flags": {
                "has_equations": False,
                "ocr_used": input_type in {"image", "pdf"},
                "low_confidence_regions": [],
                "garbled_text_detected": False,
                "rotated_page": False,
                "complex_layout": False,
                "handwritten_content": False,
            },
            "output_format_hint": "json",
        }

    def _merge_equation_lists(
        self,
        primary: List[Dict[str, Any]],
        secondary: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge equation lists while preserving the first version of each equation."""
        merged: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in [*primary, *secondary]:
            raw = self._coerce_string(item.get("raw")).casefold()
            if not raw or raw in seen:
                continue
            seen.add(raw)
            merged.append(item)
        return merged

    def _default_reading_order(
        self,
        layout_elements: List[Dict[str, Any]],
        key_value_pairs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build a simple reading-order list when none is provided."""
        reading_order: List[Dict[str, Any]] = []
        if layout_elements:
            for item in sorted(layout_elements, key=lambda element: element["reading_order"]):
                reading_order.append(
                    {
                        "reading_order": item["reading_order"],
                        "type": item["type"],
                        "text": item["text"],
                        "page": item["page"],
                    }
                )
            return reading_order

        for index, item in enumerate(key_value_pairs, start=1):
            reading_order.append(
                {
                    "reading_order": index,
                    "type": "key_value_pair",
                    "text": f'{item["key"]}: {item["value"]}'.strip(": "),
                    "page": item["page"],
                }
            )
        return reading_order

    def _build_source_spans_from_text(
        self,
        text: str,
        equations: List[Dict[str, Any]],
        key_concepts: List[str],
    ) -> List[Dict[str, Any]]:
        """Create provenance spans from text input when the model did not provide any."""
        spans: List[Dict[str, Any]] = []
        for equation in equations:
            raw = equation["raw"]
            start = text.find(raw)
            if start >= 0:
                spans.append(
                    {
                        "field": "equations",
                        "text": raw,
                        "page": 1,
                        "line": 1,
                        "char_start": start,
                        "char_end": start + len(raw),
                        "bbox": None,
                    }
                )

        for concept in key_concepts[:6]:
            concept_start = text.casefold().find(concept.casefold())
            if concept_start >= 0:
                spans.append(
                    {
                        "field": "key_concepts",
                        "text": concept,
                        "page": 1,
                        "line": 1,
                        "char_start": concept_start,
                        "char_end": concept_start + len(concept),
                        "bbox": None,
                    }
                )
        return spans

    def _normalize_common_result(
        self,
        result: Dict[str, Any],
        input_type: str,
        analysis: Optional[Dict[str, Any]] = None,
        page_count: int = 0,
    ) -> Dict[str, Any]:
        """Normalize the output into a single rich schema."""
        input_result = dict(result) if isinstance(result, dict) else {}
        base = self._build_base_result(input_type)
        normalized = dict(base)

        language = self._coerce_string(input_result.get("language"))
        if not language and analysis:
            language = self._coerce_string(analysis.get("language"), "en")
        language = language or "en"

        confidence_data = self._estimate_confidence(input_type, analysis=analysis, result=input_result)
        overall_confidence = self._normalize_probability(
            input_result.get("confidence"),
            confidence_data["confidence"],
        )
        provided_field_confidence = input_result.get("field_confidence", {})
        if not isinstance(provided_field_confidence, dict):
            provided_field_confidence = {}
        field_confidence = {}
        for key, fallback in confidence_data["field_confidence"].items():
            field_confidence[key] = self._normalize_probability(
                provided_field_confidence.get(key),
                fallback,
            )

        normalized["intent"] = self._coerce_string(input_result.get("intent"), base["intent"])
        if normalized["intent"] not in self.valid_intents:
            normalized["intent"] = base["intent"]
        normalized["secondary_intents"] = [
            item
            for item in self._coerce_string_list(input_result.get("secondary_intents"))
            if item in self.valid_secondary_intents
        ]
        normalized["asks"] = self._coerce_string_list(input_result.get("asks"))

        normalized["topic"] = self._coerce_string(input_result.get("topic"), base["topic"])
        normalized["domain"] = self._coerce_string(input_result.get("domain"), base["domain"]).lower()
        if normalized["domain"] not in self.valid_domains:
            normalized["domain"] = base["domain"]

        normalized["complexity"] = self._coerce_string(input_result.get("complexity"), base["complexity"]).lower()
        if normalized["complexity"] not in self.valid_complexities:
            normalized["complexity"] = base["complexity"]

        normalized["language"] = language
        normalized["key_concepts"] = self._coerce_string_list(input_result.get("key_concepts"))
        normalized["confidence"] = overall_confidence
        normalized["field_confidence"] = field_confidence
        normalized["equations"] = self._normalize_equations(
            input_result.get("equations"),
            field_confidence["equation_confidence"],
        )
        normalized["entities"] = self._normalize_entities(input_result.get("entities"))
        normalized["key_value_pairs"] = self._normalize_key_value_pairs(input_result.get("key_value_pairs"))
        normalized["layout_elements"] = self._normalize_layout_elements(input_result.get("layout_elements"))
        normalized["tables"] = self._normalize_tables(input_result.get("tables"))
        normalized["figures"] = self._normalize_figures(input_result.get("figures"))
        normalized["grounding"] = self._normalize_grounding(input_result.get("grounding"))
        normalized["source_spans"] = self._normalize_source_spans(input_result.get("source_spans"))
        normalized["verification_targets"] = self._normalize_boolean_map(
            input_result.get("verification_targets"),
            base["verification_targets"],
        )
        normalized["retrieval_targets"] = self._normalize_boolean_map(
            input_result.get("retrieval_targets"),
            base["retrieval_targets"],
        )
        normalized["document_metadata"] = self._normalize_document_metadata(
            input_result.get("document_metadata"),
            input_type=input_type,
            language=language,
            page_count=page_count,
        )
        normalized["quality_flags"] = self._normalize_quality_flags(
            input_result.get("quality_flags"),
            input_type=input_type,
            has_equations=bool(normalized["equations"]),
            page_count=page_count,
        )
        if not isinstance(input_result.get("reading_order"), list) or not input_result.get("reading_order"):
            normalized["reading_order"] = self._default_reading_order(
                normalized["layout_elements"],
                normalized["key_value_pairs"],
            )
        else:
            normalized["reading_order"] = input_result.get("reading_order", [])
        output_format_hint = self._coerce_string(input_result.get("output_format_hint"))
        if output_format_hint not in self.valid_output_format_hints:
            output_format_hint = self._score_output_format_hint(normalized, input_type)
        normalized["output_format_hint"] = output_format_hint
        return normalized

    def _coerce_string_list(self, value: Union[str, List[Any], None]) -> List[str]:
        """Normalize a string-or-list field into a list of strings."""
        if value is None:
            return []
        if isinstance(value, list):
            normalized: List[str] = []
            for item in value:
                text = self._coerce_string(item)
                if text and text not in normalized:
                    normalized.append(text)
            return normalized
        text = self._coerce_string(value)
        return [text] if text else []

    def _normalize_text_result(self, result: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize and enrich text output with heuristic signals."""
        normalized = self._normalize_common_result(result, input_type="text", analysis=analysis, page_count=1)

        if normalized["intent"] == "concept_explanation" and analysis["intent"] != "concept_explanation":
            normalized["intent"] = analysis["intent"]
        normalized["secondary_intents"] = self._merge_unique_strings(
            normalized["secondary_intents"],
            analysis["secondary_intents"],
        )
        normalized["asks"] = self._merge_unique_strings(
            normalized["asks"],
            analysis["asks"],
        )

        topic = normalized["topic"]
        if not topic or topic.lower() in {"main concept", "equation", "query"}:
            normalized["topic"] = analysis["topic"]

        domain_scores = analysis.get("domain_scores", {})
        max_score = max(domain_scores.values()) if isinstance(domain_scores, dict) and domain_scores else 0
        if normalized["domain"] == "general" or max_score >= 2:
            normalized["domain"] = analysis["domain"]

        complexity_order = {"basic": 0, "intermediate": 1, "advanced": 2}
        if complexity_order[analysis["complexity"]] > complexity_order[normalized["complexity"]]:
            normalized["complexity"] = analysis["complexity"]

        heuristic_equations = self._normalize_equations(
            analysis["equations_found"],
            normalized["field_confidence"]["equation_confidence"],
            default_page=1,
        )
        normalized["equations"] = self._merge_equation_lists(normalized["equations"], heuristic_equations)
        normalized["key_concepts"] = self._merge_unique_strings(
            normalized["key_concepts"],
            analysis["key_concepts"],
        )

        entities = normalized["entities"]
        for key, values in analysis["entities"].items():
            entities[key] = self._merge_unique_strings(entities.get(key, []), values)
        normalized["entities"] = entities
        for key, value in analysis["verification_targets"].items():
            normalized["verification_targets"][key] = (
                normalized["verification_targets"].get(key, False) or value
            )
        for key, value in analysis["retrieval_targets"].items():
            normalized["retrieval_targets"][key] = (
                normalized["retrieval_targets"].get(key, False) or value
            )

        if not normalized["source_spans"]:
            normalized["source_spans"] = self._build_source_spans_from_text(
                analysis.get("_source_text", ""),
                normalized["equations"],
                normalized["key_concepts"],
            )

        normalized["document_metadata"]["page_count"] = 1
        normalized["document_metadata"]["detected_language"] = analysis["language"]
        normalized["document_metadata"]["has_text_layer"] = True
        normalized["document_metadata"]["is_scanned"] = False
        normalized["quality_flags"]["has_equations"] = bool(normalized["equations"])
        normalized["quality_flags"]["ocr_used"] = False
        normalized["output_format_hint"] = self._score_output_format_hint(normalized, "text")
        normalized["_analysis"] = {
            "input_type": "text",
            "has_equations": analysis["has_equations"],
            "complexity": analysis["complexity"],
            "complexity_score": analysis["complexity_score"],
            "technical_terms_found": analysis["technical_terms_found"],
            "equations_found": analysis["equations_found"],
            "language": analysis["language"],
            "secondary_intents": analysis["secondary_intents"],
            "asks": analysis["asks"],
        }
        return normalized

    def _merge_unique_strings(self, primary: List[str], secondary: List[str]) -> List[str]:
        """Merge two string lists while preserving order."""
        merged: List[str] = []
        for item in [*primary, *secondary]:
            value = str(item).strip()
            if value and value not in merged:
                merged.append(value)
        return merged

    def _visual_text_from_result(self, result: Dict[str, Any]) -> str:
        """Collect free text from image/PDF model outputs for heuristic enrichment."""
        parts: List[str] = []
        for key in ["text", "summary", "title", "description"]:
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
        for key in ["sections", "objects"]:
            value = result.get(key)
            if isinstance(value, list):
                parts.extend(self._coerce_string_list(value))
        return "\n".join(parts).strip()

    def _normalize_visual_result(
        self,
        result: Dict[str, Any],
        input_type: str,
        supplemental_text: str,
        page_count: int,
        document_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Normalize image and PDF outputs into the shared schema."""
        if supplemental_text:
            analysis = self._analyze_text(supplemental_text, input_type=input_type)
        else:
            analysis = {
                "input_type": input_type,
                "has_equations": False,
                "equations_found": [],
                "technical_terms_found": [],
                "domain": "general",
                "domain_scores": {domain: 0 for domain in self.domain_terms},
                "complexity": "basic",
                "complexity_score": 0,
                "intent": self._infer_intent("", input_type=input_type),
                "secondary_intents": [],
                "topic": "main concept",
                "asks": [],
                "key_concepts": [],
                "language": "en",
                "entities": self._infer_entities("", [], [], []),
                "verification_targets": {
                    "sympy": False,
                    "unit_check": False,
                    "constraint_check": False,
                    "edge_case_check": False,
                },
                "retrieval_targets": {
                    "similar_problems": False,
                    "misconceptions": False,
                    "visualization_patterns": False,
                    "explanation_style": False,
                },
                "_source_text": "",
            }

        normalized = self._normalize_common_result(
            result,
            input_type=input_type,
            analysis=analysis,
            page_count=page_count,
        )

        if normalized["domain"] == "general" and analysis["domain"] != "general":
            normalized["domain"] = analysis["domain"]
        if normalized["topic"] == "main concept" and analysis["topic"] != "main concept":
            normalized["topic"] = analysis["topic"]
        normalized["intent"] = self._infer_intent("", input_type=input_type)
        normalized["secondary_intents"] = self._merge_unique_strings(
            normalized["secondary_intents"],
            analysis["secondary_intents"],
        )
        normalized["asks"] = self._merge_unique_strings(
            normalized["asks"],
            analysis["asks"],
        )

        normalized["key_concepts"] = self._merge_unique_strings(
            normalized["key_concepts"],
            analysis["key_concepts"],
        )
        heuristic_equations = self._normalize_equations(
            analysis["equations_found"],
            normalized["field_confidence"]["equation_confidence"],
            default_page=1,
        )
        normalized["equations"] = self._merge_equation_lists(normalized["equations"], heuristic_equations)

        entities = normalized["entities"]
        for key, values in analysis["entities"].items():
            entities[key] = self._merge_unique_strings(entities.get(key, []), values)
        normalized["entities"] = entities
        for key, value in analysis["verification_targets"].items():
            normalized["verification_targets"][key] = (
                normalized["verification_targets"].get(key, False) or value
            )
        for key, value in analysis["retrieval_targets"].items():
            normalized["retrieval_targets"][key] = (
                normalized["retrieval_targets"].get(key, False) or value
            )

        if document_metadata:
            normalized["document_metadata"].update(document_metadata)
        normalized["document_metadata"]["detected_language"] = analysis["language"]
        normalized["quality_flags"]["has_equations"] = bool(normalized["equations"])
        normalized["quality_flags"]["ocr_used"] = True
        normalized["output_format_hint"] = self._score_output_format_hint(normalized, input_type)
        normalized["_analysis"] = {
            "input_type": input_type,
            "has_equations": analysis["has_equations"],
            "complexity": analysis["complexity"],
            "complexity_score": analysis["complexity_score"],
            "technical_terms_found": analysis["technical_terms_found"],
            "equations_found": analysis["equations_found"],
            "language": analysis["language"],
            "secondary_intents": analysis["secondary_intents"],
            "asks": analysis["asks"],
        }
        return normalized

    def parse_text(self, text: str) -> Dict[str, Any]:
        """Parse a text query into structured JSON."""
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

        response, model_timing = self._run_vl2(
            prompt,
            max_new_tokens=self.config.get("max_new_tokens_text", 320),
        )
        result = self._normalize_text_result(self._extract_json(response), analysis)
        result["_model"] = self.model_name
        result["_timing"] = {
            **model_timing,
            "total_parse_seconds": round(perf_counter() - parse_start, 3),
        }
        return self._finalize_result(result)

    def parse_image(self, image: Image.Image, prompt_text: Optional[str] = None) -> Dict[str, Any]:
        """Parse a single image into structured JSON."""
        parse_start = perf_counter()
        user_prompt = self._coerce_string(prompt_text)
        prompt = (
            "Analyze this image and respond with valid JSON only.\n"
            "Return ONLY these keys to keep output short: intent, topic, domain, complexity, language, "
            "key_concepts, equations, text, summary.\n"
            'intent should be "ocr_extraction".\n'
            'If equations are visible, return equations as [{"raw": "...", "latex": "..."}].\n'
            "Keep text concise and summarize the main diagram/document content."
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
        visual_text = self._visual_text_from_result(model_result)
        if visual_text:
            supplemental_parts.append(visual_text)
        supplemental_text = "\n".join(supplemental_parts)
        result = self._normalize_visual_result(
            model_result,
            input_type="image",
            supplemental_text=supplemental_text,
            page_count=1,
            document_metadata={"page_count": 1, "is_scanned": True, "has_text_layer": False},
        )
        result["_model"] = self.model_name
        result["_timing"] = {
            **model_timing,
            "total_parse_seconds": round(perf_counter() - parse_start, 3),
        }
        return self._finalize_result(result)

    def _inspect_pdf_metadata(self, pdf_path: Path) -> Tuple[Dict[str, Any], str]:
        """Inspect PDF metadata and text-layer hints from the source file."""
        document = fitz.open(str(pdf_path))
        metadata = document.metadata or {}
        extracted_text_parts: List[str] = []
        has_text_layer = False
        for page in document:
            page_text = page.get_text("text").strip()
            if page_text:
                has_text_layer = True
                if len(" ".join(extracted_text_parts)) < 4000:
                    extracted_text_parts.append(page_text)

        page_count = len(document)
        document.close()
        extracted_text = "\n".join(extracted_text_parts).strip()
        inspected = {
            "title": self._coerce_string(metadata.get("title")),
            "authors": self._coerce_string_list(metadata.get("author")),
            "references": [],
            "page_count": page_count,
            "detected_language": self._detect_language(extracted_text),
            "is_scanned": not has_text_layer,
            "has_text_layer": has_text_layer,
        }
        return inspected, extracted_text

    def parse_pdf(
        self,
        pdf_path: str,
        pages: Optional[List[int]] = None,
        prompt_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Parse a PDF by converting pages to images and using DeepSeek-VL2 Tiny."""
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

        prompt = (
            "Analyze this document and respond with valid JSON only.\n"
            "Return ONLY these keys: intent, topic, domain, complexity, language, key_concepts, equations, "
            "text, summary, title, tables, figures.\n"
            'intent should be "document_extraction".\n'
            "Keep the response compact. Focus on title, structure, equations, tables, and figures."
        )
        user_prompt = self._coerce_string(prompt_text)
        if user_prompt:
            prompt = f"{prompt}\nUser instruction: {user_prompt}"

        response, model_timing = self._run_vl2(
            prompt,
            images=images,
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
        result["_model"] = self.model_name
        result["_pages_processed"] = len(images)
        result["_timing"] = {
            "pdf_render_seconds": round(pdf_render_seconds, 3),
            **model_timing,
            "total_parse_seconds": round(perf_counter() - parse_start, 3),
        }
        return self._finalize_result(result)

    def _pdf_to_images(self, pdf_path: Path, dpi: int = 150) -> List[Image.Image]:
        """Convert PDF pages to PIL images."""
        images: List[Image.Image] = []
        document = fitz.open(str(pdf_path))

        for page_num in range(len(document)):
            print(f"  Converting page {page_num + 1}/{len(document)}")
            page = document[page_num]
            pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
            image_bytes = pixmap.tobytes("png")
            images.append(Image.open(BytesIO(image_bytes)).convert("RGB"))

        document.close()
        return images

    def parse(self, input_data: Union[str, Image.Image, dict]) -> Dict:
        """Auto-detect and parse the provided input."""
        if isinstance(input_data, str):
            input_path = self._resolve_input_path(input_data)
            if input_path.exists():
                suffix = input_path.suffix.lower()
                if suffix == ".pdf":
                    return self.parse_pdf(str(input_path))
                if suffix in {".png", ".jpg", ".jpeg"}:
                    return self.parse_image(Image.open(input_path))
            return self.parse_text(input_data)

        if isinstance(input_data, Image.Image):
            return self.parse_image(input_data)

        if isinstance(input_data, dict):
            if "pdf" in input_data:
                return self.parse_pdf(input_data["pdf"])
            if "text" in input_data:
                return self.parse_text(input_data["text"])

        return {"error": "Unsupported input type"}

    def _extract_json_scalar(self, text: str, key: str) -> Optional[str]:
        """Extract a simple scalar string field from malformed JSON-like text."""
        pattern = rf'"{re.escape(key)}"\s*:\s*"([^"]+)"'
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _extract_json_list_strings(self, text: str, key: str) -> List[str]:
        """Extract a list of quoted strings from malformed JSON-like text."""
        block_match = re.search(
            rf'"{re.escape(key)}"\s*:\s*\[(.*?)\]',
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not block_match:
            return []
        return [item.strip() for item in re.findall(r'"([^"]+)"', block_match.group(1))]

    def _salvage_partial_json(self, text: str) -> Dict[str, Any]:
        """Recover key fields from truncated JSON-like output."""
        salvaged: Dict[str, Any] = {}
        for key in [
            "intent",
            "topic",
            "domain",
            "complexity",
            "language",
            "output_format_hint",
            "text",
            "summary",
            "title",
            "description",
        ]:
            value = self._extract_json_scalar(text, key)
            if value:
                salvaged[key] = value

        secondary_intents = self._extract_json_list_strings(text, "secondary_intents")
        if secondary_intents:
            salvaged["secondary_intents"] = secondary_intents

        asks = self._extract_json_list_strings(text, "asks")
        if asks:
            salvaged["asks"] = asks

        key_concepts = self._extract_json_list_strings(text, "key_concepts")
        if key_concepts:
            salvaged["key_concepts"] = key_concepts

        equation_raw = re.findall(r'"raw"\s*:\s*"([^"]+)"', text, flags=re.IGNORECASE)
        equation_latex = re.findall(r'"latex"\s*:\s*"([^"]+)"', text, flags=re.IGNORECASE)
        if equation_raw or equation_latex:
            equations: List[Dict[str, str]] = []
            max_len = max(len(equation_raw), len(equation_latex))
            for index in range(max_len):
                raw_value = equation_raw[index] if index < len(equation_raw) else ""
                latex_value = equation_latex[index] if index < len(equation_latex) else raw_value
                if raw_value or latex_value:
                    equations.append({"raw": raw_value or latex_value, "latex": latex_value or raw_value})
            if equations:
                salvaged["equations"] = equations

        entity_fields = ["symbols", "concepts", "variables", "constants", "units", "names", "dates", "amounts", "addresses"]
        entities = {}
        for field in entity_fields:
            values = self._extract_json_list_strings(text, field)
            if values:
                entities[field] = values
        if entities:
            salvaged["entities"] = entities

        if salvaged:
            salvaged["_parse_error"] = "Recovered partial JSON from malformed model output."
            salvaged["_raw_response_preview"] = text[:300]
        return salvaged

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from a model response."""
        parse_error: Optional[Exception] = None
        try:
            decoder = json.JSONDecoder()
            start = text.find("{")
            while start != -1:
                parsed, _ = decoder.raw_decode(text[start:])
                if isinstance(parsed, dict):
                    return parsed
                start = text.find("{", start + 1)
        except Exception as exc:
            parse_error = exc
        salvaged = self._salvage_partial_json(text)
        if salvaged:
            return salvaged
        if parse_error is not None:
            print(f"JSON extraction failed: {parse_error}")
        return {
            "_parse_error": "Could not parse JSON",
            "_raw_response_preview": text[:500],
        }

    def _save_result(self, result: Dict):
        """Save a parsing result locally."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"result_{timestamp}.json"

        with open(output_file, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)

        print(f"Result saved to {output_file}")

    def get_stats(self) -> Dict:
        """Return local storage statistics in MB."""
        return {
            "cache_size_mb": self._get_folder_size(self.model_cache),
            "data_size_mb": self._get_folder_size(self.data_dir),
            "outputs_size_mb": self._get_folder_size(self.output_dir),
        }

    def _get_folder_size(self, folder: Path) -> float:
        """Calculate folder size in MB."""
        if not folder.exists():
            return 0.0
        total = sum(file.stat().st_size for file in folder.glob("**/*") if file.is_file())
        return total / (1024 * 1024)
