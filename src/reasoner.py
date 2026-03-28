"""Reasoning engine for mathematical and physics problem solving."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Callable, Dict, List, Optional, Union

import requests
import yaml
from dotenv import load_dotenv
from PIL import Image

from .parser import LocalParser

logger = logging.getLogger(__name__)


class SolutionStyle(Enum):
    STEP_BY_STEP = "step_by_step"
    CONCEPT_FIRST = "concept_first"
    DERIVATION_FOCUSED = "derivation_focused"
    APPLICATION_FOCUSED = "application_focused"
    COMPARATIVE = "comparative"
    VISUALIZATION = "visualization"


class MathematicalReasoner:
    """Generate structured math and physics solutions from parsed input."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.project_root = Path(__file__).parent.parent
        self.config = self._load_config(config_path)
        load_dotenv(self.project_root / ".env")
        self.model_name = self.config.get("reasoning_api_model", "deepseek-reasoner")
        self.scene_planner_model = self.config.get("scene_planner_model", "deepseek-reasoner")
        self.manim_code_model = self.config.get("manim_code_model", "deepseek-reasoner")
        self.manim_layout_refiner_model = self.config.get(
            "manim_layout_refiner_model",
            self.manim_code_model,
        )
        self.api_base_url = self.config.get("reasoning_api_base_url", "https://api.deepseek.com").rstrip("/")
        self.api_timeout_seconds = int(self.config.get("reasoning_api_timeout_seconds", 180))
        self.api_retries = int(self.config.get("reasoning_api_retries", 3))
        self.api_retry_backoff_seconds = float(
            self.config.get("reasoning_api_retry_backoff_seconds", 2.0)
        )
        self.api_key = self._resolve_api_key()
        self.output_dir = self.project_root / self.config.get("output_dir", "outputs")
        self.cache_dir = self.project_root / self.config.get("cache_dir", "cache")
        self.prompt_assets_dir = self.project_root / self.config.get("prompt_assets_dir", "prompt_assets")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_assets_dir.mkdir(parents=True, exist_ok=True)
        self.solution_templates = self._load_templates()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(self.project_root / config_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def _resolve_api_key(self) -> str:
        key = (
            os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("deepseek")
            or os.getenv("deepdeek")
        )
        if not key:
            raise RuntimeError(
                "DeepSeek API key not found. Set DEEPSEEK_API_KEY, deepseek, or deepdeek in the environment or .env."
            )
        return key.strip().strip("'").strip('"')

    def _load_templates(self) -> Dict[str, str]:
        return {
            "step_by_step": (
                "You are an expert mathematics and physics tutor.\n"
                "Provide a detailed, rigorous, pedagogically clear solution.\n\n"
                "Problem: {problem}\nContext: {context}\nEquations: {equations}\n"
                "Key Concepts: {concepts}\nDomain: {domain}\nComplexity: {complexity}\n"
                "Primary Intent: {intent}\nSecondary Intents: {secondary_intents}\n"
                "Requested Tasks: {asks}\nVerification Targets: {verification_targets}\n"
                "Retrieval Targets: {retrieval_targets}"
            ),
            "derivation_focused": (
                "You are an expert mathematician and theoretical physicist.\n"
                "Provide a careful derivation with justified steps and physical interpretation.\n\n"
                "Problem: {problem}\nEquations: {equations}\nStarting Point: {starting_point}\n"
                "Goal: {goal}\nContext: {context}\nRequested Tasks: {asks}"
            ),
            "concept_first": (
                "You are an educator.\nExplain the concept thoroughly before solving.\n\n"
                "Topic: {topic}\nUser Question: {question}\nPrerequisites: {prerequisites}\n"
                "Context: {context}\nRequested Tasks: {asks}"
            ),
            "comparative": (
                "You are an expert analyst.\nCompare relevant methods or interpretations.\n\n"
                "Topic: {topic}\nAspects to Compare: {aspects}\nContext: {context}\nRequested Tasks: {asks}"
            ),
            "application_focused": (
                "You are a practitioner.\nConnect theory to worked applications.\n\n"
                "Concept: {concept}\nDomain: {domain}\nEquations: {equations}\n"
                "Context: {context}\nRequested Tasks: {asks}"
            ),
            "visualization": (
                "You are a visualization-focused tutor.\nEmphasize intuition, geometry, and plots.\n\n"
                "Concept: {concept}\nEquations: {equations}\nParameters: {parameters}\n"
                "Context: {context}\nRequested Tasks: {asks}"
            ),
            "scene_planner": (
                "Create a second-pass Scene Planner prompt for a downstream Manim Community Edition code generator.\n"
                "Your output will be passed to another model that writes Manim code.\n"
                "Do not write code here; produce only the planning prompt.\n\n"
                "Parsed Topic: {topic}\n"
                "Domain: {domain}\n"
                "Complexity: {complexity}\n"
                "Intent: {intent}\n"
                "Secondary Intents: {secondary_intents}\n"
                "Parser-Grounded Facts:\n{grounded_facts}\n"
                "Problem Statement: {problem}\n"
                "Context: {context}\n"
                "Equations: {equations}\n"
                "Key Concepts: {concepts}\n"
                "Parameters: {parameters}\n"
                "Solution Understanding: {understanding}\n"
                "Solution Steps Summary: {step_summaries}\n"
                "Final Answer: {final_answer}\n"
                "Verification Notes: {verification}\n"
                "Extensions: {extensions}\n"
                "Scene Planner Template Reference:\n{scene_planner_template}\n"
                "Reasoning Markdown:\n{reasoning_markdown}\n"
            ),
            "manim_code_generator": (
                "Generate a complete executable Manim Community Edition Python script from the Scene Planner.\n"
                "Use the Scene Planner as the main production spec.\n"
                "Use the few-shot example and layout guidance as style and engineering constraints, not as content to copy blindly.\n\n"
                "Parsed Topic: {topic}\n"
                "Domain: {domain}\n"
                "Parser-Grounded Facts:\n{grounded_facts}\n"
                "Problem Statement: {problem}\n"
                "Key Concepts: {concepts}\n"
                "Equations: {equations}\n"
                "Final Answer: {final_answer}\n"
                "Reasoning Markdown:\n{reasoning_markdown}\n\n"
                "Scene Planner:\n{scene_planner}\n\n"
                "Scene Planner Template Reference:\n{scene_planner_template}\n\n"
                "Few-Shot Reference:\n{few_shot_reference}\n\n"
                "Layout Guidance:\n{layout_guidance}\n\n"
                "No-Overlap Rules:\n{no_overlap_guidance}\n"
            ),
            "manim_layout_refiner": (
                "Rewrite this Manim Community Edition script to eliminate overlap and crowding while preserving the same lesson content.\n"
                "Return the full corrected script only.\n\n"
                "Parsed Topic: {topic}\n"
                "Scene Planner:\n{scene_planner}\n\n"
                "Scene Planner Template Reference:\n{scene_planner_template}\n\n"
                "Few-Shot Reference:\n{few_shot_reference}\n\n"
                "Layout Guidance:\n{layout_guidance}\n\n"
                "No-Overlap Rules:\n{no_overlap_guidance}\n\n"
                "Existing Generated Code:\n{generated_code}\n"
            ),
        }

    def _read_prompt_asset(self, name: str) -> str:
        path = self.prompt_assets_dir / name
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8").strip()

    def determine_solution_style(self, parsed_input: Dict[str, Any]) -> SolutionStyle:
        intent = str(parsed_input.get("intent", "")).lower()
        secondary = {str(item).lower() for item in parsed_input.get("secondary_intents", [])}
        complexity = str(parsed_input.get("complexity", "")).lower()
        if intent == "derivation" or "derivation" in secondary:
            return SolutionStyle.DERIVATION_FOCUSED
        if intent == "comparison" or "comparison" in secondary:
            return SolutionStyle.COMPARATIVE
        if intent == "equation_visualization" or "equation_visualization" in secondary:
            return SolutionStyle.VISUALIZATION
        if intent == "application" or "application" in secondary:
            return SolutionStyle.APPLICATION_FOCUSED
        if intent == "concept_explanation" and complexity == "advanced":
            return SolutionStyle.CONCEPT_FIRST
        return SolutionStyle.STEP_BY_STEP

    def prepare_prompt(self, parsed_input: Dict[str, Any], style: SolutionStyle) -> str:
        template = self.solution_templates[style.value]
        prompt = template.format(
            problem=self._extract_problem_statement(parsed_input),
            context=self._build_context(parsed_input),
            equations=self._format_equations(parsed_input.get("equations", [])),
            concepts=", ".join(parsed_input.get("key_concepts", [])[:6]) or "None",
            domain=parsed_input.get("domain", "general"),
            complexity=parsed_input.get("complexity", "intermediate"),
            intent=parsed_input.get("intent", "concept_explanation"),
            secondary_intents=", ".join(parsed_input.get("secondary_intents", [])) or "None",
            asks="; ".join(parsed_input.get("asks", [])) or "None",
            verification_targets=self._format_targets(parsed_input.get("verification_targets", {})),
            retrieval_targets=self._format_targets(parsed_input.get("retrieval_targets", {})),
            topic=parsed_input.get("topic", "the concept"),
            question=parsed_input.get("_source_text", ""),
            prerequisites=self._list_prerequisites(parsed_input),
            starting_point=self._find_starting_point(parsed_input),
            goal=self._find_goal(parsed_input),
            aspects=self._find_comparison_aspects(parsed_input),
            concept=parsed_input.get("topic", "the concept"),
            parameters=self._extract_parameters(parsed_input),
        )
        contract = "\n".join(
            [
                "Output requirements:",
                "- Use Markdown.",
                "- Use these exact top-level headers in order:",
                "  ## Understanding",
                "  ## Prerequisites",
                "  ## Step-by-Step Solution",
                "  ## Key Insights",
                "  ## Final Answer",
                "  ## Verification",
                "  ## Extensions",
                "- In '## Step-by-Step Solution', include numbered steps.",
                "- Use LaTeX with $$...$$ for important equations.",
                "- If the request includes implications, address them explicitly.",
            ]
        )
        return f"{prompt}\n\n{contract}"

    def _extract_problem_statement(self, parsed_input: Dict[str, Any]) -> str:
        source = str(parsed_input.get("_source_text", "")).strip()
        if source:
            return source
        asks = [str(item).strip() for item in parsed_input.get("asks", []) if str(item).strip()]
        if asks:
            return " ".join(asks)
        topic = str(parsed_input.get("topic", "")).strip()
        return f"Explain and solve: {topic}" if topic else "Explain and solve the uploaded problem."

    def _build_context(self, parsed_input: Dict[str, Any]) -> str:
        parts: List[str] = []
        if parsed_input.get("domain") and parsed_input.get("domain") != "general":
            parts.append(f"Domain: {parsed_input['domain']}.")
        if parsed_input.get("complexity"):
            parts.append(f"Depth: {parsed_input['complexity']}.")
        if parsed_input.get("key_concepts"):
            parts.append(f"Concepts: {', '.join(parsed_input['key_concepts'][:6])}.")
        if parsed_input.get("asks"):
            parts.append(f"Tasks: {'; '.join(parsed_input['asks'])}.")
        return " ".join(parts) or "Use the parser output as context."

    def _build_grounded_fact_sheet(self, parsed_input: Dict[str, Any]) -> str:
        facts: List[str] = []
        topic = str(parsed_input.get("topic", "")).strip()
        if topic:
            facts.append(f"- Topic: {topic}")
        domain = str(parsed_input.get("domain", "")).strip()
        if domain:
            facts.append(f"- Domain: {domain}")
        concepts = [str(item).strip() for item in parsed_input.get("key_concepts", []) if str(item).strip()]
        if concepts:
            facts.append(f"- Key concepts: {', '.join(concepts[:8])}")
        equations = []
        for item in parsed_input.get("equations", [])[:8]:
            if isinstance(item, dict):
                value = str(item.get("latex") or item.get("raw") or "").strip()
            else:
                value = str(item).strip()
            if value:
                equations.append(value)
        if equations:
            facts.append("- Explicit equations from parser:")
            facts.extend(f"  - {value}" for value in equations)
        asks = [str(item).strip() for item in parsed_input.get("asks", []) if str(item).strip()]
        if asks:
            facts.append(f"- User-visible asks/text: {'; '.join(asks[:3])}")
        entities = parsed_input.get("entities", {}) if isinstance(parsed_input.get("entities"), dict) else {}
        variables = [str(item).strip() for item in entities.get("variables", []) if str(item).strip()]
        if variables:
            facts.append(f"- Variables/symbols: {', '.join(variables[:8])}")
        facts.append("- Treat these parser-grounded facts as authoritative.")
        facts.append(
            "- Ignore any example circuit values, numeric parameters, or story details that are not explicitly present above unless they are direct algebraic restatements of the same equations."
        )
        facts.append(
            "- If the parser-grounded facts describe only general laws or formulas, keep the animation general instead of turning it into a worked numerical example."
        )
        return "\n".join(facts)

    def _format_equations(self, equations: List[Any]) -> str:
        rendered: List[str] = []
        for item in equations[:6]:
            value = item.get("latex") or item.get("raw") if isinstance(item, dict) else str(item)
            if value:
                rendered.append(f"$${value}$$")
        return "\n".join(rendered) if rendered else "None"

    def _format_targets(self, targets: Dict[str, Any]) -> str:
        enabled = [key for key, value in targets.items() if bool(value)]
        return ", ".join(enabled) if enabled else "None"

    def _list_prerequisites(self, parsed_input: Dict[str, Any]) -> str:
        domain = str(parsed_input.get("domain", "")).lower()
        base = {
            "physics": ["calculus", "algebra", "physical laws"],
            "mathematics": ["algebra", "calculus", "symbolic manipulation"],
            "engineering": ["calculus", "basic physics", "system modeling"],
            "cs": ["discrete math", "algebra", "algorithmic reasoning"],
        }.get(domain, ["basic mathematics", "logical reasoning"])
        if parsed_input.get("complexity") == "advanced":
            base.append("multi-step derivation skills")
        return ", ".join(base)

    def _find_starting_point(self, parsed_input: Dict[str, Any]) -> str:
        equations = parsed_input.get("equations", [])
        if equations:
            first = equations[0]
            return str(first.get("raw") or first.get("latex")) if isinstance(first, dict) else str(first)
        concepts = parsed_input.get("key_concepts", [])
        return f"Begin from the definition of {concepts[0]}" if concepts else "Start from first principles"

    def _find_goal(self, parsed_input: Dict[str, Any]) -> str:
        asks = parsed_input.get("asks", [])
        if asks:
            return "; ".join(str(item) for item in asks)
        return "Reach a complete and correct understanding."

    def _find_comparison_aspects(self, parsed_input: Dict[str, Any]) -> str:
        text = str(parsed_input.get("_source_text", "")).lower()
        aspects = [label for key, label in [
            ("accuracy", "accuracy"),
            ("efficiency", "efficiency"),
            ("complexity", "complexity"),
            ("applicability", "scope of applicability"),
        ] if key in text]
        return ", ".join(aspects) if aspects else "method, assumptions, and trade-offs"

    def _extract_parameters(self, parsed_input: Dict[str, Any]) -> str:
        entities = parsed_input.get("entities", {})
        values: List[str] = []
        for key in ["variables", "constants", "symbols"]:
            values.extend(str(item) for item in entities.get(key, []))
        return ", ".join(values[:8]) if values else "the main variables and parameters"

    def _generate_text(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> tuple[str, str, Dict[str, Union[str, float, bool, int]]]:
        start = perf_counter()
        active_model_name = model_name or self.model_name
        base_max_tokens = int(
            max_tokens
            or self.config.get(
                "reasoning_api_max_tokens",
                max(int(self.config.get("reasoning_max_new_tokens", 2048)), 8192),
            )
        )
        token_attempts = [base_max_tokens]
        if base_max_tokens < 16384:
            token_attempts.append(16384)

        text = ""
        reasoning_content = ""
        usage: Dict[str, Any] = {}
        max_tokens_used = base_max_tokens

        for max_tokens in token_attempts:
            payload = {
                "model": active_model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                        or (
                            "You are a world-class mathematics and physics reasoning assistant. "
                            "Produce correct, structured, pedagogically strong solutions."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "max_tokens": max_tokens,
            }
            last_error: Optional[Exception] = None
            for attempt in range(self.api_retries):
                try:
                    response = requests.post(
                        f"{self.api_base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                            "Connection": "close",
                        },
                        json=payload,
                        timeout=(30, self.api_timeout_seconds),
                    )
                    response.raise_for_status()
                    body = response.json()
                    choice = ((body.get("choices") or [{}])[0] or {}).get("message") or {}
                    text = str(choice.get("content") or "").strip()
                    reasoning_content = str(choice.get("reasoning_content") or "").strip()
                    usage = body.get("usage") or {}
                    max_tokens_used = max_tokens
                    last_error = None
                    break
                except requests.exceptions.RequestException as exc:
                    last_error = exc
                    if attempt + 1 >= self.api_retries:
                        break
                    sleep(self.api_retry_backoff_seconds * (2 ** attempt))
            if last_error:
                raise last_error
            if text:
                break

        elapsed = perf_counter() - start
        return text, reasoning_content, {
            "model_name": active_model_name,
            "model_loaded_now": False,
            "model_load_seconds": 0.0,
            "generation_seconds": round(elapsed, 3),
            "decode_seconds": 0.0,
            "reasoning_call_seconds": round(elapsed, 3),
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
            "reasoning_content_chars": len(reasoning_content),
            "max_tokens_requested": max_tokens_used,
        }

    def generate_solution(
        self,
        parsed_input: Dict[str, Any],
        include_reasoning_trace: bool = False,
    ) -> Dict[str, Any]:
        total_start = perf_counter()
        style = self.determine_solution_style(parsed_input)
        prompt = self.prepare_prompt(parsed_input, style)
        text, reasoning_content, timing = self._generate_text(prompt)
        solution = self._structure_solution(text, style)
        if include_reasoning_trace:
            solution = self.enhance_with_reasoning_trace(solution, reasoning_content=reasoning_content)
        solution["_metadata"] = {
            "style": style.value,
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "prompt_length_chars": len(prompt),
            "generation_time": timing["generation_seconds"],
            "total_time": round(perf_counter() - total_start, 3),
            "parsed_input_summary": {
                "domain": parsed_input.get("domain"),
                "complexity": parsed_input.get("complexity"),
                "intent": parsed_input.get("intent"),
                "secondary_intents": parsed_input.get("secondary_intents", []),
                "asks": parsed_input.get("asks", []),
                "has_equations": bool(parsed_input.get("equations")),
            },
            "_timing": timing,
        }
        solution["_prompt"] = prompt
        return solution

    def _build_solution_markdown(self, parsed_input: Dict[str, Any], solution: Dict[str, Any]) -> str:
        markdown = [
            "# XplainAI Reasoning",
            "",
            f"**Topic:** {parsed_input.get('topic', 'N/A')}",
            f"**Domain:** {parsed_input.get('domain', 'N/A')}",
            f"**Complexity:** {parsed_input.get('complexity', 'N/A')}",
            f"**Intent:** {parsed_input.get('intent', 'N/A')}",
            "",
            "## Parsed Context",
            "",
            f"- Key concepts: {', '.join(parsed_input.get('key_concepts', [])[:8]) or 'None'}",
            f"- Asks: {'; '.join(parsed_input.get('asks', [])) or 'None'}",
            "",
            "## Solution",
            "",
            solution.get("full_text", "") or "No solution text available.",
            "",
        ]
        return "\n".join(markdown).strip()

    def _clean_code_response(self, text: str) -> str:
        value = str(text or "").strip()
        fenced = re.search(r"```(?:python)?\s*(.*?)```", value, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            value = fenced.group(1).strip()
        return value.strip()

    def _manim_code_has_overlap_risk(self, code: str) -> bool:
        value = str(code or "")
        if not value.strip():
            return True
        if "BoxLayoutScene" in value and "class BoxLayoutScene" not in value:
            return True
        required_voiceover_markers = [
            "from manim_voiceover import VoiceoverScene",
            "from manim_voiceover.services.coqui import CoquiService",
            "from pydub import AudioSegment",
            "import imageio_ffmpeg",
            "AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()",
            "VoiceoverScene",
            "self.set_speech_service(",
            "with self.voiceover(",
        ]
        if any(marker not in value for marker in required_voiceover_markers):
            return True
        required_helper_defs = [
            "def fit_to_box",
            "def keep_inside_box",
            "def place_in_box",
            "def mobjects_overlap",
            "def resolve_overlap",
        ]
        if any(helper not in value for helper in required_helper_defs):
            return True
        risky_patterns = [
            r"Transform\(\s*[A-Za-z_][\w]*\[\d+\]\s*,\s*[A-Za-z_][\w]*\[\d+\]",
            r"place_in_box\([^\n]+\)\s*\n\s*[A-Za-z_][\w]*\.shift\(",
            r"\.shift\((?:UP|DOWN)\s*\*\s*[0-9.]+\)",
        ]
        if any(re.search(pattern, value, flags=re.MULTILINE) for pattern in risky_patterns):
            return True
        if value.count("place_in_box(") >= 3 and value.count(".shift(") >= 3:
            return True
        if value.count("place_in_box(") >= 3 and "resolve_overlap(" not in value:
            return True
        return False

    def _summarize_solution_steps(self, solution: Dict[str, Any]) -> str:
        steps = solution.get("steps", [])
        if not isinstance(steps, list) or not steps:
            return "No structured steps available."

        parts: List[str] = []
        for step in steps[:6]:
            if not isinstance(step, dict):
                continue
            number = step.get("step_number", len(parts) + 1)
            title = str(step.get("title") or f"Step {number}").strip()
            description = str(step.get("description") or "").strip()
            summary = re.sub(r"\s+", " ", description)[:220]
            if summary:
                parts.append(f"Step {number}: {title}. {summary}")
            else:
                parts.append(f"Step {number}: {title}.")
        return "\n".join(parts) if parts else "No structured steps available."

    def prepare_scene_planner_prompt(self, parsed_input: Dict[str, Any], solution: Dict[str, Any]) -> str:
        template = self.solution_templates["scene_planner"]
        reasoning_markdown = self._build_solution_markdown(parsed_input, solution)
        prompt = template.format(
            topic=parsed_input.get("topic", "the concept"),
            domain=parsed_input.get("domain", "general"),
            complexity=parsed_input.get("complexity", "intermediate"),
            intent=parsed_input.get("intent", "concept_explanation"),
            secondary_intents=", ".join(parsed_input.get("secondary_intents", [])) or "None",
            grounded_facts=self._build_grounded_fact_sheet(parsed_input),
            problem=self._extract_problem_statement(parsed_input),
            context=self._build_context(parsed_input),
            equations=self._format_equations(parsed_input.get("equations", [])),
            concepts=", ".join(parsed_input.get("key_concepts", [])[:8]) or "None",
            parameters=self._extract_parameters(parsed_input),
            understanding=solution.get("understanding", "") or "Not available.",
            step_summaries=self._summarize_solution_steps(solution),
            final_answer=solution.get("final_answer", "") or "Not available.",
            verification=solution.get("verification", "") or "Not available.",
            extensions="; ".join(solution.get("extensions", [])) or "None",
            scene_planner_template=(
                self._read_prompt_asset("scene_planner_template.md") or "Not available."
            ),
            reasoning_markdown=reasoning_markdown,
        )
        contract = "\n".join(
            [
                "Output requirements:",
                "- Title the output `Scene Planner`.",
                "- Produce a single polished prompt for a downstream Manim code generator.",
                "- Do not generate Manim code yourself.",
                "- The prompt must be self-contained and executable as instructions for code generation.",
                "- Follow the provided Scene Planner Template Reference closely.",
                "- Treat the provided reasoning markdown as the main source of truth for the animation plan.",
                "- Parser-Grounded Facts are more authoritative than the reasoning markdown whenever they conflict.",
                "- If the reasoning markdown introduces a worked example or numeric setup that is not present in Parser-Grounded Facts, discard it.",
                "- Use only facts that are explicitly supported by the parsed context or reasoning markdown.",
                "- If a numeric value, circuit parameter, label, or graph detail is missing, mark it as 'not specified' instead of inventing it.",
                "- Start with a strong role definition for the downstream code generator.",
                "- Include a concise problem context section with the final mathematical results to visualize.",
                "- Include a strict scene-by-scene animation plan in the correct order.",
                "- Use scene names and explicit scene goals.",
                "- Include visual design rules, colors, labels, camera behavior, and annotation guidance.",
                "- Assume the downstream Manim video is narrated unless the user explicitly asks for a silent animation.",
                "- Include layout and overlap-prevention guidance that preserves a box-based layout workflow.",
                "- Include technical requirements using Manim Community Edition primitives such as Axes, MathTex, VGroup, Brace, plot/plot_line_graph, Create, Transform, ReplacementTransform, FadeIn, FadeOut.",
                "- If intervals, piecewise behavior, or derived values are present, make them explicit and tabular.",
                "- If the source does not support a detail, say to avoid inventing it.",
                "- If the reasoning markdown implies a signals-and-systems style visualization, explicitly describe activation points, interval values, area shading, and final boxed result.",
                "- End with implementation constraints for clean, modular, executable code.",
                "- Return plain text only, not JSON and not fenced code blocks.",
            ]
        )
        return f"{prompt}\n\n{contract}"

    def generate_scene_planner(
        self,
        parsed_input: Dict[str, Any],
        solution: Dict[str, Any],
    ) -> Dict[str, Any]:
        total_start = perf_counter()
        prompt = self.prepare_scene_planner_prompt(parsed_input, solution)
        text, reasoning_content, timing = self._generate_text(
            prompt,
            system_prompt=(
                "You are an expert animation planner for mathematical explanations. Convert solved reasoning "
                "markdown into a high-fidelity Scene Planner prompt for a downstream Manim Community Edition "
                "code generator. Favor exact structure, scene ordering, visible math, and explicit visual intent."
            ),
            max_tokens=int(self.config.get("scene_planner_max_tokens", 4096)),
            model_name=self.scene_planner_model,
        )
        return {
            "text": text,
            "_metadata": {
                "model": self.scene_planner_model,
                "timestamp": datetime.now().isoformat(),
                "prompt_length_chars": len(prompt),
                "total_time": round(perf_counter() - total_start, 3),
                "_timing": timing,
                "reasoning_signal_present": bool(reasoning_content),
            },
            "_prompt": prompt,
        }

    def prepare_manim_code_prompt(
        self,
        parsed_input: Dict[str, Any],
        solution: Dict[str, Any],
        scene_planner: Dict[str, Any],
    ) -> str:
        template = self.solution_templates["manim_code_generator"]
        reasoning_markdown = self._build_solution_markdown(parsed_input, solution)
        prompt = template.format(
            topic=parsed_input.get("topic", "the concept"),
            domain=parsed_input.get("domain", "general"),
            grounded_facts=self._build_grounded_fact_sheet(parsed_input),
            problem=self._extract_problem_statement(parsed_input),
            concepts=", ".join(parsed_input.get("key_concepts", [])[:8]) or "None",
            equations=self._format_equations(parsed_input.get("equations", [])),
            final_answer=solution.get("final_answer", "") or "Not available.",
            reasoning_markdown=reasoning_markdown,
            scene_planner=str(scene_planner.get("text") or "").strip(),
            scene_planner_template=(
                self._read_prompt_asset("scene_planner_template.md") or "Not available."
            ),
            few_shot_reference=self._read_prompt_asset("manim_few_shot_example.py") or "Not available.",
            layout_guidance=self._read_prompt_asset("manim_layout_guidance.md") or "Not available.",
            no_overlap_guidance=self._read_prompt_asset("manim_no_overlap_rules.md") or "Not available.",
        )
        contract = "\n".join(
            [
                "Output requirements:",
                "- Return only executable Python code for Manim Community Edition.",
                "- Do not return markdown fences, prose, JSON, or explanations.",
                "- Include imports, constants, helper functions, scene class, and construct method.",
                "- If any scene class inherits from a helper base such as `BoxLayoutScene`, define that helper base class in the same file before the scene class.",
                "- Treat the Scene Planner as authoritative and use the Scene Planner Template Reference only to understand its structure and intended level of detail.",
                "- Parser-Grounded Facts override any conflicting or more specific detail that may appear in the reasoning markdown or Scene Planner.",
                "- You MUST follow the narrated few-shot pattern using `VoiceoverScene`, `CoquiService`, `AudioSegment`, and `imageio_ffmpeg` unless the user explicitly requests a silent animation.",
                "- The scene class must inherit from `VoiceoverScene` directly or through a helper base class defined in the same file.",
                "- Include the runtime ffmpeg wiring exactly: `AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()`.",
                "- Call `self.set_speech_service(CoquiService(...))` inside `construct()` or setup code before narration starts.",
                "- Use `with self.voiceover(text=\"...\") as tracker:` blocks throughout the lesson so the final MP4 contains narration audio.",
                "- Use a box-based layout system with layout_box(...) and scene box dictionaries.",
                "- Add stable `self.next_section(\"...\")` markers for each major logical scene or clip so the frontend can rerender from a chosen section onward.",
                "- Use modern Manim animation syntax such as `self.play(mobject.animate.rotate(...), run_time=...)`; do not use deprecated method-passing patterns like `self.play(mobject.rotate, angle, ...)`.",
                "- Apply this layout order: fixed scene boxes first, fit each object to its box, clamp inside the box, collision-check as a final pass.",
                "- Enforce a strict no-overlap policy for all visible major objects.",
                "- Use only Manim Community Edition-safe colors: prefer hex strings like `\"#00BCD4\"` or clearly supported constants such as `BLUE`, `GREEN`, `RED`, `YELLOW`, `WHITE`, `GRAY`, `ORANGE`, `PURPLE`, `TEAL`.",
                "- Do not use unsupported bare color names such as `CYAN` unless you define them yourself.",
                "- Never keep two dense equations in the same box at once.",
                "- Never use `.shift()` after `place_in_box(...)` as the primary way to separate major formula groups.",
                "- For multi-step algebra, create distinct boxes or use a vertical stack helper instead of manual offsets.",
                "- Do not use indexed submobject transforms between long formulas when a full-group replacement is safer.",
                "- Add helpers such as `stack_in_box(...)`, `replace_in_box(...)`, or equivalent safe layout utilities.",
                "- Avoid primary layout via chained relative positioning like repeated .shift() or .next_to() for major scene structure.",
                "- Include helper methods equivalent to fit_to_box, keep_inside_box, place_in_box, mobjects_overlap, and resolve_overlap.",
                "- Do not treat collision helpers as optional; the final script must define them.",
                "- After placing multiple major mobjects in a scene, call `resolve_overlap(...)` as a final safety step whenever crowding is possible.",
                "- Use fade-based swaps in tight regions instead of aggressive in-place transforms when appropriate.",
                "- Follow the Scene Planner faithfully and do not invent missing numeric details.",
                "- If the Scene Planner leaves a visual detail unspecified, implement the safest minimal version.",
                "- Every `VGroup` must contain only Mobjects. Never place raw coordinates, numpy arrays, or scalar values inside `VGroup`.",
                "- Build formula mobjects explicitly before wrapping them in `SurroundingRectangle` or grouped animations.",
                "- If you use `get_part_by_tex(...)` on `MathTex` or `Tex`, make sure the target token is isolated with `substrings_to_isolate=[...]` or use a fallback-safe highlight strategy instead of assuming the part exists.",
                "- Before finalizing, self-check for runnable structure: valid imports, one scene class, only Mobjects in groups, no markdown fences, and no placeholder comments like TODO.",
                "- Keep the code modular, readable, and directly runnable.",
            ]
        )
        return f"{prompt}\n\n{contract}"

    def prepare_manim_layout_refiner_prompt(
        self,
        parsed_input: Dict[str, Any],
        scene_planner: Dict[str, Any],
        generated_code: str,
    ) -> str:
        template = self.solution_templates["manim_layout_refiner"]
        return template.format(
            topic=parsed_input.get("topic", "the concept"),
            scene_planner=str(scene_planner.get("text") or "").strip(),
            scene_planner_template=self._read_prompt_asset("scene_planner_template.md") or "Not available.",
            few_shot_reference=self._read_prompt_asset("manim_few_shot_example.py") or "Not available.",
            layout_guidance=self._read_prompt_asset("manim_layout_guidance.md") or "Not available.",
            no_overlap_guidance=self._read_prompt_asset("manim_no_overlap_rules.md") or "Not available.",
            generated_code=generated_code,
        )

    def refine_manim_code_layout(
        self,
        parsed_input: Dict[str, Any],
        scene_planner: Dict[str, Any],
        generated_code: str,
    ) -> Dict[str, Any]:
        total_start = perf_counter()
        prompt = self.prepare_manim_layout_refiner_prompt(parsed_input, scene_planner, generated_code)
        text, reasoning_content, timing = self._generate_text(
            prompt,
            system_prompt=(
                "You are a Manim repair specialist. Rewrite the script so it is clean, executable, narrated, and "
                "strictly avoids overlap. Preserve the lesson content, but reorganize scene boxes, replacements, "
                "and formula staging to eliminate crowding. The final script must follow the narrated few-shot "
                "pattern with VoiceoverScene, CoquiService, AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe(), "
                "and with self.voiceover(...) blocks. It must also include helper methods equivalent to fit_to_box, "
                "keep_inside_box, place_in_box, mobjects_overlap, and resolve_overlap, and must use resolve_overlap "
                "as a final safety step where crowding could occur."
            ),
            max_tokens=int(self.config.get("manim_layout_refiner_max_tokens", 8192)),
            model_name=self.manim_layout_refiner_model,
        )
        code = self._clean_code_response(text)
        return {
            "text": code,
            "_metadata": {
                "model": self.manim_layout_refiner_model,
                "timestamp": datetime.now().isoformat(),
                "prompt_length_chars": len(prompt),
                "total_time": round(perf_counter() - total_start, 3),
                "_timing": timing,
                "reasoning_signal_present": bool(reasoning_content),
            },
            "_prompt": prompt,
        }

    def generate_manim_code(
        self,
        parsed_input: Dict[str, Any],
        solution: Dict[str, Any],
        scene_planner: Dict[str, Any],
    ) -> Dict[str, Any]:
        total_start = perf_counter()
        prompt = self.prepare_manim_code_prompt(parsed_input, solution, scene_planner)
        text, reasoning_content, timing = self._generate_text(
            prompt,
            system_prompt=(
                "You are an expert Manim Community Edition engineer. Generate production-quality educational "
                "animation scripts that are executable, faithful to the provided Scene Planner, and robust in layout. "
                "Use the provided few-shot reference and layout guidance to shape structure and collision handling. "
                "You must internally self-review for common Manim mistakes before returning code, especially non-Mobjects "
                "inside VGroup, broken imports, or scene classes that ignore the requested layout helpers."
            ),
            max_tokens=int(self.config.get("manim_code_max_tokens", 8192)),
            model_name=self.manim_code_model,
        )
        code = self._clean_code_response(text)
        refined = None
        if self._manim_code_has_overlap_risk(code):
            refined = self.refine_manim_code_layout(parsed_input, scene_planner, code)
            code = self._clean_code_response((refined or {}).get("text", code))
        return {
            "text": code,
            "_metadata": {
                "model": self.manim_code_model,
                "timestamp": datetime.now().isoformat(),
                "prompt_length_chars": len(prompt),
                "total_time": round(perf_counter() - total_start, 3),
                "_timing": timing,
                "reasoning_signal_present": bool(reasoning_content),
                "layout_refiner_used": bool(refined),
                "layout_refiner_model": (refined or {}).get("_metadata", {}).get("model"),
                "layout_refiner_timing": (refined or {}).get("_metadata", {}).get("_timing", {}),
            },
            "_prompt": prompt,
            "_layout_refiner": refined,
        }

    def _structure_solution(self, text: str, style: SolutionStyle) -> Dict[str, Any]:
        sections = self._split_into_sections(text)
        return {
            "solution_style": style.value,
            "full_text": text,
            "understanding": sections.get("Understanding", "").strip(),
            "prerequisites": sections.get("Prerequisites", "").strip(),
            "steps": self._parse_steps(sections.get("Step-by-Step Solution", "")),
            "equations": self._extract_equations_from_text(text),
            "key_insights": self._parse_list_section(sections.get("Key Insights", "")),
            "final_answer": sections.get("Final Answer", "").strip(),
            "verification": sections.get("Verification", "").strip(),
            "extensions": self._parse_list_section(sections.get("Extensions", "")),
        }

    def _split_into_sections(self, text: str) -> Dict[str, str]:
        matches = list(re.finditer(r"^##\s+(.+?)\s*$", text, flags=re.MULTILINE))
        if not matches:
            return {"Step-by-Step Solution": text.strip()}
        sections: Dict[str, str] = {}
        for index, match in enumerate(matches):
            title = match.group(1).strip()
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            sections[title] = text[start:end].strip()
        return sections

    def _parse_steps(self, text: str) -> List[Dict[str, Any]]:
        if not text.strip():
            return []
        matches = list(re.finditer(r"^(?:Step\s+(\d+)|(\d+)\.)[:\-]?\s*(.*)$", text, flags=re.MULTILINE))
        if not matches:
            return [{"step_number": 1, "title": "Solution", "description": text.strip(), "equations": self._extract_equations_from_text(text)}]
        steps: List[Dict[str, Any]] = []
        for index, match in enumerate(matches):
            step_number = int(match.group(1) or match.group(2))
            title = (match.group(3) or "").strip() or f"Step {step_number}"
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            description = text[start:end].strip()
            steps.append(
                {
                    "step_number": step_number,
                    "title": title,
                    "description": description,
                    "equations": self._extract_equations_from_text(description),
                }
            )
        return steps

    def _parse_list_section(self, text: str) -> List[str]:
        items = [re.sub(r"^\s*[-*]\s*", "", line).strip() for line in text.splitlines()]
        items = [item for item in items if item]
        return items or ([text.strip()] if text.strip() else [])

    def _extract_equations_from_text(self, text: str) -> List[Dict[str, str]]:
        equations: List[Dict[str, str]] = []
        for pattern in [r"\$\$(.*?)\$\$", r"\\\[(.*?)\\\]", r"\\\((.*?)\\\)"]:
            for match in re.findall(pattern, text, flags=re.DOTALL):
                latex = match.strip()
                if latex:
                    equations.append({"latex": latex, "context": self._equation_context(text, latex)})
        return equations

    def _equation_context(self, text: str, equation: str) -> str:
        index = text.find(equation)
        if index == -1:
            return ""
        start = max(0, index - 100)
        end = min(len(text), index + len(equation) + 100)
        return text[start:end].replace(equation, "[EQUATION]").strip()

    def enhance_with_reasoning_trace(
        self,
        solution: Dict[str, Any],
        reasoning_content: str = "",
    ) -> Dict[str, Any]:
        solution["reasoning_trace"] = {
            "strategy": solution.get("solution_style", "step_by_step"),
            "step_summaries": [
                {"step": step["step_number"], "focus": step.get("title") or f"Step {step['step_number']}"}
                for step in solution.get("steps", [])
            ],
            "alternative_approaches": solution.get("extensions", [])[:3],
            "reasoning_signal_present": bool(reasoning_content),
        }
        return solution

    def save_solution_bundle(self, result: Dict[str, Any]) -> Dict[str, str]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.output_dir / f"solution_{timestamp}.json"
        md_path = self.output_dir / f"solution_{timestamp}.md"
        parsed = result.get("parsed_input", {})
        solution = result.get("solution", {})
        markdown = self._build_solution_markdown(parsed, solution)
        with open(md_path, "w", encoding="utf-8") as handle:
            handle.write(markdown + "\n")
        saved = {"json": str(json_path), "markdown": str(md_path)}

        scene_planner = result.get("scene_planner", {})
        scene_planner_text = str(scene_planner.get("text") or "").strip() if isinstance(scene_planner, dict) else ""
        if scene_planner_text:
            scene_planner_path = self.output_dir / f"scene_planner_{timestamp}.md"
            with open(scene_planner_path, "w", encoding="utf-8") as handle:
                handle.write(scene_planner_text + "\n")
            saved["scene_planner"] = str(scene_planner_path)

        manim_code = result.get("manim_code", {})
        manim_code_text = str(manim_code.get("text") or "").strip() if isinstance(manim_code, dict) else ""
        if manim_code_text:
            manim_code_path = self.output_dir / f"manim_code_{timestamp}.py"
            with open(manim_code_path, "w", encoding="utf-8") as handle:
                handle.write(manim_code_text + "\n")
            saved["manim_code"] = str(manim_code_path)

        pipeline_metadata = result.setdefault("pipeline_metadata", {})
        pipeline_metadata["saved_files"] = saved
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)

        return saved


class SolutionOrchestrator:
    """Orchestrate parser -> reasoning pipeline."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.parser = LocalParser(config_path)
        self.reasoner = MathematicalReasoner(config_path)
        self.cache_enabled = bool(self.reasoner.config.get("enable_cache", True))
        self.cache_dir = self.reasoner.cache_dir

    def process(
        self,
        input_data: Union[str, Path, Image.Image, Dict[str, Any]],
        input_type: str = "auto",
        prompt_text: str = "",
        include_reasoning_trace: bool = False,
        generate_scene_planner: bool = False,
        generate_manim_code: bool = False,
        progress_callback: Optional[Callable[[str, str], None]] = None,
    ) -> Dict[str, Any]:
        start = perf_counter()
        self._emit_progress(progress_callback, "cache", "Checking cached pipeline result...")
        cache_key = self._hash_input(
            input_data,
            input_type,
            prompt_text,
            include_reasoning_trace=include_reasoning_trace,
            generate_scene_planner=generate_scene_planner or generate_manim_code,
            generate_manim_code=generate_manim_code,
        )
        if self.cache_enabled:
            cached = self._get_cached(cache_key)
            if cached:
                self._emit_progress(progress_callback, "cache", "Loaded cached pipeline result.")
                return cached
        if isinstance(input_data, dict) and "intent" in input_data:
            self._emit_progress(progress_callback, "parse", "Using pre-parsed input and skipping parser stage.")
            parsed = dict(input_data)
        else:
            parser_target = self.parser.__class__.__name__
            self._emit_progress(
                progress_callback,
                "parse",
                f"Starting parser stage with {parser_target}...",
            )
            parsed = self._parse_input(input_data, input_type, prompt_text)
            self._emit_progress(progress_callback, "parse", "Parser stage complete.")
        if "error" in parsed:
            return parsed
        if not parsed.get("_source_text"):
            parsed["_source_text"] = self._build_source_text(input_data, prompt_text)
        self._emit_progress(
            progress_callback,
            "reason",
            f"Calling DeepSeek reasoning API ({self.reasoner.model_name})...",
        )
        solution = self.reasoner.generate_solution(parsed, include_reasoning_trace=include_reasoning_trace)
        self._emit_progress(progress_callback, "reason", "Reasoning step complete.")
        if generate_scene_planner or generate_manim_code:
            self._emit_progress(
                progress_callback,
                "scene_planner",
                f"Calling DeepSeek scene planner API ({self.reasoner.scene_planner_model})...",
            )
            scene_planner = self.reasoner.generate_scene_planner(parsed, solution)
            self._emit_progress(progress_callback, "scene_planner", "Scene planner step complete.")
        else:
            scene_planner = None
        if generate_manim_code and isinstance(scene_planner, dict):
            self._emit_progress(
                progress_callback,
                "manim_code",
                f"Calling DeepSeek Manim code generator ({self.reasoner.manim_code_model})...",
            )
            manim_code = self.reasoner.generate_manim_code(parsed, solution, scene_planner or {})
            if bool((manim_code or {}).get("_metadata", {}).get("layout_refiner_used")):
                self._emit_progress(
                    progress_callback,
                    "manim_code",
                    "Layout repair/refiner was applied to the generated Manim code.",
                )
            self._emit_progress(progress_callback, "manim_code", "Manim code generation complete.")
        else:
            manim_code = None
        result = {
            "parsed_input": parsed,
            "solution": solution,
            "pipeline_metadata": {
                "parser_model": parsed.get("_model"),
                "reasoner_model": self.reasoner.model_name,
                "scene_planner_model": self.reasoner.scene_planner_model if scene_planner else None,
                "manim_code_model": self.reasoner.manim_code_model if manim_code else None,
                "parse_timing": parsed.get("_timing", {}),
                "reasoning_timing": solution.get("_metadata", {}).get("_timing", {}),
                "scene_planner_timing": (
                    scene_planner.get("_metadata", {}).get("_timing", {})
                    if isinstance(scene_planner, dict)
                    else {}
                ),
                "manim_code_timing": (
                    manim_code.get("_metadata", {}).get("_timing", {})
                    if isinstance(manim_code, dict)
                    else {}
                ),
                "total_pipeline_seconds": round(perf_counter() - start, 3),
            },
        }
        if scene_planner:
            result["scene_planner"] = scene_planner
        if manim_code:
            result["manim_code"] = manim_code
        self._emit_progress(progress_callback, "save", "Saving pipeline bundle and artifacts...")
        result["pipeline_metadata"]["saved_files"] = self.reasoner.save_solution_bundle(result)
        if self.cache_enabled:
            self._cache_result(cache_key, result)
        self._emit_progress(progress_callback, "save", "Pipeline bundle saved.")
        return result

    def _emit_progress(
        self,
        progress_callback: Optional[Callable[[str, str], None]],
        stage: str,
        message: str,
    ) -> None:
        if progress_callback is not None:
            progress_callback(stage, message)

    def _parse_input(self, input_data: Union[str, Path, Image.Image, Dict[str, Any]], input_type: str, prompt_text: str) -> Dict[str, Any]:
        if input_type == "json":
            return self._load_parsed_json(input_data)
        if input_type == "text":
            return self.parser.parse_text(str(input_data))
        if input_type == "image":
            image = input_data if isinstance(input_data, Image.Image) else Image.open(input_data)
            return self.parser.parse_image(image, prompt_text=prompt_text)
        if input_type == "pdf":
            return self.parser.parse_pdf(str(input_data), prompt_text=prompt_text)
        if isinstance(input_data, dict):
            return dict(input_data)
        path_candidate = Path(str(input_data))
        if path_candidate.exists():
            if path_candidate.suffix.lower() == ".json":
                return self._load_parsed_json(path_candidate)
            if path_candidate.suffix.lower() == ".pdf":
                return self.parser.parse_pdf(str(path_candidate), prompt_text=prompt_text)
            if path_candidate.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                return self.parser.parse_image(Image.open(path_candidate), prompt_text=prompt_text)
        return self.parser.parse_text(str(input_data))

    def _load_parsed_json(self, input_data: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(input_data, dict):
            payload = dict(input_data)
        else:
            json_path = Path(str(input_data))
            with open(json_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)

        if isinstance(payload.get("parsed_input"), dict):
            payload = payload["parsed_input"]

        if not isinstance(payload, dict) or "intent" not in payload:
            return {"error": "JSON input must be a parser output or a solution bundle containing `parsed_input`."}

        return payload

    def _build_source_text(self, input_data: Union[str, Path, Image.Image, Dict[str, Any]], prompt_text: str) -> str:
        if prompt_text.strip():
            return prompt_text.strip()
        if isinstance(input_data, str):
            candidate = Path(input_data)
            if candidate.exists():
                if candidate.suffix.lower() == ".json":
                    return f"Use the stored parser output from {candidate.name} and provide a full solution."
                return f"Analyze the uploaded file: {candidate.name}"
            return input_data
        if isinstance(input_data, Path):
            if input_data.suffix.lower() == ".json":
                return f"Use the stored parser output from {input_data.name} and provide a full solution."
            return f"Analyze the uploaded file: {input_data.name}"
        if isinstance(input_data, dict):
            asks = [str(item).strip() for item in input_data.get("asks", []) if str(item).strip()]
            if asks:
                return " ".join(asks)
            topic = str(input_data.get("topic", "")).strip()
            return topic or json.dumps(input_data, ensure_ascii=False)
        return "Analyze the uploaded problem and provide a full solution."

    def _hash_input(
        self,
        input_data: Any,
        input_type: str,
        prompt_text: str,
        *,
        include_reasoning_trace: bool = False,
        generate_scene_planner: bool = False,
        generate_manim_code: bool = False,
    ) -> str:
        payload = json.dumps(input_data, sort_keys=True, ensure_ascii=False) if isinstance(input_data, dict) else str(input_data)
        raw = json.dumps(
            {
                "input": payload,
                "input_type": input_type,
                "prompt_text": prompt_text,
                "reasoning_model": self.reasoner.model_name,
                "scene_planner_model": self.reasoner.scene_planner_model,
                "manim_code_model": self.reasoner.manim_code_model,
                "include_reasoning_trace": include_reasoning_trace,
                "generate_scene_planner": generate_scene_planner,
                "generate_manim_code": generate_manim_code,
                "reasoning_api_max_tokens": int(
                    self.reasoner.config.get(
                        "reasoning_api_max_tokens",
                        max(int(self.reasoner.config.get("reasoning_max_new_tokens", 2048)), 8192),
                    )
                ),
                "scene_planner_max_tokens": int(
                    self.reasoner.config.get("scene_planner_max_tokens", 4096)
                ),
                "manim_code_max_tokens": int(
                    self.reasoner.config.get("manim_code_max_tokens", 8192)
                ),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _get_cached(self, key: str) -> Optional[Dict[str, Any]]:
        cache_file = self.cache_dir / f"{key}.json"
        if not cache_file.exists():
            return None
        with open(cache_file, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _cache_result(self, key: str, result: Dict[str, Any]) -> None:
        with open(self.cache_dir / f"{key}.json", "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
