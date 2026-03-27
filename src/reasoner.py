"""Reasoning engine for mathematical and physics problem solving."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Union

import torch
import yaml
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        self.model_name = self.config.get(
            "reasoning_model",
            "deepseek-ai/deepseek-math-7b-instruct",
        )
        self.model_cache = self.project_root / self.config.get("model_cache", "models/cache")
        self.output_dir = self.project_root / self.config.get("output_dir", "outputs")
        self.cache_dir = self.project_root / self.config.get("cache_dir", "cache")
        self.model_cache.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._get_device()
        self.model_dtype = self._get_model_dtype()
        self.solution_templates = self._load_templates()
        self.model = None
        self.tokenizer = None

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(self.project_root / config_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def _get_device(self) -> str:
        if self.config.get("use_gpu") and torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _get_model_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.device == "cuda" else torch.float32

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
        }

    def _ensure_model_loaded(self) -> Dict[str, Union[str, float, bool]]:
        if self.model is not None and self.tokenizer is not None:
            return {"model_name": self.model_name, "model_loaded_now": False, "model_load_seconds": 0.0}

        start = perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=str(self.model_cache),
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: Dict[str, Any] = {
            "cache_dir": str(self.model_cache),
            "torch_dtype": self.model_dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if self.device == "cuda":
            kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()
        return {
            "model_name": self.model_name,
            "model_loaded_now": True,
            "model_load_seconds": round(perf_counter() - start, 3),
        }

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

    def _chat_prompt(self, prompt: str) -> str:
        system = (
            "You are a world-class mathematics and physics reasoning assistant. "
            "Produce correct, structured, pedagogically strong solutions."
        )
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"System: {system}\n\nUser: {prompt}\n\nAssistant:"

    def _generate_text(self, prompt: str) -> tuple[str, Dict[str, Union[str, float, bool]]]:
        start = perf_counter()
        load_timing = self._ensure_model_loaded()
        prepared = self._chat_prompt(prompt)
        inputs = self.tokenizer(
            prepared,
            return_tensors="pt",
            truncation=True,
            max_length=int(self.config.get("reasoning_input_max_length", 4096)),
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        gen_start = perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=int(self.config.get("reasoning_max_new_tokens", 2048)),
                do_sample=float(self.config.get("reasoning_temperature", 0.2)) > 0,
                temperature=max(float(self.config.get("reasoning_temperature", 0.2)), 1e-5),
                top_p=float(self.config.get("reasoning_top_p", 0.95)),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        gen_seconds = perf_counter() - gen_start
        decode_start = perf_counter()
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        decode_seconds = perf_counter() - decode_start
        return text, {
            **load_timing,
            "generation_seconds": round(gen_seconds, 3),
            "decode_seconds": round(decode_seconds, 3),
            "reasoning_call_seconds": round(perf_counter() - start, 3),
        }

    def generate_solution(
        self,
        parsed_input: Dict[str, Any],
        include_reasoning_trace: bool = False,
    ) -> Dict[str, Any]:
        total_start = perf_counter()
        style = self.determine_solution_style(parsed_input)
        prompt = self.prepare_prompt(parsed_input, style)
        text, timing = self._generate_text(prompt)
        solution = self._structure_solution(text, style)
        if include_reasoning_trace:
            solution = self.enhance_with_reasoning_trace(solution)
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

    def enhance_with_reasoning_trace(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        solution["reasoning_trace"] = {
            "strategy": solution.get("solution_style", "step_by_step"),
            "step_summaries": [
                {"step": step["step_number"], "focus": step.get("title") or f"Step {step['step_number']}"}
                for step in solution.get("steps", [])
            ],
            "alternative_approaches": solution.get("extensions", [])[:3],
        }
        return solution

    def save_solution_bundle(self, result: Dict[str, Any]) -> Dict[str, str]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.output_dir / f"solution_{timestamp}.json"
        md_path = self.output_dir / f"solution_{timestamp}.md"
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        parsed = result.get("parsed_input", {})
        solution = result.get("solution", {})
        markdown = [
            "# XplainAI Solution",
            "",
            f"**Topic:** {parsed.get('topic', 'N/A')}",
            f"**Domain:** {parsed.get('domain', 'N/A')}",
            f"**Complexity:** {parsed.get('complexity', 'N/A')}",
            f"**Style:** {solution.get('solution_style', 'N/A')}",
            "",
            solution.get("full_text", "No solution text available."),
            "",
        ]
        with open(md_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(markdown))
        return {"json": str(json_path), "markdown": str(md_path)}


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
    ) -> Dict[str, Any]:
        start = perf_counter()
        cache_key = self._hash_input(input_data, input_type, prompt_text)
        if self.cache_enabled:
            cached = self._get_cached(cache_key)
            if cached:
                return cached
        parsed = dict(input_data) if isinstance(input_data, dict) and "intent" in input_data else self._parse_input(input_data, input_type, prompt_text)
        if "error" in parsed:
            return parsed
        if not parsed.get("_source_text"):
            parsed["_source_text"] = self._build_source_text(input_data, prompt_text)
        solution = self.reasoner.generate_solution(parsed, include_reasoning_trace=include_reasoning_trace)
        result = {
            "parsed_input": parsed,
            "solution": solution,
            "pipeline_metadata": {
                "parser_model": parsed.get("_model"),
                "reasoner_model": self.reasoner.model_name,
                "parse_timing": parsed.get("_timing", {}),
                "reasoning_timing": solution.get("_metadata", {}).get("_timing", {}),
                "total_pipeline_seconds": round(perf_counter() - start, 3),
            },
        }
        result["pipeline_metadata"]["saved_files"] = self.reasoner.save_solution_bundle(result)
        if self.cache_enabled:
            self._cache_result(cache_key, result)
        return result

    def _parse_input(self, input_data: Union[str, Path, Image.Image, Dict[str, Any]], input_type: str, prompt_text: str) -> Dict[str, Any]:
        if input_type == "text":
            return self.parser.parse_text(str(input_data))
        if input_type == "image":
            image = input_data if isinstance(input_data, Image.Image) else Image.open(input_data)
            return self.parser.parse_image(image, prompt_text=prompt_text)
        if input_type == "pdf":
            return self.parser.parse_pdf(str(input_data), prompt_text=prompt_text)
        if isinstance(input_data, dict):
            return self.parser.parse(input_data)
        path_candidate = Path(str(input_data))
        if path_candidate.exists():
            if path_candidate.suffix.lower() == ".pdf":
                return self.parser.parse_pdf(str(path_candidate), prompt_text=prompt_text)
            if path_candidate.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                return self.parser.parse_image(Image.open(path_candidate), prompt_text=prompt_text)
        return self.parser.parse_text(str(input_data))

    def _build_source_text(self, input_data: Union[str, Path, Image.Image, Dict[str, Any]], prompt_text: str) -> str:
        if isinstance(input_data, str):
            candidate = Path(input_data)
            if candidate.exists():
                return prompt_text.strip() or f"Analyze the uploaded file: {candidate.name}"
            return input_data
        if isinstance(input_data, Path):
            return prompt_text.strip() or f"Analyze the uploaded file: {input_data.name}"
        if isinstance(input_data, dict):
            return json.dumps(input_data, ensure_ascii=False)
        return prompt_text.strip() or "Analyze the uploaded problem and provide a full solution."

    def _hash_input(self, input_data: Any, input_type: str, prompt_text: str) -> str:
        payload = json.dumps(input_data, sort_keys=True, ensure_ascii=False) if isinstance(input_data, dict) else str(input_data)
        raw = json.dumps(
            {
                "input": payload,
                "input_type": input_type,
                "prompt_text": prompt_text,
                "reasoning_model": self.reasoner.model_name,
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
