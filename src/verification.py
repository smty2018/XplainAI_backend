"""Post-solution verification helpers for math and engineering outputs."""

from __future__ import annotations

import re
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )
except ImportError:  # pragma: no cover - dependency is expected but handled defensively
    sp = None  # type: ignore[assignment]
    parse_expr = None  # type: ignore[assignment]
    standard_transformations = ()  # type: ignore[assignment]
    implicit_multiplication_application = None  # type: ignore[assignment]
    convert_xor = None  # type: ignore[assignment]


class VerificationEngine:
    """Run lightweight math and unit checks before a solution is visualized."""

    VERSION = "post-solution-simple-v1"
    _RESERVED_NAMES = {
        "Eq",
        "E",
        "I",
        "N",
        "O",
        "Q",
        "S",
        "acos",
        "asin",
        "atan",
        "cos",
        "exp",
        "ln",
        "log",
        "oo",
        "pi",
        "sin",
        "sqrt",
        "tan",
        "zoo",
    }
    _COMMON_UNIT_TOKENS = {
        "a",
        "amp",
        "amps",
        "cm",
        "ft",
        "g",
        "h",
        "hz",
        "j",
        "joule",
        "joules",
        "kg",
        "km",
        "m",
        "ma",
        "mh",
        "min",
        "mm",
        "mol",
        "n",
        "newton",
        "newtons",
        "ohm",
        "pa",
        "rad",
        "s",
        "sec",
        "t",
        "v",
        "volt",
        "volts",
        "w",
        "watt",
        "watts",
        "ω",
    }

    def __init__(self) -> None:
        # Keep verification lightweight enough to run on every solution before visualization.
        self.sympy_available = sp is not None and parse_expr is not None
        self._transformations = standard_transformations
        if implicit_multiplication_application is not None and convert_xor is not None:
            self._transformations = standard_transformations + (
                implicit_multiplication_application,
                convert_xor,
            )

    def run(self, parsed_input: Dict[str, Any], solution: Dict[str, Any]) -> Dict[str, Any]:
        # Verify the proposed solution before it is used for planning or animation.
        start = perf_counter()
        requested = self._normalize_requested_checks(
            parsed_input.get("verification_targets"),
            parsed_input,
        )
        sympy_report = self._run_sympy_solver(parsed_input, solution, enabled=requested["sympy"])
        unit_report = self._run_unit_checker(parsed_input, solution, enabled=requested["unit_check"])

        mismatches = list(sympy_report.get("mismatches", []))
        warnings: List[str] = []
        for report in (sympy_report, unit_report):
            warnings.extend(str(note) for note in report.get("notes", []) if str(note).strip())

        overall_status = "pass"
        if sympy_report.get("status") == "fail":
            overall_status = "fail"
        elif any(report.get("status") == "warn" for report in (sympy_report, unit_report)):
            overall_status = "warn"
        elif all(report.get("status") == "not_applicable" for report in (sympy_report, unit_report)):
            overall_status = "warn"
            warnings.append("No verification checks were applicable to this solution.")

        summary = self._build_summary(overall_status, sympy_report, unit_report, mismatches)
        return {
            "requested_checks": requested,
            "sympy": sympy_report,
            "unit_check": unit_report,
            "mismatches": mismatches,
            "summary": summary,
            "status": overall_status,
            "warnings": self._dedupe_strings(warnings),
            "_timing": {
                "verification_seconds": round(perf_counter() - start, 3),
                "version": self.VERSION,
            },
        }

    def _normalize_requested_checks(self, raw: Any, parsed_input: Dict[str, Any]) -> Dict[str, bool]:
        base = {
            "sympy": False,
            "unit_check": False,
            "constraint_check": False,
            "edge_case_check": False,
        }
        if isinstance(raw, dict):
            for key in list(base):
                base[key] = bool(raw.get(key, False))

        equations = parsed_input.get("equations", [])
        entities = parsed_input.get("entities", {}) if isinstance(parsed_input.get("entities"), dict) else {}
        units = entities.get("units", []) if isinstance(entities, dict) else []
        if not base["sympy"] and bool(equations):
            base["sympy"] = True
        if not base["unit_check"] and bool(units):
            base["unit_check"] = True
        return base

    def _run_sympy_solver(
        self,
        parsed_input: Dict[str, Any],
        solution: Dict[str, Any],
        *,
        enabled: bool,
    ) -> Dict[str, Any]:
        # The symbolic path independently solves parser-grounded equations, then compares that truth to the LLM's claimed answer.
        if not enabled:
            return self._skipped_report("sympy", "SymPy verification was not requested for this input.")
        if not self.sympy_available:
            return {
                "enabled": True,
                "status": "warn",
                "parsed_equations": [],
                "variables": [],
                "simplified_forms": [],
                "solutions": [],
                "llm_assignments": {},
                "mismatches": [],
                "notes": ["SymPy is not available in the current environment."],
            }

        equation_texts = self._collect_equation_texts(parsed_input)
        if not equation_texts:
            return {
                "enabled": True,
                "status": "not_applicable",
                "parsed_equations": [],
                "variables": [],
                "simplified_forms": [],
                "solutions": [],
                "llm_assignments": {},
                "mismatches": [],
                "notes": ["No parser equations were available for symbolic verification."],
            }

        variable_names = self._collect_variables(parsed_input)
        local_dict = self._build_local_dict(equation_texts + [str(solution.get("final_answer", ""))], variable_names)
        parsed_equations: List[Any] = []
        parsed_equation_strings: List[str] = []
        parse_errors: List[str] = []

        for raw_equation in equation_texts:
            try:
                parsed_equation = self._parse_equation(raw_equation, local_dict)
                if parsed_equation is None:
                    continue
                parsed_equations.append(parsed_equation)
                parsed_equation_strings.append(str(parsed_equation))
            except Exception as exc:  # pragma: no cover - depends on parser output variety
                parse_errors.append(f"Could not parse equation `{raw_equation}`: {exc}")

        if not parsed_equations:
            return {
                "enabled": True,
                "status": "warn",
                "parsed_equations": [],
                "variables": variable_names,
                "simplified_forms": [],
                "solutions": [],
                "llm_assignments": {},
                "mismatches": [],
                "notes": parse_errors or ["The parser equations could not be converted into symbolic form."],
            }

        symbol_names = variable_names or self._infer_variable_names_from_equations(parsed_equations)
        symbols = [local_dict[name] for name in symbol_names if name in local_dict]
        simplified_forms = [self._simplify_equation_repr(item) for item in parsed_equations]
        solution_dicts, solve_notes = self._solve_equations(parsed_equations, symbols)
        llm_assignments = self._extract_llm_assignments(
            solution.get("final_answer", ""),
            solution.get("full_text", ""),
            symbol_names,
            local_dict,
        )
        comparison_status, mismatches, comparison_notes = self._compare_solution_claims(
            solution_dicts,
            llm_assignments,
            str(solution.get("final_answer", "")),
        )

        status = "pass"
        if comparison_status == "mismatch":
            status = "fail"
        elif comparison_status in {"unknown", "partial"} or parse_errors:
            status = "warn"
        elif not solution_dicts:
            status = "warn"

        notes = parse_errors + solve_notes + comparison_notes
        return {
            "enabled": True,
            "status": status,
            "parsed_equations": parsed_equation_strings,
            "variables": symbol_names,
            "simplified_forms": simplified_forms,
            "solutions": [self._stringify_solution_dict(item) for item in solution_dicts],
            "llm_assignments": {key: str(value) for key, value in llm_assignments.items()},
            "mismatches": mismatches,
            "notes": self._dedupe_strings(notes),
        }

    def _run_unit_checker(
        self,
        parsed_input: Dict[str, Any],
        solution: Dict[str, Any],
        *,
        enabled: bool,
    ) -> Dict[str, Any]:
        # This is intentionally a heuristic unit check for now; it is meant to catch obvious mismatches, not replace full dimensional analysis.
        if not enabled:
            return self._skipped_report("unit_check", "Unit checking was not requested for this input.")

        parser_units = [
            str(item).strip()
            for item in (((parsed_input.get("entities") or {}) if isinstance(parsed_input.get("entities"), dict) else {}).get("units") or [])
            if str(item).strip()
        ]
        if not parser_units:
            return {
                "enabled": True,
                "status": "not_applicable",
                "units_found": [],
                "checks": [],
                "notes": ["No explicit units were detected by the parser."],
            }

        equation_texts = self._collect_equation_texts(parsed_input)
        final_answer_text = str(solution.get("final_answer", "") or "")
        checks: List[str] = []
        notes: List[str] = ["Simplified heuristic unit checker only."]

        normalized_units = self._dedupe_strings(parser_units)
        for index, equation_text in enumerate(equation_texts, start=1):
            lhs_units, rhs_units = self._extract_units_by_side(equation_text, normalized_units)
            if lhs_units and rhs_units and lhs_units != rhs_units:
                checks.append(
                    f"Equation {index} has different unit sets on each side: {sorted(lhs_units)} vs {sorted(rhs_units)}."
                )
            if self._has_mixed_addition_units(equation_text, normalized_units):
                checks.append(f"Equation {index} appears to mix incompatible units within an addition or subtraction.")

        if final_answer_text and not self._text_mentions_any_unit(final_answer_text, normalized_units):
            checks.append("The final answer does not visibly mention the units detected in the parsed problem.")

        status = "pass" if not checks else "warn"
        return {
            "enabled": True,
            "status": status,
            "units_found": normalized_units,
            "checks": checks,
            "notes": notes,
        }

    def _collect_equation_texts(self, parsed_input: Dict[str, Any]) -> List[str]:
        equations = parsed_input.get("equations", [])
        texts: List[str] = []
        if not isinstance(equations, list):
            return texts
        for item in equations:
            if isinstance(item, dict):
                candidate = str(item.get("raw") or item.get("latex") or "").strip()
            else:
                candidate = str(item).strip()
            if candidate:
                texts.append(candidate)
        return texts

    def _collect_variables(self, parsed_input: Dict[str, Any]) -> List[str]:
        entities = parsed_input.get("entities", {})
        if not isinstance(entities, dict):
            return []
        variables = entities.get("variables", [])
        values = [self._normalize_symbol_name(str(item)) for item in variables if str(item).strip()]
        return self._dedupe_strings([item for item in values if item and item not in self._RESERVED_NAMES])

    def _build_local_dict(self, texts: Sequence[str], variable_names: Sequence[str]) -> Dict[str, Any]:
        local_dict: Dict[str, Any] = {
            "sin": sp.sin if sp is not None else None,
            "cos": sp.cos if sp is not None else None,
            "tan": sp.tan if sp is not None else None,
            "asin": sp.asin if sp is not None else None,
            "acos": sp.acos if sp is not None else None,
            "atan": sp.atan if sp is not None else None,
            "sqrt": sp.sqrt if sp is not None else None,
            "log": sp.log if sp is not None else None,
            "ln": sp.log if sp is not None else None,
            "exp": sp.exp if sp is not None else None,
            "pi": sp.pi if sp is not None else None,
            "E": sp.E if sp is not None else None,
            "oo": sp.oo if sp is not None else None,
        }
        all_names = set(variable_names)
        for text in texts:
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", self._normalize_math_text(text)):
                if token not in self._RESERVED_NAMES:
                    all_names.add(token)
        for name in sorted(all_names):
            local_dict[name] = sp.Symbol(name) if sp is not None else name
        return local_dict

    def _parse_equation(self, text: str, local_dict: Dict[str, Any]) -> Optional[Any]:
        normalized = self._normalize_math_text(text)
        if not normalized:
            return None
        if "=" in normalized:
            lhs_text, rhs_text = normalized.split("=", 1)
            lhs = parse_expr(lhs_text.strip(), local_dict=local_dict, transformations=self._transformations)
            rhs = parse_expr(rhs_text.strip(), local_dict=local_dict, transformations=self._transformations)
            return sp.Eq(lhs, rhs)
        return parse_expr(normalized, local_dict=local_dict, transformations=self._transformations)

    def _normalize_math_text(self, text: str) -> str:
        value = str(text or "").strip()
        if not value:
            return ""
        replacements = {
            "$": "",
            "\\left": "",
            "\\right": "",
            "\\cdot": "*",
            "\\times": "*",
            "\\div": "/",
            "−": "-",
            "–": "-",
            "×": "*",
            "·": "*",
        }
        for source, target in replacements.items():
            value = value.replace(source, target)
        value = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", value)
        value = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", value)
        value = re.sub(r"([A-Za-z]+)_\{([^{}]+)\}", r"\1_\2", value)
        value = re.sub(r"([A-Za-z]+)_([A-Za-z0-9]+)", r"\1_\2", value)
        value = value.replace("{", "(").replace("}", ")")
        value = value.replace("^", "**")
        value = re.sub(r"\s+", " ", value).strip()
        return value

    def _infer_variable_names_from_equations(self, equations: Sequence[Any]) -> List[str]:
        names: List[str] = []
        for equation in equations:
            for symbol in sorted(getattr(equation, "free_symbols", []), key=lambda item: str(item)):
                names.append(str(symbol))
        return self._dedupe_strings(names)

    def _simplify_equation_repr(self, equation: Any) -> str:
        if sp is None:
            return str(equation)
        if isinstance(equation, sp.Equality):
            return str(sp.simplify(equation.lhs - equation.rhs))
        return str(sp.simplify(equation))

    def _solve_equations(self, equations: Sequence[Any], symbols: Sequence[Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
        if sp is None:
            return [], ["SymPy is not available."]
        notes: List[str] = []
        if not symbols:
            return [], ["No variables were identified for symbolic solving."]
        try:
            if len(symbols) == 1 and len(equations) == 1:
                target = equations[0]
                symbol = symbols[0]
                if isinstance(target, sp.Equality):
                    solution_set = sp.solveset(target, symbol, domain=sp.S.Reals)
                else:
                    solution_set = sp.solveset(sp.Eq(target, 0), symbol, domain=sp.S.Reals)
                if solution_set == sp.S.EmptySet:
                    return [], ["SymPy found no real solution."]
                if isinstance(solution_set, sp.ConditionSet):
                    notes.append("SymPy could not reduce the equation to a finite real solution set.")
                    return [], notes
                if isinstance(solution_set, sp.FiniteSet):
                    return [{str(symbol): value} for value in list(solution_set)], notes
                notes.append(f"SymPy returned solution set `{solution_set}`.")
                return [], notes

            solved = sp.solve(list(equations), list(symbols), dict=True)
            if solved == []:
                notes.append("SymPy returned no explicit solution for the supplied system.")
                return [], notes
            if isinstance(solved, dict):
                return [solved], notes
            if isinstance(solved, list):
                normalized: List[Dict[str, Any]] = []
                for item in solved:
                    if isinstance(item, dict):
                        normalized.append({str(key): value for key, value in item.items()})
                    elif len(symbols) == 1:
                        normalized.append({str(symbols[0]): item})
                return normalized, notes
            return [], notes
        except Exception as exc:  # pragma: no cover - depends on symbolic complexity
            return [], [f"SymPy solving failed: {exc}"]

    def _extract_llm_assignments(
        self,
        final_answer_text: str,
        full_text: str,
        variable_names: Sequence[str],
        local_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        search_text = final_answer_text or full_text
        normalized = self._normalize_math_text(search_text)
        assignments: Dict[str, Any] = {}

        for name in variable_names:
            pattern = rf"(?<![A-Za-z0-9_]){re.escape(name)}\s*=\s*([^\n,;]+)"
            match = re.search(pattern, normalized)
            if not match:
                continue
            rhs = match.group(1).strip().rstrip(".")
            rhs = rhs.split(" and ")[0].strip()
            rhs = rhs.split(" or ")[0].strip()
            try:
                assignments[name] = parse_expr(rhs, local_dict=local_dict, transformations=self._transformations)
            except Exception:
                continue

        return assignments

    def _compare_solution_claims(
        self,
        sympy_solutions: Sequence[Dict[str, Any]],
        llm_assignments: Dict[str, Any],
        final_answer_text: str,
    ) -> Tuple[str, List[str], List[str]]:
        if sp is None:
            return "unknown", [], ["SymPy is unavailable, so comparison could not run."]

        notes: List[str] = []
        mismatches: List[str] = []
        lowered_final_answer = str(final_answer_text or "").lower()

        if not sympy_solutions:
            if "no real solution" in lowered_final_answer or "no solution" in lowered_final_answer:
                notes.append("The LLM final answer states there is no real solution, which matches SymPy's result.")
                return "match", [], notes
            notes.append("SymPy did not produce an explicit solution to compare against the LLM final answer.")
            return "unknown", [], notes

        if not llm_assignments:
            notes.append("Could not extract variable assignments from the LLM final answer.")
            return "unknown", [], notes

        matched_count = 0
        for variable, llm_value in llm_assignments.items():
            matched = False
            for candidate in sympy_solutions:
                candidate_value = candidate.get(variable)
                if candidate_value is None:
                    continue
                try:
                    if sp.simplify(llm_value - candidate_value) == 0:
                        matched = True
                        matched_count += 1
                        break
                except Exception:
                    if str(llm_value) == str(candidate_value):
                        matched = True
                        matched_count += 1
                        break
            if not matched:
                mismatches.append(
                    f"LLM final answer claims {variable} = {llm_value}, but SymPy returned {self._format_solution_values(sympy_solutions, variable)}."
                )

        if mismatches:
            return "mismatch", mismatches, notes

        solved_variables = {key for solution in sympy_solutions for key in solution}
        missing_variables = sorted(name for name in solved_variables if name not in llm_assignments)
        if missing_variables:
            notes.append(
                "The LLM final answer matched part of the symbolic solution but omitted values for: "
                + ", ".join(missing_variables)
            )
            return "partial", [], notes

        if len(sympy_solutions) > 1:
            notes.append("The LLM final answer matches one of multiple symbolic solutions.")
        else:
            notes.append("LLM final answer matches symbolic solution.")
        return "match", [], notes

    def _extract_units_by_side(self, equation_text: str, units: Sequence[str]) -> Tuple[set[str], set[str]]:
        normalized = self._normalize_math_text(equation_text)
        if "=" not in normalized:
            mentioned = self._units_in_text(normalized, units)
            return mentioned, set()
        lhs, rhs = normalized.split("=", 1)
        return self._units_in_text(lhs, units), self._units_in_text(rhs, units)

    def _units_in_text(self, text: str, units: Sequence[str]) -> set[str]:
        lowered = text.lower()
        found = set()
        for unit in units:
            token = unit.lower()
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", lowered):
                found.add(unit)
        return found

    def _text_mentions_any_unit(self, text: str, units: Sequence[str]) -> bool:
        return bool(self._units_in_text(text, units))

    def _has_mixed_addition_units(self, equation_text: str, units: Sequence[str]) -> bool:
        normalized = self._normalize_math_text(equation_text)
        for segment in re.split(r"=", normalized):
            if "+" not in segment and "-" not in segment:
                continue
            mentioned = self._units_in_text(segment, units)
            if len(mentioned) > 1:
                return True
        return False

    def _build_summary(
        self,
        overall_status: str,
        sympy_report: Dict[str, Any],
        unit_report: Dict[str, Any],
        mismatches: Sequence[str],
    ) -> str:
        if overall_status == "pass":
            return "Verification passed. The independent checks did not find a mathematical mismatch."
        if overall_status == "fail":
            mismatch_text = mismatches[0] if mismatches else "The independent checks found a mismatch."
            return f"Verification failed. {mismatch_text}"
        sympy_status = sympy_report.get("status", "unknown")
        unit_status = unit_report.get("status", "unknown")
        return (
            "Verification completed with warnings. "
            f"SymPy status: {sympy_status}. Unit-check status: {unit_status}."
        )

    def _stringify_solution_dict(self, solution_dict: Dict[str, Any]) -> Dict[str, str]:
        return {str(key): str(value) for key, value in solution_dict.items()}

    def _format_solution_values(self, solutions: Sequence[Dict[str, Any]], variable: str) -> str:
        values = []
        for item in solutions:
            if variable in item:
                values.append(str(item[variable]))
        return ", ".join(values) if values else "no value"

    def _normalize_symbol_name(self, value: str) -> str:
        normalized = self._normalize_math_text(value)
        normalized = re.sub(r"[^A-Za-z0-9_]", "_", normalized)
        normalized = re.sub(r"_+", "_", normalized).strip("_")
        return normalized

    def _dedupe_strings(self, values: Sequence[str]) -> List[str]:
        seen = set()
        result: List[str] = []
        for value in values:
            item = str(value).strip()
            if not item or item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result

    def _skipped_report(self, name: str, message: str) -> Dict[str, Any]:
        return {
            "enabled": False,
            "status": "not_requested",
            "notes": [message],
            "name": name,
        }
