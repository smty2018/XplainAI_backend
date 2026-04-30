"""Helpers for repairing generated Manim scripts after compile failures.

The repair flow is intentionally staged:
1. find the broken region,
2. try a small local fix,
3. continue the file when it looks truncated,
4. only then ask for a full rewrite.
"""

from __future__ import annotations

import ast
import re
import textwrap
from typing import Any, Callable, Dict, List, Optional, Tuple


def parse_compile_error_line(compile_error: str) -> Optional[int]:
    """Extract the failing line number from py_compile output when possible."""
    match = re.search(r"line\s+(\d+)", str(compile_error or ""))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def python_code_compiles_in_memory(code: str) -> bool:
    """Return True when Python can parse the candidate code string."""
    try:
        ast.parse(str(code or ""))
    except SyntaxError:
        return False
    return True


def count_leading_spaces(line: str) -> int:
    """Return the indentation width for one source line."""
    return len(line) - len(line.lstrip(" "))


def compile_error_looks_truncated(compile_error: str, code: str, line_number: Optional[int]) -> bool:
    """Heuristically detect truncated code paths that should prefer continuation repair."""
    error_text = str(compile_error or "").lower()
    lines = code.splitlines()
    if not lines:
        return True
    last_line_number = len(lines)
    tail = lines[-1].rstrip()
    if line_number is not None and line_number >= last_line_number:
        if any(
            phrase in error_text
            for phrase in [
                "was never closed",
                "unterminated string literal",
                "unexpected eof",
                "eof while scanning",
            ]
        ):
            return True
    return tail.endswith(("=", "(", "[", "{", ",", "\\", "font", "MathTex", "Text"))


def find_scope_start_from_valid_prefix(lines: List[str], line_number: int) -> Optional[int]:
    """Use the largest valid AST prefix to recover the nearest class/function boundary."""
    prefix_end = min(max(line_number - 1, 1), len(lines))
    for end in range(prefix_end, 0, -1):
        snippet = "\n".join(lines[:end])
        try:
            tree = ast.parse(snippet)
        except SyntaxError:
            continue

        last_scope_start: Optional[int] = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                node_line = int(getattr(node, "lineno", 1))
                if last_scope_start is None or node_line >= last_scope_start:
                    last_scope_start = node_line
        return last_scope_start
    return None


def find_scope_start_by_indentation(lines: List[str], line_number: int) -> int:
    """Fallback scope finder using class/def boundaries and indentation."""
    error_index = max(0, min(line_number - 1, len(lines) - 1))
    for index in range(error_index, -1, -1):
        stripped = lines[index].lstrip()
        if stripped.startswith(("def ", "class ")):
            return index + 1
    return 1


def extract_block_repair_region(code: str, compile_error: str) -> Dict[str, Any]:
    """Choose a focused block around the failing line for local repair."""
    lines = code.splitlines()
    if not lines:
        return {
            "line_number": 1,
            "block_start_line": 1,
            "block_end_line": 1,
            "block_text": "",
            "context_before": "",
            "context_after": "",
            "block_indent": 0,
            "looks_truncated": True,
        }

    line_number = parse_compile_error_line(compile_error) or len(lines)
    line_number = max(1, min(line_number, len(lines)))
    error_index = line_number - 1
    looks_truncated = compile_error_looks_truncated(compile_error, code, line_number)

    scope_start_line = (
        find_scope_start_from_valid_prefix(lines, line_number)
        or find_scope_start_by_indentation(lines, line_number)
    )
    scope_start_index = max(0, scope_start_line - 1)

    current_indent = count_leading_spaces(lines[error_index]) if lines[error_index].strip() else 0
    block_start_index = error_index
    for index in range(error_index, scope_start_index - 1, -1):
        stripped = lines[index].strip()
        if not stripped:
            if index < error_index:
                block_start_index = index + 1
                break
            continue
        indent = count_leading_spaces(lines[index])
        if index != error_index and stripped.startswith(("def ", "class ")):
            block_start_index = index
            break
        if index != error_index and stripped.endswith(":") and indent < current_indent:
            block_start_index = index + 1
            break
        block_start_index = index

    block_end_index = len(lines)
    if not looks_truncated:
        base_indent = count_leading_spaces(lines[block_start_index]) if lines[block_start_index].strip() else current_indent
        for index in range(error_index + 1, len(lines)):
            stripped = lines[index].strip()
            if not stripped:
                block_end_index = index
                break
            indent = count_leading_spaces(lines[index])
            if indent < base_indent and not stripped.startswith(("#", ")", "]", "}")):
                block_end_index = index
                break

    context_before = "\n".join(lines[max(scope_start_index, block_start_index - 10):block_start_index])
    context_after = "\n".join(lines[block_end_index:min(len(lines), block_end_index + 10)])
    block_lines = lines[block_start_index:block_end_index]
    block_indent = 0
    for line in block_lines:
        if line.strip():
            block_indent = count_leading_spaces(line)
            break

    return {
        "line_number": line_number,
        "block_start_line": block_start_index + 1,
        "block_end_line": block_end_index,
        "block_text": "\n".join(block_lines),
        "context_before": context_before,
        "context_after": context_after,
        "block_indent": block_indent,
        "looks_truncated": looks_truncated,
    }


def normalize_repaired_block(block_text: str, block_indent: int) -> str:
    """Re-indent a replacement block so it fits back into the original script."""
    cleaned = str(block_text or "").replace("\r\n", "\n").strip("\n")
    if not cleaned:
        return ""
    cleaned = textwrap.dedent(cleaned)
    return textwrap.indent(cleaned, " " * max(0, int(block_indent)))


def splice_code_block(code: str, replacement_block: str, start_line: int, end_line: int) -> str:
    """Replace an inclusive line range in a script with a repaired block."""
    lines = code.splitlines()
    prefix = lines[: max(0, start_line - 1)]
    suffix = lines[max(0, end_line):]
    replacement_lines = replacement_block.splitlines() if replacement_block else []
    return "\n".join(prefix + replacement_lines + suffix)


class ManimCompileRepairEngine:
    """Coordinate block, continuation, and full-file compile repair strategies."""

    def __init__(self, sanitize_code: Callable[[str], str]):
        self.sanitize_code = sanitize_code

    def attempt_repair(
        self,
        code: str,
        compile_error: str,
        *,
        repair_block: Optional[Callable[..., Dict[str, Any]]] = None,
        continue_from_tail: Optional[Callable[..., Dict[str, Any]]] = None,
        repair_full_script: Optional[Callable[..., Dict[str, Any]]] = None,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Try focused repair paths first and fall back to full-file repair last."""
        sanitized_code = self.sanitize_code(code)
        region = extract_block_repair_region(sanitized_code, compile_error)

        if repair_block and region.get("block_text"):
            # A local block repair keeps most of the generated file untouched.
            repaired_block = repair_block(
                broken_block=str(region.get("block_text") or ""),
                compile_error=compile_error,
                block_start_line=int(region.get("block_start_line") or 1),
                block_end_line=int(region.get("block_end_line") or 1),
                context_before=str(region.get("context_before") or ""),
                context_after=str(region.get("context_after") or ""),
            )
            normalized_block = normalize_repaired_block(
                (repaired_block or {}).get("text", ""),
                int(region.get("block_indent") or 0),
            )
            if normalized_block:
                candidate = splice_code_block(
                    sanitized_code,
                    normalized_block,
                    int(region.get("block_start_line") or 1),
                    int(region.get("block_end_line") or 1),
                )
                candidate = self.sanitize_code(candidate)
                if python_code_compiles_in_memory(candidate):
                    return candidate, {
                        "mode": "block",
                        "region": region,
                        "repair_metadata": (repaired_block or {}).get("_metadata", {}),
                    }

        if bool(region.get("looks_truncated")) and continue_from_tail:
            # Continuation is cheaper than a full rewrite when the file simply stops mid-call.
            lines = sanitized_code.splitlines()
            last_intact_line = max(1, int(region.get("block_start_line") or len(lines)) - 1)
            prefix_lines = lines[:last_intact_line]
            prefix_context = "\n".join(prefix_lines[max(0, len(prefix_lines) - 30):])
            truncated_tail = "\n".join(lines[last_intact_line:])
            tail_indent = int(region.get("block_indent") or 0)
            continuation = continue_from_tail(
                prefix_context=prefix_context,
                truncated_tail=truncated_tail,
                compile_error=compile_error,
                last_intact_line=last_intact_line,
            )
            continuation_text = normalize_repaired_block(
                (continuation or {}).get("text", ""),
                tail_indent,
            )
            if continuation_text:
                candidate = "\n".join(prefix_lines + continuation_text.splitlines())
                candidate = self.sanitize_code(candidate)
                if python_code_compiles_in_memory(candidate):
                    return candidate, {
                        "mode": "continuation",
                        "region": region,
                        "repair_metadata": (continuation or {}).get("_metadata", {}),
                    }

        if not repair_full_script:
            return None, {}

        # Full-file repair is the last resort because it is the most expensive and least predictable.
        repaired = repair_full_script(
            generated_code=sanitized_code,
            compile_error=compile_error,
        )
        repaired_code = self.sanitize_code((repaired or {}).get("text", ""))
        if not repaired_code:
            return None, {}
        if not python_code_compiles_in_memory(repaired_code):
            return None, {
                "mode": "full_script",
                "region": region,
                "repair_metadata": (repaired or {}).get("_metadata", {}),
                "accepted": False,
            }
        return repaired_code, {
            "mode": "full_script",
            "region": region,
            "repair_metadata": (repaired or {}).get("_metadata", {}),
            "accepted": True,
        }
