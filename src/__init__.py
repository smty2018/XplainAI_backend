"""xplainai source package."""

# Re-export the main entry points so the rest of the app can import from
# `src` without remembering the lower-level module layout.
from .parser import LocalParser
from .parser_replicate_vl2 import ReplicateDeepSeekVL2Parser
from .reasoner import MathematicalReasoner, SolutionOrchestrator, SolutionStyle

__all__ = [
    "LocalParser",
    "ReplicateDeepSeekVL2Parser",
    "MathematicalReasoner",
    "SolutionOrchestrator",
    "SolutionStyle",
]
