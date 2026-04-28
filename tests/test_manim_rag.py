"""Tests for the local Chroma-backed Manim knowledge base."""

import importlib.util
import tempfile
import sys
import unittest
from pathlib import Path
from chromadb.api.client import SharedSystemClient


def load_manim_rag_module():
    """Import the Manim RAG module directly from the repository path."""
    # The tests import the module by path so they stay stable even if the repo
    # isn't installed as a package in the active environment.
    root = Path(__file__).resolve().parents[1]
    module_path = root / "src" / "manim_rag.py"
    spec = importlib.util.spec_from_file_location("manim_rag_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestManimKnowledgeBase(unittest.TestCase):
    """Check that indexing and retrieval work against the local seed corpus."""

    @classmethod
    def setUpClass(cls):
        # Use a temporary Chroma directory so local developer data never affects these assertions.
        cls.root = Path(__file__).resolve().parents[1]
        cls.module = load_manim_rag_module()
        cls.persist_dir = Path(tempfile.mkdtemp(prefix="test_chroma_manim_kb_"))
        # Each test class gets its own throwaway Chroma dir so retrieval tests
        # don't accidentally depend on a developer's existing local index.
        cls.kb = cls.module.ManimKnowledgeBase(
            cls.root,
            {
                "prompt_assets_dir": "prompt_assets",
                "manim_rag_enabled": True,
                "manim_rag_seed_dir": "data/manim_kb",
                "manim_rag_persist_dir": str(cls.persist_dir),
                "manim_rag_collection_name": "test_manim_knowledge",
                "manim_rag_top_k_scene_planner": 4,
                "manim_rag_top_k_manim_code": 4,
                "manim_rag_chunk_chars": 900,
                "manim_rag_chunk_overlap": 120,
                "manim_rag_embedding_dimension": 256,
            },
        )

    @classmethod
    def tearDownClass(cls):
        try:
            SharedSystemClient.clear_system_cache()
        except Exception:
            pass
        cls.kb = None

    def test_index_has_seed_documents(self):
        """The seed corpus should produce a non-empty index."""
        # This is the first guardrail: if the seed corpus never indexed, the
        # retrieval quality tests below are not worth trusting.
        status = self.kb.get_status()
        self.assertTrue(status["enabled"])
        self.assertGreater(status["document_count"], 0)
        self.assertGreater(status["chunk_count"], 0)

    def test_voiceover_query_returns_narration_guidance(self):
        """Narration queries should pull voiceover-related material."""
        result = self.kb.retrieve(
            "Need VoiceoverScene CoquiService AudioSegment next_section narration guidance",
            top_k=3,
            stage="manim_code",
        )
        combined = " ".join(hit["text"] for hit in result["hits"]).lower()
        self.assertIn("voiceoverscene", combined)
        self.assertTrue("coquiservice" in combined or "audiosegment" in combined or "next_section" in combined)

    def test_layout_query_returns_overlap_guidance(self):
        """Layout queries should pull overlap-avoidance guidance."""
        result = self.kb.retrieve(
            "Need box layout place_in_box resolve_overlap no overlap guidance",
            top_k=3,
            stage="manim_code",
        )
        combined = " ".join(hit["text"] for hit in result["hits"]).lower()
        self.assertTrue("resolve_overlap" in combined or "place_in_box" in combined)

    def test_graph_query_returns_axes_patterns(self):
        """Graph queries should return axes or plotting references."""
        result = self.kb.retrieve(
            "Need axes plot graph vertex dashed lines annotations for a quadratic scene",
            top_k=3,
            stage="scene_planner",
        )
        combined = " ".join(hit["text"] for hit in result["hits"]).lower()
        self.assertTrue("axes" in combined or "plot" in combined or "graph" in combined)

    def test_scene_planner_retrieval_uses_problem_context(self):
        """Scene-planner retrieval should include the parsed problem context in its query."""
        parsed_input = {
            "topic": "quadratic equations",
            "domain": "mathematics",
            "intent": "concept_explanation",
            "asks": ["solve x^2 + 5x + 10 = 0"],
            "equations": [{"raw": "x^2 + 5x + 10 = 0"}],
        }
        solution = {
            "final_answer": "x = (-5 +- i*sqrt(15))/2",
            "verification": "Verification completed with warnings.",
        }
        result = self.kb.retrieve_for_scene_planner(parsed_input, solution)
        self.assertEqual(result["stage"], "scene_planner")
        self.assertGreater(len(result["hits"]), 0)
        self.assertIn("quadratic equations", result["query"].lower())

    def test_manim_code_retrieval_prefers_code_or_examples(self):
        """Code-generation retrieval should favor concrete code or examples."""
        # The codegen stage should lean toward concrete example snippets, not
        # just prose guidance, because that tends to produce more runnable output.
        parsed_input = {
            "topic": "quadratic equations",
            "domain": "mathematics",
            "equations": [{"raw": "x^2 + 5x + 10 = 0"}],
        }
        solution = {"final_answer": "x = (-5 +- i*sqrt(15))/2"}
        scene_planner = {
            "text": "Scene Planner with voiceover, graph, coefficient labels, next_section markers, and final boxed answers."
        }
        result = self.kb.retrieve_for_manim_code(parsed_input, solution, scene_planner)
        self.assertEqual(result["stage"], "manim_code")
        self.assertGreater(len(result["hits"]), 0)
        self.assertTrue(
            any(hit["doc_type"] == "example_code" for hit in result["hits"])
            or any("few_shot" in hit["source_name"].lower() for hit in result["hits"])
        )


if __name__ == "__main__":
    unittest.main()
