"""Chroma-backed retrieval helpers for Manim planning and code generation."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:  # pragma: no cover - dependency handled at runtime
    chromadb = None  # type: ignore[assignment]
    Settings = None  # type: ignore[assignment]


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|==|!=|<=|>=|[+\-*/=^()]")


@dataclass
class KnowledgeChunk:
    """Store one text chunk and the metadata needed to retrieve it later."""

    chunk_id: str
    document: str
    metadata: Dict[str, Any]


class SimpleHashEmbedding:
    """Create simple deterministic embeddings without an external service."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = max(64, int(dimension))

    def embed(self, text: str) -> List[float]:
        tokens = self._tokenize(text)
        vector = [0.0] * self.dimension
        if not tokens:
            return vector

        for token in tokens:
            self._accumulate(vector, token, 1.0)
        for first, second in zip(tokens, tokens[1:]):
            self._accumulate(vector, f"{first}__{second}", 0.35)

        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 1e-9:
            return vector
        return [value / norm for value in vector]

    def embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        return [self.embed(text) for text in texts]

    def _tokenize(self, text: str) -> List[str]:
        normalized = str(text or "").lower()
        return TOKEN_RE.findall(normalized)

    def _accumulate(self, vector: List[float], token: str, weight: float) -> None:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
        index = int.from_bytes(digest[:4], "big") % self.dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign * weight


class ManimKnowledgeBase:
    """Load, index, and query the local Manim knowledge base."""

    VERSION = "manim-rag-v1"

    def __init__(self, project_root: Path, config: Dict[str, Any]) -> None:
        # Keep all retrieval paths together so indexing and querying use the same local sources.
        # This KB is intentionally local-first: prompt assets + scraped docs/examples + a simple embedding path that works offline.
        self.project_root = Path(project_root)
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("manim_rag_enabled", True))
        self.prompt_assets_dir = self.project_root / self.config.get("prompt_assets_dir", "prompt_assets")
        self.seed_data_dir = self.project_root / self.config.get("manim_rag_seed_dir", "data/manim_kb")
        self.persist_dir = self.project_root / self.config.get("manim_rag_persist_dir", "cache/chroma_manim_kb")
        self.collection_name = str(self.config.get("manim_rag_collection_name", "manim_knowledge"))
        self.chunk_chars = int(self.config.get("manim_rag_chunk_chars", 1200))
        self.chunk_overlap = int(self.config.get("manim_rag_chunk_overlap", 180))
        self.embedding_dimension = int(self.config.get("manim_rag_embedding_dimension", 384))
        self.top_k_scene_planner = int(self.config.get("manim_rag_top_k_scene_planner", 4))
        self.top_k_manim_code = int(self.config.get("manim_rag_top_k_manim_code", 4))
        self._embedding = SimpleHashEmbedding(self.embedding_dimension)
        self._manifest_path = self.persist_dir / f"{self.collection_name}_manifest.json"
        self._client = None
        self._collection = None
        self._last_sync_stats: Dict[str, Any] = {}
        self.disabled_reason: Optional[str] = None

        if self.enabled and chromadb is not None:
            self._setup()
        else:
            if not self.enabled:
                self.disabled_reason = "manim_rag_enabled is false in the current config."
            elif chromadb is None:
                self.disabled_reason = (
                    "chromadb is not installed in the current Python interpreter "
                    f"({sys.executable})."
                )
            self.enabled = False


    def _setup(self) -> None:
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        settings = Settings(anonymized_telemetry=False, allow_reset=False) if Settings else None
        self._client = chromadb.PersistentClient(path=str(self.persist_dir), settings=settings)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.sync_index()

    def sync_index(self) -> Dict[str, Any]:
        # Repeated sync calls are safe during application startup and refresh cycles.
        if not self.enabled or self._collection is None:
            self._last_sync_stats = {
                "enabled": False,
                "status": "disabled",
                "reason": self.disabled_reason or "Knowledge base is disabled or unavailable.",
                "document_count": 0,
                "chunk_count": 0,
                "sources": [],
                "version": self.VERSION,
            }
            return dict(self._last_sync_stats)

        start = perf_counter()
        chunks = self._load_chunks()
        current_ids = [chunk.chunk_id for chunk in chunks]
        previous_ids = set(self._load_manifest_ids())
        stale_ids = sorted(previous_ids - set(current_ids))
        if stale_ids:
            self._collection.delete(ids=stale_ids)

        if chunks:
            documents = [chunk.document for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            embeddings = self._embedding.embed_many(documents)
            self._collection.upsert(
                ids=current_ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

        self._write_manifest_ids(current_ids)
        source_names = sorted({chunk.metadata.get("source_name", "") for chunk in chunks if chunk.metadata.get("source_name")})
        self._last_sync_stats = {
            "enabled": True,
            "status": "ready",
            "document_count": len(source_names),
            "chunk_count": len(chunks),
            "sources": source_names,
            "version": self.VERSION,
            "sync_seconds": round(perf_counter() - start, 3),
        }
        return dict(self._last_sync_stats)

    def get_status(self) -> Dict[str, Any]:
        if not self._last_sync_stats:
            return self.sync_index()
        return dict(self._last_sync_stats)

    def retrieve_for_scene_planner(
        self,
        parsed_input: Dict[str, Any],
        solution: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Scene-planner retrieval leans toward pedagogy and layout patterns more than raw API syntax.
        query = self._build_scene_planner_query(parsed_input, solution)
        return self.retrieve(query, top_k=self.top_k_scene_planner, stage="scene_planner")

    def retrieve_for_manim_code(
        self,
        parsed_input: Dict[str, Any],
        solution: Dict[str, Any],
        scene_planner: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Codegen retrieval is a little more concrete: examples, implementation snippets, and safe Manim usage patterns.
        query = self._build_manim_code_query(parsed_input, solution, scene_planner)
        return self.retrieve(query, top_k=self.top_k_manim_code, stage="manim_code")

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 4,
        stage: str = "general",
    ) -> Dict[str, Any]:
        # Retrieve a larger candidate set first, then rerank and deduplicate the final prompt context.
        if not self.enabled or self._collection is None:
            return {
                "enabled": False,
                "status": "disabled",
                "stage": stage,
                "query": query,
                "hits": [],
                "sync": dict(self._last_sync_stats),
            }

        safe_query = str(query or "").strip()
        if not safe_query:
            return {
                "enabled": True,
                "status": "empty_query",
                "stage": stage,
                "query": "",
                "hits": [],
                "sync": dict(self._last_sync_stats),
            }

        search_k = max(top_k * 5, top_k + 4)
        results = self._collection.query(
            query_embeddings=[self._embedding.embed(safe_query)],
            n_results=search_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        hits: List[Dict[str, Any]] = []
        for document, metadata, distance in zip(docs, metas, distances):
            metadata = dict(metadata or {})
            score = max(0.0, 1.0 - float(distance or 0.0))
            score += self._stage_bonus(stage, metadata)
            hits.append(
                {
                    "source_name": metadata.get("source_name", ""),
                    "source_path": metadata.get("source_path", ""),
                    "doc_type": metadata.get("doc_type", ""),
                    "stage_hint": metadata.get("stage_hint", ""),
                    "tags": metadata.get("tags", ""),
                    "score": round(score, 4),
                    "text": str(document or "").strip(),
                }
            )

        hits.sort(key=lambda item: item["score"], reverse=True)
        deduped = self._dedupe_hits(hits)
        selected = deduped[:top_k]
        if stage == "manim_code" and not any(hit.get("doc_type") == "example_code" for hit in selected):
            example_hit = next((hit for hit in deduped if hit.get("doc_type") == "example_code"), None)
            if example_hit is not None:
                if len(selected) >= top_k and selected:
                    selected = selected[:-1] + [example_hit]
                else:
                    selected.append(example_hit)
        return {
            "enabled": True,
            "status": "ready",
            "stage": stage,
            "query": safe_query,
            "hits": selected,
            "sync": dict(self._last_sync_stats),
        }

    def format_for_prompt(self, retrieval: Optional[Dict[str, Any]], heading: str) -> str:
        if not isinstance(retrieval, dict):
            return f"{heading}\n- Not available."
        if not retrieval.get("enabled"):
            return f"{heading}\n- Chroma knowledge base is disabled or unavailable."
        hits = retrieval.get("hits") or []
        if not hits:
            return f"{heading}\n- No relevant snippets were retrieved."

        lines = [heading]
        for index, hit in enumerate(hits, start=1):
            lines.append(
                f"[{index}] Source: {hit.get('source_name', 'unknown')} | type: {hit.get('doc_type', 'unknown')} | tags: {hit.get('tags', 'none')} | score: {hit.get('score', 0):.3f}"
            )
            lines.append(self._clip_text(str(hit.get("text", "")).strip(), limit=750))
        return "\n".join(lines)

    def _load_chunks(self) -> List[KnowledgeChunk]:
        chunks: List[KnowledgeChunk] = []
        for path in self._source_paths():
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="utf-8", errors="replace")
            for index, chunk_text in enumerate(self._chunk_text(text, suffix=path.suffix.lower())):
                if not chunk_text.strip():
                    continue
                chunk_id = self._chunk_id_for(path, index)
                metadata = {
                    "source_name": path.name,
                    "source_path": str(path),
                    "doc_type": self._doc_type_for(path),
                    "stage_hint": self._stage_hint_for(path),
                    "tags": ",".join(self._tags_for(path, chunk_text)),
                    "version": self.VERSION,
                }
                chunks.append(KnowledgeChunk(chunk_id=chunk_id, document=chunk_text.strip(), metadata=metadata))
        return chunks

    def _source_paths(self) -> List[Path]:
        candidates: List[Path] = []
        for base_dir in [self.seed_data_dir, self.prompt_assets_dir]:
            if not base_dir.exists():
                continue
            for path in sorted(base_dir.rglob("*")):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in {".md", ".txt", ".py"}:
                    continue
                candidates.append(path)
        return candidates

    def _chunk_text(self, text: str, *, suffix: str) -> List[str]:
        cleaned = str(text or "").strip()
        if not cleaned:
            return []

        delimiter = "\n\n" if suffix in {".md", ".txt"} else "\n\n"
        blocks = [block.strip() for block in cleaned.split(delimiter) if block.strip()]
        if not blocks:
            blocks = [cleaned]

        chunks: List[str] = []
        current = ""
        for block in blocks:
            candidate = f"{current}{delimiter if current else ''}{block}".strip()
            if current and len(candidate) > self.chunk_chars:
                chunks.append(current.strip())
                current = block
            else:
                current = candidate
        if current.strip():
            chunks.append(current.strip())

        final_chunks: List[str] = []
        for chunk in chunks:
            if len(chunk) <= self.chunk_chars:
                final_chunks.append(chunk)
                continue
            start = 0
            while start < len(chunk):
                end = min(len(chunk), start + self.chunk_chars)
                final_chunks.append(chunk[start:end].strip())
                if end >= len(chunk):
                    break
                start = max(end - self.chunk_overlap, start + 1)
        return final_chunks

    def _chunk_id_for(self, path: Path, index: int) -> str:
        raw = f"{self.VERSION}:{path.as_posix()}:{index}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _doc_type_for(self, path: Path) -> str:
        lower_name = path.name.lower()
        if path.suffix.lower() == ".py":
            return "example_code"
        if "template" in lower_name:
            return "template"
        if "guide" in lower_name or "rules" in lower_name:
            return "guide"
        return "reference"

    def _stage_hint_for(self, path: Path) -> str:
        lower_name = path.name.lower()
        if "scene_planner" in lower_name:
            return "scene_planner"
        if "few_shot" in lower_name or path.suffix.lower() == ".py":
            return "manim_code"
        return "both"

    def _tags_for(self, path: Path, chunk_text: str) -> List[str]:
        lower_name = path.name.lower()
        lower_text = chunk_text.lower()
        tags: List[str] = []
        for key in [
            "voiceover",
            "coquiservice",
            "next_section",
            "layout",
            "overlap",
            "axes",
            "plot",
            "graph",
            "mathex",
            "brace",
            "surroundingrectangle",
            "transform",
            "replacementtransform",
        ]:
            if key in lower_name or key in lower_text:
                tags.append(key)
        if "scene planner" in lower_text:
            tags.append("scene_planner")
        return sorted(set(tags)) or ["general"]

    def _build_scene_planner_query(self, parsed_input: Dict[str, Any], solution: Dict[str, Any]) -> str:
        asks = "; ".join(str(item) for item in parsed_input.get("asks", [])[:3])
        equations = " | ".join(
            str(item.get("raw") or item.get("latex") or "").strip()
            for item in parsed_input.get("equations", [])[:3]
            if isinstance(item, dict)
        )
        return "\n".join(
            [
                "Need Manim scene planning guidance for a narrated educational math animation.",
                f"Topic: {parsed_input.get('topic', '')}",
                f"Domain: {parsed_input.get('domain', '')}",
                f"Intent: {parsed_input.get('intent', '')}",
                f"Asks: {asks or 'not specified'}",
                f"Equations: {equations or 'not specified'}",
                f"Final answer: {solution.get('final_answer', '') or 'not specified'}",
                "Prioritize scene ordering, voiceover, next_section markers, box-based layout, overlap prevention, and graphing patterns when relevant.",
            ]
        )

    def _build_manim_code_query(
        self,
        parsed_input: Dict[str, Any],
        solution: Dict[str, Any],
        scene_planner: Dict[str, Any],
    ) -> str:
        planner_text = self._clip_text(str(scene_planner.get("text") or "").strip(), limit=1200)
        return "\n".join(
            [
                "Need executable Manim Community Edition code patterns and safe API usage.",
                f"Topic: {parsed_input.get('topic', '')}",
                f"Domain: {parsed_input.get('domain', '')}",
                f"Final answer: {solution.get('final_answer', '') or 'not specified'}",
                "Required techniques: VoiceoverScene, CoquiService, AudioSegment ffmpeg wiring, next_section markers, box-based layout, resolve_overlap, safe formula replacements, and example code snippets that demonstrate these patterns.",
                f"Scene planner summary: {planner_text or 'not specified'}",
            ]
        )

    def _stage_bonus(self, stage: str, metadata: Dict[str, Any]) -> float:
        stage_hint = str(metadata.get("stage_hint", "both"))
        doc_type = str(metadata.get("doc_type", ""))
        bonus = 0.0
        if stage_hint == stage:
            bonus += 0.08
        elif stage_hint == "both":
            bonus += 0.04

        if stage == "manim_code" and doc_type == "example_code":
            bonus += 0.20
        elif stage == "scene_planner" and doc_type in {"guide", "template", "reference"}:
            bonus += 0.04
        return bonus

    def _dedupe_hits(self, hits: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        deduped: List[Dict[str, Any]] = []
        for hit in hits:
            key = hit.get("source_name") or hit.get("source_path") or hit.get("text")
            if key in seen:
                continue
            seen.add(key)
            deduped.append(hit)
        return deduped

    def _clip_text(self, text: str, *, limit: int = 700) -> str:
        normalized = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3].rstrip() + "..."

    def _load_manifest_ids(self) -> List[str]:
        if not self._manifest_path.exists():
            return []
        try:
            payload = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        ids = payload.get("ids", [])
        return [str(item) for item in ids if str(item).strip()]

    def _write_manifest_ids(self, ids: Sequence[str]) -> None:
        payload = {"version": self.VERSION, "ids": list(ids)}
        self._manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
