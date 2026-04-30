"""Scrape Manim docs and example scenes into the local KB seed directory.

The scraper writes normalized markdown and code files so the RAG index can use
them later without needing live web access.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag


DEFAULT_DOC_SEEDS = [
    "https://docs.manim.community/en/stable/",
    "https://docs.manim.community/en/stable/tutorials_guides.html",
    "https://docs.manim.community/en/stable/examples.html",
]
DEFAULT_GITHUB_TREE_URL = "https://github.com/ManimCommunity/manim/tree/main/example_scenes"
DOCS_PREFIX = "https://docs.manim.community/en/stable/"
DEFAULT_OUTPUT_DIR = Path("data/manim_kb/scraped")


class ManimKnowledgeScraper:
    """Download Manim docs and examples into the local knowledge-base seed folder."""

    USER_AGENT = "xplainai-manim-kb-scraper/1.0"

    def __init__(
        self,
        output_dir: Path | str = DEFAULT_OUTPUT_DIR,
        *,
        session: Optional[requests.Session] = None,
        timeout_seconds: int = 30,
    ) -> None:
        # Store the output layout once so the rest of the scraper can write files consistently.
        self.output_dir = Path(output_dir)
        self.docs_dir = self.output_dir / "docs"
        self.examples_dir = self.output_dir / "examples" / "github"
        self.gallery_dir = self.output_dir / "examples" / "gallery"
        self.meta_dir = self.output_dir / "_meta"
        self.timeout_seconds = max(5, int(timeout_seconds))
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": self.USER_AGENT})

    def scrape_all(
        self,
        *,
        doc_seed_urls: Optional[Sequence[str]] = None,
        github_tree_url: str = DEFAULT_GITHUB_TREE_URL,
        max_doc_pages: int = 250,
        max_example_files: int = 200,
    ) -> Dict[str, Any]:
        """Run both scrape paths and write one manifest that records what was captured."""
        # Record a manifest for each scrape so the indexed inputs remain auditable.
        self._ensure_dirs()
        docs_result = self.scrape_docs(doc_seed_urls or DEFAULT_DOC_SEEDS, max_pages=max_doc_pages)
        github_result = self.scrape_github_examples(github_tree_url, max_files=max_example_files)
        manifest = {
            "docs": docs_result,
            "github_examples": github_result,
            "output_dir": str(self.output_dir),
        }
        manifest_path = self.meta_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return manifest

    def scrape_docs(self, seed_urls: Sequence[str], *, max_pages: int = 250) -> Dict[str, Any]:
        queue: deque[str] = deque()
        visited: set[str] = set()
        written: List[str] = []
        gallery_files: List[str] = []

        for url in seed_urls:
            normalized = self._normalize_docs_url(url)
            if normalized:
                queue.append(normalized)

        while queue and len(written) < max_pages:
            url = queue.popleft()
            if url in visited:
                continue
            visited.add(url)

            html = self._fetch_text(url)
            title, markdown, links = self._extract_docs_page(url, html)
            if markdown.strip():
                output_path = self._docs_output_path(url)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(markdown, encoding="utf-8")
                written.append(str(output_path))
                # Split the example gallery into per-example assets for more useful retrieval chunks.
                if url.rstrip("/").endswith("/examples.html"):
                    gallery_files.extend(self._write_gallery_assets(markdown, source_url=url))

            for link in links:
                normalized = self._normalize_docs_url(link)
                if normalized and normalized not in visited and normalized not in queue:
                    queue.append(normalized)

        return {
            "seed_urls": list(seed_urls),
            "pages_written": len(written),
            "files": written,
            "gallery_files_written": len(gallery_files),
            "gallery_files": gallery_files,
            "max_pages": max_pages,
        }

    def scrape_github_examples(self, tree_url: str, *, max_files: int = 200) -> Dict[str, Any]:
        owner, repo, ref, base_path = self._parse_github_tree_url(tree_url)
        pending: deque[str] = deque([base_path])
        written: List[str] = []

        # Traverse the GitHub contents API breadth-first to cover nested example directories.
        while pending and len(written) < max_files:
            current_path = pending.popleft()
            api_url = self._github_contents_api_url(owner, repo, current_path, ref)
            payload = self._fetch_json(api_url)
            if not isinstance(payload, list):
                raise RuntimeError(f"Unexpected GitHub contents payload for {api_url}")

            for item in payload:
                item_type = str(item.get("type", "")).strip().lower()
                item_path = str(item.get("path", "")).strip()
                if item_type == "dir" and item_path:
                    pending.append(item_path)
                    continue
                if item_type != "file":
                    continue
                if not item_path.endswith(".py"):
                    continue
                download_url = str(item.get("download_url", "")).strip()
                if not download_url:
                    continue
                code = self._fetch_text(download_url)
                relative = Path(item_path).relative_to(base_path)
                output_path = self.examples_dir / relative
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(code, encoding="utf-8")
                written.append(str(output_path))
                if len(written) >= max_files:
                    break

        index_path = self.examples_dir / "README.md"
        index_lines = [
            "# Scraped Manim Community Example Scenes",
            "",
            f"Source tree: {tree_url}",
            "",
            "Downloaded files:",
        ]
        index_lines.extend(f"- `{Path(path).name}`" for path in written)
        index_path.write_text("\n".join(index_lines).strip() + "\n", encoding="utf-8")

        return {
            "tree_url": tree_url,
            "files_written": len(written),
            "files": written,
            "max_files": max_files,
        }

    def _ensure_dirs(self) -> None:
        for path in [self.docs_dir, self.examples_dir, self.gallery_dir, self.meta_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def _fetch_text(self, url: str) -> str:
        response = self.session.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        response.encoding = response.encoding or "utf-8"
        return response.text

    def _fetch_json(self, url: str) -> Any:
        response = self.session.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def _normalize_docs_url(self, url: str) -> Optional[str]:
        if not url:
            return None
        resolved = urljoin(DOCS_PREFIX, str(url).strip())
        parsed = urlparse(resolved)
        if parsed.scheme not in {"http", "https"}:
            return None
        clean = parsed._replace(fragment="", query="")
        normalized = urlunparse(clean)
        if not normalized.startswith(DOCS_PREFIX):
            return None
        blocked_suffixes = (
            ".png",
            ".svg",
            ".jpg",
            ".jpeg",
            ".gif",
            ".css",
            ".js",
            ".inv",
            ".woff",
            ".woff2",
        )
        blocked_paths = (
            "/_sources/",
            "/_static/",
            "/genindex",
            "/py-modindex",
            "/search.html",
        )
        lower = normalized.lower()
        if any(lower.endswith(suffix) for suffix in blocked_suffixes):
            return None
        if any(segment in lower for segment in blocked_paths):
            return None
        return normalized

    def _extract_docs_page(self, url: str, html: str) -> Tuple[str, str, List[str]]:
        soup = BeautifulSoup(html, "html.parser")
        title = self._page_title(soup)
        container = (
            soup.select_one("article.bd-article")
            or soup.select_one("article[role='main']")
            or soup.select_one("main article")
            or soup.select_one("main")
            or soup.body
        )
        if container is None:
            return title, "", []

        # Remove navigation and presentation elements so the stored content stays article-focused.
        for tag in list(container.select("script, style, nav, aside, footer, button, form")):
            tag.decompose()

        markdown_lines = [f"# {title}", "", f"Source: {url}", ""]
        markdown_lines.extend(self._node_list_to_markdown(container))
        markdown = self._collapse_blank_lines(markdown_lines)
        links = self._extract_docs_links(soup, current_url=url)
        return title, markdown, links

    def _extract_docs_links(self, soup: BeautifulSoup, *, current_url: str) -> List[str]:
        links: List[str] = []
        for anchor in soup.select("a[href]"):
            href = str(anchor.get("href", "")).strip()
            if not href or href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
                continue
            resolved = urljoin(current_url, href)
            normalized = self._normalize_docs_url(resolved)
            if normalized:
                links.append(normalized)
        seen: set[str] = set()
        ordered: List[str] = []
        for link in links:
            if link in seen:
                continue
            seen.add(link)
            ordered.append(link)
        return ordered

    def _page_title(self, soup: BeautifulSoup) -> str:
        meta_title = soup.select_one("h1")
        if meta_title and meta_title.get_text(" ", strip=True):
            return meta_title.get_text(" ", strip=True)
        if soup.title and soup.title.get_text(" ", strip=True):
            return soup.title.get_text(" ", strip=True)
        return "Manim Documentation"

    def _node_list_to_markdown(self, container: Tag) -> List[str]:
        lines: List[str] = []
        for child in container.children:
            if isinstance(child, NavigableString):
                text = " ".join(str(child).split())
                if text:
                    lines.append(text)
                continue
            if isinstance(child, Tag):
                lines.extend(self._tag_to_markdown(child))
        return lines

    def _tag_to_markdown(self, tag: Tag, *, depth: int = 0) -> List[str]:
        name = tag.name.lower()
        if name in {"script", "style", "nav", "aside", "footer"}:
            return []
        if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(name[1])
            text = tag.get_text(" ", strip=True)
            return [f"{'#' * level} {text}", ""] if text else []
        if name == "p":
            text = tag.get_text(" ", strip=True)
            return [text, ""] if text else []
        if name in {"ul", "ol"}:
            lines: List[str] = []
            marker = "-" if name == "ul" else "1."
            for item in tag.find_all("li", recursive=False):
                item_text = item.get_text(" ", strip=True)
                if item_text:
                    indent = "  " * depth
                    lines.append(f"{indent}{marker} {item_text}")
            return lines + ([""] if lines else [])
        if name == "pre":
            code = tag.get_text("\n", strip=False).rstrip()
            if not code:
                return []
            language = self._infer_code_language(tag)
            return [f"```{language}".rstrip(), code, "```", ""]
        if name == "table":
            text = tag.get_text(" | ", strip=True)
            return [text, ""] if text else []
        if name in {"div", "section", "article", "main", "span"}:
            lines: List[str] = []
            for child in tag.children:
                if isinstance(child, NavigableString):
                    text = " ".join(str(child).split())
                    if text:
                        lines.append(text)
                    continue
                if isinstance(child, Tag):
                    lines.extend(self._tag_to_markdown(child, depth=depth))
            return lines
        text = tag.get_text(" ", strip=True)
        return [text, ""] if text else []

    def _infer_code_language(self, tag: Tag) -> str:
        classes = tag.get("class", []) or []
        class_text = " ".join(str(item) for item in classes)
        match = re.search(r"language-([a-z0-9_+-]+)", class_text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
        return ""

    def _collapse_blank_lines(self, lines: Iterable[str]) -> str:
        output: List[str] = []
        previous_blank = False
        for raw_line in lines:
            line = self._clean_text(str(raw_line).rstrip())
            is_blank = not line
            if is_blank and previous_blank:
                continue
            output.append(line)
            previous_blank = is_blank
        return "\n".join(output).strip() + "\n"

    def _clean_text(self, value: str) -> str:
        text = str(value or "")
        text = text.replace("Â¶", "").replace("¶", "").replace("\xa0", " ")
        return text

    def _docs_output_path(self, url: str) -> Path:
        parsed = urlparse(url)
        relative = parsed.path.replace("/en/stable/", "", 1).strip("/")
        if not relative:
            return self.docs_dir / "index.md"
        if relative.endswith(".html"):
            relative = relative[:-5] + ".md"
        elif relative.endswith("/"):
            relative = relative + "index.md"
        elif "." not in Path(relative).name:
            relative = relative + ".md"
        return self.docs_dir / Path(relative)

    def _write_gallery_assets(self, markdown: str, *, source_url: str) -> List[str]:
        examples = self._split_example_gallery_markdown(markdown)
        if not examples:
            return []

        written: List[str] = []
        index_lines = [
            "# Example Gallery Index",
            "",
            f"Source: {source_url}",
            "",
            "Extracted examples:",
        ]

        for example in examples:
            slug = self._slugify(example["name"])
            category_slug = self._slugify(example["category"]) or "general"
            base_dir = self.gallery_dir / category_slug
            base_dir.mkdir(parents=True, exist_ok=True)

            md_path = base_dir / f"{slug}.md"
            md_lines = [
                f"# Example: {example['name']}",
                "",
                f"Category: {example['category']}",
                f"Source: {source_url}",
                "",
            ]
            if example.get("references"):
                md_lines.append(f"References: {example['references']}")
                md_lines.append("")
            md_lines.append("```python")
            md_lines.append(example["code"].rstrip())
            md_lines.append("```")
            md_path.write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")
            written.append(str(md_path))

            py_path = base_dir / f"{slug}.py"
            py_path.write_text(example["code"].rstrip() + "\n", encoding="utf-8")
            written.append(str(py_path))

            index_lines.append(f"- `{example['category']}` / `{example['name']}`")

        index_path = self.gallery_dir / "README.md"
        index_path.write_text("\n".join(index_lines).strip() + "\n", encoding="utf-8")
        written.append(str(index_path))
        return written

    def _split_example_gallery_markdown(self, markdown: str) -> List[Dict[str, str]]:
        lines = [self._clean_text(line) for line in str(markdown or "").splitlines()]
        examples: List[Dict[str, str]] = []
        current_category = "Uncategorized"
        index = 0

        # At this point the docs page is already flattened to markdown, so we
        # rebuild structure using the gallery's headings and "Example:" labels.
        while index < len(lines):
            raw_line = lines[index].strip()
            if raw_line.startswith("## "):
                current_category = raw_line[3:].strip() or current_category
                index += 1
                continue
            if not raw_line.startswith("Example: "):
                index += 1
                continue

            name = raw_line.replace("Example:", "", 1).strip()
            section: List[str] = []
            index += 1
            while index < len(lines):
                probe = lines[index].strip()
                if probe.startswith("Example: ") or probe.startswith("## "):
                    break
                section.append(lines[index])
                index += 1

            code_blocks = self._extract_code_blocks(section)
            best_code = self._select_best_code_block(code_blocks)
            if not best_code:
                continue

            references = ""
            for line in section:
                stripped = line.strip()
                if stripped.startswith("References:"):
                    references = stripped.replace("References:", "", 1).strip()
                    break

            examples.append(
                {
                    "category": current_category,
                    "name": name,
                    "references": references,
                    "code": best_code,
                }
            )

        return examples

    def _extract_code_blocks(self, section_lines: Sequence[str]) -> List[str]:
        blocks: List[str] = []
        current: List[str] = []
        in_code = False
        for raw in section_lines:
            line = str(raw)
            if line.strip().startswith("```"):
                if in_code:
                    blocks.append("\n".join(current).strip())
                    current = []
                    in_code = False
                else:
                    in_code = True
                continue
            if in_code:
                current.append(line.rstrip("\n"))
        return [block for block in blocks if block.strip()]

    def _select_best_code_block(self, blocks: Sequence[str]) -> str:
        best = ""
        best_score = -1
        for block in blocks:
            score = 0
            if "class " in block and "Scene" in block:
                score += 5
            if "from manim import" in block or "from manim import *" in block:
                score += 3
            if "def construct" in block:
                score += 2
            score += min(len(block), 2000) / 1000.0
            if score > best_score:
                best = block
                best_score = score
        return best

    def _slugify(self, value: str) -> str:
        cleaned = self._clean_text(value).lower()
        cleaned = re.sub(r"[^a-z0-9]+", "-", cleaned).strip("-")
        return cleaned or "item"

    def _parse_github_tree_url(self, url: str) -> Tuple[str, str, str, str]:
        match = re.match(
            r"^https://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/tree/(?P<ref>[^/]+)/(?P<path>.+)$",
            str(url).strip(),
        )
        if not match:
            raise ValueError(f"Unsupported GitHub tree URL: {url}")
        return (
            match.group("owner"),
            match.group("repo"),
            match.group("ref"),
            match.group("path"),
        )

    def _github_contents_api_url(self, owner: str, repo: str, path: str, ref: str) -> str:
        return f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={ref}"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line arguments for the scraper script."""
    parser = argparse.ArgumentParser(description="Scrape Manim docs and examples into the local KB seed directory.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to write scraped content into.")
    parser.add_argument("--max-doc-pages", type=int, default=250, help="Maximum number of docs pages to crawl.")
    parser.add_argument("--max-example-files", type=int, default=200, help="Maximum number of example scene files to download.")
    return parser


def main() -> None:
    """Run the scraper and print the saved manifest."""
    args = build_arg_parser().parse_args()
    scraper = ManimKnowledgeScraper(output_dir=args.output_dir)
    manifest = scraper.scrape_all(
        max_doc_pages=args.max_doc_pages,
        max_example_files=args.max_example_files,
    )
    # Printing the manifest keeps ad-hoc local runs easy to sanity-check.
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
