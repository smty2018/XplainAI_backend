"""Tests for the Manim docs and example scraper."""

import json
import tempfile
import unittest
from pathlib import Path

from scraper.manim_scraper import ManimKnowledgeScraper


class FakeResponse:
    """Minimal response object for scraper tests."""

    def __init__(self, *, text="", json_data=None, status_code=200):
        self.text = text
        self._json_data = json_data
        self.status_code = status_code
        self.encoding = "utf-8"

    def raise_for_status(self):
        """Mirror the part of requests.Response used by the scraper."""
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._json_data


class FakeSession:
    """Minimal session object that returns predefined responses."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.headers = {}

    def get(self, url, timeout=None):
        """Return the canned response so tests stay deterministic and offline."""
        if url not in self.mapping:
            raise RuntimeError(f"Unexpected URL: {url}")
        return self.mapping[url]


class ManimScraperTests(unittest.TestCase):
    """Check crawling, parsing, and gallery-splitting behavior."""

    def setUp(self):
        # Use an isolated workspace for each test because the scraper writes many files.
        self.temp_dir = Path(tempfile.mkdtemp(prefix="manim_scraper_test_"))

    def _make_scraper(self, mapping):
        """Build the scraper with a fake session and an isolated output directory."""
        # Use a fake session to test crawl behavior deterministically without network access.
        session = FakeSession(mapping)
        return ManimKnowledgeScraper(output_dir=self.temp_dir / "scraped", session=session)

    def test_normalize_docs_url_filters_external_and_assets(self):
        """Only documentation pages should stay in the crawl set."""
        scraper = self._make_scraper({})
        self.assertEqual(
            scraper._normalize_docs_url("/en/stable/tutorials_guides.html#section"),
            "https://docs.manim.community/en/stable/tutorials_guides.html",
        )
        self.assertIsNone(scraper._normalize_docs_url("https://example.com/outside"))
        self.assertIsNone(scraper._normalize_docs_url("https://docs.manim.community/en/stable/_static/file.js"))

    def test_extract_docs_links_keeps_related_pages(self):
        """Related docs links should be kept and unrelated links should be dropped."""
        scraper = self._make_scraper({})
        html = """
        <html><body>
          <a href="/en/stable/tutorials_guides.html">Guides</a>
          <a href="examples.html#intro">Examples</a>
          <a href="https://example.com/ignore">Outside</a>
          <a href="/en/stable/_static/logo.svg">Asset</a>
        </body></html>
        """
        title, markdown, links = scraper._extract_docs_page("https://docs.manim.community/en/stable/", html)
        self.assertTrue(title)
        self.assertIn("Source: https://docs.manim.community/en/stable/", markdown)
        self.assertEqual(
            links,
            [
                "https://docs.manim.community/en/stable/tutorials_guides.html",
                "https://docs.manim.community/en/stable/examples.html",
            ],
        )

    def test_docs_markdown_preserves_headings_lists_and_code(self):
        """Markdown conversion should keep headings, lists, and fenced code blocks."""
        scraper = self._make_scraper({})
        html = """
        <html>
          <body>
            <article class="bd-article">
              <h1>Voiceovers</h1>
              <p>Use narration with scenes.</p>
              <ul><li>Install package</li><li>Set service</li></ul>
              <pre class="language-python"><code>from manim import *\nprint("hi")</code></pre>
            </article>
          </body>
        </html>
        """
        _, markdown, _ = scraper._extract_docs_page("https://docs.manim.community/en/stable/voiceovers.html", html)
        self.assertIn("# Voiceovers", markdown)
        self.assertIn("- Install package", markdown)
        self.assertIn("```python", markdown)
        self.assertIn('print("hi")', markdown)

    def test_parse_github_tree_url(self):
        """GitHub tree URLs should split cleanly into repository parts."""
        scraper = self._make_scraper({})
        owner, repo, ref, path = scraper._parse_github_tree_url(
            "https://github.com/ManimCommunity/manim/tree/main/example_scenes"
        )
        self.assertEqual((owner, repo, ref, path), ("ManimCommunity", "manim", "main", "example_scenes"))

    def test_scrape_docs_writes_multiple_related_pages(self):
        """The docs scraper should save both the seed page and linked pages."""
        mapping = {
            "https://docs.manim.community/en/stable/": FakeResponse(
                text="""
                <html><body><article class="bd-article">
                <h1>Home</h1><p>Welcome.</p>
                <a href="/en/stable/tutorials_guides.html">Guides</a>
                </article></body></html>
                """
            ),
            "https://docs.manim.community/en/stable/tutorials_guides.html": FakeResponse(
                text="""
                <html><body><article class="bd-article">
                <h1>Tutorials</h1><p>Guides page.</p>
                </article></body></html>
                """
            ),
        }
        scraper = self._make_scraper(mapping)
        result = scraper.scrape_docs(
            [
                "https://docs.manim.community/en/stable/",
                "https://docs.manim.community/en/stable/tutorials_guides.html",
            ],
            max_pages=5,
        )
        self.assertEqual(result["pages_written"], 2)
        self.assertTrue((scraper.docs_dir / "index.md").exists())
        self.assertTrue((scraper.docs_dir / "tutorials_guides.md").exists())

    def test_scrape_github_examples_recurses_and_downloads_python(self):
        """GitHub example scraping should recurse into folders and save Python files."""
        mapping = {
            "https://api.github.com/repos/ManimCommunity/manim/contents/example_scenes?ref=main": FakeResponse(
                json_data=[
                    {"type": "dir", "path": "example_scenes/subdir"},
                    {
                        "type": "file",
                        "path": "example_scenes/hello.py",
                        "download_url": "https://raw.example/hello.py",
                    },
                    {
                        "type": "file",
                        "path": "example_scenes/notes.txt",
                        "download_url": "https://raw.example/notes.txt",
                    },
                ]
            ),
            "https://api.github.com/repos/ManimCommunity/manim/contents/example_scenes/subdir?ref=main": FakeResponse(
                json_data=[
                    {
                        "type": "file",
                        "path": "example_scenes/subdir/nested.py",
                        "download_url": "https://raw.example/nested.py",
                    }
                ]
            ),
            "https://raw.example/hello.py": FakeResponse(text="class Hello(Scene):\n    pass\n"),
            "https://raw.example/nested.py": FakeResponse(text="class Nested(Scene):\n    pass\n"),
        }
        scraper = self._make_scraper(mapping)
        result = scraper.scrape_github_examples(
            "https://github.com/ManimCommunity/manim/tree/main/example_scenes",
            max_files=10,
        )
        self.assertEqual(result["files_written"], 2)
        self.assertTrue((scraper.examples_dir / "hello.py").exists())
        self.assertTrue((scraper.examples_dir / "subdir" / "nested.py").exists())
        self.assertTrue((scraper.examples_dir / "README.md").exists())

    def test_scrape_all_writes_manifest(self):
        """The top-level scrape command should write a manifest file."""
        mapping = {
            "https://docs.manim.community/en/stable/": FakeResponse(
                text="""
                <html><body><article class="bd-article">
                <h1>Docs Home</h1><p>Intro.</p>
                </article></body></html>
                """
            ),
            "https://docs.manim.community/en/stable/tutorials_guides.html": FakeResponse(
                text="""
                <html><body><article class="bd-article">
                <h1>Tutorials</h1><p>Guide.</p>
                </article></body></html>
                """
            ),
            "https://docs.manim.community/en/stable/examples.html": FakeResponse(
                text="""
                <html><body><article class="bd-article">
                <h1>Examples</h1><p>Gallery.</p>
                </article></body></html>
                """
            ),
            "https://api.github.com/repos/ManimCommunity/manim/contents/example_scenes?ref=main": FakeResponse(
                json_data=[
                    {
                        "type": "file",
                        "path": "example_scenes/sample.py",
                        "download_url": "https://raw.example/sample.py",
                    }
                ]
            ),
            "https://raw.example/sample.py": FakeResponse(text="class Sample(Scene):\n    pass\n"),
        }
        scraper = self._make_scraper(mapping)
        manifest = scraper.scrape_all(max_doc_pages=10, max_example_files=10)
        manifest_path = scraper.meta_dir / "manifest.json"
        self.assertTrue(manifest_path.exists())
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["docs"]["pages_written"], manifest["docs"]["pages_written"])
        self.assertEqual(saved["github_examples"]["files_written"], 1)

    def test_split_example_gallery_markdown_extracts_clean_examples(self):
        """Gallery splitting should keep the clean Python block for each example."""
        scraper = self._make_scraper({})
        markdown = """
# Example Gallery

## Basic Concepts

Example: ManimCELogo

```
from
manim
import
*
```

```python
from manim import *

class ManimCELogo(Scene):
    def construct(self):
        self.add(Text("ok"))
```

References: Text

Example: BraceAnnotation

```python
from manim import *

class BraceAnnotation(Scene):
    def construct(self):
        self.add(Brace(Line(LEFT, RIGHT)))
```
"""
        # The first fenced block is intentionally junky; the parser should keep
        # the real Python example instead of the broken tokenized snippet.
        examples = scraper._split_example_gallery_markdown(markdown)
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0]["name"], "ManimCELogo")
        self.assertEqual(examples[0]["category"], "Basic Concepts")
        self.assertIn("class ManimCELogo(Scene):", examples[0]["code"])
        self.assertEqual(examples[0]["references"], "Text")
        self.assertIn("class BraceAnnotation(Scene):", examples[1]["code"])

    def test_write_gallery_assets_creates_md_and_py_files(self):
        """Gallery assets should be saved as both markdown and Python files."""
        scraper = self._make_scraper({})
        markdown = """
# Example Gallery

## Basic Concepts

Example: MovingAround

```python
from manim import *

class MovingAround(Scene):
    def construct(self):
        square = Square()
        self.add(square)
```
"""
        written = scraper._write_gallery_assets(
            markdown,
            source_url="https://docs.manim.community/en/stable/examples.html",
        )
        self.assertTrue(any(path.endswith("movingaround.md") for path in written))
        self.assertTrue(any(path.endswith("movingaround.py") for path in written))
        self.assertTrue((scraper.gallery_dir / "basic-concepts" / "movingaround.py").exists())
        self.assertTrue((scraper.gallery_dir / "README.md").exists())


if __name__ == "__main__":
    unittest.main()
