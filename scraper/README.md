# Manim KB Scraper

This folder contains the standalone scraper that pulls Manim community docs and
example scene code into the local knowledge-base seed directory.

Default targets:

- `https://docs.manim.community/en/stable/`
- `https://docs.manim.community/en/stable/tutorials_guides.html`
- `https://docs.manim.community/en/stable/examples.html`
- `https://github.com/ManimCommunity/manim/tree/main/example_scenes`

The scraper writes output into:

- `data/manim_kb/scraped/docs/`
- `data/manim_kb/scraped/examples/github/`
- `data/manim_kb/scraped/_meta/manifest.json`

Because the Chroma loader already ingests `.md` and `.py` files recursively from
`data/manim_kb/`, newly scraped content is automatically available to the RAG
layer on the next sync/run.

## Run

```bash
python -m scraper.manim_scraper
```

Optional flags:

```bash
python -m scraper.manim_scraper --max-doc-pages 40 --max-example-files 25
python -m scraper.manim_scraper --output-dir data/manim_kb/scraped
```

## Notes

- The docs crawler stays within `https://docs.manim.community/en/stable/`.
- The GitHub example scraper uses the GitHub contents API to walk
  `example_scenes/` recursively and download `.py` files.
- Network access is required for a real scrape.
