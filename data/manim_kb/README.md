# Manim Knowledge Base Seed

This directory seeds the local Chroma RAG collection used by the Streamlit and orchestration pipeline.

You can drop additional material here, including:

- community docs you have permission to store
- example scenes
- layout notes
- graphing examples
- voiceover patterns

The indexer also ingests files from `prompt_assets/`, so this folder is meant for extra Manim-focused knowledge that should sit beside the prompt templates.

Scraped community docs/examples can be generated via:

- `python -m scraper.manim_scraper`

They will be written under `data/manim_kb/scraped/`.

Supported file types:

- `.md`
- `.txt`
- `.py`
