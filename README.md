# XplainAI

XplainAI turns text, images, and PDFs into educational animated explanations.
It parses the input, reasons through the solution, retrieves relevant Manim guidance through RAG, creates a scene plan, generates Manim code, and can render a narrated video locally. The generated Manim code can also be edited in place, so you can rerender only the changed animation code instead of regenerating the whole pipeline every time.

## What XplainAI does

- accepts plain text, screenshots, and PDFs as input
- extracts the important mathematical or technical content
- generates a solution and verification trace
- uses a local Chroma-based RAG layer for Manim examples, docs, and layout guidance
- produces a scene planner and executable Manim code
- optionally renders the final animation locally with narration
- lets you edit the generated code directly and rerender from that edited code

## Main parts of the repo

- `streamlit_app.py`
  Streamlit frontend for the full workflow.
- `api.py`
  FastAPI wrapper for the parser endpoints.
- `src/`
  Core pipeline logic: parser, reasoner, verification, RAG, and compile repair.
- `prompt_assets/`
  Prompt templates and layout guidance used by the pipeline.
- `data/manim_kb/`
  Local Manim knowledge base seed files used by RAG.

<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/df6ea86c-a698-4f23-96bd-fcf122e3c2f8" />


## Recommended environment

The current full render pipeline is most stable on Windows with Python 3.10.

Before installing, make sure you have:

- Python `3.10`
- Git
- a Replicate API token (for the Replicate-hosted DeepSeek-VL2 parser)
- a DeepSeek API key (for reasoning, scene planning, Manim code generation, and repair)
- MiKTeX installed if you want LaTeX-heavy Manim scenes to render reliably

## Step-by-step installation

### 1. Clone the repository

```powershell
git clone <your-repo-url>
cd xplainai
```

### 2. Create a virtual environment

```powershell
py -3.10 -m venv venv
```

### 3. Activate the virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation on your machine, you can still run everything with the venv Python directly:

```powershell
.\venv\Scripts\python.exe --version
```

### 4. Upgrade pip

```powershell
.\venv\Scripts\python.exe -m pip install --upgrade pip
```

### 5. Install Python dependencies

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 6. Create a `.env` file

Create a `.env` file in the project root with at least:

```env
REPLICATE_API_TOKEN=your_replicate_token_here
DEEPSEEK_API_KEY=your_deepseek_key_here
```

The parser also accepts these alternate Replicate variable names if needed:

- `REPLICATE_TOKEN`
- `tokenreplicate`
- `replicate`

The reasoner also checks these alternate DeepSeek variable names:

- `deepseek`
- `deepdeek`

### 7. Start the Streamlit app

Always prefer running Streamlit through the venv Python so the app uses the same interpreter as your installed dependencies:

```powershell
.\venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Then open the local URL shown in the terminal, usually:

- [http://localhost:8501](http://localhost:8501)

## First run expectations

On the first run, a few things can take extra time:

- the local Chroma index may sync the Manim knowledge base
- the first parser or reasoning call depends on external APIs
- the first narrated render may download or warm up TTS assets
- the first full Manim render can take noticeably longer than later runs

This is normal for the initial setup.

## How to use XplainAI

1. Open the Streamlit app.
2. Choose whether your input is text, image, or PDF.
3. Upload the file or paste the text.
4. Run the pipeline.
5. Review:
   - parsed JSON
   - solution
   - verification
   - RAG context
   - scene planner
   - generated Manim code
6. If needed, edit the generated Manim code directly in the app and rerender without regenerating the entire pipeline.

## Local RAG setup

XplainAI uses a local Chroma-backed knowledge base for Manim guidance.

It indexes the project’s prompt assets and Manim knowledge-base seed files to retrieve:

- Manim examples
- layout guidance
- scene-planning patterns
- safe code-generation references

If you add more `.md`, `.txt`, or `.py` knowledge-base seed files, the app will pick them up on the next sync.

## Running the parser API only

If you only want the parser service and not the full Streamlit workflow:

```powershell
.\venv\Scripts\python.exe -m uvicorn api:app --host 0.0.0.0 --port 8000
```

Useful endpoints:

- [http://localhost:8000/docs](http://localhost:8000/docs)
- [http://localhost:8000/health](http://localhost:8000/health)

Available parser routes:

- `POST /parse/text`
- `POST /parse/image`
- `POST /parse/pdf`

## Troubleshooting

### Streamlit says RAG is disabled

Usually this means the app is running in the wrong Python interpreter or `chromadb` is missing from that interpreter.

Use:

```powershell
.\venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

not:

```powershell
streamlit run streamlit_app.py
```

### Manim render fails on LaTeX

Install or update MiKTeX, then try again. Some generated scenes rely on LaTeX rendering for formulas and labels.

### PowerShell blocks `Activate.ps1`

You can skip activation and use the venv Python directly in every command:

```powershell
.\venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

### The first render is slow

That is expected. The app may be warming up Manim, ffmpeg, TTS, or local caches.

## License

See `LICENSE`.
