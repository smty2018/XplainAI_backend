"""FastAPI wrapper for the Replicate-backed parser."""

from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError

from src.parser_replicate_vl2 import ReplicateAPIError, ReplicateDeepSeekVL2Parser


class TextParseRequest(BaseModel):
    text: str


def _safe_filename_suffix(filename: Optional[str], fallback: str) -> str:
    name = (filename or "").strip()
    if not name:
        return fallback
    suffix = Path(name).suffix.lower()
    return suffix or fallback


@lru_cache(maxsize=1)
def get_parser() -> ReplicateDeepSeekVL2Parser:
    return ReplicateDeepSeekVL2Parser()


app = FastAPI(
    title="XplainAI Replicate Parser API",
    version="1.0.0",
    description=(
        "Upload images or PDFs and parse them with the Replicate-hosted DeepSeek-VL2 parser."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "XplainAI Replicate Parser API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": [
            "/parse/text",
            "/parse/image",
            "/parse/pdf",
        ],
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/parse/text")
def parse_text(payload: TextParseRequest) -> Dict[str, Any]:
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="`text` cannot be empty.")

    try:
        return get_parser().parse_text(text)
    except ReplicateAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected parser failure: {exc}") from exc


@app.post("/parse/image")
def parse_image(
    file: UploadFile = File(...),
    prompt: str = Form(""),
) -> Dict[str, Any]:
    try:
        image = Image.open(file.file).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc

    try:
        result = get_parser().parse_image(image, prompt_text=prompt or None)
        result["_uploaded_filename"] = file.filename or "image"
        return result
    except ReplicateAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected parser failure: {exc}") from exc


@app.post("/parse/pdf")
def parse_pdf(
    file: UploadFile = File(...),
    prompt: str = Form(""),
) -> Dict[str, Any]:
    suffix = _safe_filename_suffix(file.filename, ".pdf")
    if suffix != ".pdf":
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as handle:
            temp_path = Path(handle.name)
            handle.write(file.file.read())

        result = get_parser().parse_pdf(str(temp_path), prompt_text=prompt or None)
        result["_uploaded_filename"] = file.filename or "document.pdf"
        return result
    except ReplicateAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected parser failure: {exc}") from exc
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)
