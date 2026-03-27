# XplainAI_backend

## API

This repo now includes a hosted-friendly FastAPI wrapper around the Replicate parser in [api.py](/Users/Sheetali/Documents/xplainai/api.py).

### Run locally

```bash
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

Then open:

- `http://localhost:8000/docs` for the interactive Swagger UI
- `http://localhost:8000/health` for a health check

### Endpoints

- `POST /parse/text`
- `POST /parse/image`
- `POST /parse/pdf`

### Required env var

Set one of these before starting:

- `REPLICATE_API_TOKEN`
- `REPLICATE_TOKEN`
- `tokenreplicate`
- `replicate`

### Quick test

```bash
curl -X POST "http://localhost:8000/parse/image" ^
  -F "file=@test_1.png"
```

```bash
curl -X POST "http://localhost:8000/parse/pdf" ^
  -F "file=@t2.pdf"
```

## Hosting

The easiest path is Render.

### Render setup

1. Push this repo to GitHub.
2. In Render, create a new Web Service from that repo.
3. Use:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn api:app --host 0.0.0.0 --port $PORT`
4. Add your `REPLICATE_API_TOKEN` as a secret environment variable.
5. Deploy, then share:
   - `https://your-service.onrender.com/docs`

Your friends can test directly from the Swagger UI without writing code.
