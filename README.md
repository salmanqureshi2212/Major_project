# FastAPI Inference Backend (Brain + Heart)

Minimal FastAPI backend that loads two Keras `.h5` models once at startup and exposes two image inference endpoints.

## Setup

Put your model files here:

- `models/brain_model.h5`
- `models/heart_model.h5`

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## API

- `POST /api/brain_abnormalities` (multipart form field name: `file`)
- `POST /api/heart_abnormalities` (multipart form field name: `file`)

Each returns an annotated output image (`image/png`).

Example curl:

```bash
curl -X POST "http://localhost:8000/api/brain_abnormalities" \
  -F "file=@some_image.jpg" \
  --output out.png
```

