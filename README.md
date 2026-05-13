# [Console Controller Classifier API](https://huggingface.co/Medo-Id)

A simple FastAPI server that uses a FastAI image classification model to identify console controllers from base64-encoded image uploads.

> The model and server are hosted on Hugging Face. Test it directly: `https://medo-id-testing.hf.space/classify`

## Features

- FastAPI-based HTTP API
- FastAI model loaded from `export.pkl`
- `GET /` health check endpoint
- `POST /classify` endpoint for image classification

## Requirements

- Python 3.13+
- `fastapi`
- `fastai`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Run locally

```bash
uvicorn main:app --reload
```

## API

### Health check

```http
GET /
```

Response:

```json
{ "status": "ok" }
```

### Classify image

```http
POST /classify
Content-Type: application/json

{
  "data": ["data:image/png;base64,<BASE64_IMAGE_DATA>"]
}
```

Response:

```json
{
  "controller_name": 0.95,
  "other_label": 0.05
}
```

The API returns a probability score for each model label.

## Notes

- Ensure `export.pkl` is present in the project root.
- The classifier expects a base64-encoded image string.
