from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from fastai.vision.all import PILImage, load_learner

import base64

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_methods=["*"],
  allow_headers=["*"],
)

learn = load_learner('export.pkl')

@app.get("/")
def root():
  return { "status": "ok" }

@app.post("/classify")
async def classify(request: Request):
    body = await request.json()
    data = body["data"]
    if isinstance(data, list):
        data = data[0]

    image_bytes = base64.b64decode(data.split(",")[1])
    pil_image = PILImage.create(image_bytes)  # ← fix here

    pred, idx, probs = learn.predict(pil_image)
    return {
        str(label): float(prob)
        for label, prob in zip(learn.dls.vocab, probs)
    }