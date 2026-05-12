from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

app = FastAPI()

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


MODEL_NAME = "itz-dhanu/NeoPharmabert"

print("Loading model from HuggingFace...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
print(" Model loaded!")

class Request(BaseModel):
    abstract: str

@app.get("/")
def home():
    return FileResponse("index.html")

@app.post("/classify")
def classify(req: Request):
    inputs = tokenizer(
        req.abstract,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().tolist()

    preds = [p > 0.5 for p in probs]

    return {
        "patient":  bool(preds[0]),
        "reporter": bool(preds[1]),
        "drug":     bool(preds[2]),
        "event":    bool(preds[3]),
        "confidence": {
            "patient":  round(probs[0], 4),
            "reporter": round(probs[1], 4),
            "drug":     round(probs[2], 4),
            "event":    round(probs[3], 4),
        }
    }
