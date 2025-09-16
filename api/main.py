from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Pre-load all trained models
MODELS = {
    "biobert": "./coaid_model_biobert",
    "minilm": "./coaid_model_minilm",
    "biobert+arg": "./coaid_model_biobert_arg"
}

# Load tokenizers & models once (efficient)
tokenizers = {name: AutoTokenizer.from_pretrained(path) for name, path in MODELS.items()}
models = {name: AutoModelForSequenceClassification.from_pretrained(path) for name, path in MODELS.items()}

# FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class ClaimRequest(BaseModel):
    claim: str
    model_name: str = "biobert"  # default model

# -------------------------------
# Endpoint: Analyze with one model
# -------------------------------
@app.post("/analyze")
async def analyze(request: ClaimRequest):
    model_key = request.model_name.lower()
    if model_key not in MODELS:
        return {"error": f"Invalid model_name. Choose from {list(MODELS.keys())}"}

    tokenizer = tokenizers[model_key]
    model = models[model_key]

    inputs = tokenizer(request.claim, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred].item()

    label_map = {0: "Fake Information", 1: "True Information"}

    return {
        "claim": request.claim,
        "model_used": model_key,
        "prediction": label_map[pred],
        "confidence": confidence,
    }

# -------------------------------
# Endpoint: Compare all models
# -------------------------------
@app.post("/compare")
async def compare(request: ClaimRequest):
    label_map = {0: "Fake Information", 1: "True Information"}
    results = {}

    for model_key in MODELS:
        tokenizer = tokenizers[model_key]
        model = models[model_key]

        inputs = tokenizer(request.claim, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred].item()

        results[model_key] = {
            "prediction": label_map[pred],
            "confidence": confidence,
        }

    return {
        "claim": request.claim,
        "results": results,
    }
