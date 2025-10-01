# api/main_codespaces.py - Optimized for GitHub Codespaces
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy
from typing import Optional, List
import logging
import asyncio

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HealthMisinfoAPI")

# Load spaCy model (optional)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None
    logger.warning("spaCy model not loaded")

# PyG imports for GNN (with fallback)
try:
    from torch_geometric.nn import GCNConv, GATConv
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    GCNConv = None
    GATConv = None
    HAS_TORCH_GEOMETRIC = False
    logger.warning("PyTorch Geometric not available")

# -------------------------------
# Model Definitions
# -------------------------------
class BioBERTClassifier(nn.Module):
    def __init__(self, model_name="dmis-lab/biobert-base-cased-v1.1", num_labels=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        pooled_output = self.dropout(pooled_output)
        return self.fc(pooled_output)

# Import exact ARG/ARG_GNN classes from original definitions
from api.main import BioBERT_ARG, BioBERT_ARG_GNN

# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI(
    title="Health Misinformation Detector API",
    description="AI-powered platform for detecting health misinformation (Codespaces Deployment)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# OPTIONS for preflight
@app.options("/predict")
async def predict_options():
    return {"message": "OK"}

@app.options("/{path:path}")
async def catch_all_options():
    return {"message": "OK"}

# -------------------------------
# Global Variables
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = None
models = {}

# -------------------------------
# Request/Response Models
# -------------------------------
class PredictionRequest(BaseModel):
    text: str
    model_name: str = "BioBERT"

# ...existing code...
class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    label: str
    probabilities: dict
    rationales: Optional[List[List[float]]] = None  # <-- Accept floats
    model_used: str
# ...existing code...

# -------------------------------
# Load Models on Startup
# -------------------------------
@app.on_event("startup")
async def load_models():
    global tokenizer, models
    logger.info("🚀 Starting model loading from saved_models...")
    logger.info(f"📱 Device: {device}, CUDA available: {torch.cuda.is_available()}")
    try:
        # Tokenizer (shared)
        tokenizer = AutoTokenizer.from_pretrained(
            "dmis-lab/biobert-base-cased-v1.1", use_fast=True
        )
        logger.info("📝 Tokenizer loaded.")

        # -------------------------------
        # Load BioBERT
        # -------------------------------
        logger.info("🧠 Loading BioBERT model...")
        biobert_model = BioBERTClassifier()
        biobert_model.load_state_dict(
            torch.load("saved_models/BioBERT/best_model.pt", map_location=device)
        )
        biobert_model.to(device)
        biobert_model.eval()
        models["BioBERT"] = biobert_model

        # -------------------------------
        # Load BioBERT_ARG
        # -------------------------------
        logger.info("🧠 Loading BioBERT_ARG model...")
        biobert_arg_model = BioBERT_ARG()
        biobert_arg_model.load_state_dict(
            torch.load("saved_models/BioBERT_ARG/best_model.pt", map_location=device)
        )
        biobert_arg_model.to(device)
        biobert_arg_model.eval()
        models["BioBERT_ARG"] = biobert_arg_model

        # -------------------------------
        # Load BioBERT_ARG_GNN
        # -------------------------------
        logger.info("🧠 Loading BioBERT_ARG_GNN model...")
        biobert_arg_gnn_model = BioBERT_ARG_GNN()
        biobert_arg_gnn_model.load_state_dict(
            torch.load("saved_models/BioBERT_ARG_GNN/best_model.pt", map_location=device)
        )
        biobert_arg_gnn_model.to(device)
        biobert_arg_gnn_model.eval()
        models["BioBERT_ARG_GNN"] = biobert_arg_gnn_model

        logger.info(f"✅ Models loaded successfully: {list(models.keys())}")
        logger.info("🚀 Server started at http://0.0.0.0:8000")
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        raise

# -------------------------------
# Endpoints
# -------------------------------
@app.get("/")
def root():
    return {
        "message": "Health Misinformation Detector API",
        "version": "1.0.0 (Codespaces)",
        "models_available": list(models.keys()),
        "device": str(device),
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "device": str(device),
        "gpu_available": torch.cuda.is_available()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, response: Response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"

    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")
    if request.model_name not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model_name}' not available. Available: {list(models.keys())}"
        )

    # ...existing code...
    try:
        logger.info(f"🔍 Prediction request using {request.model_name}")
        inputs = tokenizer(
            request.text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        model = models[request.model_name]
        with torch.no_grad():
            outputs = model(**inputs)
            # Handle tuple output for ARG models
            if isinstance(outputs, tuple):
                logits, rationales = outputs
            else:
                logits = outputs
                rationales = None

            probabilities = F.softmax(logits, dim=-1)
            confidence, prediction = torch.max(probabilities, dim=-1)

        prob_dict = {
            "misinformation": float(probabilities[0][0]),
            "reliable": float(probabilities[0][1])
        }
        predicted_label = "reliable" if prediction.item() == 1 else "misinformation"

        logger.info(f"✅ Prediction: {predicted_label} ({confidence.item():.3f})")
        return PredictionResponse(
            prediction=float(prediction.item()),
            confidence=float(confidence.item()),
            label=predicted_label,
            probabilities=prob_dict,
            rationales=rationales.tolist() if rationales is not None else None,
            model_used=request.model_name
        )
# ...existing code...
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test")
def test_endpoint():
    return {"message": "API is working!", "models_loaded": len(models), "device": str(device)}

# -------------------------------
# Run Server
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
