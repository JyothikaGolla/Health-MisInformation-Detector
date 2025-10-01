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
logger = logging.getLogger(__name__)

# Load spaCy model
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
# Model Definitions (Simplified for Codespaces)
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

# Create FastAPI app
app = FastAPI(
    title="Health Misinformation Detector API",
    description="AI-powered platform for detecting health misinformation (Codespaces Deployment)",
    version="1.0.0"
)


# Configure CORS for frontend access - Allow all for Codespace development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Must be False when using "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware to force CORS headers for all responses
@app.middleware("http")
async def add_cors_headers(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# Add explicit OPTIONS handler for preflight requests
@app.options("/predict")
async def predict_options():
    return {"message": "OK"}

@app.options("/{path:path}")
async def catch_all_options():
    return {"message": "OK"}

# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = None
models = {}

# Request/Response models
class PredictionRequest(BaseModel):
    text: str
    model_name: str = "BioBERT"

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    label: str
    probabilities: dict
    rationales: Optional[List[List[int]]] = None
    model_used: str

@app.on_event("startup")
async def load_models():
    """Load models from HuggingFace Hub instead of local files."""
    global tokenizer, models
    
    logger.info("🚀 Starting model loading from HuggingFace...")
    logger.info(f"📱 Device: {device}")
    logger.info(f"💾 CUDA available: {torch.cuda.is_available()}")
    
    try:
        # Load tokenizer
        logger.info("📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1", use_fast=True)
        
        # Load base BioBERT model
        logger.info("🧠 Loading BioBERT model...")
        biobert_model = BioBERTClassifier(model_name="dmis-lab/biobert-base-cased-v1.1")
        biobert_model.to(device)
        biobert_model.eval()
        
        # For Codespaces demo, use same model for all variants
        # In production, you'd load your fine-tuned models here
        models["BioBERT"] = biobert_model
        models["BioBERT_ARG"] = biobert_model  # Placeholder - you can add different models later
        models["BioBERT_ARG_GNN"] = biobert_model  # Placeholder - you can add different models later
        
        logger.info(f"✅ Successfully loaded {len(models)} models!")
        logger.info(f"📊 Available models: {list(models.keys())}")
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}")
        raise e

@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "Health Misinformation Detector API",
        "description": "AI-powered platform for detecting health misinformation",
        "version": "1.0.0 (GitHub Codespaces)",
        "deployment": "GitHub Codespaces with 8GB RAM",
        "models_available": list(models.keys()) if models else [],
        "device": str(device),
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "device": str(device),
        "gpu_available": torch.cuda.is_available(),
        "ram_info": "8GB available in Codespaces"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, response: Response):
    """Make prediction on health claim."""
    # Add CORS headers manually as backup
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Please wait for startup to complete.")
    
    if request.model_name not in models:
        available_models = list(models.keys())
        raise HTTPException(
            status_code=400, 
            detail=f"Model '{request.model_name}' not available. Available models: {available_models}"
        )
    
    try:
        logger.info(f"🔍 Processing prediction with {request.model_name}")
        
        # Tokenize input
        inputs = tokenizer(
            request.text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        # Get model prediction
        model = models[request.model_name]
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = F.softmax(outputs, dim=-1)
            confidence, prediction = torch.max(probabilities, dim=-1)
        
        # Convert to response format
        prob_dict = {
            "misinformation": float(probabilities[0][0]),
            "reliable": float(probabilities[0][1])
        }
        
        # Determine label
        predicted_label = "reliable" if prediction.item() == 1 else "misinformation"
        
        logger.info(f"✅ Prediction complete: {predicted_label} ({confidence.item():.3f})")
        
        return PredictionResponse(
            prediction=float(prediction.item()),
            confidence=float(confidence.item()),
            label=predicted_label,
            probabilities=prob_dict,
            rationales=None,  # Could add rationale extraction later
            model_used=request.model_name
        )
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Add some sample endpoints for testing
@app.get("/test")
def test_endpoint():
    """Simple test endpoint."""
    return {
        "message": "API is working!",
        "models_loaded": len(models),
        "device": str(device)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)