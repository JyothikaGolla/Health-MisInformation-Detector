# api/main.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy

# PyG imports for GNN
try:
    from torch_geometric.nn import GCNConv, GATConv
except ImportError:
    GCNConv = None
    GATConv = None

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None

# -------------------------------
# Model Definitions
# -------------------------------
class BioBERTClassifier(nn.Module):
    def __init__(self, model_name="dmis-lab/biobert-base-cased-v1.1", num_labels=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h = outputs.last_hidden_state[:, 0, :]
        return self.fc(h)


class BioBERT_ARG(nn.Module):
    def __init__(self, model_name="dmis-lab/biobert-base-cased-v1.1", num_labels=2, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.rationale_layer = nn.Linear(self.bert.config.hidden_size, 1)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
        rationale_logits = self.rationale_layer(last_hidden).squeeze(-1)
        rationale_probs = torch.sigmoid(rationale_logits) * attention_mask
        pooled = (rationale_probs.unsqueeze(-1) * last_hidden).sum(dim=1) / (
            rationale_probs.sum(dim=1, keepdim=True) + 1e-8
        )
        h = F.relu(self.fc1(pooled))
        h = self.dropout(h)
        return self.fc2(h), rationale_probs


class BioBERT_ARG_GNN(nn.Module):
    def __init__(self, model_name="dmis-lab/biobert-base-cased-v1.1", num_labels=2,
                 gnn_type="gcn", gnn_layers=2, gnn_hidden=128, fc_hidden=256, dropout=0.2):
        super().__init__()
        if GCNConv is None or GATConv is None:
            raise ImportError("torch-geometric required for BioBERT_ARG_GNN")

        self.bert = AutoModel.from_pretrained(model_name)
        self.gnn_type = gnn_type.lower()
        self.gnn_layers = nn.ModuleList()
        in_dim = self.bert.config.hidden_size
        for _ in range(gnn_layers):
            out_dim = gnn_hidden
            if gnn_type == "gat":
                self.gnn_layers.append(GATConv(in_dim, out_dim // 4, heads=4, concat=True))
            else:
                self.gnn_layers.append(GCNConv(in_dim, out_dim))
            in_dim = out_dim

        self.fc1 = nn.Linear(self.bert.config.hidden_size + gnn_hidden, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_labels)
        self.dropout = nn.Dropout(dropout)
        self.rationale_layer = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, edge_index_list=None,
                num_nodes_list=None, subtoken_to_word_list=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = outputs.last_hidden_state
        cls_emb = last_hidden[:, 0, :]
        batch_size = input_ids.size(0)
        device = last_hidden.device
        gnn_graph_feats = []
        rationale_scores_list = []

        for i in range(batch_size):
            num_nodes = num_nodes_list[i] if num_nodes_list else 0
            submap = subtoken_to_word_list[i].to(device) if subtoken_to_word_list else None
            seq_hidden = last_hidden[i]
            rationale_logits = self.rationale_layer(seq_hidden).squeeze(-1)
            rationale_probs = torch.sigmoid(rationale_logits)
            rationale_scores_list.append(rationale_probs)
            seq_hidden = seq_hidden * rationale_probs.unsqueeze(-1)

            if num_nodes == 0 or submap is None:
                pooled = torch.zeros(
                    (self.gnn_layers[-1].out_channels
                     if hasattr(self.gnn_layers[-1], "out_channels")
                     else seq_hidden.size(1)),
                    device=device
                )
                gnn_graph_feats.append(pooled)
                continue

            node_feats = []
            for j in range(num_nodes):
                mask = (submap == j)
                pooled_node = seq_hidden[mask].mean(dim=0) if mask.any() else torch.zeros(seq_hidden.size(1), device=device)
                node_feats.append(pooled_node)

            node_feats = torch.stack(node_feats, dim=0)
            eidx = edge_index_list[i].to(device) if (edge_index_list and edge_index_list[i].numel() > 0) \
                   else torch.empty((2, 0), dtype=torch.long, device=device)
            x = node_feats
            for gnn in self.gnn_layers:
                if eidx.numel() == 0:
                    x = gnn.lin(x) if hasattr(gnn, "lin") else x
                else:
                    x = gnn(x, eidx)
                x = F.relu(x)
            pooled = x.mean(dim=0)
            gnn_graph_feats.append(pooled)

        gnn_graph_feats = torch.stack(gnn_graph_feats, dim=0)
        h = torch.cat([cls_emb, gnn_graph_feats], dim=1)
        h = self.dropout(F.relu(self.fc1(h)))
        return self.fc2(h), torch.stack(rationale_scores_list, dim=0)


# -------------------------------
# Text Processing
# -------------------------------
def build_graph_from_spacy(doc, adj_window=2, add_dep_edges=True, add_ent_edges=True):
    edges = []
    if add_dep_edges:
        for tok in doc:
            if tok.i != tok.head.i:
                edges.extend([[tok.i, tok.head.i], [tok.head.i, tok.i]])
    for i in range(len(doc)):
        for j in range(1, adj_window + 1):
            if i + j < len(doc):
                edges.extend([[i, i + j], [i + j, i]])
    if add_ent_edges:
        for ent in doc.ents:
            for t1 in range(ent.start, ent.end):
                for t2 in range(ent.start, ent.end):
                    if t1 != t2:
                        edges.append([t1, t2])
    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def preprocess_text_for_inference(text, tokenizer, max_len=128):
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    if nlp is None:
        return input_ids, attention_mask, None, None, None

    doc = nlp(text)
    token_spans = [(t.idx, t.idx + len(t.text)) for t in doc]
    offsets = enc["offset_mapping"].squeeze(0).tolist()
    subtoken_to_word = []
    for start, end in offsets:
        if start == 0 and end == 0:
            subtoken_to_word.append(-1)
            continue
        mapped = -1
        for tidx, (ts, te) in enumerate(token_spans):
            if (start < te) and (end > ts):
                mapped = tidx
                break
        subtoken_to_word.append(mapped)
    subtoken_to_word = torch.tensor(subtoken_to_word, dtype=torch.long)
    edge_index = build_graph_from_spacy(doc)
    num_nodes = len(doc)
    return input_ids, attention_mask, edge_index, num_nodes, subtoken_to_word


# -------------------------------
# FastAPI Setup
# -------------------------------
app = FastAPI(
    title="Health Misinformation Detector",
    description="AI-powered platform for detecting health misinformation using BioBERT, ARG, and GNN models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],  # Allow all origins for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InferenceRequest(BaseModel):
    text: str
    model_name: str = "BioBERT"  # Changed from model_type to avoid conflict
    
    class Config:
        # Fix the Pydantic protected namespace warning
        protected_namespaces = ()

@app.on_event("startup")
def load_resources():
    global tokenizer, models, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1", use_fast=True)
    models = {}
    base_dir = "saved_models"
    for model_name in ["BioBERT", "BioBERT_ARG", "BioBERT_ARG_GNN"]:
        model_dir = os.path.join(base_dir, model_name)
        if not os.path.exists(model_dir):
            continue
        # Use original HF model name for initialization
        if model_name == "BioBERT":
            model = BioBERTClassifier(model_name="dmis-lab/biobert-base-cased-v1.1")
        elif model_name == "BioBERT_ARG":
            model = BioBERT_ARG(model_name="dmis-lab/biobert-base-cased-v1.1")
        else:
            model = BioBERT_ARG_GNN(model_name="dmis-lab/biobert-base-cased-v1.1")
        # Load saved PyTorch weights
        model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pt"), map_location=device))
        model.to(device)
        model.eval()
        models[model_name] = model


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "Health Misinformation Detector API",
        "description": "AI-powered platform for detecting health misinformation",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs",
            "openapi": "/openapi.json"
        },
        "models_available": list(models.keys()) if 'models' in globals() else [],
        "status": "active"
    }


@app.get("/health")
def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "models_loaded": len(models) if 'models' in globals() else 0,
        "device": str(device) if 'device' in globals() else "unknown"
    }


@app.post("/predict")
def predict(req: InferenceRequest):
    """Predict if a health claim is misinformation or reliable."""
    if req.model_name not in models:
        return {"error": f"Model {req.model_name} not available. Available models: {list(models.keys())}"}

    model = models[req.model_name]
    input_ids, attention_mask, edge_index, num_nodes, subtoken_to_word = preprocess_text_for_inference(
        req.text, tokenizer
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        if req.model_name == "BioBERT":
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1).cpu().tolist()
            prediction = int(torch.argmax(logits, dim=1).item())
            confidence = float(max(probs[0]))
            return {
                "prediction": prediction,
                "confidence": confidence,
                "label": "reliable" if prediction == 1 else "misinformation",
                "probabilities": {"misinformation": probs[0][0], "reliable": probs[0][1]},
                "model_used": req.model_name
            }

        elif req.model_name == "BioBERT_ARG":
            logits, rationale_scores = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1).cpu().tolist()
            prediction = int(torch.argmax(logits, dim=1).item())
            confidence = float(max(probs[0]))
            return {
                "prediction": prediction,
                "confidence": confidence,
                "label": "reliable" if prediction == 1 else "misinformation",
                "probabilities": {"misinformation": probs[0][0], "reliable": probs[0][1]},
                "rationales": rationale_scores.cpu().tolist(),
                "model_used": req.model_name
            }

        else:  # BioBERT_ARG_GNN
            logits, rationale_scores = model(
                input_ids, attention_mask,
                edge_index_list=[edge_index],
                num_nodes_list=[num_nodes],
                subtoken_to_word_list=[subtoken_to_word]
            )
            probs = torch.softmax(logits, dim=1).cpu().tolist()
            prediction = int(torch.argmax(logits, dim=1).item())
            confidence = float(max(probs[0]))
            return {
                "prediction": prediction,
                "confidence": confidence,
                "label": "reliable" if prediction == 1 else "misinformation",
                "probabilities": {"misinformation": probs[0][0], "reliable": probs[0][1]},
                "rationales": rationale_scores.cpu().tolist(),
                "model_used": req.model_name
            }
# uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload