import numpy as np
from typing import List, Tuple, Dict, Any
import os

# Try to import transformers and load BioBERT; fall back gracefully
_MODEL = None
_TOKENIZER = None

def _load_model():
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return
    try:
        from transformers import AutoTokenizer, AutoModel
        name = os.getenv('BIOBERT_MODEL', 'dmis-lab/biobert-base-cased-v1.1')
        _TOKENIZER = AutoTokenizer.from_pretrained(name)
        _MODEL = AutoModel.from_pretrained(name)
    except Exception:
        _MODEL, _TOKENIZER = None, None

def _mean_pool(last_hidden_state, attention_mask):
    import torch
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    return masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

def encode_text(text: str, rationales: List[str]) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Encode with BioBERT if available; otherwise deterministic mock.
    Returns: (vector[768], content_score[0..1], highlights)
    """
    _load_model()
    t = (text or "").strip()
    r_concat = " \n".join(rationales or [])
    full = (t + "\n" + r_concat).strip()
    if _MODEL is None:
        # fallback
        rng = np.random.default_rng(abs(hash(full)) % (2**32))
        vec = rng.normal(0, 1, 768).astype('float32')
    else:
        from transformers import AutoTokenizer
        import torch
        toks = _TOKENIZER(full, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            out = _MODEL(**toks)
            vec_t = _mean_pool(out.last_hidden_state, toks['attention_mask'])
        vec = vec_t[0].cpu().numpy().astype('float32')

    medical_terms = ["trial","randomized","cohort","vaccine","placebo","symptom","risk","effect","double-blind","meta-analysis"]
    score = min(1.0, sum(w in full.lower() for w in medical_terms)/6.0 + 0.2)
    highlights = {"key_terms": [w for w in medical_terms if w in full.lower()]}
    return vec, float(score), highlights
