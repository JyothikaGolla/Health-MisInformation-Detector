import json, os, random
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
import requests

API = os.getenv('API_URL', 'http://localhost:8000')

SAMPLES = [
    {"claim": "Turmeric cures cancer", "meta": {"postId": "0"}, "label": 0},
    {"claim": "Randomized trial shows vaccine efficacy", "meta": {"postId": "1"}, "label": 1},
    {"claim": "Drinking bleach eliminates viruses", "meta": {"postId": "0"}, "label": 0},
]

def map_verdict_to_score(v):
    return {"fake": 0.0, "uncertain": 0.5, "true": 1.0}.get(v, 0.5)

def main():
    y_true, y_score, preds = [], [], []
    for s in SAMPLES:
        r = requests.post(f"{API}/analyze", json={"claim": s["claim"], "meta": s["meta"]}, timeout=30).json()
        y_true.append(s['label'])
        score = map_verdict_to_score(r.get('verdict', 'uncertain'))
        y_score.append(score)
        preds.append(1 if score >= 0.66 else 0)
    f1 = f1_score(y_true, preds)
    try:
        auroc = roc_auc_score(y_true, y_score)
    except Exception:
        auroc = float('nan')
    print({"F1": f1, "AUROC": auroc, "n": len(SAMPLES)})

if __name__ == "__main__":
    main()
