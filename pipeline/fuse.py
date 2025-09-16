import numpy as np
from typing import Tuple

def fuse_and_classify(content_vec: np.ndarray, prop_vec: np.ndarray, content_score: float, prop_score: float) -> Tuple[str, float]:
    """Concatenate + simple rule-based classifier (replace with ML model)."""
    risk = 0.6 * (1.0 - content_score) + 0.4 * prop_score
    confidence = float(min(0.99, 0.5 + abs(content_score - prop_score)))
    if risk >= 0.66:
        return "fake", confidence
    if risk <= 0.33:
        return "true", confidence
    return "uncertain", confidence
