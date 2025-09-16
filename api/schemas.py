from pydantic import BaseModel
from typing import List, Dict, Any

class AnalysisOut(BaseModel):
    id: int
    claim: str
    verdict: str
    confidence: float
    class Config:
        from_attributes = True
