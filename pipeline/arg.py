import os, random, re
from typing import List
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

def _fallback_rationales(text: str, k: int) -> List[str]:
    seeds = [
        "WHO guidance suggests randomized trials do not support the claim.",
        "Meta-analysis in PubMed indicates low-quality evidence for efficacy.",
        "CDC reports show no causal link between the intervention and outcome.",
        "Cochrane review finds insufficient evidence; small sample sizes.",
        "Observational bias likely; confounders not controlled.",
    ]
    random.shuffle(seeds)
    outs = [f"Rationale {i+1}: {seeds[i % len(seeds)]}" for i in range(k)]
    # append simple pseudo-citations
    cite = "(e.g., WHO; PubMed: 12345678)"
    return [o + ' ' + cite for o in outs]

def generate_rationales(text: str, k: int = 3) -> List[str]:
    """OpenAI-backed rationale generation with graceful fallback.
    Each rationale attempts to include a generic citation hint.
    Replace with your retrieval-augmented flow for real PubMed IDs.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    provider = os.getenv('ARG_PROVIDER', 'openai')
    if not api_key or provider != 'openai' or OpenAI is None:
        return _fallback_rationales(text, k)

    try:
        client = OpenAI(api_key=api_key)
        prompt = f"""
You are generating factual, **grounded** rationales to assess a health claim.
Claim: {text}
Return {k} bullet points, each with a brief citation-like hint (WHO/CDC/Cochrane or PubMed ID if known). Keep each under 30 words.
"""
        # Using responses API; adapt model name as needed
        resp = client.chat.completions.create(model=os.getenv('OPENAI_MODEL','gpt-4o-mini'), messages=[{"role":"user","content":prompt}], temperature=0.2)
        content = resp.choices[0].message.content.strip()
        # Split bullets
        lines = [re.sub(r'^[-*\s]+','',ln).strip() for ln in content.split('\n') if ln.strip()]
        return lines[:k] if lines else _fallback_rationales(text, k)
    except Exception:
        return _fallback_rationales(text, k)
