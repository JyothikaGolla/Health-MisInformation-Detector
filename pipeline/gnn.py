import numpy as np
from typing import Dict, Any, Tuple
import json, os

DATA_POSTS = os.path.join(os.path.dirname(__file__), "../data/sample_posts.json")
DATA_EDGES = os.path.join(os.path.dirname(__file__), "../data/sample_edges.json" )

def _pyg_embed(post_id: str, edges):
    # Minimal illustrative GNN using torch_geometric if available
    try:
        import torch
        from torch_geometric.data import Data
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception:
        return None

    # Map post_id nodes to indices and build a toy graph (posts only)
    nodes = sorted(list({e.get('target') for e in edges} | {post_id}))
    idx = {n:i for i,n in enumerate(nodes)}
    # Build edge index where each edge target is the post
    sources = [idx.get(e.get('target', post_id)) for e in edges]
    targets = [idx[post_id] for _ in edges]
    import torch
    if len(sources)==0:
        sources=[idx[post_id]]; targets=[idx[post_id]]
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    x = torch.randn(len(nodes), 8)

    class TinyGNN(nn.Module):
        def __init__(self, in_dim=8, h=16, out=8):
            super().__init__()
            try:
                from torch_geometric.nn import GCNConv
                self.conv1 = GCNConv(in_dim, h)
                self.conv2 = GCNConv(h, out)
                self.using_pyg = True
            except Exception:
                self.lin1 = nn.Linear(in_dim, h)
                self.lin2 = nn.Linear(h, out)
                self.using_pyg = False
        def forward(self, x, edge_index):
            if hasattr(self, 'using_pyg') and self.using_pyg:
                from torch_geometric.nn import GCNConv
                x = self.conv1(x, edge_index); x = F.relu(x)
                x = self.conv2(x, edge_index)
            else:
                x = F.relu(self.lin1(x)); x = self.lin2(x)
            return x
    model = TinyGNN()
    with torch.no_grad():
        emb = model(x, edge_index)
    return emb[idx[post_id]].numpy().astype('float32')

def gnn_embed(meta: Dict[str, Any]) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    try:
        posts = json.load(open(DATA_POSTS))
        edges = json.load(open(DATA_EDGES))
    except Exception:
        posts, edges = [], []

    post_id = str(meta.get("postId", "0"))
    shares = sum(1 for e in edges if e.get("target") == post_id)
    influencers = sum(1 for e in edges if e.get("weight", 0) > 0.8 and e.get("target") == post_id)

    vec = _pyg_embed(post_id, edges)
    if vec is None:
        vec = np.array([shares, influencers, shares * 0.1 + influencers * 0.9, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32')

    # Risk score from propagation
    norm = float(1.0 / (1.0 + shares + influencers))
    score = max(0.0, 1.0 - norm)
    cues = {"shares": shares, "influencers": influencers}
    return vec, score, cues
