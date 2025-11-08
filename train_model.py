# train_model.py
import os
import argparse
import numpy as np # numerical computing library
import pandas as pd # data manipulation library
from tqdm import tqdm # displays progress bar during processing
import torch # for DL computations
import torch.nn as nn # for creating custom layers, models, architectures
import torch.nn.functional as F

# Dataset - loads your samples
# DataLoader - batches and shuffles them efficiently during training
from torch.utils.data import Dataset, DataLoader

# AutoTokenizer - creates tokens and IDs
# AutoModel - loads pre-trained models
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW

# for performance evaluation 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# PyG imports
try:
    from torch_geometric.nn import GCNConv, GATConv
except ImportError:
    raise ImportError("Please install torch-geometric to use GNN features.")

# spaCy for dependency parsing and NER (Named Entity Recognition)
import spacy # parsing and entity recognition
nlp = spacy.load("en_core_web_sm")

# ---------------- Dataset ----------------
class HealthDataset(Dataset):
    def __init__(self, dataframe, tokenizer, text_columns, label_column="label", max_len=128,
                 use_gnn=False, adj_window=2, add_dep_edges=True, add_ent_edges=True):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_columns = text_columns
        self.label_column = label_column
        self.max_len = max_len
        self.use_gnn = use_gnn
        self.adj_window = adj_window
        self.add_dep_edges = add_dep_edges
        self.add_ent_edges = add_ent_edges

    def __len__(self):
        return len(self.data)

    def build_graph_from_spacy(self, doc):
        edges = []
        if self.add_dep_edges:
            for tok in doc:
                if tok.i != tok.head.i:
                    edges.extend([[tok.i, tok.head.i], [tok.head.i, tok.i]])
        for i in range(len(doc)):
            for j in range(1, self.adj_window + 1):
                if i + j < len(doc):
                    edges.extend([[i, i + j], [i + j, i]])
        if self.add_ent_edges:
            for ent in doc.ents:
                for t1 in range(ent.start, ent.end):
                    for t2 in range(ent.start, ent.end):
                        if t1 != t2:
                            edges.append([t1, t2])
        if len(edges) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def __getitem__(self, idx):
        text = " ".join(str(self.data.iloc[idx].get(col, "")) for col in self.text_columns if col in self.data.columns)
        label = int(self.data.iloc[idx][self.label_column])

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=True,
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        offsets = enc["offset_mapping"].squeeze(0).tolist()

        doc = nlp(text)
        token_spans = [(t.idx, t.idx + len(t.text)) for t in doc]
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

        if self.use_gnn:
            edge_index = self.build_graph_from_spacy(doc)
            num_nodes = len(doc)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            num_nodes = 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long),
            "edge_index": edge_index,
            "num_nodes": num_nodes,
            "subtoken_to_word": subtoken_to_word,
        }

def collate_fn(batch):
    collated = {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "edge_index": [b["edge_index"].to(torch.long) for b in batch],
        "num_nodes": [int(b["num_nodes"]) for b in batch],
        "subtoken_to_word": [b["subtoken_to_word"] for b in batch],
    }
    return collated

# ---------------- Models ----------------
class BioBERTClassifier(nn.Module):
    def __init__(self, model_name="dmis-lab/biobert-base-cased-v1.1", num_labels=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask) # [batch_size, sequence_length, hidden_size]
        h = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        return self.fc(h)

class BioBERT_ARG(nn.Module):
    def __init__(self, model_name="dmis-lab/biobert-base-cased-v1.1", num_labels=2, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        
        # Token-level rationale scorer
        self.rationale_layer = nn.Linear(self.bert.config.hidden_size, 1)
        
        # Classification layers
        self.fc1 = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
        
        # Compute rationale probabilities
        rationale_logits = self.rationale_layer(last_hidden).squeeze(-1)  # (batch, seq_len)
        rationale_probs = torch.sigmoid(rationale_logits) * attention_mask  # mask out padding
        
        # Weighted sum of token embeddings
        pooled = (rationale_probs.unsqueeze(-1) * last_hidden).sum(dim=1) / (rationale_probs.sum(dim=1, keepdim=True) + 1e-8)
        
        h = F.relu(self.fc1(pooled))
        h = self.dropout(h)
        return self.fc2(h)

class BioBERT_ARG_GNN(nn.Module):
    def __init__(self, model_name="dmis-lab/biobert-base-cased-v1.1", num_labels=2,
                 gnn_type="gcn", gnn_layers=2, gnn_hidden=128, fc_hidden=256, dropout=0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.gnn_type = gnn_type.lower()
        self.gnn_layers = nn.ModuleList()
        in_dim = self.bert.config.hidden_size
        for l in range(gnn_layers):
            out_dim = gnn_hidden
            if gnn_type == "gat":
                self.gnn_layers.append(GATConv(in_dim, out_dim // 4, heads=4, concat=True))
            else:
                self.gnn_layers.append(GCNConv(in_dim, out_dim))
            in_dim = out_dim
        self.fc1 = nn.Linear(self.bert.config.hidden_size + gnn_hidden, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_labels)
        self.dropout = nn.Dropout(dropout)
        
        # Token-level rationale scorer
        self.rationale_layer = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, edge_index_list=None, num_nodes_list=None, subtoken_to_word_list=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = outputs.last_hidden_state
        cls_emb = last_hidden[:, 0, :]
        batch_size = input_ids.size(0)
        device = last_hidden.device

        gnn_graph_feats = []
        for i in range(batch_size):
            num_nodes = num_nodes_list[i]
            submap = subtoken_to_word_list[i].to(device)
            seq_hidden = last_hidden[i]

            # Apply ARG rationale weights
            rationale_logits = self.rationale_layer(seq_hidden).squeeze(-1)
            rationale_probs = torch.sigmoid(rationale_logits)
            seq_hidden = seq_hidden * rationale_probs.unsqueeze(-1)

            if num_nodes == 0:
                pooled = torch.zeros((self.gnn_layers[-1].out_channels if hasattr(self.gnn_layers[-1], "out_channels") else seq_hidden.size(1)), device=device)
                gnn_graph_feats.append(pooled)
                continue

            node_feats = []
            for j in range(num_nodes):
                mask = (submap == j)
                pooled_node = seq_hidden[mask].mean(dim=0) if mask.any() else torch.zeros(seq_hidden.size(1), device=device)
                node_feats.append(pooled_node)
            node_feats = torch.stack(node_feats, dim=0)

            eidx = edge_index_list[i].to(device) if (edge_index_list and edge_index_list[i].numel() > 0) else torch.empty((2, 0), dtype=torch.long, device=device)
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
        return self.fc2(h)

# ---------------- Training / Eval ----------------
def compute_metrics(preds, labels):
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return acc, precision, recall, f1

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        outputs = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            edge_index_list=batch.get("edge_index"),
            num_nodes_list=batch.get("num_nodes"),
            subtoken_to_word_list=batch.get("subtoken_to_word")
        )
        labels = batch["labels"].to(device)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            outputs = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                edge_index_list=batch.get("edge_index"),
                num_nodes_list=batch.get("num_nodes"),
                subtoken_to_word_list=batch.get("subtoken_to_word")
            )
            preds.extend(outputs.argmax(dim=1).cpu().numpy())
            trues.extend(batch["labels"].cpu().numpy())
    return compute_metrics(preds, trues)

def train_and_save_model(model_name, model_cls, train_loader, val_loader, device, args):
    model_folder = os.path.join(args.output_dir, model_name)
    os.makedirs(model_folder, exist_ok=True)
    log_file = os.path.join(model_folder, f"{model_name}_metrics.csv")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

    model = model_cls(model_name=args.tokenizer_name, num_labels=args.num_labels).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05*total_steps), num_training_steps=total_steps)

    best_val_f1 = 0
    import csv
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch","train_loss","val_acc","val_precision","val_recall","val_f1"])
        for epoch in range(args.epochs):
            print(f"[{model_name}] Epoch {epoch+1}/{args.epochs}")
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
            val_acc, val_precision, val_recall, val_f1 = eval_epoch(model, val_loader, device)
            print(f"Loss: {train_loss:.4f}, Acc: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
            writer.writerow([epoch+1, train_loss, val_acc, val_precision, val_recall, val_f1])

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), os.path.join(model_folder, "best_model.pt"))
                try:
                    model.bert.save_pretrained(model_folder)
                    tokenizer.save_pretrained(model_folder)
                except:
                    pass

# ---------------- Main ----------------
def main(args):
    df = pd.read_csv(args.dataset) # load the dataset 
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True) # convert text -> tokens -> numerical IDs that can be fed into BioBERT.

    # data pre-processing
    dataset = HealthDataset(
        df, tokenizer, text_columns=args.text_columns, label_column=args.label_column,
        max_len=args.max_len, use_gnn=args.use_gnn, adj_window=args.adj_window,
        add_dep_edges=args.add_dep_edges, add_ent_edges=args.add_ent_edges
    )

    # split the dataset - 80% training and 20% testing
    train_size = int(0.8*len(dataset)) # training set size
    val_size = len(dataset) - train_size # validation set size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size]) # randomly divides data into 2 sets
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # here - shuffle=True -> randomly shuffles the training data order during each epoch
    # collate_fn -> a custom function that merges multiple samples into a single batch

    # check if GPU (cuda) is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models_to_train = [
        ("BioBERT", BioBERTClassifier),
        ("BioBERT_ARG", BioBERT_ARG),
        ("BioBERT_ARG_GNN", BioBERT_ARG_GNN)
    ]

    for name, cls in models_to_train:
        train_and_save_model(name, cls, train_loader, val_loader, device, args)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default="dmis-lab/biobert-base-cased-v1.1")
    parser.add_argument("--text_columns", nargs="+", default=["link", "title", "description", "Our Review Summary", "Why This Matters"])
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--use_gnn", action="store_true")
    parser.add_argument("--adj_window", type=int, default=2)
    parser.add_argument("--add_dep_edges", action="store_true")
    parser.add_argument("--add_ent_edges", action="store_true")
    parser.add_argument("--gnn_type", type=str, default="gcn", choices=["gcn","gat"])
    parser.add_argument("--gnn_layers", type=int, default=2)
    parser.add_argument("--gnn_hidden", type=int, default=128)
    parser.add_argument("--fc_hidden", type=int, default=256) # fully connected components - combines all learned features to make the final binary classification
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="saved_models")
    args = parser.parse_args()
    main(args)
