import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    AutoConfig,
    Trainer,
    TrainingArguments,
)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.data import Dataset
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from tqdm import tqdm

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("./dataset.csv")

# Keep required fields for frontend
df = df[
    [
        "link",
        "title",
        "description",
        "original_title",
        "rating",
        "category",
        "tags",
        "source_link",
        "Our Review Summary",
        "Why This Matters",
        "label",
    ]
].dropna()

# Combine text for classification
df["text"] = (
    df["title"].astype(str)
    + " "
    + df["description"].astype(str)
    + " "
    + df["original_title"].astype(str)
    + " "
    + df["Our Review Summary"].astype(str)
    + " "
    + df["Why This Matters"].astype(str)
    + " "
    + df["tags"].astype(str)
)

# Rationales = review summary + why this matters + tags
df["rationale_text"] = (
    df["Our Review Summary"].astype(str)
    + " "
    + df["Why This Matters"].astype(str)
    + " "
    + df["tags"].astype(str)
)

X_train, X_test, y_train, y_test = train_test_split(
    df, df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# -------------------------------
# Dataset Class
# -------------------------------
class NewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=256, use_rationale=False):
        self.texts = dataframe["text"].tolist()
        self.labels = dataframe["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_rationale = use_rationale
        self.rationales = dataframe["rationale_text"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx]),
        }
        if self.use_rationale:
            rationale_encoding = self.tokenizer(
                self.rationales[idx],
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt",
            )
            item["rationale_labels"] = rationale_encoding["attention_mask"].squeeze()
        return item


# -------------------------------
# Metrics
# -------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # unwrap HuggingFace output
        logits = logits[0]
    if logits.ndim > 1 and logits.shape[1] > 1:
        preds = np.argmax(logits, axis=1)
    else:
        preds = (logits > 0).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# -------------------------------
# Custom Models
# -------------------------------
class BioBERT_ARG(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            model_name, output_hidden_states=True
        )
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.rationale_head = nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, labels=None, rationale_labels=None):
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0]

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        rationale_logits = self.rationale_head(sequence_output).squeeze(-1)
        if rationale_labels is not None:
            rationale_loss_fn = BCEWithLogitsLoss()
            rationale_loss = rationale_loss_fn(rationale_logits, rationale_labels.float())
            loss = loss + 0.5 * rationale_loss if loss is not None else rationale_loss

        return {"loss": loss, "logits": logits, "rationale_logits": rationale_logits}


class GCNClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# -------------------------------
# Training Function
# -------------------------------
def train_and_evaluate(model_name, save_dir, custom_model=None, use_rationale=False):
    print(f"\n🚀 Training model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = NewsDataset(X_train, tokenizer, use_rationale=use_rationale)
    test_dataset = NewsDataset(X_test, tokenizer, use_rationale=use_rationale)

    if custom_model:
        model = custom_model
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )

    training_args = TrainingArguments(
        output_dir=f"./result_{save_dir}",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=200,
        weight_decay=0.01,
        logging_dir=f"./logs_{save_dir}",
        logging_steps=20,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        save_safetensors=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"📊 Final Evaluation for {model_name}: {metrics}")

    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"✅ Model saved at {save_dir}")

    return metrics


# -------------------------------
# Run Models
# -------------------------------
results = {}

# 1. BioBERT
results["BioBERT"] = train_and_evaluate(
    "dmis-lab/biobert-base-cased-v1.1", "./health_model_biobert"
)

# 2. MiniLM
results["MiniLM"] = train_and_evaluate(
    "sentence-transformers/all-MiniLM-L6-v2", "./health_model_minilm"
)

# 3. BioBERT+ARG
arg_model = BioBERT_ARG("dmis-lab/biobert-base-cased-v1.1")
results["BioBERT+ARG"] = train_and_evaluate(
    "dmis-lab/biobert-base-cased-v1.1",
    "./health_model_biobert_arg",
    custom_model=arg_model,
    use_rationale=True,
)

# -------------------------------
# 4. BioBERT+ARG+GNN
# -------------------------------
print("\n🚀 Training BioBERT+ARG+GNN...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
bert = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1").to(device)
bert.eval()

# Encode texts into embeddings
with torch.no_grad():
    embeddings = []
    for text in tqdm(df["text"].tolist(), desc="Encoding with BioBERT"):
        enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
        out = bert(**enc).last_hidden_state[:, 0, :].cpu()
        embeddings.append(out)
    embeddings = torch.cat(embeddings, dim=0)

# Build a simple fully connected graph (toy example: connect each node to its k nearest)
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(embeddings.numpy())
k = 5
edge_index = []
for i in range(sim_matrix.shape[0]):
    neighbors = np.argsort(sim_matrix[i])[-k:]
    for j in neighbors:
        if i != j:
            edge_index.append([i, j])
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

labels = torch.tensor(df["label"].values, dtype=torch.long)

data = Data(x=embeddings, edge_index=edge_index, y=labels)
data = train_test_split_edges(data)

gnn_model = GCNClassifier(in_channels=embeddings.shape[1], hidden_channels=128, num_classes=2).to(device)
optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
best_state = None
best_acc = 0

for epoch in range(10):
    gnn_model.train()
    optimizer.zero_grad()
    out = gnn_model(data.x.to(device), data.edge_index.to(device))
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask].to(device))
    loss.backward()
    optimizer.step()

    gnn_model.eval()
    with torch.no_grad():
        preds = out.argmax(dim=1).cpu()
        acc = accuracy_score(data.y[data.val_mask].cpu(), preds[data.val_mask].cpu())
    if acc > best_acc:
        best_acc = acc
        best_state = gnn_model.state_dict()

# Evaluate final GNN
gnn_model.load_state_dict(best_state)
gnn_model.eval()
with torch.no_grad():
    out = gnn_model(data.x.to(device), data.edge_index.to(device))
    preds = out.argmax(dim=1).cpu().numpy()
    y_true = data.y.cpu().numpy()

mask_np = data.test_mask.cpu().numpy()
acc = accuracy_score(y_true[mask_np], preds[mask_np])
prec, rec, f1, _ = precision_recall_fscore_support(y_true[mask_np], preds[mask_np], average='binary', zero_division=0)

results["BioBERT+ARG+GNN"] = {
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1,
}

# -------------------------------
# Compare Models
# -------------------------------
print("\n===============================")
print("📊 Model Comparison")
print("===============================")
for model_name, metrics in results.items():
    print(model_name, ":", metrics)
print("===============================")
