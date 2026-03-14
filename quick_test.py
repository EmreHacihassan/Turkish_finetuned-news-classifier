"""Quick training test for the Turkish News Classification model."""
import sys
import os
import csv
import json
import time
import random
import warnings
import contextlib
import platform
from pathlib import Path
from collections import Counter
from typing import Dict, List

warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report

print('=' * 70)
print('QUICK TRAINING TEST - RTX 4070 Laptop GPU')
print('=' * 70)

# ═══════════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════════
BASE_DIR = Path('.')
DATA_DIR = BASE_DIR / 'data'
MODEL_NAME = 'dbmdz/distilbert-base-turkish-cased'
MAX_LEN = 256
BATCH_SIZE = 32
GRAD_ACCUM = 2
LABEL_SMOOTH = 0.05
LR = 2e-5
SEED = 42

# Set seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda:0')
torch.backends.cudnn.benchmark = True

print(f'Device: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

# ═══════════════════════════════════════════════════════════════════════
# CATEGORY MAPPING
# ═══════════════════════════════════════════════════════════════════════
CATEGORY_MAP = {
    'Türkiye': 'Gündem', 'turkiye': 'Gündem', 'Gündem': 'Gündem',
    'siyaset': 'Gündem', 'Polemik': 'Gündem',
    'Dünya': 'Dünya', 'dunya': 'Dünya', 'Dünyadan': 'Dünya',
    'BBC': 'Dünya', 'english': 'Dünya',
    'Spor': 'Spor', 'spor': 'Spor', 'futbol': 'Spor', 'Futbol': 'Spor',
    'basketbol': 'Spor', 'Basketbol': 'Spor',
    'Ekonomi': 'Ekonomi', 'ekonomi': 'Ekonomi', 'Para': 'Ekonomi',
    'Borsa': 'Ekonomi', 'Döviz': 'Ekonomi',
    'Teknoloji': 'Teknoloji', 'bilim_ve_teknoloji': 'Teknoloji',
    'Sağlık': 'Sağlık', 'saglik': 'Sağlık',
    'Sanat': 'Kültür-Sanat', 'Kültür-Sanat': 'Kültür-Sanat',
    'Magazin': 'Kültür-Sanat', 'sinema': 'Kültür-Sanat',
    'Eğitim': 'Eğitim', 'egitim': 'Eğitim',
    'Yaşam': 'Yaşam', 'yasam': 'Yaşam', 'Seyahat': 'Yaşam',
    'Otomobil': 'Otomobil', 'otomobil': 'Otomobil',
    'Savunma Sanayi': 'Gündem',
}
SKIP_TOPICS = {'', 'diger', 'Diğer', 'General', 'AIRPORT'}

# ═══════════════════════════════════════════════════════════════════════
# LOAD DATA (subset for quick test)
# ═══════════════════════════════════════════════════════════════════════
def load_csv(path: Path, limit: int = None) -> List[Dict]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if limit and len(rows) >= limit:
                break
            topic = row.get('topic', '').strip()
            if topic in SKIP_TOPICS:
                continue
            category = CATEGORY_MAP.get(topic)
            if category is None:
                continue
            title = (row.get('title') or '').strip()
            abstract = (row.get('abstract') or '').strip()
            text = f'{title}. {abstract}' if abstract else title
            if len(text) < 10:
                continue
            rows.append({'text': text, 'category': category})
    return rows

print('\nLoading data (5000 train, 500 val samples)...')
train_rows = load_csv(DATA_DIR / 'train.csv', limit=5000)
val_rows = load_csv(DATA_DIR / 'validation.csv', limit=500)
print(f'  Train: {len(train_rows):,} samples')
print(f'  Val: {len(val_rows):,} samples')

# Create label mapping
categories = sorted(set(r['category'] for r in train_rows + val_rows))
cat2id = {c: i for i, c in enumerate(categories)}
id2cat = {i: c for c, i in cat2id.items()}
num_labels = len(categories)
print(f'  Categories: {num_labels}')

train_dist = Counter(r['category'] for r in train_rows)
print('\n  Distribution:')
for cat in categories:
    print(f'    {cat:<16}: {train_dist.get(cat, 0):>5}')

# ═══════════════════════════════════════════════════════════════════════
# DATASET & DATALOADER
# ═══════════════════════════════════════════════════════════════════════
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], max_length=self.max_len,
                             padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
        }

print('\nLoading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_texts = [r['text'] for r in train_rows]
train_labels = [cat2id[r['category']] for r in train_rows]
val_texts = [r['text'] for r in val_rows]
val_labels = [cat2id[r['category']] for r in val_rows]

train_ds = NewsDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_ds = NewsDataset(val_texts, val_labels, tokenizer, MAX_LEN)

# Weighted sampler for class imbalance
class_counts = np.array([train_dist.get(id2cat[i], 1) for i in range(num_labels)])
class_weights = 1.0 / np.maximum(class_counts, 1)
class_weights = class_weights / class_weights.sum() * num_labels
sample_weights = np.array([class_weights[l] for l in train_labels])
sampler = WeightedRandomSampler(torch.from_numpy(sample_weights).float(), len(train_ds), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0, pin_memory=True)

print(f'  Train batches: {len(train_loader)}')
print(f'  Val batches: {len(val_loader)}')

# ═══════════════════════════════════════════════════════════════════════
# MODEL, LOSS, OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════
class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.0):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    def forward(self, logits, targets):
        log_probs = torch.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        smooth = -log_probs.mean(dim=-1)
        return (self.confidence * nll + self.smoothing * smooth).mean()

print('\nLoading model...')
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
model.to(device)
print(f'  Parameters: {sum(p.numel() for p in model.parameters()):,}')

criterion = LabelSmoothingCE(smoothing=LABEL_SMOOTH)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

total_steps = (len(train_loader) // GRAD_ACCUM) * 2  # 2 epochs
warmup_steps = int(total_steps * 0.06)
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
scaler = GradScaler('cuda')

print(f'  Total steps: {total_steps}')
print(f'  Warmup steps: {warmup_steps}')

# ═══════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()
            preds_all.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    p, l = np.array(preds_all), np.array(labels_all)
    return {
        'loss': total_loss / len(loader),
        'acc': accuracy_score(l, p),
        'f1': f1_score(l, p, average='macro', zero_division=0),
    }

print('\n' + '=' * 70)
print('TRAINING (2 epochs on 5000 samples)')
print('=' * 70)

torch.cuda.empty_cache()
t_start = time.time()

for epoch in range(1, 3):
    print(f'\nEpoch {epoch}/2')
    print('-' * 40)

    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        with autocast('cuda', dtype=torch.float16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels) / GRAD_ACCUM

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * GRAD_ACCUM

        if (step + 1) % 50 == 0:
            avg_loss = total_loss / (step + 1)
            print(f'  Step {step+1:>4}/{len(train_loader)} | Loss: {avg_loss:.4f}')

    # Evaluate
    val_metrics = evaluate(model, val_loader, criterion, device)
    elapsed = time.time() - t_start

    print(f'\n  Train Loss : {total_loss/len(train_loader):.4f}')
    print(f'  Val Loss   : {val_metrics["loss"]:.4f}')
    print(f'  Val Acc    : {val_metrics["acc"]:.4f}')
    print(f'  Val F1     : {val_metrics["f1"]:.4f}')
    print(f'  Time       : {elapsed:.1f}s')
    print(f'  VRAM       : {torch.cuda.memory_allocated(0)/1024**3:.2f} GB')

# Final evaluation with classification report
print('\n' + '=' * 70)
print('FINAL EVALUATION')
print('=' * 70)

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        all_preds.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy())
        all_labels.extend(batch['labels'].numpy())

target_names = [id2cat[i] for i in range(num_labels)]
print(classification_report(all_labels, all_preds, target_names=target_names, digits=3))

total_time = time.time() - t_start
print(f'Total time: {total_time:.1f}s')
print(f'Samples/sec: {(len(train_rows) * 2) / total_time:.1f}')

# Quick inference test
print('\n' + '=' * 70)
print('INFERENCE TEST')
print('=' * 70)

test_sentences = [
    "Galatasaray derbide Fenerbahçe'yi 3-1 mağlup etti",
    "Dolar kuru 35 TL'yi aştı, Merkez Bankası faiz kararını açıkladı",
    "Yapay zeka alanında yeni bir çip geliştirildi",
    "Cumhurbaşkanı yeni kabineyi açıkladı",
]

model.eval()
for text in test_sentences:
    enc = tokenizer(text, max_length=MAX_LEN, padding='max_length', truncation=True, return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1).squeeze()
    pred_id = probs.argmax().item()
    print(f'  [{id2cat[pred_id]:<14}] ({probs[pred_id]*100:5.1f}%) {text[:50]}...')

print('\n' + '=' * 70)
print('TEST COMPLETED SUCCESSFULLY!')
print('=' * 70)
