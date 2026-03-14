"""
Türkçe Haber Sınıflandırma — DistilBERT Fine-Tuning
====================================================
Veri: batubayk/TR-News
Model: dbmdz/distilbert-base-turkish-cased
"""

import os
import csv
import json
import random
import numpy as np
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import classification_report, accuracy_score, f1_score

# ─── Ayarlar ───────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_NAME = "dbmdz/distilbert-base-turkish-cased"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
SEED = 42

csv.field_size_limit(10**7)

# ─── Kategori eşleme: 122 ham etiketi 10 ana kategoriye indir ───
CATEGORY_MAP = {
    # Türkiye / Gündem / Siyaset
    "Türkiye": "Gündem", "turkiye": "Gündem", "Gündem": "Gündem",
    "siyaset": "Gündem", "Polemik": "Gündem",
    "2019 Yerel Seçim": "Gündem", "23 Haziran seçimleri": "Gündem",
    "secim_2015": "Gündem", "secim": "Gündem", "Seçim": "Gündem",
    "Seçim 2015": "Gündem", "yerel_yonetimler": "Gündem",
    "H. Bunu Konuşuyor": "Gündem", "15 Temmuz davaları": "Gündem",
    "Haberin Var Mı?": "Gündem", "Kadına Şiddet": "Gündem",
    "sokak": "Gündem", "Ortak Gelecek": "Gündem",

    # Dünya
    "Dünya": "Dünya", "dunya": "Dünya", "Dünyadan": "Dünya",
    "BBC": "Dünya", "english": "Dünya",

    # Spor
    "Spor": "Spor", "spor": "Spor",
    "futbol": "Spor", "Futbol": "Spor",
    "basketbol": "Spor", "Basketbol": "Spor",
    "Voleybol": "Spor", "Tenis": "Spor", "Hentbol": "Spor",
    "Motor Sporları": "Spor", "Olimpiyat": "Spor",
    "diger_sporlar": "Spor", "İddaa": "Spor",
    "EURO 2016": "Spor", "2016 Avrupa Şampiyonası": "Spor",
    "2018 Dünya Kupası": "Spor", "Dünya Kupası 2018": "Spor",
    "2016 Rio Olimpiyatları": "Spor", "foto_spor": "Spor",
    "2018 DK Tarihçe": "Spor",

    # Ekonomi
    "Ekonomi": "Ekonomi", "ekonomi": "Ekonomi",
    "Makro Ekonomi": "Ekonomi", "Para": "Ekonomi",
    "Emlak": "Ekonomi", "Enerji": "Ekonomi",
    "Döviz": "Ekonomi", "Borsa": "Ekonomi", "Altın": "Ekonomi",
    "Sigorta": "Ekonomi", "Sosyal Güvenlik": "Ekonomi",
    "Bitcoin": "Ekonomi", "Girişimcilik": "Ekonomi",
    "Akdeniz Ekonomi Forumu": "Ekonomi",
    "Ege Ekonomik Forum 2017": "Ekonomi",
    "AIRPORT": "Ekonomi",

    # Teknoloji & Bilim
    "Teknoloji": "Teknoloji", "bilim_ve_teknoloji": "Teknoloji",
    "uzay": "Teknoloji",

    # Sağlık
    "Sağlık": "Sağlık", "saglik": "Sağlık",

    # Kültür & Sanat & Magazin
    "Sanat": "Kültür-Sanat", "Kültür-Sanat": "Kültür-Sanat",
    "kultur-sanat": "Kültür-Sanat", "foto_kultur_sanat": "Kültür-Sanat",
    "Müzik": "Kültür-Sanat", "konser": "Kültür-Sanat",
    "Televizyon": "Kültür-Sanat", "sinema": "Kültür-Sanat",
    "kitap": "Kültür-Sanat", "Biyografi": "Kültür-Sanat",
    "Fiskos": "Kültür-Sanat", "Magazin": "Kültür-Sanat",
    "Cemiyet Hayatı": "Kültür-Sanat", "Medya": "Kültür-Sanat",
    "Röportajlar": "Kültür-Sanat", "Özel Röportajlar": "Kültür-Sanat",
    "Özel": "Kültür-Sanat", "etkinlik": "Kültür-Sanat",

    # Eğitim
    "Eğitim": "Eğitim", "egitim": "Eğitim",

    # Yaşam
    "Yaşam": "Yaşam", "yasam": "Yaşam", "İş-Yaşam": "Yaşam",
    "calisma_yasami": "Yaşam", "Seyahat": "Yaşam", "gezi": "Yaşam",
    "Turizm": "Yaşam", "cevre": "Yaşam",
    "surdurulebilir_yasam": "Yaşam", "Ramazan": "Yaşam",
    "Tarifler": "Yaşam", "Yemek Yapma": "Yaşam",
    "Nasıl Yapılır": "Yaşam", "Alışveriş": "Yaşam",
    "Güvenli Alışveriş": "Yaşam", "Moda": "Yaşam", "moda": "Yaşam",
    "Evde Ek İş": "Yaşam", "Sıfır Atık": "Yaşam",
    "Merak Edilenler": "Yaşam", "Nedir": "Yaşam",
    "Astroloji": "Yaşam", "astroloji": "Yaşam",
    "Şipşak": "Yaşam", "Takılın": "Yaşam",
    "MOTY 2019": "Yaşam",

    # Otomobil
    "Otomobil": "Otomobil", "otomobil": "Otomobil",
    "Savunma Sanayi": "Otomobil",
}

# Kategorilere dönüşmeyen, çok küçük veya boş olanlar atlanır
SKIP_TOPICS = {"", "diger", "Diğer", "General", "Yazı Dizisi",
               "yazi_dizileri", "pazar_yazilari",
               "cumhuriyet_ege", "cumhuriyet_pazar"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Veri yükleme ──────────────────────────────────────────
def load_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            topic = row.get("topic", "").strip()
            if topic in SKIP_TOPICS:
                continue
            category = CATEGORY_MAP.get(topic)
            if category is None:
                continue
            title = (row.get("title") or "").strip()
            abstract = (row.get("abstract") or "").strip()
            text = f"{title}. {abstract}" if abstract else title
            if len(text) < 10:
                continue
            rows.append({"text": text, "category": category})
    return rows


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def evaluate(model, dataloader, device):
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    return np.array(preds_all), np.array(labels_all)


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1) Veri yükle
    print("\n[1/6] Veri yükleniyor...")
    train_rows = load_csv(DATA_DIR / "train.csv")
    val_rows = load_csv(DATA_DIR / "validation.csv")
    test_rows = load_csv(DATA_DIR / "test.csv")
    print(f"  Train: {len(train_rows):,}  Val: {len(val_rows):,}  Test: {len(test_rows):,}")

    # Kategori -> id
    categories = sorted(set(r["category"] for r in train_rows))
    cat2id = {c: i for i, c in enumerate(categories)}
    id2cat = {i: c for c, i in cat2id.items()}
    num_labels = len(categories)
    print(f"  Kategoriler ({num_labels}): {categories}")

    # Label mapping kaydet
    label_map = {"cat2id": cat2id, "id2cat": {str(k): v for k, v in id2cat.items()}}
    with open(MODEL_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    # Dağılım
    dist = Counter(r["category"] for r in train_rows)
    for c in categories:
        print(f"    {c}: {dist[c]:,}")

    # 2) Tokenizer
    print("\n[2/6] Tokenizer yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 3) Datasets
    print("[3/6] Dataset'ler hazırlanıyor...")
    train_texts = [r["text"] for r in train_rows]
    train_labels = [cat2id[r["category"]] for r in train_rows]
    val_texts = [r["text"] for r in val_rows]
    val_labels = [cat2id[r["category"]] for r in val_rows]
    test_texts = [r["text"] for r in test_rows]
    test_labels = [cat2id[r["category"]] for r in test_rows]

    train_ds = NewsDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_ds = NewsDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    test_ds = NewsDataset(test_texts, test_labels, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, num_workers=0)

    # 4) Model
    print("\n[4/6] Model yükleniyor...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    )
    model.to(device)

    # 5) Eğitim
    print("\n[5/6] Eğitim başlıyor...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, WARMUP_STEPS, total_steps)

    best_f1 = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if (step + 1) % 500 == 0:
                avg = total_loss / (step + 1)
                print(f"  Epoch {epoch+1}/{EPOCHS} | Step {step+1}/{len(train_loader)} | Loss: {avg:.4f}")

        # Validation
        preds, labels_arr = evaluate(model, val_loader, device)
        acc = accuracy_score(labels_arr, preds)
        f1 = f1_score(labels_arr, preds, average="macro")
        print(f"  Epoch {epoch+1} Val — Acc: {acc:.4f} | F1-macro: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            print(f"  ✓ En iyi model kaydedildi (F1: {f1:.4f})")

    # 6) Test
    print("\n[6/6] Test sonuçları...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    preds, labels_arr = evaluate(model, test_loader, device)
    target_names = [id2cat[i] for i in range(num_labels)]
    print(classification_report(labels_arr, preds, target_names=target_names, digits=4))

    test_acc = accuracy_score(labels_arr, preds)
    test_f1 = f1_score(labels_arr, preds, average="macro")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1-macro: {test_f1:.4f}")
    print(f"\nModel kaydedildi: {MODEL_DIR}")


if __name__ == "__main__":
    main()
