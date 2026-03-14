"""
Classifier modülü — Eğitilmiş DistilBERT ile haber sınıflandırma
"""

import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = Path(__file__).parent / "model"


class NewsClassifier:
    def __init__(self, model_dir=None):
        model_dir = Path(model_dir) if model_dir else MODEL_DIR

        with open(model_dir / "label_map.json", "r", encoding="utf-8") as f:
            label_map = json.load(f)
        self.id2cat = {int(k): v for k, v in label_map["id2cat"].items()}
        self.cat2id = label_map["cat2id"]

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        """Tek bir metni sınıflandır. Kategori ve güven skoru döndür."""
        enc = self.tokenizer(
            text, max_length=256, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze()

        top_idx = torch.argmax(probs).item()
        return {
            "category": self.id2cat[top_idx],
            "confidence": round(probs[top_idx].item(), 4),
            "all_scores": {
                self.id2cat[i]: round(p.item(), 4) for i, p in enumerate(probs)
            },
        }

    def predict_batch(self, texts: list) -> list:
        """Birden fazla metni sınıflandır."""
        return [self.predict(t) for t in texts]

    @property
    def categories(self):
        return sorted(self.id2cat.values())
