# 🗞️ Türkçe Haber Sınıflandırma Projesi

## Proje Özeti
DistilBERT tabanlı Türkçe haber sınıflandırma modeli.
122+ ham kategoriyi 10 ana kategoriye indirgeyen fine-tuning pipeline.

## Donanım ve Ortam
- **GPU:** RTX 4070 Laptop — 8 GB VRAM (dizüstü versiyonu, masaüstü 4070'ten yaklaşık %15 daha yavaş)
- **OS:** Windows + WSL
- **Python:** 3.11.9
- **PyTorch:** CUDA 12.4 sürümü kurulu olmalı (`--index-url https://download.pytorch.org/whl/cu124`)
- **Ortam:** `.venv` klasöründe sanal ortam

## Kritik Kural: CUDA Kontrolü
Her oturumda ilk iş CUDA'nın aktif olduğunu doğrula:
```python
import torch
print(torch.cuda.is_available())  # True olmalı
print(torch.cuda.get_device_name(0))  # RTX 4070 görmeli
```
`False` dönüyorsa eğitimi başlatma — CPU'da 274K örnekle saatler sürer.

## Klasör Yapısı
```
news/
├── data/
│   ├── train.csv        # 274,106 satır
│   ├── validation.csv   # 14,415 satır
│   └── test.csv         # 15,192 satır
├── model/
│   ├── best/            # En yüksek val F1'li model buraya kaydedilir
│   ├── checkpoint-epoch-1/   # Her epoch tam state (resume için)
│   ├── checkpoint-epoch-2/
│   ├── label_map.json   # cat2id / id2cat mapping
│   └── training_info.json
├── logs/
│   └── data_hashes.json # Veri bütünlüğü için MD5 hash'ler
├── train_final.ipynb    # Ana notebook — bunu kullan
├── training_curves.png
├── confusion_matrix.png
├── error_analysis.csv   # Yanlış tahminler
└── requirements.txt
```

## Model ve Veri
- **Model:** `dbmdz/distilbert-base-turkish-cased` (68M parametre)
- **Veri:** `batubayk/TR-News` — HuggingFace'ten indiriliyor
- **Görev:** Sequence Classification (10 sınıf)
- **Tokenizer:** Aynı model adıyla yükleniyor, değiştirme

## 10 Kategori
| ID | Kategori | Train Sayısı |
|----|----------|--------------|
| 0 | Dünya | 40,952 |
| 1 | Ekonomi | 20,433 |
| 2 | Eğitim | 5,082 |
| 3 | Gündem | 122,827 ← en büyük |
| 4 | Kültür-Sanat | 11,536 |
| 5 | Otomobil | 1,513 ← en küçük |
| 6 | Sağlık | 16,175 |
| 7 | Spor | 30,871 |
| 8 | Teknoloji | 7,011 |
| 9 | Yaşam | 17,706 |

Gündem/Otomobil oranı ~81:1 → WeightedRandomSampler bunu dengeler, elle müdahale etme.

## Hiperparametreler (Sabit Tut)
```python
MAX_LEN      = 256    # Değiştirme — başlık+özet için yeterli
BATCH_SIZE   = 32     # 8 GB VRAM sınırı, artırma
GRAD_ACCUM   = 2      # Efektif batch = 64
EPOCHS       = 4
LR           = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
LABEL_SMOOTH = 0.05
MAX_GRAD_NORM = 1.0
USE_AMP      = True   # fp16, RTX 4070 destekler — kapat
```
VRAM sıkışırsa `BATCH_SIZE=16, GRAD_ACCUM=4` yap — efektif batch aynı kalır.

## Düzeltilmiş Category Mapping Kararları
Bunlar bilinçli kararlar, geri alma:
- `"Savunma Sanayi"` → `"Gündem"` (Otomobil'den taşındı — daha mantıklı)
- `"AIRPORT"` → SKIP listesinde (anlamlı kategori değil)
- `"BBC"`, `"english"` → `"Dünya"` (doğru)

## Checkpoint Sistemi
Her epoch sonunda `model/checkpoint-epoch-N/` altına tam state kaydedilir:
- `pytorch_model.bin` — ağırlıklar
- `training_state.pt` — optimizer, scheduler, history
- En iyi F1'de `model/best/` da güncellenir

**Eğitim yarıda kesilirse resume:**
```python
epoch, history = FullCheckpoint.resume(
    Path("model/checkpoint-epoch-2"),
    model, optimizer, scheduler
)
```

## Beklenen Performans (RTX 4070 Laptop)
- Epoch süresi: ~12-18 dakika
- 4 epoch toplam: ~60-75 dakika
- Beklenen test F1-macro: %88-92
- VRAM kullanımı: ~5-6 GB (8 GB limitin altında)

## W&B Entegrasyonu (Opsiyonel)
`Config` sınıfında `use_wandb=True` yapınca aktif olur.
Önce terminal'de: `wandb login`
Proje adı: `tr-news-distilbert`

## Dikkat Edilecekler
- `num_workers=4` Windows'ta sorun çıkarırsa `0` yap
- `pin_memory=True` sadece CUDA varsa aktif, elle değiştirme
- `persistent_workers` sadece `num_workers > 0` ise True olmalı
- `error_analysis.csv` her test sonrası üzerine yazılır, önemliyse kopyala
- Confusion matrix'te Gündem/Dünya ve Ekonomi/Gündem karışması normaldir

## Sık Kullanılan Komutlar
```bash
# Projeyi başlat
cd "C:\Users\LENOVO\Desktop\Aktif Projeler\news"
claude

# Sanal ortamı aktifleştir (WSL'de)
source .venv/bin/activate

# CUDA kurulumu (bir kez)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Bağımlılıkları kur
pip install -r requirements.txt
```

## Yapılacaklar / Eksikler
- [ ] Hyperparameter search (Optuna) henüz eklenmedi
- [ ] ONNX export — production deploy için
- [ ] DVC ile veri versiyonlama
- [ ] Docker container
- [ ] Per-class threshold optimizasyonu (Otomobil/Eğitim için)
