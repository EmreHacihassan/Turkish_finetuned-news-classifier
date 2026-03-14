# Türkçe Haber Sınıflandırma Projesi — Plan

## Amaç
DistilBERT modelini Türkçe haber metinleri üzerinde fine-tune ederek haber kategorisi tahmini yapmak.
Web'den ücretsiz scraping ile çekilen güncel haberleri bu model ile otomatik sınıflandırıp
bir web arayüzünde kategoriye göre listelemek.

---

## 1. Veri Seti
- **Kaynak:** `batubayk/TR-News` (Hugging Face)
- **İndirme:** `huggingface-cli download batubayk/TR-News --repo-type=dataset --local-dir ./data`
- **Format:** Türkçe haber başlık + metin + kategori etiketi
- **Hedef:** Kategoriler arasında train/val/test bölünmesi

## 2. Ortam & Gereksinimler
- Python 3.11.9 (mevcut)
- torch 2.10.0 (mevcut)
- transformers 4.57.6 (mevcut)
- scikit-learn 1.8.0 (mevcut)
- Flask 3.1.2 (mevcut)
- beautifulsoup4 4.14.3 (mevcut)
- **Kurulacak:** datasets, accelerate

## 3. Fine-Tuning Pipeline
1. Veri setini yükle ve incele (kategori dağılımı, örnek sayısı)
2. Tokenizer: `dbmdz/distilbert-base-turkish-cased`
3. Train/val/test split (%80/%10/%10)
4. Eğitim parametreleri:
   - Epochs: 3
   - Batch size: 16
   - Learning rate: 2e-5
   - Weight decay: 0.01
   - Warmup steps: 500
5. Metrikler: accuracy, f1-macro, classification report
6. Model kayıt: `./model/` klasörüne

## 4. Haber Scraper
- **Kaynak:** Ücretsiz, JS gerektirmeyen haber siteleri (RSS feed veya HTML parse)
- **Yöntem:** requests + BeautifulSoup
- **Çıktı:** Başlık, özet/metin, kaynak URL, tarih
- **Sınıflandırma:** Fine-tuned model ile kategori tahmini

## 5. Web Arayüzü (Flask)
- Ana sayfa: Kategoriler sidebar/tab olarak
- Kategori seçince: O kategorideki haberler listelenir
- Her haber kartı: Başlık, özet, kaynak link, tahmin edilen kategori + güven skoru
- Tasarım: Minimalist, Türkçe, responsive
- Taze haberleri çekip sınıflandıran "Haberleri Güncelle" butonu

## 6. Dosya Yapısı
```
news/
├── PLAN.md               ← Bu dosya
├── data/                 ← İndirilen veri seti
├── model/                ← Eğitilmiş model
├── train.py              ← Fine-tuning scripti
├── scraper.py            ← Haber çekme modülü
├── classifier.py         ← Model inference modülü
├── app.py                ← Flask web uygulaması
├── templates/
│   └── index.html        ← Ana sayfa şablonu
└── static/
    └── style.css         ← Stil dosyası
```

## 7. Adımlar
- [x] Plan oluştur
- [ ] datasets & accelerate kur
- [ ] Veri setini indir
- [ ] Veri setini incele
- [ ] Fine-tuning scriptini yaz
- [ ] Modeli eğit
- [ ] Scraper yaz
- [ ] Classifier modülü yaz
- [ ] Flask arayüzü yaz
- [ ] Test et
