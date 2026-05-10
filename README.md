# AI vs Human Metin Sınıflandırma (NLP Projesi)

##  Proje Genel Bakış
Bu proje, makine öğrenmesi teknikleri kullanılarak belirli bir metnin insan tarafından mı yazıldığını yoksa bir yapay zeka modeli tarafından mı üretildiğini tespit etmeyi amaçlamaktadır. İstanbul Sabahattin Zaim Üniversitesi BIM432 Natural Language Processing dersi için geliştirilmiştir.

## Veri Seti
- **Ana Veri Seti:** Model eğitim ve değerlendirme hattı `data/AI_Human.csv` dosyası üzerinden çalışmaktadır.
- **Dengeli Alt Küme:** GitHub üzerinde hızlı inceleme ve örnekleme için `data/dataset_sample.csv` dosyası da eklenmiştir (100 İnsan / 100 AI metni).
- **Kaynak:** Kaggle AI vs Human Text Dataset

##  Metodoloji

**1. Ön İşleme (Preprocessing):**
- Küçük harfe çevirme
- Noktalama işaretlerinin kaldırılması
- NLTK `word_tokenize` ile kelime ayrıştırma
- Etkisiz kelimelerin (Stopword) NLTK ile çıkarılması

**2. Özellik Çıkarımı (Feature Extraction):**
- TF-IDF (Term Frequency-Inverse Document Frequency) kullanılmıştır.
- Özellik uzayı en etkili 5.000 özellikle sınırlandırılmıştır.
- Unigram + bigram özellikleri (`ngram_range=(1, 2)`) eklenmiştir; böylece tek kelimelerin yanında iki kelimelik kalıplar da modele dahil edilmiştir.

**3. Modeller:**
- Lojistik Regresyon (Temel Model / Baseline)
- Destek Vektör Makineleri (Support Vector Machine - LinearSVC)

**4. Eğitim / Test Ayrımı:**
- Her iki model artık aynı train/test bölünmesi üzerinde değerlendirilmektedir.
- `split_data()` fonksiyonu ile `stratify=y` kullanılarak sınıf oranları korunmuştur.

**5. Cross-Validation:**
- Overfitting kontrolü için 5-Fold Stratified Cross-Validation eklenmiştir.
- Cross-validation çıktıları `results/figures/cross_validation.png` ve `results/figures/cross_validation.txt` dosyalarına kaydedilmektedir.

## 📈 Sonuçlar
| Model | Doğruluk (Accuracy) | F1-Skoru |
|-------|---------------------|----------|
| **Lojistik Regresyon** | %99.34 | %99.34 |
| **SVM (LinearSVC)** | **%99.77** | **%99.77** |

### Önceki Sonuçlarla Karşılaştırma
| Metrik | LR Önceki | LR Güncel | SVM Önceki | SVM Güncel |
|--------|-----------|-----------|------------|------------|
| Accuracy | %99.14 | **%99.34** | %99.72 | **%99.77** |
| Toplam Hata | 840 | **642** | 269 | **228** |
| False Positives | 267 | **172** | 59 | **42** |
| False Negatives | 573 | **470** | 210 | **186** |

##  Temel Çıkarımlar (Key Insights)
- **Model Verimliliği:** SVM, TF-IDF tarafından oluşturulan yüksek boyutlu (high-dimensional) dizi verilerini ele almadaki matematiksel üstünlüğü sayesinde Lojistik Regresyon'u geride bırakmıştır.
- **Yapay Zeka Karakteristiği:** AI tarafından üretilen metinler, şablonlara bağlı ciddi bir yapısal katılık (rigidity) gösterir. "Additionally" (ek olarak), "conclusion" (sonuç olarak) gibi resmi akademik kelimelere güçlü şekilde bağımlıdır.
- **İnsan Karakteristiği:** İnsan eliyle yazılan metinler; sözdizimsel olarak çok daha değişken bir yapıya sahiptir. "Almost" (neredeyse), "would" (yapardı) gibi kesin olmayan (informal) ifadeleri bol barındırır.

##  Hata Analizi (Error Analysis)
Modellerin yanlış tahminleri `perform_error_analysis()` fonksiyonu ile otomatik olarak incelenmektedir:
- **Yanlış Pozitifler (False Positives):** Makale formatına sıkı sıkıya bağlı kalarak resmi, yapısal şablonlar kullanan insan metinleri bazen AI sanılmaktadır.
- **Yanlış Negatifler (False Negatives):** Bilerek resmi olmayan gündelik bir dil kullanan, kişisel bir giriş yapan ("benim adım...") veya noktalama işareti hataları barındıran AI metinleri, modeli yanıltıp İnsan oyu alabilmektedir. 

Detaylı hata analiz dosyaları:
- `results/error_analysis_lr.txt`
- `results/error_analysis_svm.txt`

##  Kaydedilen Model ve Tahmin Script'i
Pipeline çalıştırıldığında en iyi performans veren SVM modeli ve TF-IDF vectorizer diske kaydedilir:
- `results/svm_model.joblib`
- `results/tfidf_vectorizer.joblib`

Yeni bir metin üzerinde tahmin yapmak için `predict.py` kullanılabilir.

## Projeyi Çalıştırma

1. **Gereksinimlerin Kurulması:**
```bash
pip install -r requirements.txt
```

2. **Tüm Projenin (Pipeline) Çalıştırılması:**
```bash
python -m src.run
```
Bu komut:
- Veriyi yükler ve sınıf dağılımını gösterir
- Metinleri ön işler
- Unigram + bigram TF-IDF özelliklerini çıkarır
- Veriyi stratified olarak train/test şeklinde böler
- Logistic Regression ve SVM modellerini aynı test seti üzerinde değerlendirir
- Confusion matrix, feature importance, model karşılaştırma ve cross-validation grafiklerini üretir
- Error analysis raporlarını oluşturur
- SVM modelini ve TF-IDF vectorizer'ı `results/` klasörüne kaydeder

3. **Yeni Metin Tahmini:**
```bash
python -m src.predict
```

Bu script kaydedilmiş model varsa doğrudan yükler; yoksa modeli yeniden eğitip kaydeder.

##  Sınırlandırmalar (Limitations)
Modelimizin elde ettiği %99.77 başarı oranı yüksek olsa da, belirli akademik sınırlar çerçevesinde değerlendirilmelidir:
- **Veri Seti Bağımlılığı:** Yüksek doğruluk oranı, kullanılan Kaggle veri setinin yazım örüntülerine bağlı olabilir. Farklı kaynaklardan gelen metinlerde performans yeniden test edilmelidir.
- **Stil Odaklı Tespit (Style-based Detection):** Model, kavramsal veya bilgisel derinlik yerine yazarın yapısal stiline (resmi/samimi kelime örgüsü) odaklanmaktadır.
- **Gelecek Modellerin Tehdidi:** Yüksek doğruluk oranı mevcut verilere dayanır. Gelecek LLM'ler (yeni nesil yapay zekalar), şu andaki yapısal izleri ve "robotik resmiyetleri" silerek bu klasik tespit mekanizmasını atlatabilir (bypass detection).

## 📁 Proje Klasör Yapısı

```text
project/
├── data/
│   ├── AI_Human.csv
│   └── dataset_sample.csv
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── run.py
│   └── predict.py
├── results/
│   ├── figures/
│   │   ├── class_distribution.png
│   │   ├── cm_logistic_regression.png
│   │   ├── cm_svm.png
│   │   ├── cross_validation.png
│   │   ├── cross_validation.txt
│   │   ├── fi_logistic_regression.png
│   │   ├── fi_svm.png
│   │   ├── model_comparison.png
│   │   ├── text_length_by_class.png
│   │   └── text_length_distribution.png
│   ├── metrics.txt
│   ├── error_analysis.txt
│   ├── error_analysis_lr.txt
│   ├── error_analysis_svm.txt
│   ├── svm_model.joblib
│   └── tfidf_vectorizer.joblib
├── report/
│   └── project_report.pdf
├── requirements.txt
└── README.md
```

##  Rapor (Report)
Proje raporunun PDF kopyası `report/project_report.pdf` konumunda yer almaktadır.
