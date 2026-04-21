# AI vs Human Metin Sınıflandırma (NLP Projesi)

## 📌 Proje Genel Bakış
Bu proje, makine öğrenmesi teknikleri kullanılarak belirli bir metnin insan tarafından mı yazıldığını yoksa bir yapay zeka modeli tarafından mı üretildiğini tespit etmeyi amaçlamaktadır. İstanbul Sabahattin Zaim Üniversitesi BIM432 Natural Language Processing dersi için geliştirilmiştir.

## 📊 Veri Seti
- **Dengeli Alt Küme:** GitHub yükleme sınırlarını aşmamak ve modeller için tarafsız bir örneklem göstermek adına dengeli bir veri seti (`dataset_sample.csv` - 100 İnsan / 100 AI metni) repo'ya eklenmiştir.
- **Tam Model Eğitimi:** Gerçek klasifikasyon hattı (pipeline) muazzam büyüklükte bir ana veri setinde eğitilmiştir.
- **Kaynak:** Kaggle AI vs Human Text Dataset

## ⚙️ Metodoloji

**1. Ön İşleme (Preprocessing):**
- Küçük harfe çevirme
- Noktalama işaretlerinin kaldırılması
- Etkisiz kelimelerin (Stopword) NLTK ile çıkarılması

**2. Özellik Çıkarımı (Feature Extraction):**
- TF-IDF (Term Frequency-Inverse Document Frequency) kullanılarak kelime uzayı en etkili 5.000 özellikle sınırlandırılmıştır.

**3. Modeller:**
- Lojistik Regresyon (Temel Model / Baseline)
- Destek Vektör Makineleri (Support Vector Machine - LinearSVC)

## 📈 Sonuçlar
| Model | Doğruluk (Accuracy) | F1-Skoru |
|-------|---------------------|----------|
| **Lojistik Regresyon** | %99.14 | %99.14 |
| **SVM (LinearSVC)** | **%99.72** | **%99.72** |

## 🔍 Temel Çıkarımlar (Key Insights)
- **Model Verimliliği:** SVM, TF-IDF tarafından oluşturulan yüksek boyutlu (high-dimensional) dizi verilerini ele almadaki matematiksel üstünlüğü sayesinde Lojistik Regresyon'u geride bırakmıştır.
- **Yapay Zeka Karakteristiği:** AI tarafından üretilen metinler, şablonlara bağlı ciddi bir yapısal katılık (rigidity) gösterir. "Additionally" (ek olarak), "conclusion" (sonuç olarak) gibi resmi akademik kelimelere güçlü şekilde bağımlıdır.
- **İnsan Karakteristiği:** İnsan eliyle yazılan metinler; sözdizimsel olarak çok daha değişken bir yapıya sahiptir. "Almost" (neredeyse), "would" (yapardı) gibi kesin olmayan (informal) ifadeleri bol barındırır.

## 🧪 Hata Analizi (Error Analysis)
Modellerin yanlış tahminlerinin (`evaluate_model.py` üzerinden) derinlemesine incelenmesi şu bulguları ortaya çıkarmıştır:
- **Yanlış Pozitifler (False Positives):** Makale formatına sıkı sıkıya bağlı kalarak resmi, yapısal şablonlar kullanan insan metinleri bazen AI sanılmaktadır.
- **Yanlış Negatifler (False Negatives):** Bilerek resmi olmayan gündelik bir dil kullanan, kişisel bir giriş yapan ("benim adım...") veya noktalama işareti hataları barındıran AI metinleri, modeli yanıltıp İnsan oyu alabilmektedir. 

*(Detaylı hata analiz logları `results/error_analysis.txt` dosyasında bulunabilir)*

## 🚀 Projeyi Çalıştırma

1. **Gereksinimlerin Kurulması:**
```bash
pip install -r requirements.txt
```

2. **Tüm Projenin (Pipeline) Çalıştırılması:**
```bash
python run.py
```
*(Bu komut veriyi ön işler, TF-IDF özelliklerini çıkarır, modelleri eğitir, değerlendirir ve grafikleri/matrisleri otomatik olarak `results` altındaki `figures/` klasörüne kaydeder).*

## ⚠️ Sınırlandırmalar (Limitations)
Modelimizin elde ettiği %99.7 başarı oranı muazzam olsa da, belirli akademik sınırlar çerçevesinde değerlendirilmelidir:
- **Küçük ve Sınırlı Veri Seti (Small Dataset):** Model dar ve dış etkenlerden yalıtılmış bir örneklem üzerinde eğitilmiştir. Dolayısıyla yüksek Accuracy, kısmen overfitting (aşırı öğrenme) veya veri benzerliğinden kaynaklanabilir.
- **Stil Odaklı Tespit (Style-based Detection):** Model, kavramsal veya bilgisel derinlik yerine yazarın yapısal stiline (resmi/samimi kelime örgüsü) odaklanmaktadır.
- **Gelecek Modellerin Tehdidi:** Yüksek doğruluk oranı mevcut verilere dayanır. Gelecek LLM'ler (yeni nesil yapay zekalar), şu andaki yapısal izleri ve "robotik resmiyetleri" silerek bu klasik tespit mekanizmasını atlatabilir (bypass detection).

## 📁 Proje Klasör Yapısı

```text
project/
├── data/
│   └── dataset_sample.csv
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── train_model.py
│   └── evaluate_model.py
├── results/
│   ├── figures/
│   │   ├── cm_logistic_regression.png
│   │   ├── cm_svm.png
│   │   ├── fi_logistic_regression.png
│   │   ├── fi_svm.png
│   │   └── model_comparison.png
│   ├── metrics.txt
│   └── error_analysis.txt
├── report/
│   └── project_report.pdf
├── requirements.txt
└── README.md
```

## 📄 Rapor (Report)
Tartışma parametrelerini de içeren tam proje raporunun PDF kopyası `/report/project_report.pdf` dosyasındadır.
