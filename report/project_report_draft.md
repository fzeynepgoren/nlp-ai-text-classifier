# AI vs Human Metin Sınıflandırma: Proje Rapor Taslağı
*(Aşağıdaki metni kopyalayıp Word/Google Docs'a yapıştırarak "project_report.pdf" (Proje Raporu) dosyanı kolayca oluşturabilirsin)*

---

## 1. Introduction (Giriş)
Büyük Dil Modellerinin (Large Language Models - LLM; örn. GPT-4, Claude vb.) gelişimi, metinsel bilgi üretimi alanında devrim yaratmıştır. Ancak bu modellerin sağladığı benzeri görülmemiş kolaylıkların yanı sıra, akademik dürüstlük, dezenformasyon ve dijital içeriklerin güvenilirliği konusunda ciddi problemler ortaya çıkmıştır. Bir metnin gerçekten bir insan tarafından mı yazıldığını, yoksa yapay zeka tarafından mı üretildiğini ayırabilmek (AI Text Detection), NLP alanında en kritik zorluklardan biri haline gelmiştir. Bu projenin amacı, metinlerin kaynağını tespit eden, eğitim ve profesyonel ağlarda doğruluğu sağlayacak sağlam ve güvenilir bir ikili sınıflandırma (binary classification) modeli geliştirmektir.

## 2. Related Work (İlgili Çalışmalar)
AI tarafından üretilen metinleri tespit etmek zorludur çünkü LLM'ler aslında insan dilindeki örüntüleri kusursuz bir şekilde taklit etmek üzere özel olarak eğitilir(optimize edilir). Erken dönem tespit mekanizmaları genelde "şaşkınlık metrikleri" (perplexity) veya cümle uzunluklarındaki ani değişimler (burstiness) gibi basit yöntemlere bel bağlamıştır. Ne var ki, son nesil yapay zeka sistemleri bu tür basit istatistikleri kolayca aşabilmektedir. Nitekim son araştırmalar, yapay zekanın kullandığı yapısal formülleri çıkartmada Destek Vektör Makineleri (SVM) ve TF-IDF kombinasyonu gibi yüksek boyutlu lexik analizlerin; karmaşık derin öğrenme (Deep Learning) modellerinden bütçe-performans açısından çok daha iyi sonuçlar verebileceğini göstermektedir.

## 3. Dataset Description (Veri Seti Tanımı)
Modelimiz "AI vs Human Text Dataset" tabanlı geniş çaplı bir metin veri setinde eğitilmiştir. Github ve Rapor testi için yaratılan örneklem (dataset_sample.csv), sınıflar arası herhangi bir tahminsel önyargıyı engellemek için mükemmel bir dengeye (100 AI vs 100 İnsan metni) sahiptir.
- **Sınıf 0:** İnsan elinden çıkmış (Human-written)
- **Sınıf 1:** AI tarafından üretilmiş (AI-generated)

Veri seti üzerinde `exploration.ipynb` aracılığıyla yapılan Keşifçi Veri Analizi (EDA) sürecinde, iki sınıfın metin uzunlukları arasında devasa küresel çaplı farklar bulunmamıştır. Ancak İnsan metinlerinin kelime sayı varyanslarının daha dağınık olduğu; AI metinlerinin ise dar bir ortanca değere sıkı sıkıya bağlı olduğu (standart uzunluklar ürettiği) tespit edilmiştir.

## 4. Methodology (Metodoloji)
Bu projenin sınıflandırma altyapısı klasik, sağlam NLP mimarilerinden beslenerek oluşturulmuştur:
- **Pre-processing (Ön İşleme):** Tüm metinlerde standartlaşma sağlamak için NLTK modülü üzerinden bütün harfler küçük harfe dönüştürülmüş, noktalama işaretleri kaldırılmış ve anlam ifade etmeyen bağlaç/edat tarzı İngilizce durdurma kelimeleri (stopwords) silinmiştir.
- **Feature Extraction (Özellik Çıkarımı):** Metinlerin makine tarafından işlenebilmesi için TF-IDF (Terim Frekansı-Ters Doküman Frekansı) kullanılmıştır. Bu teknoloji ile kelimeler sadece geçme sıklığına göre değil, belgenin bütününe kattığı özgün değere göre değerlendirilmiş ve maksimum 5.000 özellik boyutunda sınırlandırılan seyrek bir matris elde edilmiştir.
- **Model Selection (Model Seçimi):** İlk aşamada, anlaşılırlığı ve kontrol gücü yüksek bir temel model olan Lojistik Regresyon (Logistic Regression) tercih edilmiştir. Daha sonrasında ise TF-IDF'in oluşturduğu bu 5.000 boyutlu özellik uzayında iki sınıf arasındaki ayırıcı çizgiyi (hyperplane) çizmeyi matematiksel olarak en iyi başaran Destek Vektör Makineleri (LinearSVC) devreye sokulmuştur.

## 5. Experiments (Deneyler)
Modellerimiz test veri setine karşı büyük ve ağır çaplı bir çapraz doğrulama eğitiminden geçirilmiştir. Burada tasarlanan en vizyoner deneylerden biri; model başarısını metinler "Ön-İşlem (Preprocessing) uygulanmışken" ve "uygulanmamışken (Raw Text)" karşılaştırmak olmuştur.
Yapılan deney büyüleyici bir paradoksu ortaya koymuştur: Ön işleme uygulanmamış modelin doğruluk puanı (%99.10), ön işleme uygulanan versiyonu (%98.90) geride bırakmıştır!
Nedeni gayet açıktır: Geleneksel NLP yaklaşımının aksine, büyük dil modelleri genellikle tamamen kusursuz harf/noktalama kuralları uygular. Yapay Zekanın virgülleri kelimesi kelimesine matematiksel bir hassasiyetle koyuyor oluşu, tespit modelimiz için devasa bir ipucudur (feature). Noktalama işaretlerini ön işleme ile uçurmak, modelimize AI'ı yakalaması için aslında çok yardımcı olan "kusursuz gramer" imzasını elinden almış ve tespiti bir tık zorlaştırmıştır.

## 6. Results (Sonuçlar)
İki model tarafından alınan değerlendirme skorları (test kısmı için), muazzam bir tahmin kapasitesi ve dayanıklılık sunmaktadır:

| Metrik | Lojistik Regresyon | Destek Vektör Ekipmanları (SVM) |
|--------|---------------------|-----------------|
| Accuracy (Doğruluk) | %99.14 | %99.72 |
| Precision (Kesinlik) | %99.14 | %99.72 |
| Recall (Duyarlılık) | %99.14 | %99.72 |
| F1-Score | %99.14 | %99.72 |

*(Karışıklık matrisi sonuçları incelendiğinde, SVM'in False Positive yani gerçek insanlara yapay zeka iftirası atma eğilimini Lojistik regresyona göre çok ciddi oranda azalttığı `results/figures` klasöründe görülmektedir).*
Bununla birlikte karışıklık matrisi (confusion matrix), modelin son derece resmi bir dille yazılmış insan metinlerini yapay zeka üretimi olarak yanlış sınıflandırma eğiliminde olduğunu da açıkça ortaya koymaktadır. Ancak elde edilen bu %99.7'lik yüksek doğruluk oranının (high accuracy), veri setinin nispeten sınırlı boyutu ve dış etkenlerden yalıtılmış kontrollü doğasından kısmen etkilenmiş olma (overfitting) ihtimali göz ardı edilmemelidir.

## 7. Discussion (Tartışma ve İçgörüler)
Projeden alınan hata metinleri ve ağırlık şemaları sonucunda aşağıdaki çıkarımlar elde edilmiştir:
- **Modelin Öğrendiği Farklılıklar (Feature Importance):** Elde edilen lineer özellik katsayılarına bakıldığında AI metinleri "Additionally" (ek olarak), "Conclusion" (sonuç olarak) ve "Essential" (temel/gerekli) kelimelerinin kullanımıyla hemen açığa çıkmaktadır. Bu kelimeler Yapay Zekanın formal, sıkıcı akademik makale yazma eğiliminin yapı taşlarıdır. "Going", "would", "almost" (neredeyse) kelimeleri ise kararsızlığı, net olmayan ve resmiyetten uzak gramer yapısını işaret ettiği için İnsan yazarlarına aittir.
- **Neden Hata Yaptı? (Hata Analizi):** İnsan metnini Yapay Zeka zannederek yanılan False Positive (Yanlış Pozitif) tespitlerinde görülen en büyük şablon; insanların essay (makale) yazarken gereğinden fazla bağlaç ("İlk olarak...", "İkinci noktaysa...", "Sonuçta...") kullanıp çok resmi (robotic) davranmasıdır. Yapay zeka olmasını fark edemediği metinler ise tamamen AI'a "samimi ol", "kişisel konuş" gibi komutlar yedirildiği metinlerdir.

## 8. Conclusion (Sonuç)
Genel olarak, bu çalışmada Yapay Zekaya karşı insani değerleri saptamayı hedefleyen algoritma %99.7 düzeyinde şaşırtıcı ve muazzam bir Accuracy yakalamıştır. Ancak modelimiz bu projede kullanılan veri setinde üstün bir performans sergilese de, algoritmaların gerçek dünyadaki çok daha çeşitli ve bağımsız metin kaynaklarına başarıyla genellenebilme (generalization) düzeyi halihazırda belirsizliğini korumaktadır.

Bu projenin bizlere kazandırdığı asıl başarı; makine öğrenimi sisteminin AI izlerini saptarken yapısal bağlaçlarını referans alması ya da AI'ların en çok "Hiç Gramer Hatası Yapmama" gerçeğiyle yakalanması tespiti olmuştur. Doğrusal modeller (özellikle SVM), bu sorunu devasa özellik uzaylarında lineer hesaplamalarla inanılmaz iyi ayırabilmektedir. Gelecek çalışmalarda, daha gelişmiş Transformer ve Deep Learning modelleri kullanılarak yapay zekanın "samimi olma" gibi gayri-resmi dil denemelerinin de tamamen çözülmesi planlanmaktadır.
