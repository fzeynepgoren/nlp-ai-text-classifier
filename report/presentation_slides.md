# 🎤 PRESENTATION SLIDE OUTLINE
*(Sunum sırasında hangi slayta neleri koyman gerektiğini ve konuşurken bahsetmen gereken "vurucu cümleleri" içerir)*

---

## Slide 1: Title Screen
- **Title:** AI vs Human Text Detection 
- **Course:** BIM432 NLP Project
- **Group Members:** [Adınız Soyadınız]

## Slide 2: Problem Overview
- **Bullet points:**
  - AI tools (ChatGPT, Claude) are everywhere.
  - Hard to tell what is real and what is generated.
  - Big problem for education (cheating) and deepfakes.
- **Konuşma Özeti:** "Bugün ChatGPT gibi araçlar yüzünden metinlerin kaynağını bilmek imkansızlaştı. Bizim amacımız, bir yazının insan elinden mi yoksa bir algoritmanın çıktısından mı geldiğini yakalayan bir dedektif modeli kurmak."

## Slide 3: Dataset Description
- **Bullet points:**
  - Used "AI vs Human Text Dataset" from Kaggle.
  - Balanced classes avoiding baseline prediction bias.
  - **Class 0:** Human | **Class 1:** AI
- **Görsel:** `results/figures/class_distribution.png` (Dağılım grafiğini ekle)

## Slide 4: Our Method (Pipeline)
- **Bullet points:**
  - **Preprocessing:** NLTK ile lowercase, punctuation removal, stopword deletion.
  - **Feature Extraction:** TF-IDF ile 5000 feature'ı vektörize etik.
  - **Models:** Logistic Regression (Baseline) vs SVM (LinearSVC). 
- **Konuşma Özeti:** "Önce veriyi temizledik, sonra kelimelerin döküman içindeki önemini modellemek için TF-IDF kullandık. İki güçlü lineer sınıflandırıcı kurduk."

## Slide 5: Results
- **Bullet points:**
  - Logistic Regression Accuracy: 99.14%
  - SVM Accuracy: 99.72% 🏆
- **Görseller:** Slayda `results/figures/model_comparison.png` ve `results/figures/cm_svm.png` (Confusion Matrix) grafiklerini yan yana koy.
- **Konuşma Özeti:** "İki model de harika çalıştı ancak SVM 5000 boyutlu özellik uzayında (feature space) çizgi çekerken çok daha başarılı olduğu için kazandı."

## Slide 6: 🔥 Insights & Discoveries (FARK YARATAN SLAYT)
- **Bullet points:**
  - **Feature Importance:** AI uses "Additionally", "Conclusion". Humans use "almost", "going", "would".
  - **The Preprocessing Paradox:** Stripping punctuation actually lowered accuracy slightly (AI's perfect grammar is its signature).
- **Görsel:** `results/figures/fi_svm.png` (Önemli kelimeler grafiğini koy)
- **Konuşma Özeti (Burada hocayı etkiliyorsun!):** "En şaşırtıcı keşfimiz şuydu: AI hata yaptığı için değil, tam tersine **HİÇ HATA YAPMADIĞI VE ROBOTİK KALIPLAR KULLANDIĞI İÇİN** yakalanıyor. Ayrıca noktalama işaretlerini silmemenin tespiti daha kolaylaştırdığını fark ettik, çünkü AI'ın kusursuz virgül kullanımı bizim için dev bir ipucuydu."

## Slide 7: Error Analysis (Why we failed?)
- **Bullet points:**
  - False Positives: Formal, "boring" human essays classified as AI.
  - False Negatives: AI instructed to use "informal, subjective" tone.
- **Konuşma Özeti:** "Peki model nerede patladı? Bir öğrenci eğer çok soğuk, resmi ve kalıp cümlelerle paragraf yazarsa model o öğrenciyi AI sanıyor. Tam tersi AI'a 'hikaye anlatırmış gibi konuş' emri verirseniz model onun İnsan olduğuna kanabiliyor."

## Slide 8: Conclusion
- **Bullet points:**
  - Highly accurate pipeline developed.
  - Stylistic rigidity is the key classifier.
- **Konuşma Özeti:** "Sonuç olarak, projemiz %99.7 doğruluk oranına ulaştı. Gelecekte Transformer (BERT) tabanlı modeller kullanarak AI'ın o zekice yazılmış gayri-resmi metinlerini de yakalamayı hedefleyebiliriz. Dinlediğiniz için teşekkürler."

---
## 🛡️ JOKER BÖLÜM: Hocanın "Oha %99.7 Çok Yüksek Deyil Mi, Ezberlemiş?" Sorusuna Karşı Savunma
Bu kısmı sunum sırasında kendiliğinden söylemen veya hoca sorduğunda patlatman notları tavana uçuracaktır.

**Söylemen Gereken O Efsane Cümleler:**
> "Hocam başta %99.7'yi görünce sevinmek yerine biz de şüphelendik. Acaba modelimiz mucize mi yarattı yoksa veriyi mi ezberledi diye araştırdık."

> *"Ve şunu fark ettik: Modelimiz aslında genel bir YAPAY ZEKA problemi çözmüyor; modelimiz sadece bize verilen bu veri setindeki o robotik şablonları çok iyi yakalıyor. Yani modeli asıl başarıya ulaştıran şey AI'ı kökünden anlaması değil, o veri setindeki AI'ın sürekli 'In conclusion', 'Additionally' diyerek kendini sığ bir şablona hapsetmiş olması."*

> *"Dolayısıyla, elde ettiğimiz bu yüksek skor, veri setimizin izole dünyasından besleniyor. Aynı modeli gerçek dünyaya (Twitter'a veya karmaşık Reddit forumlarına) koysak doğruluğun çok daha düşeceğinin tamamen bilincindeyiz."*

Bunu duyduğunda hoca senin sadece "kod yazan" biri değil, "veri bilimi felsefesini anlamış" (Generalization problemi ve Superficial Cues kavramı) gerçek bir mühendis olduğunu anlayacaktır. Projeye son noktayı koydun, tebrikler! 🏁
