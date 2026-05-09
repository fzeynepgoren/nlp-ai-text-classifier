import pandas as pd
import joblib
from src.preprocessing import preprocess
from src.feature_extraction import get_tfidf_features
from src.train_model import split_data, train_logistic_regression, train_svm
from src.evaluate_model import evaluate, plot_confusion_matrix, compare_models, plot_feature_importance, perform_error_analysis, cross_validate_models

# 1. Veriyi yükle
print("Veri yükleniyor...")
df = pd.read_csv("data/AI_Human.csv")
print(f"Dataset boyutu: {df.shape}")

# Sınıf dağılımını göster
print(f"\nSınıf Dağılımı:")
print(f"  İnsan (0): {(df['generated'] == 0.0).sum()} ({(df['generated'] == 0.0).mean()*100:.1f}%)")
print(f"  AI    (1): {(df['generated'] == 1.0).sum()} ({(df['generated'] == 1.0).mean()*100:.1f}%)")

# Orijinal metinleri hata analizi için sakla
original_texts = df['text'].copy()

# 2. Ön işleme
print("\nPreprocessing yapılıyor...")
df['text'] = df['text'].apply(preprocess)

# 3. Feature extraction (TF-IDF)
print("TF-IDF özellikleri çıkarılıyor...")
X, vectorizer = get_tfidf_features(df['text'])
y = df['generated']

# ==========================================
# 4. TEK BİR YERDE STRATIFIED SPLIT
# ==========================================
print("Veri stratified olarak bölünüyor (80/20)...")
X_train, X_test, y_train, y_test, text_test = split_data(X, y, texts=original_texts)
print(f"  Eğitim seti: {X_train.shape[0]} örnek")
print(f"  Test seti:   {X_test.shape[0]} örnek")
print(f"  Test seti sınıf dağılımı: İnsan={int((y_test == 0.0).sum())}, AI={int((y_test == 1.0).sum())}")

# ==========================================
# MODEL 1: Logistic Regression
# ==========================================
print("\n🔹 Logistic Regression eğitiliyor...")
lr_model = train_logistic_regression(X_train, y_train)
lr_pred, lr_acc = evaluate(lr_model, X_test, y_test, "Logistic Regression")
plot_confusion_matrix(y_test, lr_pred, "Logistic Regression", "results/figures/cm_logistic_regression.png")
plot_feature_importance(lr_model, vectorizer, model_name="Logistic Regression", save_path="results/figures/fi_logistic_regression.png")
perform_error_analysis(y_test, lr_pred, text_test, model_name="Logistic Regression", save_path="results/error_analysis_lr.txt")

# ==========================================
# MODEL 2: SVM
# ==========================================
print("\n🔹 SVM eğitiliyor...")
svm_model = train_svm(X_train, y_train)
svm_pred, svm_acc = evaluate(svm_model, X_test, y_test, "SVM")
plot_confusion_matrix(y_test, svm_pred, "SVM", "results/figures/cm_svm.png")
plot_feature_importance(svm_model, vectorizer, model_name="SVM", save_path="results/figures/fi_svm.png")
perform_error_analysis(y_test, svm_pred, text_test, model_name="SVM", save_path="results/error_analysis_svm.txt")

# ==========================================
# MODEL KARŞILAŞTIRMASI
# ==========================================
print("\n📊 Model Karşılaştırması:")
print(f"  Logistic Regression: {lr_acc:.4f}")
print(f"  SVM:                 {svm_acc:.4f}")

results = {
    "Logistic Regression": lr_acc,
    "SVM": svm_acc
}
compare_models(results, "results/figures/model_comparison.png")

# ==========================================
# MODELLERI KAYDET (predict.py için)
# ==========================================
joblib.dump(svm_model, "results/svm_model.joblib")
joblib.dump(vectorizer, "results/tfidf_vectorizer.joblib")
print("\n💾 SVM modeli ve TF-IDF vectorizer kaydedildi (predict.py için).")

# ==========================================
# CROSS-VALIDATION (Overfitting Kontrolü)
# ==========================================
cross_validate_models(X, y, save_path="results/figures/cross_validation.png")

print("\n✅ Tüm grafikler results/figures/ klasörüne, error analysis dosyaları ise results/ klasörüne kaydedildi!")
