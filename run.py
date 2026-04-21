import pandas as pd
from src.preprocessing import preprocess
from src.feature_extraction import get_tfidf_features
from src.train_model import train_model, train_svm
from src.evaluate_model import evaluate, plot_confusion_matrix, compare_models, plot_feature_importance, perform_error_analysis

# 1. Veriyi yükle
print("Veri yükleniyor...")
df = pd.read_csv("data/AI_Human.csv")
print(f"Dataset boyutu: {df.shape}")

# Orijinal metinleri hata analizi için sakla
original_texts = df['text'].copy()

# 2. Ön işleme
print("Preprocessing yapılıyor...")
df['text'] = df['text'].apply(preprocess)

# 3. Feature extraction (TF-IDF)
print("TF-IDF özellikleri çıkarılıyor...")
X, vectorizer = get_tfidf_features(df['text'])
y = df['generated']

# ==========================================
# MODEL 1: Logistic Regression
# ==========================================
print("\n🔹 Logistic Regression eğitiliyor...")
lr_model, X_test, y_test, text_test = train_model(X, y, texts=original_texts)
lr_pred, lr_acc = evaluate(lr_model, X_test, y_test, "Logistic Regression")
plot_confusion_matrix(y_test, lr_pred, "Logistic Regression", "results/figures/cm_logistic_regression.png")
plot_feature_importance(lr_model, vectorizer, model_name="Logistic Regression", save_path="results/figures/fi_logistic_regression.png")
perform_error_analysis(y_test, lr_pred, text_test, model_name="Logistic Regression", save_path="results/error_analysis_lr.txt")

# ==========================================
# MODEL 2: SVM
# ==========================================
print("\n🔹 SVM eğitiliyor...")
svm_model, X_test_svm, y_test_svm, text_test_svm = train_svm(X, y, texts=original_texts)
svm_pred, svm_acc = evaluate(svm_model, X_test_svm, y_test_svm, "SVM")
plot_confusion_matrix(y_test_svm, svm_pred, "SVM", "results/figures/cm_svm.png")
plot_feature_importance(svm_model, vectorizer, model_name="SVM", save_path="results/figures/fi_svm.png")
perform_error_analysis(y_test_svm, svm_pred, text_test_svm, model_name="SVM", save_path="results/error_analysis_svm.txt")

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

print("\n✅ Tüm grafikler results/figures/ klasörüne, error analysis dosyaları ise results/ klasörüne kaydedildi!")
