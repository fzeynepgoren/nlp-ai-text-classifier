"""
Eğitilmiş modeli kullanarak yeni bir metnin İnsan mı yoksa AI tarafından mı yazıldığını tahmin eder.
Kullanım: python predict.py
"""
import joblib
import os
import pandas as pd
from src.preprocessing import preprocess
from src.feature_extraction import get_tfidf_features
from src.train_model import split_data, train_logistic_regression, train_svm


def load_or_train_model():
    """Kaydedilmiş model varsa yükle, yoksa eğit ve kaydet."""
    model_path = "results/svm_model.joblib"
    vectorizer_path = "results/tfidf_vectorizer.joblib"

    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        print("Kaydedilmiş model yükleniyor...")
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
    else:
        print("Model bulunamadı, eğitim başlatılıyor...")
        df = pd.read_csv("data/AI_Human.csv")
        df['text'] = df['text'].apply(preprocess)
        X, vectorizer = get_tfidf_features(df['text'])
        y = df['generated']
        X_train, X_test, y_train, y_test, _ = split_data(X, y)
        model = train_svm(X_train, y_train)

        os.makedirs("results", exist_ok=True)
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        print(f"Model kaydedildi: {model_path}")

    return model, vectorizer


def predict_text(text, model, vectorizer):
    """Tek bir metin için tahmin yap."""
    cleaned = preprocess(text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    label = "🤖 AI tarafından yazılmış" if prediction == 1.0 else "✍️ İnsan tarafından yazılmış"
    return label, prediction


if __name__ == "__main__":
    model, vectorizer = load_or_train_model()

    print("\n" + "=" * 60)
    print("  AI vs İnsan Metin Sınıflandırıcı - Tahmin Aracı")
    print("=" * 60)
    print("Çıkmak için 'q' yazın.\n")

    while True:
        text = input("📝 Metni girin: ").strip()
        if text.lower() == 'q':
            print("Çıkılıyor...")
            break
        if not text:
            print("Boş metin girdiniz, tekrar deneyin.\n")
            continue

        label, pred = predict_text(text, model, vectorizer)
        print(f"   → Sonuç: {label}\n")
