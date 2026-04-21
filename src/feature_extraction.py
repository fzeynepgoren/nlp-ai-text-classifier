from sklearn.feature_extraction.text import TfidfVectorizer


def get_tfidf_features(texts):
    """TF-IDF ile metin özellik çıkarımı (max 5000 feature)."""
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer
