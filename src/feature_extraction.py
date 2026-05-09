from sklearn.feature_extraction.text import TfidfVectorizer


def get_tfidf_features(texts):
    """TF-IDF ile metin özellik çıkarımı (unigram + bigram, max 5000 feature)."""
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    return X, vectorizer
