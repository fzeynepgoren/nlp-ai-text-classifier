import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

stop_words = set(stopwords.words('english'))


def preprocess(text):
    """Metni temizle: küçük harf, noktalama sil, stopword sil (word_tokenize ile)."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words]
    return " ".join(words)
