from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def train_model(X, y, texts=None):
    """Logistic Regression modeli eğit."""
    if texts is not None:
        X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(X, y, texts, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        text_test = None

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model, X_test, y_test, text_test


def train_svm(X, y, texts=None):
    """SVM (Support Vector Machine) modeli eğit."""
    if texts is not None:
        X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(X, y, texts, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        text_test = None

    model = LinearSVC(max_iter=2000)
    model.fit(X_train, y_train)

    return model, X_test, y_test, text_test
