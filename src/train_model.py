from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def split_data(X, y, texts=None, test_size=0.2, random_state=42):
    """Veriyi stratified olarak tek bir yerde böl (train/test).
    Stratify parametresi sınıf oranlarını korur."""
    if texts is not None:
        X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
            X, y, texts, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        text_test = None

    return X_train, X_test, y_train, y_test, text_test


def train_logistic_regression(X_train, y_train):
    """Logistic Regression modeli eğit."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train):
    """SVM (Support Vector Machine) modeli eğit."""
    model = LinearSVC(max_iter=2000)
    model.fit(X_train, y_train)
    return model
