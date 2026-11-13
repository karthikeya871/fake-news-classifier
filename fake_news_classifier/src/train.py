from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

def train_naive_bayes(X_train, y_train, alpha: float = 1.0):
    """Trains Multinomial Naïve Bayes Classifier."""
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train, C: float = 0.05):
    """Trains a Calibrated Support Vector Machine Classifier."""
    base_svm = LinearSVC(C=C, max_iter=2000, random_state=42)
    # CalibratedClassifierCV smooths hard boundaries into stable probabilities
    calibrated_svm = CalibratedClassifierCV(estimator=base_svm, cv=3)
    calibrated_svm.fit(X_train, y_train)
    return calibrated_svm