import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def build_vectorizer(strategy: str = 'tfidf', max_features: int = MAX_FEATURES, ngram_range: tuple = (1, 2)):
    """Returns vectorizer instance based on strategy ('bow' or 'tfidf')."""
    if strategy == 'bow':
        return CountVectorizer(max_features=10000, ngram_range=ngram_range, dtype=np.float32)
    elif strategy == 'tfidf':
        return TfidfVectorizer(max_features=10000, ngram_range=ngram_range, dtype=np.float32)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")