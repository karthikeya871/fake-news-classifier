from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from config import MAX_FEATURES

def build_vectorizer(strategy: str = 'tfidf', max_features: int = MAX_FEATURES, ngram_range: tuple = (1, 2)):
    """Returns vectorizer instance based on strategy ('bow' or 'tfidf')."""
    if strategy == 'bow':
        return CountVectorizer(max_features=max_features, ngram_range=ngram_range)
    elif strategy == 'tfidf':
        return TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")