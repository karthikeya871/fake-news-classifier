import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

# Import exact file paths from your existing config.py
from config import TRUE_CSV_PATH, FAKE_CSV_PATH

nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))

def load_and_merge_data() -> pd.DataFrame:
    """Loads True.csv and Fake.csv using config paths and returns a combined, shuffled DataFrame."""
    df_true = pd.read_csv(TRUE_CSV_PATH)
    df_false = pd.read_csv(FAKE_CSV_PATH)

    df_true['label'] = 0
    df_false['label'] = 1

    df = pd.concat([df_true, df_false], ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df['raw_text'] = df['title'].fillna('') + " " + df['text'].fillna('')
    return df

def preprocess_text(text: str) -> str:
    """Removes publisher signatures, URLs, punctuation, numbers, and stopwords."""
    text = str(text).lower()
    text = re.sub(r'^[A-Z\s\.\,-]+\s*\((Reuters|AP|AFP)\)\s*-\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://\S+|www\.\S+|\b[a-zA-Z0-9-]+\.(com|org|net|in|co|gov)\b', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", ' ', text)
    text = re.sub(r'\d+', '', text)
    words = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(words)

if __name__ == "__main__":
    # 1. Load & Preprocess Data
    df = load_and_merge_data()
    df['clean_text'] = df['raw_text'].apply(preprocess_text)

    # 2. Study Vocabulary & Word Frequency
    raw_vocab = set(" ".join(df['raw_text']).lower().split())
    clean_vocab = set(" ".join(df['clean_text']).split())
    print(f"Total Raw Vocabulary Size: {len(raw_vocab)}")
    print(f"Cleaned Vocabulary Size:   {len(clean_vocab)} (Noise reduced by {len(raw_vocab) - len(clean_vocab)} tokens)\n")

    real_words = pd.Series(" ".join(df[df['label'] == 0]['clean_text']).split()).value_counts().head(10)
    fake_words = pd.Series(" ".join(df[df['label'] == 1]['clean_text']).split()).value_counts().head(10)
    print("Top 10 Frequent Words (REAL):\n", real_words.to_dict())
    print("Top 10 Frequent Words (FAKE):\n", fake_words.to_dict(), "\n")

    # 3. Innovation: Compare Preprocessing & Feature-Extraction Strategies
    experiments = [
        ("1. Raw Text + Bag-of-Words (CountVectorizer)", df['raw_text'], CountVectorizer(max_features=5000)),
        ("2. Clean Text + Bag-of-Words (CountVectorizer)", df['clean_text'], CountVectorizer(max_features=5000)),
        ("3. Clean Text + TF-IDF (Unigrams)", df['clean_text'], TfidfVectorizer(max_features=5000, ngram_range=(1, 1))),
        ("4. Clean Text + TF-IDF (Unigrams + Bigrams)", df['clean_text'], TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ]

    print(f"{'Strategy':<46} | {'NB Acc':<8} | {'NB F1':<8} | {'SVM Acc':<8} | {'SVM F1':<8}")
    print("-" * 88)

    for name, text_series, vec in experiments:
        X_train, X_test, y_train, y_test = train_test_split(text_series, df['label'], test_size=0.2, random_state=42)
        X_train_vec = vec.fit_transform(X_train)
        X_test_vec = vec.transform(X_test)

        # Naïve Bayes
        nb = MultinomialNB()
        nb.fit(X_train_vec, y_train)
        nb_pred = nb.predict(X_test_vec)

        # Linear SVM
        svm = LinearSVC(C=0.05, random_state=42)
        svm.fit(X_train_vec, y_train)
        svm_pred = svm.predict(X_test_vec)

        print(f"{name:<46} | {accuracy_score(y_test, nb_pred)*100:6.2f}% | {f1_score(y_test, nb_pred)*100:6.2f}% | "
              f"{accuracy_score(y_test, svm_pred)*100:6.2f}% | {f1_score(y_test, svm_pred)*100:6.2f}%")

        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        # Plot & save heatmaps for the final experiment (NB & SVM)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        sns.heatmap(confusion_matrix(y_test, nb_pred), annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Naïve Bayes Confusion Matrix')

        sns.heatmap(confusion_matrix(y_test, svm_pred), annot=True, fmt='d', cmap='Greens', ax=axes[1])
        axes[1].set_title('Linear SVM Confusion Matrix')

        plt.savefig('heatmap.png')
        plt.show()