import os
import joblib
from sklearn.model_selection import train_test_split

from config import (
    TEST_SIZE, RANDOM_STATE, VECTORIZER_PATH,
    NB_MODEL_PATH, SVM_MODEL_PATH, MODEL_DIR
)
from src.data_loader import load_and_merge_data
from src.preprocessing import clean_text
from src.feature_extraction import build_vectorizer
from src.train import train_naive_bayes, train_svm
from src.evaluate import print_evaluation

def main():
    # 1. Load Data
    print("1/6. Loading raw datasets...")
    df = load_and_merge_data()

    # 2. Preprocess Text
    print("2/6. Cleaning text (removing noise and stopwords)...")
    df['clean_text'] = df['full_text'].apply(clean_text)

    # 3. Train/Test Split
    print("3/6. Splitting data into train (80%) and test (20%) sets...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df['clean_text'],
        df['label'],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # 4. Feature Extraction Comparison (Innovation Requirement)
    print("4/6. Transforming text data into numerical TF-IDF vectors...")

    # --- Experiment A: Bag-of-Words (CountVectorizer) ---
    bow_vectorizer = build_vectorizer(strategy='bow')
    X_train_bow = bow_vectorizer.fit_transform(X_train_raw)
    X_test_bow = bow_vectorizer.transform(X_test_raw)

    print("\n--- Strategy A: Bag-of-Words (CountVectorizer) ---")
    bow_nb = train_naive_bayes(X_train_bow, y_train)
    print_evaluation(bow_nb, X_test_bow, y_test, model_name="Naïve Bayes (Bag-of-Words)")

    # --- Strategy B: TF-IDF (Best Strategy) ---
    vectorizer = build_vectorizer(strategy='tfidf')
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    print("\n--- Strategy B: TF-IDF Vectorization ---")
    nb_model = train_naive_bayes(X_train, y_train)
    svm_model = train_svm(X_train, y_train)

    # 6. Evaluate Models
    print("6/6. Evaluating model performance...")
    print_evaluation(nb_model, X_test, y_test, model_name="Multinomial Naïve Bayes")
    print_evaluation(svm_model, X_test, y_test, model_name="Linear SVM")


    os.makedirs(MODEL_DIR, exist_ok=True)
    # Strip unnecessary 3.5M+ stop words to keep model size under 100KB and RAM under 40MB
    vectorizer.stop_words_ = set()
    joblib.dump(vectorizer, VECTORIZER_PATH, compress=9)
    joblib.dump(nb_model, NB_MODEL_PATH, compress=9)
    joblib.dump(svm_model, SVM_MODEL_PATH, compress=9)
    print("\n[SUCCESS] Pipeline executed successfully. Trained models saved in 'models/' directory.")

if __name__ == "__main__":
    main()