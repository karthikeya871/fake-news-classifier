import os
BASE_DIR=os.path.dirname(os.path.abspath(__file__))

TRUE_CSV_PATH=os.path.join(BASE_DIR,"data","True.csv")
FAKE_CSV_PATH=os.path.join(BASE_DIR,"data","Fake.csv")

MODEL_DIR=os.path.join(BASE_DIR,"models")
VECTORIZER_PATH=os.path.join(MODEL_DIR, "vectorizer.pkl")
NB_MODEL_PATH=os.path.join(MODEL_DIR, "naive_bayes_models.pkl")
SVM_MODEL_PATH=os.path.join(MODEL_DIR, "svm_model.pkl")

TEST_SIZE=0.25
RANDOM_STATE=42
MAX_FEATURES=5000
