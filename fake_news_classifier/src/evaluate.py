from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def print_evaluation(model, X_test, y_test, model_name: str = "Model"):
    """Calculates and prints model accuracy and classification metrics."""
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    print(f"\n================ {model_name} Results ================")
    print(f"Accuracy Score: {acc * 100:.2f}%\n")
    print(classification_report(y_test, predictions, target_names=["Real (0)", "Fake (1)"]))
    return acc