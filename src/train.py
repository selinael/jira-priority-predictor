import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

def main(data_path, out_path, model_type="svm"):
    # Load dataset
    df = pd.read_csv(data_path)

    # Features and labels
    X = df["Description"]
    y = df["priority"]

    # Split into train/val/test (70/15/15 stratified)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    # Choose model
    if model_type == "svm":
        clf = LinearSVC(class_weight="balanced")
    else:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")

    # Build pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2)),
        ("clf", clf)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Validate
    print("\n Validation Performance:")
    y_val_pred = pipeline.predict(X_val)
    print(classification_report(y_val, y_val_pred))

    # Test
    print("\nTest Performance:")
    y_test_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_test_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))

    # Save model
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(pipeline, out_path)
    print(f"\n Model saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to cleaned tickets CSV")
    parser.add_argument("--out", type=str, default="../models/priority_clf.joblib", help="Output model path")
    parser.add_argument("--model", type=str, choices=["svm", "logreg"], default="svm", help="Model type")
    args = parser.parse_args()

    main(args.data, args.out, args.model)
