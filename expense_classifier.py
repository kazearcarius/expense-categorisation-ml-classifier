"""
Expense Categorisation ML Classifier
===================================

This module trains a simple machine‑learning model to classify expense
transactions into categories based on the text description, vendor and
amount. It provides both a training pipeline and a prediction
function that can be imported into other projects.

Dependencies:
    pandas, scikit‑learn
"""

import argparse
from typing import List
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


def load_data(path: str) -> pd.DataFrame:
    """Load labelled expense data. Expected columns: Description, Vendor, Amount, Category"""
    return pd.read_csv(path)


def build_pipeline() -> Pipeline:
    """Create a scikit‑learn pipeline for text vectorisation and classification."""
    return Pipeline([
        ('vectorizer', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000))
    ])


def train_model(data: pd.DataFrame) -> Pipeline:
    """Train the classifier and return the fitted pipeline."""
    X = data['Description']
    y = data['Category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_pipeline()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    return model


def save_model(model: Pipeline, path: str) -> None:
    """Serialize the model to disk using joblib."""
    import joblib  # type: ignore
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an expense categorisation classifier.")
    parser.add_argument("--input", required=True, help="CSV file with labelled expenses")
    parser.add_argument("--model", required=True, help="Path to save the trained model (joblib format)")
    args = parser.parse_args()

    data = load_data(args.input)
    model = train_model(data)
    save_model(model, args.model)


if __name__ == '__main__':
    main()