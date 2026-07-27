"""
train_model.py  (Phase 3 — Questionnaire-derived features)
-----------------------------------------------------------
Generates synthetic training data by simulating questionnaire responses
and mapping them through questionnaire_extractor logic, then trains a
Random Forest Regressor and saves the model.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from questionnaire_extractor import extract_features, QUESTIONS

np.random.seed(42)
N = 5000  # number of synthetic respondents


def simulate_respondent(severity: float) -> dict:
    """
    severity ∈ [0, 1]:  0 = mentally very healthy, 1 = severely ill.
    Returns simulated questionnaire answers (0-4 per question).
    """
    answers = {}
    for q in QUESTIONS:
        # Base score proportional to severity, with noise
        raw = severity * 4.0 + np.random.normal(0, 0.8)
        answers[q["id"]] = int(np.clip(round(raw), 0, 4))
    return answers


def generate_dataset():
    records = []
    for _ in range(N):
        # Random severity; skew toward healthier range
        severity = np.clip(np.random.beta(2, 5), 0, 1)
        # Journal sentiment: healthier ppl have more positive journals
        journal_sentiment = np.clip(
            np.random.normal(0.5 - severity, 0.3), -1, 1
        )

        answers = simulate_respondent(severity)
        features = extract_features(answers, journal_sentiment)

        # Mental fitness: higher severity → lower fitness (0–100)
        noise = np.random.normal(0, 3)
        fitness = np.clip(100 - severity * 80 + noise, 0, 100)

        row = features.copy()
        row["journal_sentiment"] = round(journal_sentiment, 4)
        row["mental_fitness"] = round(fitness, 2)
        records.append(row)

    return pd.DataFrame(records)


def train():
    print("Generating questionnaire-derived synthetic dataset …")
    df = generate_dataset()

    feature_cols = [
        "Schizophrenia", "Bipolar_disorder", "Eating_disorders",
        "Anxiety_disorders", "Drug_use_disorders",
        "Depressive_disorders", "Alcohol_use_disorders",
        "journal_sentiment",
    ]

    X = df[feature_cols]
    y = df["mental_fitness"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training on {len(X_train)} samples …")
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"Test MSE  : {mse:.4f}")
    print(f"Test R²   : {r2:.4f}")

    joblib.dump({"model": model, "feature_cols": feature_cols}, "model.pkl")
    print("model.pkl saved successfully [OK]")


if __name__ == "__main__":
    train()
