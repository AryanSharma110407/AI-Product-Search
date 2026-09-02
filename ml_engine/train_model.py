"""
ml_engine/train_model.py
------------------------
Trains XGBoost and Random Forest models on the synthetic financial dataset.

Models trained:
  1. Regressor: Predicts `predicted_cash_3m` and `predicted_runway_months`
  2. Classifier: Predicts `financial_risk_level` (LOW/MEDIUM/HIGH/CRITICAL)

The best model is saved as a joblib file for use by forecast_tool.py.

Run directly to train:
    python -m ml_engine.train_model
"""

import os
import json
import csv

import numpy as np

# scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    classification_report,
    accuracy_score,
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Try XGBoost, fall back to Random Forest if not installed
try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[train_model] XGBoost not found, using Random Forest only.")

import joblib

# Paths
ENGINE_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(ENGINE_DIR, "financial_dataset.csv")
MODEL_DIR = os.path.join(ENGINE_DIR, "models")
REGRESSOR_PATH = os.path.join(MODEL_DIR, "financial_regressor.joblib")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "financial_classifier.joblib")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "risk_label_encoder.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

# Feature columns and target columns
FEATURE_COLS = [
    "current_cash",
    "monthly_revenue",
    "monthly_expenses",
    "debt_obligations",
    "burn_rate",
    "proposed_purchase_amount",
]

CATEGORICAL_FEATURES = ["purchase_category"]

REGRESSION_TARGETS = ["predicted_cash_3m", "predicted_runway_months", "cash_shortage_probability"]
CLASSIFICATION_TARGET = "financial_risk_level"


def load_dataset(filepath: str = DATASET_PATH) -> tuple:
    """Load dataset from CSV and return features (X) and targets (y_reg, y_cls)."""
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"[train_model] Loaded {len(rows)} rows from {filepath}")

    # Convert to numpy arrays
    # Encode purchase_category
    categories = list(set(r["purchase_category"] for r in rows))
    cat_to_idx = {c: i for i, c in enumerate(sorted(categories))}

    X = []
    y_reg = []
    y_cls = []

    for row in rows:
        features = [float(row[col]) for col in FEATURE_COLS]
        features.append(cat_to_idx[row["purchase_category"]])
        X.append(features)

        reg_targets = [float(row[col]) for col in REGRESSION_TARGETS]
        y_reg.append(reg_targets)

        y_cls.append(row[CLASSIFICATION_TARGET])

    X = np.array(X, dtype=np.float64)
    y_reg = np.array(y_reg, dtype=np.float64)

    # Encode classification labels
    le = LabelEncoder()
    y_cls_encoded = le.fit_transform(y_cls)

    return X, y_reg, y_cls_encoded, le, cat_to_idx


def train_regressor(X_train, X_test, y_train, y_test):
    """Train a regressor for financial state prediction."""
    if HAS_XGBOOST:
        model = XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
        )
        model_name = "XGBoost Regressor"
    else:
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        )
        model_name = "Random Forest Regressor"

    print(f"\n[train_model] Training {model_name}...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(f"\n[train_model] {model_name} Results:")
    for i, target_name in enumerate(REGRESSION_TARGETS):
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        print(f"  {target_name:30s} | MAE: {mae:>14,.2f} | R2: {r2:.4f}")

    return model, model_name


def train_classifier(X_train, X_test, y_train, y_test, le):
    """Train a classifier for financial risk level prediction."""
    if HAS_XGBOOST:
        model = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )
        model_name = "XGBoost Classifier"
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        )
        model_name = "Random Forest Classifier"

    print(f"\n[train_model] Training {model_name}...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n[train_model] {model_name} Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return model, model_name


def save_models(regressor, classifier, le, cat_to_idx, reg_name, cls_name):
    """Save trained models to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(regressor, REGRESSOR_PATH)
    joblib.dump(classifier, CLASSIFIER_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)

    metadata = {
        "regressor": reg_name,
        "classifier": cls_name,
        "feature_columns": FEATURE_COLS + ["purchase_category_encoded"],
        "regression_targets": REGRESSION_TARGETS,
        "classification_target": CLASSIFICATION_TARGET,
        "category_mapping": cat_to_idx,
        "risk_classes": list(le.classes_),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[train_model] Models saved to {MODEL_DIR}/")
    print(f"  Regressor:     {REGRESSOR_PATH}")
    print(f"  Classifier:    {CLASSIFIER_PATH}")
    print(f"  Label Encoder: {LABEL_ENCODER_PATH}")
    print(f"  Metadata:      {METADATA_PATH}")


# ---------------------------------------------------------------------------
# Run directly to train models
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Load data
    X, y_reg, y_cls, le, cat_to_idx = load_dataset()

    # Split data (80/20, time-aware would be better with real data)
    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
        X, y_reg, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )

    print(f"\n[train_model] Train set: {len(X_train)} | Test set: {len(X_test)}")

    # Train models
    regressor, reg_name = train_regressor(X_train, X_test, y_reg_train, y_reg_test)
    classifier, cls_name = train_classifier(X_train, X_test, y_cls_train, y_cls_test, le)

    # Save
    save_models(regressor, classifier, le, cat_to_idx, reg_name, cls_name)

    print("\n[train_model] Training complete.")
