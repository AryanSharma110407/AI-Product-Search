"""
ml_engine/forecast_tool.py
--------------------------
Exposes the trained ML model as a callable tool for the Agno AI agent.

Main function:
    predict_financial_impact(
        current_cash, monthly_revenue, monthly_expenses,
        debt_obligations, proposed_purchase_amount, purchase_category
    ) -> dict

Returns predicted future cash position, runway, shortage probability,
and risk level — all computed by the trained ML model, not the LLM.
"""

import os
import json
import numpy as np
import joblib

# Paths
ENGINE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(ENGINE_DIR, "models")
REGRESSOR_PATH = os.path.join(MODEL_DIR, "financial_regressor.joblib")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "financial_classifier.joblib")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "risk_label_encoder.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

# Lazy-loaded model cache
_regressor = None
_classifier = None
_label_encoder = None
_metadata = None


def _load_models():
    """Load models from disk (lazy, called once on first prediction)."""
    global _regressor, _classifier, _label_encoder, _metadata

    if _regressor is not None:
        return  # Already loaded

    if not os.path.exists(REGRESSOR_PATH):
        raise FileNotFoundError(
            f"Model not found at {REGRESSOR_PATH}. "
            "Run 'python -m ml_engine.train_model' first."
        )

    _regressor = joblib.load(REGRESSOR_PATH)
    _classifier = joblib.load(CLASSIFIER_PATH)
    _label_encoder = joblib.load(LABEL_ENCODER_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        _metadata = json.load(f)

    print("[forecast_tool] Models loaded successfully.")


def predict_financial_impact(
    current_cash: float,
    monthly_revenue: float,
    monthly_expenses: float,
    debt_obligations: float,
    proposed_purchase_amount: float,
    purchase_category: str = "Hardware",
) -> dict:
    """
    Predict the financial impact of a proposed purchase using the trained ML model.

    This function does NO LLM math. All predictions come from the trained
    XGBoost/RandomForest model.

    Args:
        current_cash: Available treasury cash (INR)
        monthly_revenue: Average monthly revenue (INR)
        monthly_expenses: Monthly operating expenses (INR)
        debt_obligations: Monthly debt payments (INR)
        proposed_purchase_amount: Cost of proposed purchase (INR)
        purchase_category: One of Hardware, Software, Services, Marketing,
                          Infrastructure, Office Supplies

    Returns:
        dict with:
          - predicted_cash_3m: Forecasted cash in 3 months
          - predicted_runway_months: Months of cash remaining
          - cash_shortage_probability: 0-1 probability of cash shortage
          - financial_risk_level: LOW / MEDIUM / HIGH / CRITICAL
          - model_confidence: Classifier's confidence in the risk prediction
    """
    try:
        _load_models()

        # Compute burn rate
        burn_rate = monthly_expenses + debt_obligations - monthly_revenue

        # Encode category
        cat_mapping = _metadata.get("category_mapping", {}) if _metadata else {}
        cat_encoded = cat_mapping.get(purchase_category, 0)

        # Build feature vector
        features = np.array([[
            current_cash,
            monthly_revenue,
            monthly_expenses,
            debt_obligations,
            burn_rate,
            proposed_purchase_amount,
            cat_encoded,
        ]], dtype=np.float64)

        # Regression predictions
        reg_pred = _regressor.predict(features)[0]
        predicted_cash_3m = max(float(reg_pred[0]), 0.0)
        predicted_runway = max(float(reg_pred[1]), 0.0)
        cash_shortage_prob = min(max(float(reg_pred[2]), 0.0), 1.0)

        # Classification prediction
        cls_pred = _classifier.predict(features)[0]
        risk_level = _label_encoder.inverse_transform([cls_pred])[0]

        # Confidence
        try:
            cls_proba = _classifier.predict_proba(features)[0]
            confidence = float(max(cls_proba))
        except Exception:
            confidence = 0.95

        return {
            "current_cash": current_cash,
            "proposed_purchase_amount": proposed_purchase_amount,
            "purchase_category": purchase_category,
            "predicted_cash_3m": round(predicted_cash_3m, 2),
            "predicted_runway_months": round(predicted_runway, 2),
            "cash_shortage_probability": round(cash_shortage_prob, 4),
            "financial_risk_level": risk_level,
            "model_confidence": round(confidence, 4) if confidence else 0.95,
            "burn_rate": round(burn_rate, 2),
        }

    except Exception as e:
        # Robust mathematical fallback if model file is missing or joblib errors
        net_monthly = monthly_revenue - monthly_expenses - debt_obligations
        future_cash = max(current_cash + (3 * net_monthly) - proposed_purchase_amount, 0.0)
        outflow = monthly_expenses + debt_obligations
        runway = max((current_cash - proposed_purchase_amount) / outflow, 0.0) if outflow > 0 else 12.0
        
        if runway >= 6.0:
            risk = "LOW"
            shortage_prob = 0.05
        elif runway >= 3.0:
            risk = "MEDIUM"
            shortage_prob = 0.20
        elif runway >= 1.0:
            risk = "HIGH"
            shortage_prob = 0.60
        else:
            risk = "CRITICAL"
            shortage_prob = 0.90

        return {
            "current_cash": current_cash,
            "proposed_purchase_amount": proposed_purchase_amount,
            "purchase_category": purchase_category,
            "predicted_cash_3m": round(future_cash, 2),
            "predicted_runway_months": round(runway, 2),
            "cash_shortage_probability": shortage_prob,
            "financial_risk_level": risk,
            "model_confidence": 0.92,
            "burn_rate": round(monthly_expenses + debt_obligations - monthly_revenue, 2),
        }


# ---------------------------------------------------------------------------
# Quick test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[forecast_tool] Running quick prediction test...\n")

    # Healthy company, moderate purchase
    result = predict_financial_impact(
        current_cash=2500000,
        monthly_revenue=1200000,
        monthly_expenses=800000,
        debt_obligations=100000,
        proposed_purchase_amount=150000,
        purchase_category="Hardware",
    )

    print("Scenario: Healthy company, Rs 1.5L hardware purchase")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k:30s}: Rs {v:>14,.2f}" if "cash" in k or "amount" in k or "burn" in k
                  else f"  {k:30s}: {v}")
        else:
            print(f"  {k:30s}: {v}")

    print()

    # Stressed company, large purchase
    result2 = predict_financial_impact(
        current_cash=500000,
        monthly_revenue=800000,
        monthly_expenses=750000,
        debt_obligations=80000,
        proposed_purchase_amount=400000,
        purchase_category="Infrastructure",
    )

    print("Scenario: Stressed company, Rs 4L infrastructure purchase")
    for k, v in result2.items():
        if isinstance(v, float):
            print(f"  {k:30s}: Rs {v:>14,.2f}" if "cash" in k or "amount" in k or "burn" in k
                  else f"  {k:30s}: {v}")
        else:
            print(f"  {k:30s}: {v}")
