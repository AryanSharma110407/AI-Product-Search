"""
ml_engine/generate_synthetic.py
-------------------------------
Generates a realistic synthetic financial dataset for training the
financial forecasting ML model.

Each row represents a company-quarter financial snapshot + a proposed
purchase action. The target variables are the predicted financial state
3 months after the purchase.

Economic relationships are modeled realistically:
  - Revenue correlates with company size and growth rate
  - Expenses scale with revenue (COGS + operating costs)
  - Debt obligations reduce available cash
  - A proposed purchase reduces cash immediately
  - Future cash = current_cash + (3 * net_monthly_income) - purchase

Run directly to generate the dataset:
    python -m ml_engine.generate_synthetic
"""

import os
import random
import csv
import math

# Output path
OUTPUT_DIR = os.path.dirname(__file__)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "financial_dataset.csv")

# Number of scenarios to generate
NUM_SCENARIOS = 12000

# Purchase categories and their typical cost ranges (INR)
PURCHASE_CATEGORIES = {
    "Hardware": (15000, 500000),
    "Software": (5000, 300000),
    "Services": (10000, 200000),
    "Marketing": (20000, 500000),
    "Infrastructure": (50000, 1000000),
    "Office Supplies": (2000, 50000),
}

# Minimum reserve ratio (company should keep at least this fraction of monthly expenses)
MIN_RESERVE_MONTHS = 2.0


def generate_company_state() -> dict:
    """
    Generate a single realistic company financial state.

    Models a company with:
      - monthly_revenue in range Rs 5L to Rs 2Cr
      - expense ratio between 55% and 95% of revenue
      - cash reserve between 1 month and 18 months of expenses
      - debt between 0% and 40% of annual revenue
    """
    # Company size (monthly revenue in INR)
    monthly_revenue = random.uniform(500000, 20000000)  # Rs 5L to Rs 2Cr

    # Expense ratio (what % of revenue goes to expenses)
    expense_ratio = random.uniform(0.55, 0.95)
    monthly_expenses = monthly_revenue * expense_ratio

    # Net monthly income (can be negative for loss-making companies)
    net_monthly = monthly_revenue - monthly_expenses

    # Cash reserves (between 1 and 18 months of expenses)
    cash_months = random.uniform(1.0, 18.0)
    current_cash = monthly_expenses * cash_months

    # Debt obligations (monthly payment: 0% to 5% of monthly revenue)
    debt_ratio = random.uniform(0.0, 0.05)
    debt_obligations = monthly_revenue * debt_ratio

    # Burn rate (net cash outflow per month after debt)
    burn_rate = monthly_expenses + debt_obligations - monthly_revenue

    return {
        "current_cash": round(current_cash, 2),
        "monthly_revenue": round(monthly_revenue, 2),
        "monthly_expenses": round(monthly_expenses, 2),
        "debt_obligations": round(debt_obligations, 2),
        "burn_rate": round(burn_rate, 2),
    }


def generate_purchase() -> dict:
    """Generate a realistic purchase proposal."""
    category = random.choice(list(PURCHASE_CATEGORIES.keys()))
    min_price, max_price = PURCHASE_CATEGORIES[category]
    amount = random.uniform(min_price, max_price)

    return {
        "proposed_purchase_amount": round(amount, 2),
        "purchase_category": category,
    }


def compute_targets(state: dict, purchase: dict) -> dict:
    """
    Compute target variables based on economic relationships.

    The future state after 3 months is calculated deterministically:
      future_cash = current_cash
                    + 3 * (revenue - expenses - debt)
                    - purchase_amount
                    + noise (small random variation for model learning)

    Risk levels are assigned based on:
      - Cash runway (months of expenses remaining)
      - Whether the purchase pushes below minimum reserve
    """
    current_cash = state["current_cash"]
    monthly_revenue = state["monthly_revenue"]
    monthly_expenses = state["monthly_expenses"]
    debt_obligations = state["debt_obligations"]
    purchase_amount = purchase["proposed_purchase_amount"]

    # Net monthly cash flow
    net_monthly = monthly_revenue - monthly_expenses - debt_obligations

    # Simulate 3-month projection with small noise
    noise_factor = random.uniform(0.92, 1.08)  # +/- 8% variation
    predicted_cash_3m = (current_cash + (3 * net_monthly * noise_factor) - purchase_amount)

    # Ensure non-negative (can't have negative cash, company would take action)
    predicted_cash_3m = max(predicted_cash_3m, 0.0)

    # Cash runway: how many months of expenses can we cover?
    total_monthly_outflow = monthly_expenses + debt_obligations
    if total_monthly_outflow > 0:
        post_purchase_cash = current_cash - purchase_amount
        predicted_runway = max(post_purchase_cash / total_monthly_outflow, 0.0)
    else:
        predicted_runway = 99.0  # Effectively infinite

    # Cash shortage probability
    # Based on: how close are we to running out?
    min_safe_cash = monthly_expenses * MIN_RESERVE_MONTHS
    if predicted_cash_3m >= min_safe_cash * 2:
        cash_shortage_prob = random.uniform(0.0, 0.05)
    elif predicted_cash_3m >= min_safe_cash:
        cash_shortage_prob = random.uniform(0.05, 0.25)
    elif predicted_cash_3m >= min_safe_cash * 0.5:
        cash_shortage_prob = random.uniform(0.25, 0.55)
    elif predicted_cash_3m > 0:
        cash_shortage_prob = random.uniform(0.55, 0.85)
    else:
        cash_shortage_prob = random.uniform(0.85, 1.0)

    # Financial risk level
    if predicted_runway >= 6.0 and cash_shortage_prob < 0.1:
        risk_level = "LOW"
    elif predicted_runway >= 3.0 and cash_shortage_prob < 0.3:
        risk_level = "MEDIUM"
    elif predicted_runway >= 1.0 and cash_shortage_prob < 0.7:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    return {
        "predicted_cash_3m": round(predicted_cash_3m, 2),
        "predicted_runway_months": round(predicted_runway, 2),
        "cash_shortage_probability": round(cash_shortage_prob, 4),
        "financial_risk_level": risk_level,
    }


def generate_dataset(num_scenarios: int = NUM_SCENARIOS) -> list:
    """Generate the full dataset as a list of dictionaries."""
    dataset = []

    for _ in range(num_scenarios):
        state = generate_company_state()
        purchase = generate_purchase()
        targets = compute_targets(state, purchase)

        row = {**state, **purchase, **targets}
        dataset.append(row)

    return dataset


def save_dataset(dataset: list, filepath: str = OUTPUT_FILE) -> None:
    """Save dataset to CSV."""
    if not dataset:
        print("[generate_synthetic] No data to save.")
        return

    fieldnames = list(dataset[0].keys())

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)

    print(f"[generate_synthetic] Saved {len(dataset)} rows to {filepath}")


def print_summary(dataset: list) -> None:
    """Print a summary of the generated dataset."""
    risk_counts = {}
    for row in dataset:
        level = row["financial_risk_level"]
        risk_counts[level] = risk_counts.get(level, 0) + 1

    print("\n[generate_synthetic] Dataset Summary:")
    print(f"  Total scenarios: {len(dataset)}")
    print(f"  Risk distribution:")
    for level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        count = risk_counts.get(level, 0)
        pct = (count / len(dataset)) * 100
        print(f"    {level:10s}: {count:5d} ({pct:5.1f}%)")

    # Sample stats
    revenues = [r["monthly_revenue"] for r in dataset]
    purchases = [r["proposed_purchase_amount"] for r in dataset]
    runways = [r["predicted_runway_months"] for r in dataset]

    print(f"\n  Revenue range:  Rs {min(revenues):>12,.0f} - Rs {max(revenues):>12,.0f}")
    print(f"  Purchase range: Rs {min(purchases):>12,.0f} - Rs {max(purchases):>12,.0f}")
    print(f"  Runway range:   {min(runways):>8.1f} - {max(runways):>8.1f} months")


# ---------------------------------------------------------------------------
# Run directly to generate the dataset
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)  # Reproducible
    print("[generate_synthetic] Generating financial dataset...")
    data = generate_dataset()
    save_dataset(data)
    print_summary(data)
