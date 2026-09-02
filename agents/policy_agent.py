"""
agents/policy_agent.py
----------------------
Policy decision reasoning agent.

Provides functions that combine deterministic financial checks with
RAG policy lookups to produce structured compliance verdicts.

This module does NOT contain the LLM agent itself — it provides
the deterministic tools that the orchestrator's LLM calls.
"""

from database.db_setup import get_department_finances, verify_financial_risk
from rag_engine.policy_rag_tool import check_company_policy


def evaluate_purchase_compliance(
    department_name: str,
    product_name: str,
    quantity: int,
    unit_price: float,
) -> dict:
    """
    Run a full compliance check on a proposed purchase.

    Combines:
      1. Department financial snapshot (deterministic DB lookup)
      2. Financial risk evaluation (deterministic Python math)
      3. Policy compliance check (RAG lookup)

    Returns a structured verdict dict with all evidence the LLM
    needs to make its final decision.
    """
    # 1. Get department finances
    finances = get_department_finances(department_name)
    if "error" in finances:
        return {"error": finances["error"], "verdict": "BLOCKED"}

    # 2. Run financial risk check (deterministic)
    risk = verify_financial_risk(department_name, quantity, unit_price)
    if "error" in risk:
        return {"error": risk["error"], "verdict": "BLOCKED"}

    # 3. Query company policy via RAG
    policy_query = (
        f"What are the procurement rules for {department_name} department "
        f"buying {product_name}? What is the spending limit? "
        f"Are there any brand restrictions?"
    )
    policy_context = check_company_policy(policy_query, top_k=3)

    # 4. Compile verdict
    total_price = quantity * unit_price

    return {
        "department": department_name,
        "product_name": product_name,
        "quantity": quantity,
        "unit_price": unit_price,
        "total_price": total_price,
        "financial_snapshot": {
            "account_balance": finances["account_balance"],
            "available_balance": finances["available_balance"],
            "autonomous_limit": finances["autonomous_limit"],
            "total_commitments": finances["total_unpaid_commitments"],
        },
        "risk_assessment": {
            "risk_level": risk["risk_level"],
            "requires_approval": risk["requires_approval"],
            "projected_balance": risk["projected_balance"],
            "reason": risk["reason"],
        },
        "policy_context": policy_context,
        "verdict": risk["risk_level"],
    }
