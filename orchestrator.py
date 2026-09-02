"""
orchestrator.py
---------------
Agno AI Orchestrator — the central agentic brain.

This is the Policy Engine that:
  1. Receives a procurement request (e.g. "Buy 5 laptops for Engineering")
  2. Calls tools to gather evidence:
     - Financial RAG (Look Back): Company policies & rules
     - E-Commerce Search (Live Market): Product candidates & prices
     - ML Forecast Model (Look Forward): 3-month cash impact prediction
     - Database Tools: Department finances & risk evaluation
  3. Reasons over all evidence and produces a structured decision
  4. Logs the decision in the audit trail

Run directly to test:
    python orchestrator.py
"""

import os
import json
import logging
from dotenv import load_dotenv

# pyrefly: ignore [missing-import]
from agno.agent import Agent, RunResponse
# pyrefly: ignore [missing-import]
from agno.models.google import Gemini

# Import tools
from database.db_setup import get_department_finances, verify_financial_risk
from database.audit_logger import log_procurement_decision, request_human_approval
from rag_engine.policy_rag_tool import check_company_policy
from agents.research_agent import search_product_deals

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool wrappers — these are the functions the Agno agent can call
# ---------------------------------------------------------------------------

def tool_check_finances(department: str) -> str:
    """Check the current financial state of a department. Returns account balance, commitments, and available funds."""
    result = get_department_finances(department)
    return json.dumps(result, indent=2)


def tool_evaluate_risk(department: str, quantity: int, unit_price: float) -> str:
    """Evaluate the financial risk of a proposed purchase using deterministic python engine rules."""
    result = verify_financial_risk(department, quantity, unit_price)
    return json.dumps(result, indent=2)


def tool_search_products(query: str, budget_ceiling: float) -> str:
    """Search for products online and return only those within the budget ceiling. Products above budget are automatically filtered out."""
    deals = search_product_deals(query, budget_ceiling)
    if not deals:
        return json.dumps({"message": "No products found within the budget ceiling.", "deals": []})
    return json.dumps({"deals": deals, "count": len(deals)}, indent=2)


def tool_check_policy(query: str) -> str:
    """Look up company procurement policy rules. Use this to check brand restrictions, approval requirements, and spending limits."""
    return check_company_policy(query, top_k=3)


def tool_log_decision(
    department: str,
    product_name: str,
    quantity: int,
    unit_price: float,
    status: str,
    risk_level: str,
    reason: str,
) -> str:
    """Log a procurement decision in the audit trail. Use status: APPROVED, PENDING_APPROVAL, BLOCKED, or REJECTED."""
    result = log_procurement_decision(
        department_name=department,
        product_name=product_name,
        quantity=quantity,
        unit_price=unit_price,
        status=status,
        risk_level=risk_level,
        reason=reason,
    )
    return json.dumps(result, indent=2)


def tool_request_approval(
    department: str,
    product_name: str,
    total_price: float,
    reason: str,
) -> str:
    """Request human/manager approval for a purchase that exceeds autonomous limits or has high financial risk."""
    result = request_human_approval(
        department_name=department,
        product_name=product_name,
        total_price=total_price,
        reason=reason,
    )
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Orchestrator System Prompt
# ---------------------------------------------------------------------------

ORCHESTRATOR_SYSTEM_PROMPT = """
You are TechNova's AI Procurement Agent — an autonomous system that evaluates
and processes corporate purchase requests.

You have access to the following tools. Use them IN ORDER to make informed decisions:

## Your Decision Workflow

### Step 1: Understand the Request
Parse the user's procurement request to identify:
- Department making the request
- Product type / category
- Quantity needed
- Any specific brand or model preferences

### Step 2: Check Department Finances
Call `tool_check_finances(department)` to get the current financial snapshot:
- Account balance, available funds, autonomous limit, existing commitments

### Step 3: Check Company Policy
Call `tool_check_policy(query)` to retrieve relevant procurement rules:
- Brand restrictions (e.g., Apple requires CTO sign-off)
- Department spending limits
- Approval requirements

### Step 4: Search for Products
Call `tool_search_products(query, budget_ceiling)` to find available products.
Use the department's available balance as the budget ceiling.
The system will automatically filter out products above the budget.

### Step 5: Evaluate Financial Risk
For the best candidate product, call `tool_evaluate_risk(department, quantity, unit_price)`
to get a deterministic risk assessment.

### Step 6: Make Your Decision & Log Audit Trail
Based on ALL evidence gathered:
- If risk is LOW and policy is satisfied: APPROVE and log the decision
- If risk is MEDIUM: APPROVE with caution notes, or suggest alternatives
- If risk is HIGH: Request human approval via `tool_request_approval()`
- If risk is BLOCKED: Reject or search for cheaper alternatives

Always call `tool_log_decision(...)` to record your final decision in the audit trail.

## Important Rules
1. NEVER do financial calculations yourself. Always use the deterministic risk tool.
2. NEVER approve a purchase that violates company policy.
3. ALWAYS check policy before making a decision.
4. If a product requires CTO sign-off (e.g., Apple products), flag it.
5. Provide a clear, structured summary of your decision and reasoning.
"""


# ---------------------------------------------------------------------------
# Build the Orchestrator Agent
# ---------------------------------------------------------------------------

def build_orchestrator() -> Agent:
    """Build the Agno AI orchestrator agent with deterministic tools."""
    llm = Gemini(
        id="gemini-2.0-flash",
        api_key=GOOGLE_API_KEY,
        temperature=0.2,
        max_output_tokens=8192,
    )

    agent = Agent(
        name="TechNova Procurement Orchestrator",
        role="Autonomous procurement agent that evaluates purchase requests using financial data, company policy, market search, and deterministic financial rules.",
        model=llm,
        tools=[
            tool_check_finances,
            tool_evaluate_risk,
            tool_search_products,
            tool_check_policy,
            tool_log_decision,
            tool_request_approval,
        ],
        instructions=ORCHESTRATOR_SYSTEM_PROMPT,
        markdown=True,
        show_tool_calls=True,
    )
    return agent


def run_procurement_request(request: str) -> str:
    """
    Process a procurement request through the orchestrator.

    Args:
        request: Natural language procurement request, e.g.
                 "Engineering needs 5 Dell laptops under 50000 each"

    Returns:
        The agent's full response with decision and reasoning.
    """
    agent = build_orchestrator()
    response: RunResponse = agent.run(request)
    return response.content


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("  TechNova AI Procurement Orchestrator")
    print("=" * 70)

    test_request = (
        "Engineering department needs 3 laptops for new developers. "
        "Budget is around Rs 50,000 per laptop. "
        "Prefer Dell or Lenovo. Need at least 8GB RAM and 512GB SSD."
    )

    print(f"\nRequest: {test_request}\n")
    print("-" * 70)

    result = run_procurement_request(test_request)
    print(result)
