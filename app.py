"""
app.py
──────
FastAPI application for the TechNova Agentic Procurement Orchestrator.

Serves:
  1. Main Web Dashboard (GET /)          — Cyberpunk UI for enterprise procurement
  2. Procurement Pipeline (POST /api/procure) — Runs full 4-tier orchestrator telemetry
  3. Legacy Product Search (POST /api/search) — Live e-commerce search
"""

import os
import json

# pyrefly: ignore [missing-import]
import markdown as md_lib
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, Request
# pyrefly: ignore [missing-import]
from fastapi.responses import HTMLResponse, JSONResponse
# pyrefly: ignore [missing-import]
from fastapi.templating import Jinja2Templates

from database.db_setup import get_department_finances, verify_financial_risk
from database.audit_logger import log_procurement_decision, request_human_approval
from rag_engine.policy_rag_tool import check_company_policy
from ml_engine.forecast_tool import predict_financial_impact
from agents.research_agent import (
    search_product_deals,
    search_product,
    research_category,
    SingleProductResult,
)

load_dotenv(override=True)

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="TechNova Agentic Procurement System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main enterprise procurement dashboard."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/departments")
async def get_departments():
    """Get list of active departments and their financial snapshots."""
    departments = ["Engineering", "Sales", "Marketing", "HR", "Operations", "Finance"]
    snapshots = []
    for dept in departments:
        snapshot = get_department_finances(dept)
        if "error" not in snapshot:
            snapshots.append(snapshot)
    return JSONResponse({"departments": snapshots})


@app.post("/api/procure")
async def run_procurement_pipeline(request: Request):
    """
    Full 4-Tier Agentic Procurement Pipeline Endpoint.

    Accepts JSON:
      {
        "department": "Engineering",
        "product_query": "Dell Inspiron 15 laptop",
        "quantity": 3,
        "unit_budget": 55000,
        "category": "Hardware"
      }

    Returns structured telemetry across all 4 tiers:
      1. Treasury Snapshot (Department finances & available balance)
      2. Policy RAG Search (Company rules & brand restrictions)
      3. Live Market Candidates (Budget-filtered products)
      4. ML Financial Forecast (Predicted 3-month cash, runway & risk score)
      5. Final Decision & Audit Log ID
    """
    body = await request.json()

    department = body.get("department", "Engineering").strip()
    product_query = body.get("product_query", "").strip()
    quantity = int(body.get("quantity", 1))
    unit_budget = float(body.get("unit_budget", 50000))
    category = body.get("category", "Hardware").strip()

    if not product_query:
        return JSONResponse({"error": "Please enter a product query."}, status_code=400)

    total_requested_budget = quantity * unit_budget

    # ── Tier 1: Department Treasury Snapshot ──────────────────────────
    finances = get_department_finances(department)
    if "error" in finances:
        return JSONResponse({"error": finances["error"]}, status_code=400)

    # ── Tier 2: Financial Policy RAG ──────────────────────────────────
    policy_query = (
        f"What are the procurement rules for {department} department "
        f"buying {product_query}? What is the spending limit? "
        f"Are there any brand restrictions or CTO approval requirements?"
    )
    policy_text = check_company_policy(policy_query, top_k=3)

    # Check for Apple CTO sign-off restriction flag
    requires_cto_signoff = "apple" in product_query.lower() or "macbook" in product_query.lower()

    # ── Tier 3: Live E-Commerce Market Search & Budget Filter ────────
    deals = search_product_deals(product_query, budget_ceiling=unit_budget)

    # Pick the best deal candidate if available, else fall back to target budget
    best_candidate = deals[0] if deals and "error" not in deals[0] else None
    actual_unit_price = best_candidate["price_inr"] if best_candidate else unit_budget
    total_cost = quantity * actual_unit_price

    # ── Tier 4: ML Financial Forecast Model (Look Forward) ─────────────
    current_cash = finances["account_balance"]
    # Estimate monthly revenue/expenses based on department account
    monthly_revenue = current_cash * 0.45
    monthly_expenses = current_cash * 0.35
    debt_obligations = current_cash * 0.04

    ml_forecast = predict_financial_impact(
        current_cash=current_cash,
        monthly_revenue=monthly_revenue,
        monthly_expenses=monthly_expenses,
        debt_obligations=debt_obligations,
        proposed_purchase_amount=total_cost,
        purchase_category=category,
    )

    # ── Tier 5: Deterministic Evaluation & Decision Engine ───────────
    risk_evaluation = verify_financial_risk(department, quantity, actual_unit_price)

    # Determine final decision status
    if requires_cto_signoff:
        final_status = "PENDING_CTO_APPROVAL"
        risk_level = "HIGH"
        reason = f"Apple products require CTO sign-off as per company policy (Section 2.1). {risk_evaluation['reason']}"
    elif risk_evaluation["risk_level"] == "BLOCKED" or ml_forecast["financial_risk_level"] == "CRITICAL":
        final_status = "BLOCKED"
        risk_level = "CRITICAL"
        reason = f"Purchase blocked due to severe financial risk: {risk_evaluation['reason']}"
    elif risk_evaluation["requires_approval"] or ml_forecast["financial_risk_level"] == "HIGH":
        final_status = "PENDING_APPROVAL"
        risk_level = "HIGH"
        reason = f"Purchase exceeds autonomous limit of Rs {finances['autonomous_limit']:,.0f}. {risk_evaluation['reason']}"
    else:
        final_status = "APPROVED"
        risk_level = ml_forecast["financial_risk_level"]
        reason = risk_evaluation["reason"]

    # ── Audit Log Recording ───────────────────────────────────────────
    log_entry = log_procurement_decision(
        department_name=department,
        product_name=best_candidate["product_name"] if best_candidate else product_query,
        quantity=quantity,
        unit_price=actual_unit_price,
        status=final_status,
        risk_level=risk_level,
        reason=reason,
    )

    # Build response payload
    return JSONResponse({
        "status": "success",
        "request": {
            "department": department,
            "product_query": product_query,
            "quantity": quantity,
            "unit_budget": unit_budget,
            "total_requested": total_requested_budget,
        },
        "treasury": finances,
        "policy_rag": {
            "query": policy_query,
            "matched_rules": policy_text,
            "requires_cto_signoff": requires_cto_signoff,
        },
        "market_deals": deals,
        "best_deal": best_candidate,
        "ml_forecast": ml_forecast,
        "decision": {
            "status": final_status,
            "risk_level": risk_level,
            "actual_unit_price": actual_unit_price,
            "total_cost": total_cost,
            "reason": reason,
            "audit_log_id": log_entry.get("log_id"),
        },
    })


@app.post("/api/search")
async def search_product_ui(request: Request):
    """Legacy UI endpoint for simple product queries."""
    body = await request.json()
    query = body.get("query", "").strip()

    if not query:
        return JSONResponse({"error": "Please enter a product to search."}, status_code=400)

    try:
        result: SingleProductResult = search_product(query)

        if result.error:
            return JSONResponse({"error": result.error}, status_code=500)

        md_lines = [
            f"## Price Comparison: {query}\n",
            "| Platform | Product | Price (₹) | MRP (₹) | Link |",
            "|----------|---------|-----------|---------|------|",
        ]
        for r in result.results:
            mrp = f"₹{r.original_mrp:,}" if r.original_mrp else "—"
            link = f"[View]({r.source_url})" if r.source_url else "—"
            md_lines.append(
                f"| {r.platform} | {r.product_name} | ₹{r.price_inr:,} | {mrp} | {link} |"
            )

        if result.best_deal:
            bd = result.best_deal
            md_lines.append(f"\n### Best Deal\n")
            md_lines.append(
                f"**{bd.product_name}** on **{bd.platform}** at **₹{bd.price_inr:,}**"
            )

        rendered_md = "\n".join(md_lines)
        result_html = md_lib.markdown(rendered_md, extensions=["tables", "fenced_code"])

        return JSONResponse({
            "result_html": result_html,
            "raw_json": result.model_dump(),
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
