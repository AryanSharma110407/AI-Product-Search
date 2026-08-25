"""
app.py
──────
FastAPI application that serves two interfaces:

1. Dashboard UI (GET /)         — Red & Black themed web page for humans
2. JSON search  (POST /api/search)       — Returns rendered HTML for the dashboard
3. Structured   (POST /api/research)     — Returns raw JSON for the orchestrator
4. Category     (POST /api/research/category) — Returns multi-candidate JSON
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

from research_agent import (
    search_product,
    research_category,
    SingleProductResult,
    CategoryResearchResult,
)

load_dotenv()

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="AI Product Search Agent")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main dashboard page."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/api/search")
async def search_product_ui(request: Request):
    """
    Dashboard endpoint — returns rendered HTML for the UI.

    Accepts JSON: { "query": "iPhone 15 128GB" }
    Returns JSON: { "result_html": "<table>...</table>", "raw_json": {...} }
    """
    body = await request.json()
    query = body.get("query", "").strip()

    if not query:
        return JSONResponse({"error": "Please enter a product to search."}, status_code=400)

    try:
        result: SingleProductResult = search_product(query)

        if result.error:
            return JSONResponse({"error": result.error}, status_code=500)

        # Build a markdown table from structured data for the UI
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
            md_lines.append(f"\n### 🏆 Best Deal\n")
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


@app.post("/api/research")
async def research_product_json(request: Request):
    """
    Orchestrator endpoint — returns raw structured JSON.

    Accepts JSON: { "query": "Dell Inspiron 15 laptop" }
    Returns JSON: SingleProductResult schema
    """
    body = await request.json()
    query = body.get("query", "").strip()

    if not query:
        return JSONResponse({"error": "Query is required."}, status_code=400)

    result: SingleProductResult = search_product(query)
    return JSONResponse(result.model_dump())


@app.post("/api/research/category")
async def research_category_json(request: Request):
    """
    Orchestrator endpoint — returns multi-candidate structured JSON for a category.

    Accepts JSON: {
        "category": "office laptops",
        "count": 3,
        "budget_hint": "under 45000"
    }
    Returns JSON: CategoryResearchResult schema
    """
    body = await request.json()
    category = body.get("category", "").strip()
    count = body.get("count", 3)
    budget_hint = body.get("budget_hint")

    if not category:
        return JSONResponse({"error": "Category is required."}, status_code=400)

    result: CategoryResearchResult = research_category(
        category=category,
        count=count,
        budget_hint=budget_hint,
    )
    return JSONResponse(result.model_dump())


# ---------------------------------------------------------------------------
# Run with:  python app.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # pyrefly: ignore [missing-import]
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
