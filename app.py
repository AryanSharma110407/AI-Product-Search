import os
import markdown
from dataclasses import dataclass

from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, Request
# pyrefly: ignore [missing-import]
from fastapi.responses import HTMLResponse, JSONResponse
# pyrefly: ignore [missing-import]
from fastapi.templating import Jinja2Templates
# pyrefly: ignore [missing-import]
from agno.agent import Agent, RunResponse
# pyrefly: ignore [missing-import]
from agno.tools.serpapi import SerpApiTools
# pyrefly: ignore [missing-import]
from agno.models.google import Gemini

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")

# ---------------------------------------------------------------------------
# Agent builder
# ---------------------------------------------------------------------------

@dataclass
class ProductSearchAgent:
    """
    A lightweight product-search agent that uses Gemini + SerpApi
    to compare prices across Indian e-commerce platforms.
    """

    def build(self) -> Agent:
        llm = Gemini(
            id="gemini-2.5-flash",
            api_key=GOOGLE_API_KEY,
            temperature=0.1,
            max_output_tokens=4096,
        )

        agent = Agent(
            name="Product Price Comparator",
            role="Search the web for current product pricing across Indian e-commerce platforms and return a structured comparison.",
            model=llm,
            tools=[
                SerpApiTools(api_key=SERP_API_KEY),
            ],
            description=[
                "You are a product search expert that finds CURRENT and VERIFIED pricing information from Indian e-commerce websites."
            ],
            instructions="""
You are a professional product-price comparison agent for Indian buyers.

**Workflow – follow every step in order:**

Step 1  SEARCH
  For the product the user asks about, run separate searches for each of these platforms:
    • "{product}" price on amazon.in
    • "{product}" price on flipkart.com
    • "{product}" price on croma.com
    • "{product}" price on reliancedigital.in
    • "{product}" price on vijaysales.com

Step 2  EXTRACT
  From the search results, pull out:
    • Exact product name / variant
    • Current selling price (₹)
    • MRP / original price if available
    • Any discount or coupon
    • Seller / platform
    • Product URL (if found)

Step 3  COMPARE
  Build a comparison table in Markdown with these columns:
    | Platform | Product Name | Price (₹) | MRP (₹) | Discount | Link |

Step 4  RECOMMEND
  Below the table, write a short "Best Deal" section:
    • Which platform has the lowest verified price
    • Any notable deals or coupons
    • A one-line recommendation

**Rules:**
- Always use Indian Rupees (₹).
- If a product is unavailable on a platform, write "Not Found" in that row.
- Never invent prices. If you cannot find a price, say so.
- Keep the output clean, short, and scannable.
""",
            markdown=True,
            show_tool_calls=False,
        )
        return agent

    def search(self, query: str) -> str:
        """Run a search query and return the agent's markdown response."""
        agent = self.build()
        response: RunResponse = agent.run(query)
        return response.content


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="Product Search Agent")
templates = Jinja2Templates(directory="templates")
agent = ProductSearchAgent()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main dashboard page."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/api/search")
async def search_product(request: Request):
    """
    Accepts JSON: { "query": "iPhone 15 128GB" }
    Returns JSON: { "result_html": "<rendered markdown>", "raw": "..." }
    """
    body = await request.json()
    query = body.get("query", "").strip()

    if not query:
        return JSONResponse({"error": "Please enter a product to search."}, status_code=400)

    try:
        raw_md = agent.search(query)
        result_html = markdown.markdown(raw_md, extensions=["tables", "fenced_code"])
        return JSONResponse({"result_html": result_html, "raw": raw_md})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# Run with:  python app.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # pyrefly: ignore [missing-import]
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
