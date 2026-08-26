"""
research_agent.py
─────────────────
Reusable product research module.

Exports two functions:
  • search_product(query)           → structured JSON for a single product
  • research_category(category, n)  → structured JSON with N candidates for a category

All outputs conform to Pydantic schemas so downstream engines
(Decision, Finance, Policy) can consume them directly.
"""

import os
import json
import time
import logging
from typing import List, Dict, Optional

from pydantic import BaseModel, Field
from dotenv import load_dotenv

# pyrefly: ignore [missing-import]
from agno.agent import Agent, RunResponse
# pyrefly: ignore [missing-import]
from agno.tools.serpapi import SerpApiTools
# pyrefly: ignore [missing-import]
from agno.models.groq import Groq

load_dotenv()

SERP_API_KEY = os.getenv("SERP_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic Schemas — the contract between Research and Decision Engine
# ---------------------------------------------------------------------------

class ProductSpecs(BaseModel):
    ram: Optional[str] = Field(None, description="e.g. '16GB'")
    storage: Optional[str] = Field(None, description="e.g. '512GB SSD'")
    cpu: Optional[str] = Field(None, description="e.g. 'Intel Core i5-1335U'")
    screen_size: Optional[str] = Field(None, description="e.g. '15.6 inch'")
    warranty: Optional[str] = Field(None, description="e.g. '1 Year Onsite'")
    extra_details: Dict[str, str] = Field(
        default_factory=dict,
        description="Any other relevant specs not captured above",
    )


class EcomProductCandidate(BaseModel):
    product_name: str = Field(..., description="Full product name including variant")
    price_inr: int = Field(..., description="Current selling price in INR")
    original_mrp: Optional[int] = Field(None, description="MRP before discount, if available")
    platform: str = Field(..., description="Source platform, e.g. 'Amazon.in'")
    source_url: str = Field("", description="Direct URL to the product listing")
    specs: ProductSpecs = Field(default_factory=ProductSpecs, description="Hardware specifications")
    availability: bool = Field(True, description="Whether the product is in stock")


class CategoryResearchResult(BaseModel):
    category: str = Field(..., description="Equipment category, e.g. 'laptops'")
    query_used: str = Field(..., description="The search query that produced these results")
    candidates: List[EcomProductCandidate] = Field(
        default_factory=list,
        description="3-5 product candidates for this category",
    )
    error: Optional[str] = Field(None, description="Error message if search failed")


class SingleProductResult(BaseModel):
    query: str = Field(..., description="The original user query")
    results: List[EcomProductCandidate] = Field(
        default_factory=list,
        description="Product listings found across platforms",
    )
    best_deal: Optional[EcomProductCandidate] = Field(
        None, description="The lowest-priced available option",
    )
    error: Optional[str] = Field(None, description="Error message if search failed")


# ---------------------------------------------------------------------------
# JSON output prompt — instructs Gemini to return strict JSON, not markdown
# ---------------------------------------------------------------------------

SINGLE_PRODUCT_PROMPT = """
You are a product-price research agent for Indian buyers.

**Your output MUST be valid JSON matching the schema below. Do NOT output markdown, tables, or prose.**

Output Schema:
{{
  "query": "<the user's original query>",
  "results": [
    {{
      "product_name": "<full product name>",
      "price_inr": <integer price in INR>,
      "original_mrp": <integer MRP or null>,
      "platform": "<e.g. Amazon.in>",
      "source_url": "<URL or empty string>",
      "specs": {{
        "ram": "<e.g. 16GB or null>",
        "storage": "<e.g. 512GB SSD or null>",
        "cpu": "<e.g. Intel Core i5 or null>",
        "screen_size": "<e.g. 15.6 inch or null>",
        "warranty": "<e.g. 1 Year or null>",
        "extra_details": {{}}
      }},
      "availability": true
    }}
  ],
  "best_deal": {{ <the result object with the lowest price_inr where availability is true, or null> }},
  "error": null
}}

**Workflow:**
1. Search for the product on: amazon.in, flipkart.com, croma.com, reliancedigital.in, vijaysales.com
2. Extract the product name, price, MRP, specs, and URL from each platform.
3. If a product is not found on a platform, skip that platform entirely (do NOT add a result with made-up data).
4. Set "best_deal" to the result with the lowest price_inr among available items.
5. Never invent prices or specs. If you cannot find data, omit the entry.
6. Return ONLY the JSON object. No markdown fences, no explanation text.
"""

CATEGORY_RESEARCH_PROMPT = """
You are a procurement research agent for Indian businesses.

**Your output MUST be valid JSON matching the schema below. Do NOT output markdown, tables, or prose.**

Output Schema:
{{
  "category": "<category name>",
  "query_used": "<the search query you used>",
  "candidates": [
    {{
      "product_name": "<full product name>",
      "price_inr": <integer price in INR>,
      "original_mrp": <integer MRP or null>,
      "platform": "<e.g. Amazon.in>",
      "source_url": "<URL or empty string>",
      "specs": {{
        "ram": "<or null>",
        "storage": "<or null>",
        "cpu": "<or null>",
        "screen_size": "<or null>",
        "warranty": "<or null>",
        "extra_details": {{}}
      }},
      "availability": true
    }}
  ],
  "error": null
}}

**Workflow:**
1. The user will give you a category (e.g. "laptops for office use under 50000").
2. Search across amazon.in, flipkart.com, croma.com for {count} different product options in that category.
3. Pick candidates from DIFFERENT brands where possible (e.g. one Dell, one HP, one Lenovo).
4. Extract real specs and prices for each candidate.
5. Never invent prices. If a search returns nothing usable, set "error" to a description and return an empty candidates list.
6. Return ONLY the JSON object. No markdown fences, no explanation text.
"""


# ---------------------------------------------------------------------------
# Agent builder — shared core
# ---------------------------------------------------------------------------

def _build_agent(system_prompt: str) -> Agent:
    """Build an Agno Agent configured for structured JSON output."""
    llm = Groq(
        id="openai/gpt-oss-120b",
        api_key=GROQ_API_KEY,
        temperature=0.1,
    )

    agent = Agent(
        name="Product Research Agent",
        role="Search the web for product pricing and return structured JSON data.",
        model=llm,
        tools=[
            SerpApiTools(api_key=SERP_API_KEY),
        ],
        description=[
            "You are a product research agent that returns ONLY valid JSON. "
            "Never return markdown. Never return prose."
        ],
        instructions=system_prompt,
        markdown=False,
        show_tool_calls=False,
    )
    return agent


def _safe_parse_json(raw: str) -> dict:
    """
    Attempt to parse raw LLM output as JSON.
    Handles common issues like markdown code fences wrapping the JSON.
    """
    text = raw.strip()

    # Strip markdown code fences if the model wrapped the output
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        first_newline = text.index("\n")
        text = text[first_newline + 1:]
    if text.endswith("```"):
        text = text[:-3].strip()

    return json.loads(text)


def _run_with_retry(agent: Agent, query: str, max_retries: int = 2) -> str:
    """
    Run the agent with exponential backoff on failure.
    Returns the raw response content string.
    """
    for attempt in range(max_retries + 1):
        try:
            response: RunResponse = agent.run(query)
            return response.content
        except Exception as e:
            logger.warning(f"Agent attempt {attempt + 1} failed: {e}")
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.info(f"Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_product(query: str) -> SingleProductResult:
    """
    Search for a single product across Indian e-commerce platforms.

    Returns a SingleProductResult with structured data.
    If the search fails entirely, the error field will be populated.

    Example:
        result = search_product("iPhone 16 128GB")
        print(result.best_deal.price_inr)
    """
    try:
        agent = _build_agent(SINGLE_PRODUCT_PROMPT)
        raw = _run_with_retry(agent, query)
        data = _safe_parse_json(raw)
        return SingleProductResult(**data)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from Gemini: {e}")
        return SingleProductResult(
            query=query,
            error=f"Gemini returned malformed JSON: {str(e)[:200]}",
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return SingleProductResult(
            query=query,
            error=f"Search failed: {str(e)[:200]}",
        )


def research_category(
    category: str,
    count: int = 3,
    budget_hint: Optional[str] = None,
) -> CategoryResearchResult:
    """
    Research a product category and return multiple candidates.

    Args:
        category:    e.g. "office laptops", "24 inch IPS monitors"
        count:       number of candidates to return (default 3)
        budget_hint: optional budget constraint, e.g. "under 50000"

    Returns a CategoryResearchResult with structured candidate data.

    Example:
        result = research_category("office laptops", count=4, budget_hint="under 45000")
        for c in result.candidates:
            print(c.product_name, c.price_inr)
    """
    prompt = CATEGORY_RESEARCH_PROMPT.replace("{count}", str(count))

    search_query = f"best {category} in India"
    if budget_hint:
        search_query += f" {budget_hint}"
    search_query += f" — find {count} different options from different brands"

    try:
        agent = _build_agent(prompt)
        raw = _run_with_retry(agent, search_query)
        data = _safe_parse_json(raw)
        return CategoryResearchResult(**data)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from Gemini: {e}")
        return CategoryResearchResult(
            category=category,
            query_used=search_query,
            error=f"Gemini returned malformed JSON: {str(e)[:200]}",
        )
    except Exception as e:
        logger.error(f"Category research failed: {e}")
        return CategoryResearchResult(
            category=category,
            query_used=search_query,
            error=f"Research failed: {str(e)[:200]}",
        )
