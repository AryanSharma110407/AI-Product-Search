# AI Procurement & Product Search Agent

An intelligent, autonomous product search and price comparison agent built for the **Razorpay Buildathon (AI Growth & Agentic Commerce track)**.

This project implements **Phase 4 (Research & Candidate Shortlisting)** of a complete bounded-autonomy procurement engine. It is decoupled as a reusable Python module that queries search engines, compiles pricing across platforms, processes details using Gemini with native JSON schemas, and returns validated JSON data for downstream engines (Decision, Finance, and Policy).

---

## 🚀 Features

- **Structured JSON Integration**: Enforces strict JSON outputs matching Pydantic schemas (`CategoryResearchResult`, `SingleProductResult`) using Gemini's native JSON mode.
- **Multi-Candidate Category Research**: Supports fetching and comparing a shortlist of 3–5 different product candidates per category (e.g. searching "office laptops" returns a list of distinct models from Dell, HP, Lenovo, etc.).
- **Multi-Platform Search**: Automatically queries platform-specific listings on Amazon India (`amazon.in`), Flipkart (`flipkart.com`), Croma (`croma.com`), and other Indian e-commerce sites using SerpApi.
- **Isolated Utility Module (`research_agent.py`)**: Decoupled from the web UI so it can be imported as a Python function directly into your main orchestrator backend.
- **Red & Black Themed Dashboard**: Translucent carbon black panels and glowing crimson red borders with step-by-step progress tracking for visual demos.
- **Robust Exception Handling**: Implements retry logic with exponential backoff for LLM rate limits and outputs fallback search results if SerpApi keys expire.

---

## 🛠️ Tech Stack

- **Agentic Framework**: [Agno AI](https://github.com/agno-ai/agno) (formerly Phidata)
- **Language Model**: Google Gemini (`gemini-2.5-flash`)
- **Web Search Engine**: SerpApi (Google Search Engine results)
- **Web Application**: FastAPI & Jinja2 Templates
- **Data Validation**: Pydantic v2
- **Styling**: Tailwind CSS (Red & Black Cyberpunk theme)

---

## 📦 Installation & Setup

1. **Clone the repository** (or navigate to the workspace directory).
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Environment Variables**:
   Create a `.env` file in the root folder and add your keys:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key
   SERP_API_KEY=your_serp_api_key
   ```
4. **Run the Application**:
   ```bash
   python app.py
   ```
5. Open your browser and navigate to `http://localhost:8000`.

---

## 🔬 Running Integration Tests

To test the structured JSON serialization of the research agent without launching the web server, run:
```bash
python test_research.py
```
This script executes:
1. A single-product price check (`search_product` for "Logitech MX Master 3S").
2. A multi-candidate category lookup (`research_category` for "24 inch IPS monitors").
3. Verification that returned outputs conform to the Pydantic schemas.

---

## 🧠 System Architecture

```
                       ORCHESTRATOR
                            │
              Calls Python Function or REST API
                            │
                            ▼
                  ┌───────────────────┐
                  │ research_agent.py │
                  └─────────┬─────────┘
                            │
                  ┌─────────┴─────────┐
                  ▼                   ▼
            SerpApi Tool        Gemini Model
           (Search Web)       (Enforce JSON)
                  │                   │
                  └─────────┬─────────┘
                            ▼
                  ┌───────────────────┐
                  │   Pydantic JSON   │
                  │   (Valid Schema)  │
                  └─────────┬─────────┘
                            ▼
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
 ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
 │   DECISION   │    │   FINANCE    │    │    POLICY    │
 │    ENGINE    │    │    ENGINE    │    │    ENGINE    │
 └──────────────┘    └──────────────┘    └──────────────┘
```

---

## 📡 API Endpoints

FastAPI exposes these interfaces:
- **`GET /`**: Renders the red-black interactive web dashboard.
- **`POST /api/search`**: Dashboard UI endpoint. Takes a query and returns rendered HTML tables for display.
- **`POST /api/research`**: Core JSON endpoint. Takes a query and returns `SingleProductResult` JSON.
- **`POST /api/research/category`**: Multi-candidate endpoint. Takes a category search payload and returns `CategoryResearchResult` JSON.
