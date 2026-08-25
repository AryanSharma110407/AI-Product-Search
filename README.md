# AI Procurement & Product Search Agent

An intelligent, autonomous product search and price comparison agent built for the **Razorpay Buildathon (AI Growth & Agentic Commerce track)**.

The project demonstrates the core research phase of a bounded-autonomy procurement engine. It takes a product search query, executes automated searches across major e-commerce platforms, processes the raw details using Gemini, and presents a comparison table with the best recommended deals.

---

## 🚀 Features

- **Multi-Platform Search**: Searches platforms like Amazon India (`amazon.in`), Flipkart (`flipkart.com`), Croma (`croma.com`), and more.
- **AI Synthesis**: Uses **Agno AI** (formerly Phidata) and **Google Gemini** (`gemini-2.5-flash`) to parse pricing, discounts, and product specs.
- **Red & Black Theme UI**: Distinct dark carbon design with crimson red accents, animated step indicators, and fully styled price comparison tables.
- **FastAPI Backend**: Single-script, lightweight Python backend handling client routing and API execution.

---

## 🛠️ Tech Stack

- **Framework**: [Agno AI](https://github.com/agno-ai/agno) (Agentic Workflow)
- **Language Model**: Google Gemini (`gemini-2.5-flash`)
- **Web Search Tool**: SerpApi (Google Search Results)
- **Web Framework**: FastAPI & Jinja2 Templates
- **Styling**: Tailwind CSS & custom styled components (Red & Black Cyberpunk theme)

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

## 🧠 Bounded Autonomy Workflow

```
       USER QUERY (e.g. "iPhone 16")
                    │
                    ▼
          ┌───────────────────┐
          │ FastAPI Endpoint  │
          └─────────┬─────────┘
                    │
                    ▼
          ┌───────────────────┐
          │  Agno AI Agent    │
          └─────────┬─────────┘
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
    SerpApi Tool        Gemini Model
   (Search Google)    (Process Results)
          │                   │
          └─────────┬─────────┘
                    ▼
          ┌───────────────────┐
          │ Markdown Output   │
          │ (Price Comparison)│
          └─────────┬─────────┘
                    ▼
           RENDERED DASHBOARD UI
```

---

