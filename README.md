# TechNova Autonomous Agentic Procurement & Financial Risk System

A state-of-the-art **4-Tier Autonomous AI Procurement Engine** built for bounded-autonomy corporate purchasing. 

Unlike standard "shopping bots", this system combines **Historical Knowledge (Financial RAG)**, **Future Quantitative State Prediction (ML Forecasting Model)**, **Live Market Search (SerpApi + Gemini)**, and **Deterministic Financial Rules** into a unified, audit-logged decision framework.

---

## 🏛️ System Architecture Overview

```
                                  USER QUERY
                   ("Procure 3 Laptops for Engineering @ ₹55,000")
                                      │
                                      ▼
                           ┌─────────────────────┐
                           │   AGNO LLM AGENT    │
                           │   (Policy Engine)   │
                           └──────────┬──────────┘
                                      │
                   Determines required evidence & tools
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
┌──────────────┐              ┌──────────────┐              ┌──────────────┐
│ Financial RAG│              │ E-Commerce   │              │ ML Forecast  │
│ (Look Back)  │              │ Market Search│              │(Look Forward)│
└──────┬───────┘              └──────┬───────┘              └──────┬───────┘
       │                             │                             │
   Retrieves                     Finds & Filters               Predicts Future
Historical Docs,               Product Candidates             Cash Runway, Cash
Policies, Reports              (Price, Specs, URLs)           Position, Risk %
       │                             │                             │
       └─────────────────────────────┼─────────────────────────────┘
                                     ▼
                           ┌───────────────────┐
                           │     EVALUATOR     │
                           │  (Risk & Safety)  │
                           └─────────┬─────────┘
                                     │
                    Is Financial Risk Acceptable?
                      /                         \
            NO (High Risk)                   YES (Low Risk)
                 │                                 │
                 ▼                                 ▼
      Trigger Re-Optimization                Approve & Execute
      (Search Cheaper Candidate)             (Audit Log / Payout)
```

---

## 🧩 The 4 Core Component Tiers

| Component Tier | Role | Technology / Mechanism | Core Question Answered |
| :--- | :--- | :--- | :--- |
| **Tier 1: Department Treasury** | **Real-Time Accounts**: Live balances & commitments | SQLite DDL (`company_finances.db`) | *"What is our available cash headroom after unpaid salaries & bills?"* |
| **Tier 2: Financial RAG** | **Look Back**: Historical policies & rules | TF-IDF Semantic Search over `rag_engine/company_policy.md` | *"What are our spending limits, brand restrictions, and CTO sign-off rules?"* |
| **Tier 3: E-Commerce Search** | **Live Market Context**: Real-time product pricing | `agents/research_agent.py` + SerpApi + Gemini | *"What products exist under our budget ceiling from Amazon, Flipkart, & Croma?"* |
| **Tier 4: ML Forecast Model** | **Look Forward**: 3-Month quantitative prediction | Trained XGBoost Regressor & Classifier (`ml_engine/`) | *"If we spend ₹X today, what will our cash runway and risk level be in 90 days?"* |
| **Tier 5: Decision & Audit Engine** | **Reasoning & Audit**: Decision & audit trail | Agno AI Agent + SQLite (`database/audit_logger.py`) | *"Is this purchase APPROVED, PENDING_APPROVAL, or BLOCKED?"* |

---

## 📂 Modular Repository Structure

```text
Build RP/
├── app.py                      # Main FastAPI server & Cyberpunk Dashboard API
├── orchestrator.py             # Agno AI Orchestrator Agent
├── company_finances.db         # SQLite Treasury Database
├── requirements.txt            # Python dependencies
├── README.md
├── .env                        # API keys (GOOGLE_API_KEY, SERP_API_KEY)
├── .gitignore
│
├── agents/                     # Agent Sub-modules
│   ├── research_agent.py       # E-Commerce Search & Budget Filter Adapter
│   └── policy_agent.py         # Policy & Decision Reasoning
│
├── ml_engine/                  # Financial ML Model (Look Forward)
│   ├── generate_synthetic.py   # Economically sound dataset generator (12,000 scenarios)
│   ├── train_model.py          # XGBoost training script (R²=0.9997, Accuracy=97.8%)
│   └── forecast_tool.py        # ML Prediction Tool callable by Agno AI
│
├── rag_engine/                 # Financial RAG System (Look Back)
│   ├── company_policy.md       # Corporate procurement policy document
│   └── policy_rag_tool.py      # TF-IDF semantic vector search tool
│
├── database/                   # Database & Audit Engine
│   ├── db_setup.py             # SQLite DDL initializer & seed data
│   └── audit_logger.py         # Procurement audit log recorder
│
└── templates/                  # Web Interface
    └── index.html              # Cyberpunk Red & Black Telemetry Dashboard UI
```

---

## 📊 ML Forecasting Model Performance

The **Look Forward ML Model** is trained on 12,000 realistic company-quarter financial scenarios with economically sound relationships:

- **Features**: `current_cash`, `monthly_revenue`, `monthly_expenses`, `debt_obligations`, `burn_rate`, `proposed_purchase_amount`, `purchase_category`
- **Targets**: `predicted_cash_3m`, `predicted_runway_months`, `cash_shortage_probability`, `financial_risk_level`

### Benchmark Results:
- **XGBoost Regressor (3-Month Cash Prediction)**: R² = **0.9997**, MAE = **₹5.5L**
- **XGBoost Regressor (Runway Months)**: R² = **0.9981**, MAE = **0.14 months**
- **XGBoost Classifier (Risk Level)**: Accuracy = **97.8%** across `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`

---

## 📦 Quick Start & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_gemini_api_key
SERP_API_KEY=your_serp_api_key
```

### 3. Initialize & Seed Database (Only once)
```bash
python -m database.db_setup
```

### 4. Generate Dataset & Train ML Model (Only once)
```bash
python -m ml_engine.generate_synthetic
python -m ml_engine.train_model
```

### 5. Launch the Web Application
```bash
python app.py
```
Open your browser and navigate to **`http://127.0.0.1:8000`**.

### 6. Test CLI Orchestrator (Optional)
```bash
python orchestrator.py
```

---

## 🎨 Cyberpunk Web Dashboard Features

- **Live 5-Step Telemetry Indicator**: Real-time percentage progress and stopwatch ticker (`Elapsed: X.Xs`).
- **Department Treasury Cards**: Shows live account balance, unpaid commitments, available headroom, and auto-approval limits.
- **RAG Policy Viewer**: Highlights retrieved policy sections and CTO sign-off warning flags (e.g. Apple MacBook purchases).
- **Market Deals Table**: Displays candidate products sorted by price with budget savings.
- **ML Cash Forecast Card**: Displays predicted 3-month cash position, runway months, shortage probability, and ML risk level badge.
- **Decision & Audit Banner**: Renders approval status badge (`APPROVED`, `PENDING_APPROVAL`, `BLOCKED`) with human-readable rationale and SQLite Audit Log ID.

---

## 📜 License & Compliance

Built for enterprise bounded-autonomy procurement demonstrations. All procurement decisions are recorded in the `procurement_logs` SQLite table for compliance review.
