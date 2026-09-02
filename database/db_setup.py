"""
database/db_setup.py
--------------------
SQLite database initializer with DDL schema and realistic seed data.

Creates `company_finances.db` with tables for:
  - financial_accounts (company bank accounts with balances & reserves)
  - departments (business units with autonomous spending limits)
  - cash_commitments (scheduled outflows: salaries, bills, vendor payments)
  - procurement_logs (audit trail for all agent purchasing decisions)

Run directly to initialize and seed the database:
    python -m database.db_setup
"""

import os
import sqlite3
from datetime import date, timedelta

# Database file lives in the project root
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "company_finances.db")


def get_connection() -> sqlite3.Connection:
    """Get a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def create_tables(conn: sqlite3.Connection) -> None:
    """Create all tables if they don't already exist."""
    cursor = conn.cursor()

    # 1. Financial Accounts
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS financial_accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_name VARCHAR(100) NOT NULL,
            current_balance DECIMAL(15, 2) NOT NULL,
            minimum_reserve DECIMAL(15, 2) NOT NULL
        )
    """)

    # 2. Departments
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS departments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(100) UNIQUE NOT NULL,
            account_id INTEGER,
            autonomous_limit DECIMAL(15, 2) NOT NULL,
            FOREIGN KEY(account_id) REFERENCES financial_accounts(id)
        )
    """)

    # 3. Cash Commitments
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cash_commitments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            department_id INTEGER,
            description VARCHAR(255) NOT NULL,
            amount DECIMAL(15, 2) NOT NULL,
            due_date DATE NOT NULL,
            type VARCHAR(50) CHECK(type IN ('SALARY', 'BILL', 'VENDOR', 'RENT', 'TAX', 'OTHER')),
            status VARCHAR(50) DEFAULT 'UNPAID',
            FOREIGN KEY(department_id) REFERENCES departments(id)
        )
    """)

    # 4. Procurement Logs (Audit Trail)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS procurement_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            department_id INTEGER,
            product_name VARCHAR(255) NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price DECIMAL(15, 2) NOT NULL,
            total_price DECIMAL(15, 2) NOT NULL,
            status VARCHAR(50) NOT NULL,
            risk_level VARCHAR(50) NOT NULL,
            reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(department_id) REFERENCES departments(id)
        )
    """)

    conn.commit()
    print("[db_setup] Tables created successfully.")


def seed_data(conn: sqlite3.Connection) -> None:
    """Insert realistic seed data for a mid-size Indian tech company."""
    cursor = conn.cursor()

    # Check if data already exists
    row = cursor.execute("SELECT COUNT(*) FROM financial_accounts").fetchone()
    if row[0] > 0:
        print("[db_setup] Database already seeded. Skipping.")
        return

    # ── Financial Accounts ──────────────────────────────────────────
    accounts = [
        ("Operating Account", 2500000.00, 500000.00),   # Rs 25L balance, Rs 5L reserve
        ("Capital Expenditure Fund", 800000.00, 200000.00),  # Rs 8L balance, Rs 2L reserve
        ("Emergency Reserve", 1500000.00, 1000000.00),  # Rs 15L balance, Rs 10L reserve
    ]
    cursor.executemany(
        "INSERT INTO financial_accounts (account_name, current_balance, minimum_reserve) VALUES (?, ?, ?)",
        accounts,
    )

    # ── Departments ─────────────────────────────────────────────────
    # account_id references: 1=Operating, 2=CapEx, 3=Emergency
    departments = [
        ("Engineering", 2, 80000.00),     # Can auto-approve up to Rs 80K from CapEx
        ("Sales", 1, 40000.00),           # Can auto-approve up to Rs 40K from Operating
        ("Marketing", 1, 30000.00),       # Can auto-approve up to Rs 30K from Operating
        ("HR", 1, 25000.00),              # Can auto-approve up to Rs 25K from Operating
        ("Operations", 2, 50000.00),      # Can auto-approve up to Rs 50K from CapEx
        ("Finance", 1, 20000.00),         # Can auto-approve up to Rs 20K from Operating
    ]
    cursor.executemany(
        "INSERT INTO departments (name, account_id, autonomous_limit) VALUES (?, ?, ?)",
        departments,
    )

    # ── Cash Commitments (Scheduled Outflows) ───────────────────────
    today = date.today()
    commitments = [
        # Engineering dept (id=1)
        (1, "Engineering Team Salaries (12 devs)", 960000.00, (today + timedelta(days=5)).isoformat(), "SALARY", "UNPAID"),
        (1, "AWS Cloud Infrastructure", 145000.00, (today + timedelta(days=10)).isoformat(), "BILL", "UNPAID"),
        (1, "GitHub Enterprise License", 28000.00, (today + timedelta(days=15)).isoformat(), "VENDOR", "UNPAID"),

        # Sales dept (id=2)
        (2, "Sales Team Salaries (8 reps)", 640000.00, (today + timedelta(days=5)).isoformat(), "SALARY", "UNPAID"),
        (2, "Salesforce CRM License", 85000.00, (today + timedelta(days=12)).isoformat(), "VENDOR", "UNPAID"),
        (2, "Client Travel Reimbursements", 45000.00, (today + timedelta(days=20)).isoformat(), "OTHER", "UNPAID"),

        # Marketing dept (id=3)
        (3, "Marketing Team Salaries (5 staff)", 400000.00, (today + timedelta(days=5)).isoformat(), "SALARY", "UNPAID"),
        (3, "Google Ads Campaign", 120000.00, (today + timedelta(days=8)).isoformat(), "VENDOR", "UNPAID"),
        (3, "Design Agency Retainer", 75000.00, (today + timedelta(days=18)).isoformat(), "VENDOR", "UNPAID"),

        # HR dept (id=4)
        (4, "HR Team Salaries (3 staff)", 195000.00, (today + timedelta(days=5)).isoformat(), "SALARY", "UNPAID"),
        (4, "Employee Insurance Premium", 210000.00, (today + timedelta(days=25)).isoformat(), "BILL", "UNPAID"),

        # Operations dept (id=5)
        (5, "Operations Team Salaries (6 staff)", 420000.00, (today + timedelta(days=5)).isoformat(), "SALARY", "UNPAID"),
        (5, "Office Rent - Bangalore", 350000.00, (today + timedelta(days=1)).isoformat(), "RENT", "UNPAID"),
        (5, "Electricity & Utilities", 45000.00, (today + timedelta(days=10)).isoformat(), "BILL", "UNPAID"),
        (5, "Security & Housekeeping", 65000.00, (today + timedelta(days=5)).isoformat(), "VENDOR", "UNPAID"),

        # Finance dept (id=6)
        (6, "Finance Team Salaries (4 staff)", 320000.00, (today + timedelta(days=5)).isoformat(), "SALARY", "UNPAID"),
        (6, "Quarterly GST Payment", 180000.00, (today + timedelta(days=28)).isoformat(), "TAX", "UNPAID"),
        (6, "Audit Firm Retainer", 50000.00, (today + timedelta(days=30)).isoformat(), "VENDOR", "UNPAID"),
    ]
    cursor.executemany(
        "INSERT INTO cash_commitments (department_id, description, amount, due_date, type, status) VALUES (?, ?, ?, ?, ?, ?)",
        commitments,
    )

    conn.commit()
    print("[db_setup] Seed data inserted successfully.")
    print(f"  Accounts:    {len(accounts)}")
    print(f"  Departments: {len(departments)}")
    print(f"  Commitments: {len(commitments)}")


def get_department_finances(department_name: str) -> dict:
    """
    Fetch a consolidated financial snapshot for a department.

    Returns a dictionary with:
      - department_name, department_id
      - account_name, account_balance, minimum_reserve
      - autonomous_limit
      - total_unpaid_commitments (sum of all UNPAID commitments)
      - available_balance (account_balance - minimum_reserve - total_unpaid_commitments)
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Get department + linked account
    row = cursor.execute("""
        SELECT d.id, d.name, d.autonomous_limit,
               a.account_name, a.current_balance, a.minimum_reserve
        FROM departments d
        JOIN financial_accounts a ON d.account_id = a.id
        WHERE LOWER(d.name) = LOWER(?)
    """, (department_name,)).fetchone()

    if not row:
        conn.close()
        return {"error": f"Department '{department_name}' not found."}

    dept_id = row["id"]

    # Sum unpaid commitments for this department
    commitment_row = cursor.execute("""
        SELECT COALESCE(SUM(amount), 0) as total
        FROM cash_commitments
        WHERE department_id = ? AND status = 'UNPAID'
    """, (dept_id,)).fetchone()

    total_commitments = float(commitment_row["total"])
    account_balance = float(row["current_balance"])
    min_reserve = float(row["minimum_reserve"])
    available = account_balance - min_reserve - total_commitments

    conn.close()

    return {
        "department_name": row["name"],
        "department_id": dept_id,
        "account_name": row["account_name"],
        "account_balance": account_balance,
        "minimum_reserve": min_reserve,
        "autonomous_limit": float(row["autonomous_limit"]),
        "total_unpaid_commitments": total_commitments,
        "available_balance": round(available, 2),
    }


def verify_financial_risk(department_name: str, quantity: int, unit_price: float) -> dict:
    """
    Deterministic financial risk evaluation for a proposed purchase.

    All math is done in Python (no LLM calculations).

    Returns a dictionary with:
      - total_price (quantity * unit_price)
      - risk_level: LOW / MEDIUM / HIGH / BLOCKED
      - requires_approval: bool
      - projected_balance: balance after purchase
      - reason: human-readable explanation
    """
    finances = get_department_finances(department_name)
    if "error" in finances:
        return finances

    total_price = quantity * unit_price
    available = finances["available_balance"]
    autonomous_limit = finances["autonomous_limit"]
    projected_balance = available - total_price

    # Decision logic
    if total_price > available:
        risk_level = "BLOCKED"
        requires_approval = True
        reason = (
            f"Purchase of Rs {total_price:,.0f} exceeds available balance of Rs {available:,.0f}. "
            f"This purchase cannot proceed without additional funding."
        )
    elif total_price > autonomous_limit:
        if projected_balance < 0:
            risk_level = "HIGH"
        elif projected_balance < finances["minimum_reserve"]:
            risk_level = "HIGH"
        else:
            risk_level = "MEDIUM"
        requires_approval = True
        reason = (
            f"Purchase of Rs {total_price:,.0f} exceeds {finances['department_name']}'s "
            f"autonomous limit of Rs {autonomous_limit:,.0f}. "
            f"Manager/CTO approval required. "
            f"Projected balance after purchase: Rs {projected_balance:,.0f}."
        )
    else:
        if projected_balance < finances["minimum_reserve"] * 0.5:
            risk_level = "MEDIUM"
            reason = (
                f"Purchase is within autonomous limit but will reduce available "
                f"balance to Rs {projected_balance:,.0f}, which is below 50% of "
                f"the safety reserve. Proceed with caution."
            )
        else:
            risk_level = "LOW"
            reason = (
                f"Purchase of Rs {total_price:,.0f} is within {finances['department_name']}'s "
                f"autonomous limit of Rs {autonomous_limit:,.0f}. "
                f"Projected balance after purchase: Rs {projected_balance:,.0f}. Auto-approved."
            )
        requires_approval = risk_level != "LOW"

    return {
        "department": finances["department_name"],
        "quantity": quantity,
        "unit_price": unit_price,
        "total_price": total_price,
        "account_balance": finances["account_balance"],
        "available_balance": available,
        "projected_balance": round(projected_balance, 2),
        "autonomous_limit": autonomous_limit,
        "risk_level": risk_level,
        "requires_approval": requires_approval,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Run directly to initialize the database
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[db_setup] Initializing database at: {DB_PATH}")
    conn = get_connection()
    create_tables(conn)
    seed_data(conn)

    # Quick verification
    print("\n[db_setup] Verification:")
    for dept in ["Engineering", "Sales", "Marketing", "HR", "Operations", "Finance"]:
        snapshot = get_department_finances(dept)
        if "error" not in snapshot:
            print(
                f"  {dept:12s} | Balance: Rs {snapshot['account_balance']:>12,.0f} | "
                f"Commitments: Rs {snapshot['total_unpaid_commitments']:>12,.0f} | "
                f"Available: Rs {snapshot['available_balance']:>12,.0f} | "
                f"Auto-limit: Rs {snapshot['autonomous_limit']:>8,.0f}"
            )

    conn.close()
    print("\n[db_setup] Done.")
