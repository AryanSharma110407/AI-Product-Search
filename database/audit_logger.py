"""
database/audit_logger.py
------------------------
Procurement audit logging and human approval simulation.

Provides tools for:
  - log_procurement_decision(): Record a purchase decision in the audit trail
  - request_human_approval(): Simulate a human-in-the-loop approval gate
  - get_procurement_history(): Retrieve past procurement decisions
"""

import sqlite3
from typing import Optional
from database.db_setup import get_connection


def log_procurement_decision(
    department_name: str,
    product_name: str,
    quantity: int,
    unit_price: float,
    status: str,
    risk_level: str,
    reason: str,
) -> dict:
    """
    Insert a procurement decision into the audit log.

    Args:
        department_name: e.g. "Engineering"
        product_name: e.g. "Dell Inspiron 15 3520"
        quantity: number of units
        unit_price: price per unit in INR
        status: one of 'APPROVED', 'PENDING_APPROVAL', 'BLOCKED', 'REJECTED'
        risk_level: one of 'LOW', 'MEDIUM', 'HIGH', 'BLOCKED'
        reason: human-readable explanation of the decision

    Returns:
        dict with log_id and confirmation message
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Look up department ID
    dept_row = cursor.execute(
        "SELECT id FROM departments WHERE LOWER(name) = LOWER(?)",
        (department_name,),
    ).fetchone()

    if not dept_row:
        conn.close()
        return {"error": f"Department '{department_name}' not found."}

    total_price = quantity * unit_price

    cursor.execute(
        """INSERT INTO procurement_logs
           (department_id, product_name, quantity, unit_price, total_price, status, risk_level, reason)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (dept_row["id"], product_name, quantity, unit_price, total_price, status, risk_level, reason),
    )
    log_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return {
        "log_id": log_id,
        "department": department_name,
        "product_name": product_name,
        "total_price": total_price,
        "status": status,
        "risk_level": risk_level,
        "message": f"Procurement decision logged (ID: {log_id}).",
    }


def request_human_approval(
    department_name: str,
    product_name: str,
    total_price: float,
    reason: str,
) -> dict:
    """
    Simulate a Human-in-the-Loop approval gate.

    In production this would send a notification to a manager.
    For the prototype, it logs the request with PENDING_APPROVAL status
    and returns instructions.

    Returns:
        dict with pending log entry and approval instructions
    """
    result = log_procurement_decision(
        department_name=department_name,
        product_name=product_name,
        quantity=1,
        unit_price=total_price,
        status="PENDING_APPROVAL",
        risk_level="HIGH",
        reason=reason,
    )

    if "error" in result:
        return result

    result["approval_instructions"] = (
        f"Purchase of {product_name} (Rs {total_price:,.0f}) for {department_name} "
        f"requires manager approval. Notification sent to department head. "
        f"Tracking ID: {result['log_id']}"
    )
    return result


def get_procurement_history(
    department_name: Optional[str] = None,
    limit: int = 20,
) -> list:
    """
    Retrieve recent procurement decisions from the audit log.

    Args:
        department_name: filter by department (optional)
        limit: max number of records to return

    Returns:
        list of procurement log dictionaries
    """
    conn = get_connection()
    cursor = conn.cursor()

    if department_name:
        rows = cursor.execute(
            """SELECT p.*, d.name as department_name
               FROM procurement_logs p
               JOIN departments d ON p.department_id = d.id
               WHERE LOWER(d.name) = LOWER(?)
               ORDER BY p.created_at DESC
               LIMIT ?""",
            (department_name, limit),
        ).fetchall()
    else:
        rows = cursor.execute(
            """SELECT p.*, d.name as department_name
               FROM procurement_logs p
               JOIN departments d ON p.department_id = d.id
               ORDER BY p.created_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()

    conn.close()

    return [dict(row) for row in rows]
