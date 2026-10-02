"""
Inventory helpers for the sunglasses store (M5 lab: planning with code execution).

  - create_inventory()    : reset the inventory table to the default product set
  - create_transactions() : reset the transactions table to a single opening-balance entry
  - seed_db()             : open the JSON-backed TinyDB store and reset both tables
  - build_schema_block()  : describe both tables (fields, types, sample rows) for the LLM prompt
  - get_current_balance() : running balance after the latest transaction
  - next_transaction_id() : next id in the TXN001, TXN002, ... sequence
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from tinydb import TinyDB
from tinydb.table import Table

DB_PATH = Path(__file__).with_name("store_db.json")
OPENING_BALANCE = 500.00

INVENTORY = [
    {"item_id": "SG001", "name": "Aviator",
     "description": "Original aviator style with an iconic teardrop metal frame and polarized lenses.",
     "quantity_in_stock": 23, "price": 80},
    {"item_id": "SG002", "name": "Wayfarer",
     "description": "Bold trapezoid acetate frame, an everyday favourite that suits most face shapes.",
     "quantity_in_stock": 14, "price": 95},
    {"item_id": "SG003", "name": "Cat Eye",
     "description": "Retro cat eye frames with upswept corners and tinted lenses for a glamorous look.",
     "quantity_in_stock": 8, "price": 110},
    {"item_id": "SG004", "name": "Sport",
     "description": "Lightweight wraparound frames for running and cycling, with impact-resistant lenses.",
     "quantity_in_stock": 15, "price": 70},
    {"item_id": "SG005", "name": "Classic",
     "description": "Classic round profile with minimalist metal frames, offering a timeless and versatile "
                    "style that fits both casual and formal wear.",
     "quantity_in_stock": 10, "price": 60},
    {"item_id": "SG006", "name": "Moon",
     "description": "Oversized round frames with gradient lenses for a bold, retro statement.",
     "quantity_in_stock": 6, "price": 120},
    {"item_id": "SG007", "name": "Clubmaster",
     "description": "Vintage browline frames with a half-rim top and metal lower rim.",
     "quantity_in_stock": 2, "price": 130},
    {"item_id": "SG008", "name": "Shield",
     "description": "One-piece shield lens with a futuristic frameless look and full UV protection.",
     "quantity_in_stock": 0, "price": 150},
]

_db: TinyDB | None = None
_db_path: Path | None = None


def get_db(path: Path | str = DB_PATH) -> TinyDB:
    """The store's TinyDB database (one shared instance, so every helper sees the same tables)."""
    global _db, _db_path
    path = Path(path).resolve()
    if _db is None or _db_path != path:
        if _db is not None:
            _db.close()
        _db, _db_path = TinyDB(path, indent=2, encoding="utf-8"), path
    return _db


def create_inventory(db: TinyDB | None = None) -> Table:
    """Reset the inventory table to the default sunglasses catalogue."""
    table = (db or get_db()).table("inventory")
    table.truncate()
    table.insert_multiple([dict(item) for item in INVENTORY])
    return table


def create_transactions(db: TinyDB | None = None) -> Table:
    """Reset the transactions table to a single opening-balance entry."""
    table = (db or get_db()).table("transactions")
    table.truncate()
    table.insert({
        "transaction_id": "TXN001",
        "customer_name": "OPENING_BALANCE",
        "transaction_summary": "Opening balance",
        "transaction_amount": OPENING_BALANCE,
        "balance_after_transaction": OPENING_BALANCE,
        "timestamp": datetime.now().isoformat(),
    })
    return table


def seed_db(path: Path | str = DB_PATH) -> tuple[TinyDB, Table, Table]:
    """Open the JSON-backed store and reset it. Returns (db, inventory_tbl, transactions_tbl)."""
    db = get_db(path)
    return db, create_inventory(db), create_transactions(db)


def load_db(path: Path | str = DB_PATH) -> tuple[TinyDB, Table, Table]:
    """Open the store keeping its current data (seeds any empty table). Same return as seed_db()."""
    db = get_db(path)
    inventory_tbl, transactions_tbl = db.table("inventory"), db.table("transactions")
    if not len(inventory_tbl):
        create_inventory(db)
    if not len(transactions_tbl):
        create_transactions(db)
    return db, inventory_tbl, transactions_tbl


# --------------------------------------------------------------------------------------
# Helpers the generated code may use
# --------------------------------------------------------------------------------------
def get_current_balance(tbl: Table) -> float:
    """Balance after the most recent transaction (0.0 if there are none)."""
    rows = tbl.all()
    if not rows:
        return 0.0
    latest = max(rows, key=lambda row: row.doc_id)
    return float(latest.get("balance_after_transaction", 0.0))


def next_transaction_id(tbl: Table, prefix: str = "TXN") -> str:
    """Next id in the sequence, e.g. TXN001 → TXN002."""
    numbers = [
        int(row["transaction_id"][len(prefix):])
        for row in tbl.all()
        if str(row.get("transaction_id", "")).startswith(prefix) and row["transaction_id"][len(prefix):].isdigit()
    ]
    return f"{prefix}{(max(numbers) + 1) if numbers else 1:03d}"


# --------------------------------------------------------------------------------------
# Schema description for the prompt
# --------------------------------------------------------------------------------------
def _type_name(values: list) -> str:
    types = sorted({type(v).__name__ for v in values if v is not None})
    return " | ".join(types) or "unknown"


def _describe_table(name: str, tbl: Table, notes: dict[str, str], samples: int) -> str:
    rows = tbl.all()
    fields = list(rows[0].keys()) if rows else list(notes)
    lines = [f"Table `{name}` ({len(rows)} rows)", "Fields:"]
    for field in fields:
        values = [row.get(field) for row in rows]
        note = f" — {notes[field]}" if field in notes else ""
        lines.append(f"- {field} ({_type_name(values)}){note}")
    lines.append(f"Sample rows (first {min(samples, len(rows))}):")
    lines.append(json.dumps(rows[:samples], indent=2))
    return "\n".join(lines)


def build_schema_block(inventory_tbl: Table, transactions_tbl: Table, samples: int = 3) -> str:
    """Live description of both tables (fields, types, notes, sample rows) for the LLM prompt."""
    inventory_notes = {
        "item_id": "unique product id, e.g. SG001",
        "name": "style name, e.g. Aviator, Classic (case varies in requests)",
        "description": "free text; frame shape words such as 'round' appear here",
        "quantity_in_stock": "units currently available",
        "price": "unit price in USD",
    }
    transaction_notes = {
        "transaction_id": "unique id, e.g. TXN001 (use next_transaction_id)",
        "customer_name": "customer, or OPENING_BALANCE for the first row",
        "transaction_summary": "short description, e.g. 'Purchase of 2 x Aviator (SG001)'",
        "transaction_amount": "money for this transaction in USD (purchases positive, refunds negative)",
        "balance_after_transaction": "running balance after this row (use get_current_balance)",
        "timestamp": "ISO-8601 date/time",
    }
    return "\n\n".join([
        _describe_table("inventory_tbl", inventory_tbl, inventory_notes, samples),
        _describe_table("transactions_tbl", transactions_tbl, transaction_notes, samples),
    ])
