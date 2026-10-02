"""
Helper module for the SQL reflection lab (M2 - Improving SQL Generation with Reflection).

Provides:
  - create_transactions_db : build products.db filled with a random product event history
  - get_schema             : describe a SQLite database's tables as text (for the LLM prompt)
  - execute_sql            : run an LLM-written SQL query (read-only) and return a DataFrame
  - print_html             : pretty display (HTML in Jupyter, plain text in a terminal)
  - check_api_keys         : fail fast with a clear message if a provider key is missing
"""

from __future__ import annotations

import html
import os
import random
import re
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

# Read OPENAI_API_KEY from a .env in this project folder, or the shared one in the
# parent ReflectDesinPatternDemo folder (project values win)
_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env")
load_dotenv(_HERE.parent / ".env")

# Emojis in the step titles must not crash a Windows console / redirected output
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass


# --------------------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------------------
BRANDS = ["Acme", "Nimbus", "Vertex", "Orion", "Zephyr"]
COLORS = ["Black", "White", "Red", "Blue", "Green", "Silver", "Navy", "Gray"]
MODEL_WORDS = ["Pro", "Lite", "Max", "Air", "Classic", "Sport", "Flex", "Edge"]
CATEGORIES = {  # category -> (min, max) launch price in USD
    "Headphones": (40, 250),
    "Backpack": (30, 120),
    "Smartwatch": (90, 400),
    "Water Bottle": (10, 40),
    "Sneakers": (50, 180),
    "Jacket": (60, 220),
    "Desk Lamp": (20, 90),
    "Keyboard": (30, 160),
}
SALE_NOTES = ["online order", "in-store purchase", "marketplace order", None]

TRANSACTIONS_DDL = """
CREATE TABLE transactions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id   INTEGER,
    product_name TEXT,
    brand        TEXT,
    category     TEXT,
    color        TEXT,
    action       TEXT,      -- insert | restock | sale | price_update
    qty_delta    INTEGER,   -- + insert/restock, - sale, 0 price_update
    unit_price   REAL,      -- price at that moment (NULL for restock)
    notes        TEXT,
    ts           DATETIME
)
"""


def create_transactions_db(
    db_path: str = "products.db",
    n_products: int = 40,
    start: str = "2025-01-01",
    days: int = 365,
    seed: int | None = 42,
) -> str:
    """
    (Re)create a SQLite database with a `transactions` event table.

    Every row is an event for a product: `insert` (initial stock), `restock`, `sale`
    (negative qty_delta) or `price_update` (qty_delta = 0). Stock levels, sales and price
    history are all derived by aggregating these events. Pass seed=None for fresh random data.
    """
    rng = random.Random(seed)
    start_dt = datetime.fromisoformat(start)

    def random_ts(day: int) -> str:
        moment = start_dt + timedelta(days=day, seconds=rng.randint(8 * 3600, 22 * 3600))
        return moment.strftime("%Y-%m-%d %H:%M:%S")

    events = []
    for product_id in range(1, n_products + 1):
        brand, color = rng.choice(BRANDS), rng.choice(COLORS)
        category = rng.choice(list(CATEGORIES))
        name = f"{brand} {rng.choice(MODEL_WORDS)} {category}"
        product = (product_id, name, brand, category, color)

        price = round(rng.uniform(*CATEGORIES[category]), 2)
        popularity = rng.uniform(0.3, 2.5)  # average sales per day
        launch_day = rng.randint(0, 30)
        stock = rng.randint(50, 200)
        events.append((*product, "insert", stock, price, "initial stock", random_ts(launch_day)))

        for day in range(launch_day + 1, days):
            if rng.random() < 1 / 60:
                new_price = round(price * rng.uniform(0.85, 1.15), 2)
                note = "price increase" if new_price > price else "seasonal discount"
                price = new_price
                events.append((*product, "price_update", 0, price, note, random_ts(day)))

            # Poisson-like number of orders for the day
            n_orders = sum(rng.random() < popularity / 4 for _ in range(4))
            for _ in range(n_orders):
                qty = rng.randint(1, 3)
                if qty > stock:
                    break
                stock -= qty
                events.append((*product, "sale", -qty, price, rng.choice(SALE_NOTES), random_ts(day)))

            if stock < 20:
                qty = rng.randint(50, 150)
                stock += qty
                events.append((*product, "restock", qty, None, "supplier restock", random_ts(day)))

    events.sort(key=lambda e: e[-1])  # chronological ids

    db_file = Path(db_path)
    db_file.unlink(missing_ok=True)
    with closing(sqlite3.connect(db_file)) as conn:
        conn.execute(TRANSACTIONS_DDL)
        conn.executemany(
            "INSERT INTO transactions (product_id, product_name, brand, category, color, action, "
            "qty_delta, unit_price, notes, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            events,
        )
        conn.commit()

    print(f"Created {db_file} with {len(events):,} events for {n_products} products.")
    return str(db_file)


def get_schema(db_path: str = "products.db") -> str:
    """Return every table's columns as text, e.g. 'Table name: transactions\\nid (INTEGER)\\n...'."""
    with closing(sqlite3.connect(_read_only_uri(db_path), uri=True)) as conn:
        tables = [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )]
        parts = []
        for table in tables:
            columns = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
            lines = [f"{col[1]} ({col[2]})" for col in columns]
            parts.append(f"Table name: {table}\n" + "\n".join(lines))
    return "\n\n".join(parts)


def _read_only_uri(db_path: str) -> str:
    path = Path(db_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {path}. Run utils.create_transactions_db() first.")
    return f"{path.as_uri()}?mode=ro"


def _clean_sql(sql: str) -> str:
    """LLMs often wrap SQL in ```sql fences; keep only the query itself."""
    sql = sql.strip()
    fenced = re.search(r"```(?:sql|sqlite)?\s*([\s\S]*?)```", sql, flags=re.IGNORECASE)
    return fenced.group(1).strip() if fenced else sql


def execute_sql(sql: str, db_path: str = "products.db") -> pd.DataFrame:
    """
    Run a query and return the result as a DataFrame.

    The database is opened read-only, so LLM-written SQL can't modify it. If the query fails,
    the error is returned as a one-row DataFrame, so it can be fed back to the LLM as feedback.
    """
    query = _clean_sql(sql)
    try:
        with closing(sqlite3.connect(_read_only_uri(db_path), uri=True)) as conn:
            return pd.read_sql_query(query, conn)
    except Exception as e:
        return pd.DataFrame({"error": [f"{type(e).__name__}: {e}"]})


# --------------------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------------------
def _in_notebook() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except ImportError:
        return False


def print_html(content: Any, title: str | None = None) -> None:
    """Display text, SQL or a DataFrame: an HTML card in Jupyter, plain text in a terminal."""
    if _in_notebook():
        from IPython.display import HTML, display

        if isinstance(content, pd.DataFrame):
            body = content.to_html(index=False, border=0)
        else:
            body = f'<pre style="white-space:pre-wrap;margin:0;">{html.escape(str(content))}</pre>'
        header = (
            f'<div style="font-weight:600;font-size:1.05em;margin-bottom:8px;">{html.escape(title)}</div>'
            if title
            else ""
        )
        display(HTML(
            '<div style="border:1px solid #8884;border-radius:8px;padding:12px 14px;margin:8px 0;">'
            f"{header}{body}</div>"
        ))
        return

    if title:
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
    print(content.to_string(index=False) if isinstance(content, pd.DataFrame) else content)


# --------------------------------------------------------------------------------------
# API keys
# --------------------------------------------------------------------------------------
_PROVIDER_KEYS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def check_api_keys(*models: str) -> None:
    """Models use aisuite's 'provider:model' names, e.g. 'openai:gpt-4.1'."""
    for model in models:
        provider = model.split(":", 1)[0].lower()
        env_var = _PROVIDER_KEYS.get(provider)
        if env_var and not os.getenv(env_var):
            raise RuntimeError(
                f"{env_var} is not set, but model '{model}' needs it. "
                "Add it to the .env file in the ReflectDesinPatternDemo folder (see .env.example)."
            )
