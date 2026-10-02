"""
Generate a synthetic coffee vending-machine sales dataset -> coffee_sales.csv

Columns: date, time, cash_type, card, price, coffee_name
Covers 2024-01-01 .. 2025-09-30, so "Q1 2024 vs Q1 2025" style questions work.

Re-run any time (output is reproducible thanks to the fixed seed):
    python generate_coffee_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START_DATE, END_DATE = "2024-01-01", "2025-09-30"
OUT_PATH = Path(__file__).with_name("coffee_sales.csv")

# Menu with 2024 base prices (USD). Prices go up ~6% in 2025.
MENU = {
    "Espresso": 2.40,
    "Americano": 2.90,
    "Americano with Milk": 3.30,
    "Cortado": 3.30,
    "Cappuccino": 3.80,
    "Latte": 3.80,
    "Hot Chocolate": 3.80,
    "Cocoa": 3.80,
}
PRICE_INCREASE_2025 = 1.06

# Relative popularity of each drink (normalised later)
POPULARITY = {
    "Espresso": 0.06,
    "Americano": 0.15,
    "Americano with Milk": 0.22,
    "Cortado": 0.09,
    "Cappuccino": 0.15,
    "Latte": 0.20,
    "Hot Chocolate": 0.07,
    "Cocoa": 0.06,
}
# Taste shifts in 2025, so the year-over-year chart has a story to tell
TREND_2025 = {"Latte": 1.30, "Cortado": 1.25, "Espresso": 0.80, "Americano": 0.90}
WINTER_DRINKS = {"Hot Chocolate", "Cocoa"}
WINTER_MONTHS = {11, 12, 1, 2, 3}

# Opening hours 07:00-21:59, weighted towards the morning and lunch peaks
HOURS = np.arange(7, 22)
HOUR_WEIGHTS = np.array([4, 9, 10, 8, 6, 6, 5, 5, 6, 5, 4, 4, 3, 3, 2], dtype=float)
HOUR_WEIGHTS /= HOUR_WEIGHTS.sum()

CARD_SHARE = 0.90
N_CUSTOMERS = 600  # pool of anonymised card holders; a few regulars buy a lot

COFFEES = list(MENU)


def daily_volume(day: pd.Timestamp, rng: np.random.Generator) -> int:
    """Number of sales on a given day."""
    base = 9.0 if day.year == 2024 else 11.5  # the business grew in 2025
    if day.dayofweek >= 5:
        base *= 0.75  # quieter weekends
    if day.month in (12, 1, 2):
        base *= 1.15  # cold months sell more hot drinks
    elif day.month in (7, 8):
        base *= 0.85  # summer dip
    return int(rng.poisson(base))


def coffee_probabilities(day: pd.Timestamp) -> np.ndarray:
    weights = np.array([POPULARITY[c] for c in COFFEES])
    if day.year == 2025:
        weights *= [TREND_2025.get(c, 1.0) for c in COFFEES]
    if day.month in WINTER_MONTHS:
        weights *= [1.8 if c in WINTER_DRINKS else 1.0 for c in COFFEES]
    return weights / weights.sum()


def generate(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    customer_ids = [f"ANON-0000-0000-{i:04d}" for i in range(1, N_CUSTOMERS + 1)]
    customer_weights = 1.0 / np.arange(1, N_CUSTOMERS + 1) ** 0.8
    customer_weights /= customer_weights.sum()

    rows = []
    for day in pd.date_range(START_DATE, END_DATE, freq="D"):
        n_sales = daily_volume(day, rng)
        if n_sales == 0:
            continue

        hours = rng.choice(HOURS, size=n_sales, p=HOUR_WEIGHTS)
        minutes = rng.integers(0, 60, size=n_sales)
        coffees = rng.choice(COFFEES, size=n_sales, p=coffee_probabilities(day))
        paid_by_card = rng.random(n_sales) < CARD_SHARE
        cards = rng.choice(customer_ids, size=n_sales, p=customer_weights)
        price_factor = PRICE_INCREASE_2025 if day.year == 2025 else 1.0

        for hour, minute, coffee, by_card, card in zip(hours, minutes, coffees, paid_by_card, cards):
            rows.append({
                "date": day.strftime("%Y-%m-%d"),
                "time": f"{hour:02d}:{minute:02d}",
                "cash_type": "card" if by_card else "cash",
                "card": card if by_card else "",
                "price": round(MENU[coffee] * price_factor, 1),
                "coffee_name": coffee,
            })

    df = pd.DataFrame(rows)
    return df.sort_values(["date", "time"], kind="stable").reset_index(drop=True)


if __name__ == "__main__":
    data = generate()
    data.to_csv(OUT_PATH, index=False, float_format="%.2f")
    print(f"Wrote {len(data):,} rows to {OUT_PATH}")
    print(data.head())
