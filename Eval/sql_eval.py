"""
Component-level eval — SqlQueryImprover (M2: SQL generation with reflection)

Objective eval with a per-example ground truth: every question has a hand-written, correct SQL
query. The agent's SQL is executed against the same database and its result is compared with
the ground-truth result ("execution accuracy").

Two components are scored on the same questions:
  - V1: generate_sql()                      (text → SQL)
  - V2: refine_sql_external_feedback()      (reflection on V1 + its real output)
so the report shows whether the reflection step actually fixes queries (or breaks good ones).

Run:  python sql_eval.py [--limit 3] [--min-ratio 0.8] [--model-generation openai:gpt-4.1-mini]
"""

from __future__ import annotations

import argparse
import math
from typing import Any

import pandas as pd

import eval_utils as eu

EVAL_SET = [
    {
        "id": "top-color-revenue",
        "question": "Which color of product has the highest total sales revenue?",
        "tests": "sales have negative qty_delta",
        "ground_truth_sql": """
            SELECT color FROM transactions WHERE action = 'sale'
            GROUP BY color ORDER BY SUM(-qty_delta * unit_price) DESC LIMIT 1""",
    },
    {
        "id": "total-units-sold",
        "question": "How many units were sold in total?",
        "tests": "sales have negative qty_delta",
        "ground_truth_sql": "SELECT SUM(-qty_delta) AS units_sold FROM transactions WHERE action = 'sale'",
    },
    {
        "id": "march-revenue",
        "question": "What was the total sales revenue in March 2025?",
        "tests": "sign + date range on ts",
        "ground_truth_sql": """
            SELECT SUM(-qty_delta * unit_price) AS revenue FROM transactions
            WHERE action = 'sale' AND ts >= '2025-03-01' AND ts < '2025-04-01'""",
    },
    {
        "id": "avg-units-per-sale",
        "question": "On average, how many units are sold per sale?",
        "tests": "sign + filter to sales",
        "ground_truth_sql": "SELECT AVG(-qty_delta) AS avg_units FROM transactions WHERE action = 'sale'",
    },
    {
        "id": "top-restock-brand",
        "question": "Which brand had the most restock events?",
        "tests": "filter on action",
        "ground_truth_sql": """
            SELECT brand FROM transactions WHERE action = 'restock'
            GROUP BY brand ORDER BY COUNT(*) DESC LIMIT 1""",
    },
    {
        "id": "stock-level",
        "question": "What is the current stock level of the Acme Flex Desk Lamp?",
        "tests": "stock = sum of ALL events (incl. initial insert)",
        "ground_truth_sql": "SELECT SUM(qty_delta) AS stock FROM transactions WHERE product_name = 'Acme Flex Desk Lamp'",
    },
    {
        "id": "current-price",
        "question": "What is the current price of the Zephyr Classic Smartwatch?",
        "tests": "latest event is a restock with NULL price",
        "ground_truth_sql": """
            SELECT unit_price FROM transactions
            WHERE product_name = 'Zephyr Classic Smartwatch' AND unit_price IS NOT NULL
            ORDER BY ts DESC, id DESC LIMIT 1""",
    },
    {
        "id": "top-3-product-ids",
        "question": "Which 3 product IDs sold the most units?",
        "tests": "sign + group by product_id + limit",
        "ground_truth_sql": """
            SELECT product_id FROM transactions WHERE action = 'sale'
            GROUP BY product_id ORDER BY SUM(-qty_delta) DESC LIMIT 3""",
    },
    {
        "id": "products-per-category",
        "question": "How many distinct products are in each category?",
        "tests": "COUNT(DISTINCT product_id) per group",
        "ground_truth_sql": """
            SELECT category, COUNT(DISTINCT product_id) AS products FROM transactions GROUP BY category""",
    },
    {
        "id": "sales-per-category",
        "question": "How many sale events were recorded in each category?",
        "tests": "count rows per group with a filter",
        "ground_truth_sql": """
            SELECT category, COUNT(*) AS sale_events FROM transactions WHERE action = 'sale' GROUP BY category""",
    },
]


# --------------------------------------------------------------------------------------
# Comparing a result with the ground truth
# --------------------------------------------------------------------------------------
def _norm(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value).strip().lower()


def _sort_key(value: Any):
    if value is None:
        return (0, 0.0, "")
    if isinstance(value, float):
        return (1, value, "")
    return (2, 0.0, value)


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=0.01)
    return a == b


def _same_multiset(a: list, b: list) -> bool:
    return len(a) == len(b) and all(_same(x, y) for x, y in zip(sorted(a, key=_sort_key), sorted(b, key=_sort_key)))


def compare_results(expected: pd.DataFrame, actual: pd.DataFrame) -> tuple[bool, str]:
    """
    Execution-accuracy check, tolerant to column names/order, row order, extra columns and
    float rounding. Every ground-truth column must appear (same values) in the agent's result,
    and the rows must match once projected onto those columns.

    Returns (flag, reason).
    """
    if list(actual.columns) == ["error"]:
        return False, f"query failed: {actual.iloc[0, 0]}"
    if len(actual) != len(expected):
        return False, f"{len(actual)} row(s) returned, expected {len(expected)}"

    exp_cols = [[_norm(v) for v in expected[c]] for c in expected.columns]
    act_cols = [[_norm(v) for v in actual.iloc[:, i]] for i in range(actual.shape[1])]

    mapping, used = [], set()
    for name, values in zip(expected.columns, exp_cols):
        match = next((i for i, a in enumerate(act_cols) if i not in used and _same_multiset(values, a)), None)
        if match is None:
            preview = ", ".join(str(v) for v in values[:3]) + ("…" if len(values) > 3 else "")
            return False, f"no column matches expected '{name}' ({preview})"
        mapping.append(match)
        used.add(match)

    exp_rows = sorted(zip(*exp_cols), key=lambda r: [_sort_key(v) for v in r])
    act_rows = sorted(zip(*(act_cols[i] for i in mapping)), key=lambda r: [_sort_key(v) for v in r])
    if not all(all(_same(x, y) for x, y in zip(e, a)) for e, a in zip(exp_rows, act_rows)):
        return False, "right values, but paired in the wrong rows"
    return True, "matches ground truth"


# --------------------------------------------------------------------------------------
# Running the eval
# --------------------------------------------------------------------------------------
def load_sql_project():
    agent = eu.import_project_module("sql", "sql_reflection_agent")
    return agent, agent.utils


def evaluate_sql_example(example: dict, agent, sql_utils, db_path: str, schema: str,
                         model_generation: str, model_evaluation: str) -> dict:
    """Run V1 and V2 for one question and compare both with the ground truth."""
    row = {"id": example["id"], "question": example["question"], "tests": example["tests"]}
    expected = sql_utils.execute_sql(example["ground_truth_sql"], db_path)
    if list(expected.columns) == ["error"]:
        raise RuntimeError(f"Ground-truth SQL for {example['id']} failed: {expected.iloc[0, 0]}")
    row["expected"] = expected.to_dict(orient="records")[:5]

    try:
        sql_v1 = agent.generate_sql(example["question"], schema, model_generation)
        df_v1 = sql_utils.execute_sql(sql_v1, db_path)
        row["v1_pass"], row["v1_reason"] = compare_results(expected, df_v1)
        row["sql_v1"] = sql_v1

        feedback, sql_v2 = agent.refine_sql_external_feedback(
            question=example["question"], sql_query=sql_v1, df_feedback=df_v1,
            schema=schema, model=model_evaluation,
        )
        df_v2 = sql_utils.execute_sql(sql_v2, db_path)
        row["v2_pass"], row["v2_reason"] = compare_results(expected, df_v2)
        row["sql_v2"], row["feedback"] = sql_v2, feedback
    except Exception as e:  # API / network problems: count as failures, keep going
        row.setdefault("v1_pass", False)
        row.setdefault("v1_reason", f"error: {e}")
        row["v2_pass"], row["v2_reason"] = False, f"error: {type(e).__name__}: {e}"
    return row


def run_eval(model_generation: str = "openai:gpt-4.1", model_evaluation: str = "openai:gpt-4.1",
             min_ratio: float = 0.8, limit: int | None = None, show: bool = True) -> eu.EvalResult:
    eu.require_api_key(model_generation, model_evaluation)
    agent, sql_utils = load_sql_project()

    eu.RESULTS_DIR.mkdir(exist_ok=True)
    db_path = str(eu.RESULTS_DIR / "products_eval.db")
    sql_utils.create_transactions_db(db_path, seed=42)  # same data as the project's default
    schema = sql_utils.get_schema(db_path)

    examples = EVAL_SET[:limit] if limit else EVAL_SET
    rows = []
    for i, example in enumerate(examples, 1):
        eu.progress(f"[sql {i}/{len(examples)}] {example['question']}")
        rows.append(evaluate_sql_example(example, agent, sql_utils, db_path, schema,
                                         model_generation, model_evaluation))

    n = len(rows)
    v1 = sum(r["v1_pass"] for r in rows) / n
    v2 = sum(r["v2_pass"] for r in rows) / n
    fixed = sum((not r["v1_pass"]) and r["v2_pass"] for r in rows)
    broke = sum(r["v1_pass"] and not r["v2_pass"] for r in rows)
    flag = v2 >= min_ratio

    table = eu.md_table(
        ["Question", "Tests", "V1", "V2", "V2 note"],
        [[r["question"], r["tests"], eu.mark(r["v1_pass"]), eu.mark(r["v2_pass"]), r["v2_reason"]] for r in rows],
    )
    failures = "\n\n".join(
        f"**{r['question']}**\n- V1: {r['v1_reason']}\n- V2: {r['v2_reason']}\n\n"
        f"```sql\n{(r.get('sql_v2') or r.get('sql_v1') or '').strip()}\n```"
        for r in rows if not r["v2_pass"]
    ) or "_None — every V2 query matched the ground truth._"

    report = f"""### Eval — SqlQueryImprover: execution accuracy (V1 → V2)
- Examples: {n}
- Models: generation `{model_generation}`, reflection `{model_evaluation}`
- V1 accuracy (generate_sql): {v1:.0%} ({sum(r['v1_pass'] for r in rows)}/{n})
- V2 accuracy (after reflection with execution feedback): {v2:.0%} ({sum(r['v2_pass'] for r in rows)}/{n})
- Reflection fixed {fixed} · broke {broke}
- Threshold (V2): {min_ratio:.0%}
- Status: {eu.status(flag)}

**Details:**

{table}

**Failed V2 queries:**

{failures}
"""
    result = eu.EvalResult(
        name="sql", title="SqlQueryImprover — execution accuracy", metric="V2 execution accuracy",
        score=v2, threshold=min_ratio, passed=flag, report=report, rows=rows,
        extra={"v1_accuracy": v1, "fixed_by_reflection": fixed, "broken_by_reflection": broke},
    )
    if show:
        eu.show_markdown(report)
    return result


def main():
    parser = argparse.ArgumentParser(description="Component-level eval for the SQL generation + reflection steps")
    parser.add_argument("--model-generation", default="openai:gpt-4.1")
    parser.add_argument("--model-evaluation", default="openai:gpt-4.1")
    parser.add_argument("--min-ratio", type=float, default=0.8, help="V2 accuracy needed to PASS (0–1)")
    parser.add_argument("--limit", type=int, help="only run the first N questions")
    args = parser.parse_args()

    result = run_eval(args.model_generation, args.model_evaluation, args.min_ratio, args.limit)
    print(f"\nSaved report: {eu.save_result(result)}")


if __name__ == "__main__":
    main()
