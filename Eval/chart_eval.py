"""
Component-level eval — ChartGenerationAgent (M2: chart generation with reflection)

Objective eval with a per-example ground truth. For each chart request the expected numbers are
computed straight from coffee_sales.csv with pandas. The agent's code is executed, the figure is
captured when it is saved, and the numbers actually drawn (bar heights, line points, pie slices,
value labels) are compared with the expected numbers. Requests like "Q1 sales" are ambiguous
(revenue? number of sales? per coffee?), so each example lists every reasonable reading.

Checks per chart (all must pass):
  code_extracted · runs · saved_to_path · title · axis_labels · rules (no plt.show, no seaborn)
  · correct_data (≥ 90% of the expected values found in the chart)

Scored for V1 (generate_chart_code) and V2 (reflect_on_image_and_regenerate).

Run:  python chart_eval.py [--limit 2] [--min-ratio 0.6] [--generation-model gpt-4o-mini]
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # the generated code only saves figures

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Wedge

import eval_utils as eu

DATA_RECALL_NEEDED = 0.9
SCALES = (1.0, 0.001, 100.0, 0.01)  # raw, thousands, percent, cents


def _q1(df):
    return df[(df["quarter"] == 1) & df["year"].isin([2024, 2025])]


def _share(series: pd.Series) -> pd.Series:
    return series / series.sum()


EVAL_SET = [
    {
        "id": "q1-2024-vs-2025",
        "instruction": "Create a plot comparing Q1 coffee sales in 2024 and 2025 using the data in coffee_sales.csv.",
        "expected": lambda df: {
            "Q1 revenue per year": _q1(df).groupby("year")["price"].sum(),
            "Q1 number of sales per year": _q1(df).groupby("year").size(),
            "Q1 revenue per coffee & year": _q1(df).groupby(["coffee_name", "year"])["price"].sum(),
            "Q1 number of sales per coffee & year": _q1(df).groupby(["coffee_name", "year"]).size(),
            "Q1 revenue per month & year": _q1(df).groupby(["year", "month"])["price"].sum(),
            "Q1 number of sales per month & year": _q1(df).groupby(["year", "month"]).size(),
        },
    },
    {
        "id": "monthly-revenue-2024",
        "instruction": "Plot the total monthly revenue for 2024 as a line chart.",
        "expected": lambda df: {
            "monthly revenue 2024": df[df["year"] == 2024].groupby("month")["price"].sum(),
        },
    },
    {
        "id": "sales-per-coffee-2025",
        "instruction": "Show the number of sales per coffee type in 2025 as a bar chart.",
        "expected": lambda df: {
            "number of sales per coffee 2025": df[df["year"] == 2025].groupby("coffee_name").size(),
        },
    },
    {
        "id": "card-vs-cash-2025",
        "instruction": "Show the share of card vs cash payments in 2025 as a pie chart.",
        "expected": lambda df: {
            "share of payments (count)": _share(df[df["year"] == 2025].groupby("cash_type").size()),
            "share of payments (revenue)": _share(df[df["year"] == 2025].groupby("cash_type")["price"].sum()),
            "number of payments": df[df["year"] == 2025].groupby("cash_type").size(),
        },
    },
    {
        "id": "avg-price-2024-vs-2025",
        "instruction": "Compare the average price of each coffee type in 2024 and 2025.",
        "expected": lambda df: {
            "average price per coffee & year": df.groupby(["coffee_name", "year"])["price"].mean(),
        },
    },
]


# --------------------------------------------------------------------------------------
# Reading the numbers a figure actually draws
# --------------------------------------------------------------------------------------
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def _finite(values) -> list[float]:
    out = []
    for v in np.ravel(np.asarray(values, dtype=object)):
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def plotted_values(fig: Figure) -> list[float]:
    """Bar sizes, pie slice fractions, line/scatter y-values and numeric labels of every axes."""
    values: list[float] = []
    for ax in fig.axes:
        for patch in ax.patches:
            if isinstance(patch, Wedge):
                values.append((patch.theta2 - patch.theta1) / 360.0)
            elif isinstance(patch, Rectangle):
                values += [patch.get_height(), patch.get_width()]  # vertical or horizontal bars
        for line in ax.lines:
            values += _finite(line.get_ydata())
        for collection in ax.collections:
            offsets = collection.get_offsets()
            if len(offsets):
                values += _finite(np.asarray(offsets)[:, 1])
        for text in ax.texts:
            values += [float(n.replace(",", "")) for n in _NUMBER.findall(text.get_text()) if n not in "-."]
    return [v for v in values if math.isfinite(v)]


def data_recall(expected: list[float], plotted: list[float]) -> float:
    """Share of expected values found in the chart (each plotted value used once), best over SCALES."""
    best = 0.0
    for scale in SCALES:
        pool = sorted(plotted)
        used = [False] * len(pool)
        matched = 0
        for target in (e * scale for e in expected):
            for i, value in enumerate(pool):
                if not used[i] and math.isclose(value, target, rel_tol=0.01, abs_tol=0.005):
                    used[i] = True
                    matched += 1
                    break
        best = max(best, matched / len(expected))
    return best


def _has_title(fig: Figure) -> bool:
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None and suptitle.get_text().strip():
        return True
    return any(ax.get_title(loc).strip() for ax in fig.axes for loc in ("center", "left", "right"))


def _has_axis_labels(fig: Figure) -> bool:
    cartesian = [ax for ax in fig.axes if ax.has_data() and not any(isinstance(p, Wedge) for p in ax.patches)]
    return all(ax.get_xlabel().strip() and ax.get_ylabel().strip() for ax in cartesian)


# --------------------------------------------------------------------------------------
# Evaluating one generated chart (like the lab's evaluate_* function: flag + Markdown)
# --------------------------------------------------------------------------------------
def evaluate_chart_code(llm_output: str, df: pd.DataFrame, out_path: str,
                        expected_options: dict[str, pd.Series]) -> tuple[bool, dict]:
    """Execute the generated code and check it. Returns (flag, details)."""
    checks: dict[str, bool] = {}
    details: dict = {"checks": checks}

    match = re.search(r"<execute_python>([\s\S]*?)</execute_python>", llm_output or "")
    code = match.group(1).strip() if match else ""
    checks["code_extracted"] = bool(code)

    saved: list[tuple[Figure, str]] = []
    original_savefig = Figure.savefig

    def spy(self, fname, *args, **kwargs):
        saved.append((self, str(fname)))
        return original_savefig(self, fname, *args, **kwargs)

    Path(out_path).unlink(missing_ok=True)
    Figure.savefig = spy
    try:
        exec(code, {"df": df.copy()})
        checks["runs"] = bool(code)
    except Exception as e:  # generated code is allowed to fail; that's what we measure
        checks["runs"] = False
        details["error"] = f"{type(e).__name__}: {e}"
    finally:
        Figure.savefig = original_savefig

    fig = saved[-1][0] if saved else None
    checks["saved_to_path"] = any(Path(p).resolve() == Path(out_path).resolve() for _, p in saved) and Path(out_path).exists()
    checks["title"] = bool(fig) and _has_title(fig)
    checks["axis_labels"] = bool(fig) and _has_axis_labels(fig)
    checks["rules"] = bool(code) and "plt.show(" not in code and "seaborn" not in code

    best_name, best_recall = None, 0.0
    if fig is not None:
        values = plotted_values(fig)
        for name, series in expected_options.items():
            recall = data_recall([float(v) for v in series.values], values)
            if recall > best_recall:
                best_name, best_recall = name, recall
    checks["correct_data"] = best_recall >= DATA_RECALL_NEEDED
    details["data_match"] = best_name
    details["data_recall"] = round(best_recall, 3)
    plt.close("all")

    return all(checks.values()), details


def _summary(flag: bool, details: dict) -> str:
    failed = [name for name, ok in details["checks"].items() if not ok]
    if flag:
        return f"✅ data: {details['data_match']}"
    reason = ", ".join(failed)
    if "error" in details:
        reason += f" ({details['error'][:60]})"
    return f"❌ {reason}"


def load_chart_project():
    agent = eu.import_project_module("chart", "reflection_agent")
    return agent, agent.utils


def run_eval(generation_model: str = "gpt-4o-mini", reflection_model: str = "o4-mini",
             min_ratio: float = 0.6, limit: int | None = None, show: bool = True) -> eu.EvalResult:
    eu.require_api_key(generation_model, reflection_model)
    agent, chart_utils = load_chart_project()
    df = chart_utils.load_and_prepare_data(str(eu.PROJECTS["chart"] / "coffee_sales.csv"))

    out_dir = eu.RESULTS_DIR / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = EVAL_SET[:limit] if limit else EVAL_SET
    rows = []
    for i, example in enumerate(examples, 1):
        eu.progress(f"[chart {i}/{len(examples)}] {example['instruction']}")
        options = example["expected"](df)
        # forward slashes: the path is pasted into generated Python code
        out_v1 = (out_dir / f"{example['id']}_v1.png").as_posix()
        out_v2 = (out_dir / f"{example['id']}_v2.png").as_posix()
        row = {"id": example["id"], "instruction": example["instruction"]}
        try:
            code_v1 = agent.generate_chart_code(example["instruction"], generation_model, out_v1)
            row["v1_pass"], row["v1"] = evaluate_chart_code(code_v1, df, out_v1, options)

            if Path(out_v1).exists():
                feedback, code_v2 = agent.reflect_on_image_and_regenerate(
                    chart_path=out_v1, instruction=example["instruction"], model_name=reflection_model,
                    out_path_v2=out_v2, code_v1=code_v1,
                )
                row["feedback"] = feedback
                row["v2_pass"], row["v2"] = evaluate_chart_code(code_v2, df, out_v2, options)
            else:
                row["v2_pass"] = False
                row["v2"] = {"checks": {"v1_chart_exists": False}, "error": "no V1 chart to reflect on"}
        except Exception as e:  # API / network problems: count as failures, keep going
            row.setdefault("v1_pass", False)
            row.setdefault("v1", {"checks": {}, "error": f"{type(e).__name__}: {e}"})
            row["v2_pass"] = False
            row["v2"] = {"checks": {"api_call": False}, "error": f"{type(e).__name__}: {e}"}
        rows.append(row)

    n = len(rows)
    v1 = sum(r["v1_pass"] for r in rows) / n
    v2 = sum(r["v2_pass"] for r in rows) / n
    flag = v2 >= min_ratio

    def check_rate(variant: str) -> str:
        names = ["code_extracted", "runs", "saved_to_path", "title", "axis_labels", "rules", "correct_data"]
        cells = []
        for name in names:
            ok = sum(r[variant]["checks"].get(name, False) for r in rows)
            cells.append(f"{ok}/{n}")
        return " | ".join(cells)

    table = eu.md_table(
        ["Request", "V1", "V2"],
        [[r["instruction"], _summary(r["v1_pass"], r["v1"]), _summary(r["v2_pass"], r["v2"])] for r in rows],
    )
    report = f"""### Eval — ChartGenerationAgent: chart correctness (V1 → V2)
- Examples: {n}
- Models: generation `{generation_model}`, reflection `{reflection_model}`
- V1 charts passing every check: {v1:.0%}
- V2 charts passing every check (after reflection): {v2:.0%}
- Threshold (V2): {min_ratio:.0%}
- Status: {eu.status(flag)}

**Checks passed per variant:**

| Variant | code extracted | runs | saved to path | title | axis labels | rules | correct data |
|---|---|---|---|---|---|---|---|
| V1 | {check_rate('v1')} |
| V2 | {check_rate('v2')} |

**Details:**

{table}

Charts are saved in `results/charts/` (`<id>_v1.png`, `<id>_v2.png`).
"""
    result = eu.EvalResult(
        name="chart", title="ChartGenerationAgent — chart correctness", metric="V2 charts passing all checks",
        score=v2, threshold=min_ratio, passed=flag, report=report, rows=rows, extra={"v1_pass_rate": v1},
    )
    if show:
        eu.show_markdown(report)
    return result


def main():
    parser = argparse.ArgumentParser(description="Component-level eval for chart generation + reflection")
    parser.add_argument("--generation-model", default="gpt-4o-mini")
    parser.add_argument("--reflection-model", default="o4-mini")
    parser.add_argument("--min-ratio", type=float, default=0.6, help="V2 pass rate needed to PASS (0–1)")
    parser.add_argument("--limit", type=int, help="only run the first N requests")
    args = parser.parse_args()

    result = run_eval(args.generation_model, args.reflection_model, args.min_ratio, args.limit)
    print(f"\nSaved report: {eu.save_result(result)}")


if __name__ == "__main__":
    main()
