"""
Run the component-level evals for every AIAgentPractice project and write one summary.

    python run_all_evals.py                    # all four evals
    python run_all_evals.py --only sql tools   # some of them
    python run_all_evals.py --limit 2          # first 2 examples of each (quick, cheap smoke run)

Each eval also has its own script (sql_eval.py, chart_eval.py, tool_selection_eval.py,
email_agent_eval.py) with model / threshold options.
"""

from __future__ import annotations

import argparse
from datetime import datetime

import eval_utils as eu

EVALS = {
    "sql": ("sql_eval", "SqlQueryImprover"),
    "chart": ("chart_eval", "ChartGenerationAgent"),
    "tools": ("tool_selection_eval", "FunctionsToTools"),
    "email": ("email_agent_eval", "EmailTool"),
}


def main():
    parser = argparse.ArgumentParser(description="Run all component-level evals")
    parser.add_argument("--only", nargs="+", choices=list(EVALS), help="evals to run (default: all)")
    parser.add_argument("--limit", type=int, help="only run the first N examples of each eval")
    args = parser.parse_args()

    results, errors = [], {}
    for key in args.only or list(EVALS):
        module_name, project = EVALS[key]
        print(f"\n######## {project} ########")
        try:
            module = __import__(module_name)
            result = module.run_eval(limit=args.limit, show=False)
            eu.show_markdown(result.report)
            print(f"Saved report: {eu.save_result(result)}")
            results.append(result)
        except Exception as e:  # one broken eval (e.g. missing key) shouldn't stop the others
            errors[project] = f"{type(e).__name__}: {e}"
            print(f"!! {project} eval failed to run: {errors[project]}")

    rows = [[r.title, r.metric, f"{r.score:.0%}", f"{r.threshold:.0%}", eu.status(r.passed)] for r in results]
    rows += [[project, "—", "—", "—", f"⚠️ not run: {message}"] for project, message in errors.items()]
    summary = f"""## Component-level eval summary — {datetime.now():%Y-%m-%d %H:%M}

{eu.md_table(["Component", "Metric", "Score", "Threshold", "Status"], rows)}
"""
    print("\n" + summary)
    eu.RESULTS_DIR.mkdir(exist_ok=True)
    path = eu.RESULTS_DIR / f"summary_{datetime.now():%Y%m%d-%H%M%S}.md"
    path.write_text(summary + "\n\n" + "\n\n---\n\n".join(r.report for r in results), encoding="utf-8")
    print(f"Saved summary: {path}")


if __name__ == "__main__":
    main()
