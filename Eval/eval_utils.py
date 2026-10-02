"""
Shared helpers for the component-level evals (M4).

  - import_project_module : import a module from one of the AIAgentPractice projects
                            (each project has its own utils.py / display_functions.py,
                            so they are imported without clashing)
  - EvalResult            : a dataset-level result: score, threshold, PASS/FAIL flag, Markdown report
  - md_table / status     : small Markdown builders for the reports
  - show_markdown         : render Markdown in Jupyter, print it in a terminal
  - save_result           : write the report (.md) and the per-example rows (.json) to results/
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

from dotenv import load_dotenv

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent  # AIAgentPractice
RESULTS_DIR = EVAL_DIR / "results"

PROJECTS = {
    "chart": ROOT / "ReflectDesinPatternDemo" / "ChartGenerationAgent",
    "sql": ROOT / "ReflectDesinPatternDemo" / "SqlQueryImprover",
    "tools": ROOT / "ToolUseDemo" / "FunctionsToTools",
    "email": ROOT / "ToolUseDemo" / "EmailTool",
}

# API keys: a .env in this folder, or the ones you already created for the other modules
load_dotenv(EVAL_DIR / ".env")
load_dotenv(ROOT / "ReflectDesinPatternDemo" / ".env")
load_dotenv(ROOT / "ToolUseDemo" / ".env")

# Emojis in the reports must not crash a Windows console / redirected output
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# Module names that several projects define for themselves
_PROJECT_HELPERS = ("utils", "display_functions")


def import_project_module(project: str, module: str) -> ModuleType:
    """Import `module` from a project folder, giving it that project's own helper modules."""
    folder = PROJECTS[project]
    if not folder.exists():
        raise FileNotFoundError(f"Project folder not found: {folder}")

    for name in _PROJECT_HELPERS:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(folder))
    try:
        return importlib.import_module(module)
    finally:
        sys.path.remove(str(folder))
        # The imported module keeps its own references; free the names for the next project
        for name in _PROJECT_HELPERS:
            sys.modules.pop(name, None)


def require_api_key(*models: str) -> None:
    """Fail fast if a model's provider key is missing ('openai:gpt-4.1', 'gpt-4o-mini', 'claude-...')."""
    for model in models:
        provider = model.split(":", 1)[0] if ":" in model else ("anthropic" if "claude" in model else "openai")
        env_var = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(provider.lower())
        if env_var and not os.getenv(env_var):
            raise RuntimeError(
                f"{env_var} is not set, but model '{model}' needs it. Add it to Eval/.env "
                "(see .env.example) or to the .env of ReflectDesinPatternDemo / ToolUseDemo."
            )


# --------------------------------------------------------------------------------------
# Results & reports
# --------------------------------------------------------------------------------------
@dataclass
class EvalResult:
    name: str             # e.g. "sql"
    title: str            # e.g. "SqlQueryImprover — execution accuracy"
    metric: str           # what `score` measures
    score: float          # 0.0–1.0
    threshold: float      # min_ratio needed to pass
    passed: bool
    report: str           # Markdown
    rows: list[dict[str, Any]] = field(default_factory=list)  # per-example details
    extra: dict[str, Any] = field(default_factory=dict)        # other metrics (e.g. V1 score)


def status(flag: bool) -> str:
    return "✅ PASS" if flag else "❌ FAIL"


def mark(flag: bool) -> str:
    return "✅" if flag else "❌"


def md_cell(value: Any, limit: int = 90) -> str:
    text = " ".join(str(value).split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(md_cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except ImportError:
        return False


def show_markdown(markdown: str) -> None:
    if _in_notebook():
        from IPython.display import Markdown, display

        display(Markdown(markdown))
    else:
        print(markdown)


def save_result(result: EvalResult) -> Path:
    """Save the Markdown report and the raw rows; returns the .md path."""
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    md_path = RESULTS_DIR / f"{result.name}_{stamp}.md"
    md_path.write_text(result.report, encoding="utf-8")
    (RESULTS_DIR / f"{result.name}_{stamp}.json").write_text(
        json.dumps(asdict(result), indent=2, default=str), encoding="utf-8"
    )
    return md_path


def progress(message: str) -> None:
    print(message, flush=True)
