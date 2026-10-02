"""
Component-level eval — FunctionsToTools (M3: turning functions into tools)

Component under test: the LLM's tool use — does it pick the right tool(s), with the right
arguments, in the right order (and no tool when none is needed)?

Objective eval with a per-example ground truth (expected tools, order and argument rules).
The project's real tool functions are replaced by recording fakes that keep the same name,
signature and docstring — so aisuite describes them to the LLM exactly as in the project —
but nothing is written to disk and no network calls are made. Fakes return fixed values
(time 14:05:09, weather 72.5°F), so we can also check that results flow into later calls.

Run:  python tool_selection_eval.py [--model openai:gpt-4.1-mini] [--min-ratio 0.8] [--limit 3]
"""

from __future__ import annotations

import argparse
import functools
from pathlib import PurePath

import eval_utils as eu

FAKE_TIME = "14:05:09"
FAKE_WEATHER = "Current: 72.5°F, High: 80.1°F, Low: 65.3°F"

EVAL_SET = [
    {
        "id": "time",
        "prompt": "What time is it?",
        "tools": ["get_current_time"],
    },
    {
        "id": "weather",
        "prompt": "Can you get the weather for my location?",
        "tools": ["get_weather_from_ip"],
    },
    {
        "id": "note",
        "prompt": "Make a txt note called groceries.txt with this list: milk, eggs, bread.",
        "tools": ["write_txt_file"],
        "args": {"write_txt_file": {"file_path": ("endswith", "groceries.txt"),
                                    "content": ("contains_all", ["milk", "eggs", "bread"])}},
    },
    {
        "id": "qr",
        "prompt": "Create a QR code that links to https://example.com using the logo at dl_logo.jpg, and call it example_qr.",
        "tools": ["generate_qr_code"],
        "args": {"generate_qr_code": {"data": ("contains", "example.com"),
                                      "filename": ("stem", "example_qr"),
                                      "image_path": ("endswith", "dl_logo.jpg")}},
    },
    {
        "id": "weather-note",
        "prompt": "Write the current weather into a note called weather.txt.",
        "tools": ["get_weather_from_ip", "write_txt_file"],
        "order": [("get_weather_from_ip", "write_txt_file")],
        "args": {"write_txt_file": {"file_path": ("endswith", "weather.txt"), "content": ("contains", "72.5")}},
    },
    {
        "id": "time-note",
        "prompt": "Save the current time in a file called time.txt.",
        "tools": ["get_current_time", "write_txt_file"],
        "order": [("get_current_time", "write_txt_file")],
        "args": {"write_txt_file": {"file_path": ("endswith", "time.txt"), "content": ("contains", "14:05")}},
    },
    {
        "id": "qr-and-time",
        "prompt": "Make a QR code for https://deeplearning.ai with my logo dl_logo.jpg named dl_qr, and also tell me what time it is.",
        "tools": ["generate_qr_code", "get_current_time"],
        "args": {"generate_qr_code": {"data": ("contains", "deeplearning.ai"), "filename": ("stem", "dl_qr")}},
        "answer": ("contains", "14:05"),
    },
    {
        "id": "no-tool-needed",
        "prompt": "What is the capital of France?",
        "tools": [],
        "answer": ("contains", "paris"),
    },
]


# --------------------------------------------------------------------------------------
# Argument rules
# --------------------------------------------------------------------------------------
def rule_holds(rule: tuple, value) -> bool:
    kind, expected = rule
    text = str(value).strip().lower()
    if kind == "contains":
        return expected.lower() in text
    if kind == "contains_all":
        return all(e.lower() in text for e in expected)
    if kind == "endswith":
        return text.replace("\\", "/").endswith(expected.lower())
    if kind == "stem":  # file name without extension, e.g. "example_qr" or "example_qr.png"
        return PurePath(text).stem == expected.lower() or text == expected.lower()
    raise ValueError(f"Unknown rule: {kind}")


def describe(rule: tuple) -> str:
    kind, expected = rule
    return f"{kind} {expected!r}"


# --------------------------------------------------------------------------------------
# Recording fakes with the real tools' names, signatures and docstrings
# --------------------------------------------------------------------------------------
def make_fake_tools(tools_module, calls: list) -> list:
    fake_results = {
        "get_current_time": lambda: FAKE_TIME,
        "get_weather_from_ip": lambda: FAKE_WEATHER,
        "write_txt_file": lambda file_path, content: file_path,
        "generate_qr_code": lambda data, filename, image_path:
            f"QR code saved as {filename}.png containing: {data[:50]}...",
    }

    def fake(real_function):
        @functools.wraps(real_function)  # keeps __name__, __doc__ and the signature for aisuite
        def recorder(**kwargs):
            calls.append({"tool": real_function.__name__, "args": kwargs})
            return fake_results[real_function.__name__](**kwargs)
        return recorder

    return [fake(getattr(tools_module, name)) for name in fake_results]


# --------------------------------------------------------------------------------------
# Evaluating one example (flag + details, like the lab's evaluate_* function)
# --------------------------------------------------------------------------------------
def evaluate_tool_calls(example: dict, calls: list[dict], answer: str) -> tuple[bool, list[tuple[str, bool]]]:
    called = [c["tool"] for c in calls]
    expected = set(example["tools"])
    checks: list[tuple[str, bool]] = []

    if expected:
        checks.append((f"called {', '.join(sorted(expected))}", expected <= set(called)))
    checks.append(("no unexpected tools" if expected else "no tools called", set(called) <= expected))

    for first, then in example.get("order", []):
        ok = first in called and then in called and called.index(first) < called.index(then)
        checks.append((f"{first} before {then}", ok))

    for tool, rules in example.get("args", {}).items():
        tool_calls = [c["args"] for c in calls if c["tool"] == tool]
        for arg, rule in rules.items():
            ok = any(rule_holds(rule, args.get(arg, "")) for args in tool_calls)
            checks.append((f"{tool}.{arg} {describe(rule)}", ok))

    if "answer" in example:
        checks.append((f"answer {describe(example['answer'])}", rule_holds(example["answer"], answer or "")))

    return all(ok for _, ok in checks), checks


def load_tools_project():
    return eu.import_project_module("tools", "tools_agent")


def run_eval(model: str = "openai:o4-mini", min_ratio: float = 0.8, limit: int | None = None,
             show: bool = True) -> eu.EvalResult:
    eu.require_api_key(model)
    tools_agent = load_tools_project()

    examples = EVAL_SET[:limit] if limit else EVAL_SET
    rows = []
    for i, example in enumerate(examples, 1):
        eu.progress(f"[tools {i}/{len(examples)}] {example['prompt']}")
        calls: list[dict] = []
        answer = ""
        try:
            response = tools_agent.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": example["prompt"]}],
                tools=make_fake_tools(tools_agent, calls),
                max_turns=5,
            )
            answer = response.choices[0].message.content or ""
            flag, checks = evaluate_tool_calls(example, calls, answer)
        except Exception as e:  # API problems or invalid tool arguments: count as a failure
            flag, checks = False, [(f"run without error ({type(e).__name__}: {e})", False)]
        rows.append({
            "id": example["id"], "prompt": example["prompt"], "expected_tools": example["tools"],
            "calls": calls, "answer": answer, "pass": flag,
            "checks": [{"check": name, "pass": ok} for name, ok in checks],
        })

    n = len(rows)
    score = sum(r["pass"] for r in rows) / n
    all_checks = [c for r in rows for c in r["checks"]]
    check_rate = sum(c["pass"] for c in all_checks) / len(all_checks)
    flag = score >= min_ratio

    table = eu.md_table(
        ["Prompt", "Expected", "Called (in order)", "Checks", "Status"],
        [[r["prompt"], ", ".join(r["expected_tools"]) or "—",
          " → ".join(c["tool"] for c in r["calls"]) or "—",
          f"{sum(c['pass'] for c in r['checks'])}/{len(r['checks'])}", eu.mark(r["pass"])] for r in rows],
    )
    failures = "\n".join(
        f"- **{r['id']}**: " + "; ".join(c["check"] for c in r["checks"] if not c["pass"])
        + (f" — calls: `{r['calls']}`" if r["calls"] else "")
        for r in rows if not r["pass"]
    ) or "_None._"

    report = f"""### Eval — FunctionsToTools: tool selection & arguments
- Examples: {n}
- Model: `{model}`
- Examples fully correct: {score:.0%} ({sum(r['pass'] for r in rows)}/{n})
- Individual checks passed: {check_rate:.0%} ({sum(c['pass'] for c in all_checks)}/{len(all_checks)})
- Threshold: {min_ratio:.0%}
- Status: {eu.status(flag)}

**Details:**

{table}

**Failed checks:**
{failures}
"""
    result = eu.EvalResult(
        name="tools", title="FunctionsToTools — tool selection & arguments", metric="examples fully correct",
        score=score, threshold=min_ratio, passed=flag, report=report, rows=rows,
        extra={"check_pass_rate": check_rate},
    )
    if show:
        eu.show_markdown(report)
    return result


def main():
    parser = argparse.ArgumentParser(description="Component-level eval for the tool-calling agent")
    parser.add_argument("--model", default="openai:o4-mini")
    parser.add_argument("--min-ratio", type=float, default=0.8, help="share of fully correct examples needed (0–1)")
    parser.add_argument("--limit", type=int, help="only run the first N prompts")
    args = parser.parse_args()

    result = run_eval(args.model, args.min_ratio, args.limit)
    print(f"\nSaved report: {eu.save_result(result)}")


if __name__ == "__main__":
    main()
