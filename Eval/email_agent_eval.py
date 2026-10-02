"""
Component-level eval — EmailTool (M3: email assistant workflow)

Component under test: the email agent (LLM + the ten email tools) acting on the inbox.

Objective eval with a per-example ground truth: the expected change to the inbox.
For each request the sample inbox is reset, the agent runs, and the inbox before/after is
compared: which emails were deleted, marked read/unread, and which emails were sent. The agent
must make the expected changes and nothing else (no deleting the wrong email, no stray replies).

Runs on its own copy of the inbox (results/emails_eval.db, port 8766), so your EmailTool inbox
and any running notebook are not touched.

Run:  python email_agent_eval.py [--model openai:gpt-4.1-mini] [--min-ratio 0.75] [--limit 3]
"""

from __future__ import annotations

import argparse
import os
import re

import eval_utils as eu

EVAL_PORT = os.getenv("EMAIL_EVAL_PORT", "8766")

# Sample inbox ids: 1 Happy Hour (eric) · 2 Quarterly report (boss, unread) · 3 Design review (alice)
# 4 Lunch (bob) · 5 Password (it) · 6 Proposal draft (sent to carol) · 7 Benefits (hr)
# 8 Team offsite (boss, read) · 9 Newsletter · 10 Invoice (dana). Unread at start: 1–5.
EVAL_SET = [
    {
        "id": "delete-happy-hour",
        "request": "Delete the happy hour email",
        "deleted": {1},
    },
    {
        "id": "boss-follow-up",
        "request": "Check for unread emails from boss@email.com, mark them as read, and send a polite follow-up.",
        "marked_read": {2},
        "sent": [("boss@email.com", None)],
    },
    {
        "id": "delete-alice",
        "request": "Delete alice@work.com email",
        "deleted": {3},
    },
    {
        "id": "mark-it-read",
        "request": "Mark the email from IT about my password as read.",
        "marked_read": {5},
    },
    {
        "id": "reply-bob",
        "request": "Reply to Bob that noon works for lunch.",
        "sent": [("bob@work.com", "noon")],
        "may_mark_read": {4},
    },
    {
        "id": "delete-all-boss",
        "request": "Delete all emails from boss@email.com.",
        "deleted": {2, 8},
    },
    {
        "id": "count-unread",
        "request": "How many unread emails do I have?",
        "answer": r"\b(5|five)\b",
    },
    {
        "id": "email-carol",
        "request": "Send an email to carol@client.com asking whether she has had a chance to review the proposal draft.",
        "sent": [("carol@client.com", "proposal")],
    },
]


def load_email_project():
    # Point the EmailTool service at a separate database and port BEFORE importing it
    os.environ["EMAIL_API_PORT"] = EVAL_PORT
    os.environ["EMAIL_API_URL"] = f"http://127.0.0.1:{EVAL_PORT}"
    eu.RESULTS_DIR.mkdir(exist_ok=True)
    os.environ["EMAIL_DB_PATH"] = str(eu.RESULTS_DIR / "emails_eval.db")

    agent = eu.import_project_module("email", "email_agent")
    service = eu.import_project_module("email", "email_service")
    if service.DB_PATH.resolve() != (eu.RESULTS_DIR / "emails_eval.db").resolve():
        raise RuntimeError("EmailTool was already imported with its own inbox; run this eval in a fresh process.")
    return agent, service


def inbox_snapshot(email_tools) -> dict[int, dict]:
    return {e["id"]: e for e in email_tools.list_all_emails()}


def inbox_diff(before: dict[int, dict], after: dict[int, dict]) -> dict:
    return {
        "deleted": sorted(set(before) - set(after)),
        "read_changes": {i: after[i]["read"] for i in set(before) & set(after) if before[i]["read"] != after[i]["read"]},
        "sent": [after[i] for i in sorted(set(after) - set(before))],
    }


# --------------------------------------------------------------------------------------
# Evaluating one request (flag + checks, like the lab's evaluate_* function)
# --------------------------------------------------------------------------------------
def evaluate_inbox_change(example: dict, diff: dict, answer: str) -> tuple[bool, list[tuple[str, bool]]]:
    checks: list[tuple[str, bool]] = []
    expected_deleted = example.get("deleted", set())
    marked_read = example.get("marked_read", set())
    may_mark_read = example.get("may_mark_read", set())
    expected_sent = example.get("sent", [])

    checks.append((f"deleted exactly {sorted(expected_deleted) or 'nothing'}", set(diff["deleted"]) == expected_deleted))

    for email_id in sorted(marked_read):
        checks.append((f"email {email_id} marked read", diff["read_changes"].get(email_id) is True))
    unexpected_reads = set(diff["read_changes"]) - marked_read - may_mark_read
    checks.append(("no other emails marked read/unread", not unexpected_reads))

    for recipient, keyword in expected_sent:
        ok = any(
            e["recipient"].lower() == recipient
            and (keyword is None or keyword in (e["subject"] + " " + e["body"]).lower())
            for e in diff["sent"]
        )
        checks.append((f"sent to {recipient}" + (f" mentioning '{keyword}'" if keyword else ""), ok))
    allowed_recipients = {recipient for recipient, _ in expected_sent}
    stray = [e["recipient"] for e in diff["sent"] if e["recipient"].lower() not in allowed_recipients]
    checks.append(("no other emails sent" if expected_sent else "no emails sent", not stray))

    if "answer" in example:
        checks.append((f"answer matches /{example['answer']}/", bool(re.search(example["answer"], answer or "", re.I))))

    return all(ok for _, ok in checks), checks


def run_eval(model: str = "openai:gpt-4.1", min_ratio: float = 0.75, limit: int | None = None,
             max_turns: int = 10, show: bool = True) -> eu.EvalResult:
    eu.require_api_key(model)
    agent, service = load_email_project()
    email_tools = agent.email_tools

    examples = EVAL_SET[:limit] if limit else EVAL_SET
    rows = []
    for i, example in enumerate(examples, 1):
        eu.progress(f"[email {i}/{len(examples)}] {example['request']}")
        service.reset_db()
        before = inbox_snapshot(email_tools)
        sequence, answer = [], ""
        try:
            response = agent.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": agent.build_prompt(example["request"])}],
                tools=agent.ALL_TOOLS,
                max_turns=max_turns,
            )
            answer = response.choices[0].message.content or ""
            sequence = agent.display_functions.tool_sequence(response)
            diff = inbox_diff(before, inbox_snapshot(email_tools))
            flag, checks = evaluate_inbox_change(example, diff, answer)
        except Exception as e:  # API problems: count as a failure
            diff = inbox_diff(before, inbox_snapshot(email_tools))
            flag, checks = False, [(f"run without error ({type(e).__name__}: {e})", False)]
        rows.append({
            "id": example["id"], "request": example["request"], "pass": flag, "tool_calls": sequence,
            "answer": answer, "diff": diff, "checks": [{"check": c, "pass": ok} for c, ok in checks],
        })
    service.reset_db()

    n = len(rows)
    score = sum(r["pass"] for r in rows) / n
    avg_calls = sum(len(r["tool_calls"]) for r in rows) / n
    flag = score >= min_ratio

    def change_summary(diff: dict) -> str:
        parts = []
        if diff["deleted"]:
            parts.append(f"deleted {diff['deleted']}")
        if diff["read_changes"]:
            parts.append("read " + ", ".join(f"{k}→{'read' if v else 'unread'}" for k, v in diff["read_changes"].items()))
        if diff["sent"]:
            parts.append("sent to " + ", ".join(e["recipient"] for e in diff["sent"]))
        return "; ".join(parts) or "no change"

    table = eu.md_table(
        ["Request", "Inbox change", "Tool calls", "Status"],
        [[r["request"], change_summary(r["diff"]), len(r["tool_calls"]), eu.mark(r["pass"])] for r in rows],
    )
    failures = "\n".join(
        f"- **{r['id']}**: " + "; ".join(c["check"] for c in r["checks"] if not c["pass"])
        + f" — tools: {' → '.join(r['tool_calls']) or 'none'}"
        for r in rows if not r["pass"]
    ) or "_None._"

    report = f"""### Eval — EmailTool: correct inbox changes
- Examples: {n}
- Model: `{model}`
- Requests handled correctly (expected changes, nothing else): {score:.0%} ({sum(r['pass'] for r in rows)}/{n})
- Average tool calls per request: {avg_calls:.1f}
- Threshold: {min_ratio:.0%}
- Status: {eu.status(flag)}

**Details:**

{table}

**Failed checks:**
{failures}
"""
    result = eu.EvalResult(
        name="email", title="EmailTool — correct inbox changes", metric="requests handled correctly",
        score=score, threshold=min_ratio, passed=flag, report=report, rows=rows,
        extra={"avg_tool_calls": avg_calls},
    )
    if show:
        eu.show_markdown(report)
    return result


def main():
    parser = argparse.ArgumentParser(description="Component-level eval for the email assistant agent")
    parser.add_argument("--model", default="openai:gpt-4.1")
    parser.add_argument("--min-ratio", type=float, default=0.75, help="share of correct requests needed (0–1)")
    parser.add_argument("--limit", type=int, help="only run the first N requests")
    args = parser.parse_args()

    result = run_eval(args.model, args.min_ratio, args.limit)
    print(f"\nSaved report: {eu.save_result(result)}")


if __name__ == "__main__":
    main()
