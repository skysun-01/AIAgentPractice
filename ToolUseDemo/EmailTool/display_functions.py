"""
Helper module for the M3 lab "Email assistant workflow".

Provides:
  - pretty_print_chat_completion : show every step of a tool-calling run
                                   (tool calls → tool results → final answer)

It also loads OPENAI_API_KEY from a .env in this folder or the parent ToolUseDemo folder.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env")
load_dotenv(_HERE.parent / ".env")

# Emojis in the output must not crash a Windows console / redirected output
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except ImportError:
        return False


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a dict message (tool results) or an SDK message object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _format_args(arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            return arguments
    return ", ".join(f"{k}={v!r}" for k, v in (arguments or {}).items())


def _format_tool_content(content: Any) -> str:
    # aisuite stores tool results JSON-encoded; show strings plainly and pretty-print the rest
    if isinstance(content, str):
        try:
            decoded = json.loads(content)
            return decoded if isinstance(decoded, str) else json.dumps(decoded, indent=2)
        except json.JSONDecodeError:
            return content
    return str(content)


def _steps(response: Any) -> list[tuple[str, str]]:
    """Turn a chat completion into (label, text) steps, in the order they happened."""
    choice = response.choices[0]
    messages = list(getattr(choice, "intermediate_messages", None) or [choice.message])
    final_index = max(
        (i for i, m in enumerate(messages) if _get(m, "role") != "tool" and _get(m, "content")),
        default=-1,
    )

    steps = []
    for i, msg in enumerate(messages):
        if _get(msg, "role") == "tool":
            name = _get(msg, "name") or "tool"
            steps.append((f"📦 Tool result · {name}", _format_tool_content(_get(msg, "content"))))
            continue
        for call in _get(msg, "tool_calls") or []:
            function = _get(call, "function")
            name, args = _get(function, "name"), _get(function, "arguments")
            steps.append(("🔧 Tool call", f"{name}({_format_args(args)})"))
        content = _get(msg, "content")
        if content:
            label = "✅ Final response" if i == final_index else "💬 Assistant"
            steps.append((label, content))
    return steps


def tool_sequence(response: Any) -> list[str]:
    """Names of the tools the model called, in order."""
    return [text.split("(", 1)[0] for label, text in _steps(response) if label == "🔧 Tool call"]


def pretty_print_chat_completion(response: Any) -> None:
    """Show the tool calls, tool results and final answer of a chat completion."""
    steps = _steps(response)
    sequence = tool_sequence(response)
    summary = " → ".join(sequence) if sequence else "no tools called"

    if not _in_notebook():
        print("\n" + "=" * 80)
        print(f"Tool sequence: {summary}")
        print("=" * 80)
        for label, text in steps:
            print(f"\n{label}\n{text}")
        return

    from IPython.display import HTML, display

    cards = "".join(
        '<div style="border:1px solid #8884;border-radius:8px;padding:10px 12px;margin:6px 0;">'
        f'<div style="font-weight:600;margin-bottom:4px;">{html.escape(label)}</div>'
        f'<pre style="white-space:pre-wrap;margin:0;">{html.escape(str(text))}</pre></div>'
        for label, text in steps
    )
    display(HTML(
        f'<div style="margin:8px 0;"><div style="margin-bottom:6px;"><b>Tool sequence:</b> '
        f"{html.escape(summary)}</div>{cards}</div>"
    ))
