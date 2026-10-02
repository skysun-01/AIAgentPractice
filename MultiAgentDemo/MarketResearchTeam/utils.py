"""
Logging helpers for the market research team: each agent's steps are shown as styled
blocks in Jupyter (HTML), or as readable text in a terminal.

  - log_agent_title_html(title, icon)
  - log_tool_call_html(tool_name, arguments)
  - log_tool_result_html(result)
  - log_final_summary_html(content)   # HTML, Markdown or plain text
  - log_unexpected_html()

It also loads OPENAI_API_KEY / TAVILY_API_KEY from a .env in this folder or the parent
MultiAgentDemo folder.
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env")
load_dotenv(_HERE.parent / ".env")

# Emojis must not crash a Windows console or redirected output
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

MAX_RESULT_CHARS = 3000  # keep long tool results readable


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except ImportError:
        return False


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        try:  # pretty-print JSON strings (tool arguments, tool results)
            return json.dumps(json.loads(value), indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            return value
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def _truncate(text: str) -> str:
    return text if len(text) <= MAX_RESULT_CHARS else text[:MAX_RESULT_CHARS] + f"\n… ({len(text):,} chars total)"


def _looks_like_html(text: str) -> bool:
    return bool(re.match(r"\s*<[a-zA-Z]", text))


def _html_to_text(text: str) -> str:
    text = re.sub(r"<img[^>]*src=\"([^\"]+)\"[^>]*>", r"[image: \1]", text)
    text = re.sub(r"<(br|/p|/h\d|/li)\s*/?>", "\n", text)
    return html.unescape(re.sub(r"<[^>]+>", "", text)).strip()


def _card(body: str, accent: str) -> None:
    from IPython.display import HTML, display

    display(HTML(
        f'<div style="border-left:4px solid {accent};border-radius:6px;padding:10px 14px;'
        f'margin:8px 0;background:rgba(127,127,127,0.06);">{body}</div>'
    ))


def log_agent_title_html(title: str, icon: str = "🤖") -> None:
    if _in_notebook():
        from IPython.display import HTML, display

        display(HTML(f'<h2 style="margin:18px 0 6px 0;">{icon} {html.escape(title)}</h2>'))
    else:
        print("\n" + "=" * 80 + f"\n{icon} {title}\n" + "=" * 80)


def log_tool_call_html(tool_name: str, arguments: Any) -> None:
    args = _as_text(arguments)
    if _in_notebook():
        _card(f"<b>🔧 Tool call:</b> <code>{html.escape(tool_name)}</code>"
              f'<pre style="white-space:pre-wrap;margin:6px 0 0 0;">{html.escape(args)}</pre>', "#1f6feb")
    else:
        print(f"\n🔧 Tool call: {tool_name}\n{args}")


def log_tool_result_html(result: Any) -> None:
    text = _truncate(_as_text(result))
    if _in_notebook():
        _card(f'<b>📦 Result</b><pre style="white-space:pre-wrap;margin:6px 0 0 0;">{html.escape(text)}</pre>',
              "#8b949e")
    else:
        print(f"\n📦 Result\n{text}")


def log_final_summary_html(content: Any) -> None:
    text = content if isinstance(content, str) else _as_text(content)
    if not _in_notebook():
        print("\n✅ Output\n" + (_html_to_text(text) if _looks_like_html(text) else text))
        return
    if _looks_like_html(text):
        _card(text, "#2da44e")
        return
    try:  # Markdown → HTML so headings/lists render inside the card
        import markdown

        body = markdown.markdown(text, extensions=["tables"])
    except ImportError:
        body = f'<pre style="white-space:pre-wrap;margin:0;">{html.escape(text)}</pre>'
    _card("<b>✅ Output</b>" + body, "#2da44e")


def log_unexpected_html() -> None:
    message = "⚠️ Unexpected: the model returned neither content nor tool calls."
    if _in_notebook():
        _card(f"<b>{message}</b>", "#cf222e")
    else:
        print("\n" + message)
