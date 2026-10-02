"""
Display helper for the customer service agent lab.

  - print_html : show text, code, JSON or Python objects as a titled card in Jupyter,
                 or as plain text in a terminal

It also loads OPENAI_API_KEY from a .env in this folder or the parent CodeAsActionDemo folder.
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

# Emojis / $ signs etc. must not crash a Windows console or redirected output
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


def _to_text(content: Any) -> str:
    if content is None:
        return "(nothing returned)"
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, indent=2, default=str)
    except TypeError:
        return str(content)


def print_html(content: Any, title: str | None = None) -> None:
    """Display content under a title: an HTML card in Jupyter, plain text in a terminal."""
    text = _to_text(content)
    if not _in_notebook():
        if title:
            print("\n" + "=" * 80)
            print(title)
            print("=" * 80)
        print(text)
        return

    from IPython.display import HTML, display

    header = (
        f'<div style="font-weight:600;font-size:1.05em;margin-bottom:8px;">{html.escape(title)}</div>'
        if title
        else ""
    )
    display(HTML(
        '<div style="border:1px solid #8884;border-radius:8px;padding:12px 14px;margin:8px 0;">'
        f'{header}<pre style="white-space:pre-wrap;margin:0;">{html.escape(text)}</pre></div>'
    ))
