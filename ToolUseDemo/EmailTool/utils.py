"""
Helpers for the email assistant lab.

  - test_* functions : call the email service endpoints directly (no LLM) and show the result,
                       to sanity-check the backend before handing control to the agent
  - reset_database   : restore the sample inbox
  - print_html       : pretty display (HTML card in Jupyter, plain text in a terminal)
"""

from __future__ import annotations

import html
import json
import sys
from typing import Any

import requests

import email_service

# Make sure the email service is up before any helper is used
BASE_URL = email_service.ensure_running()

# Emojis in the output must not crash a Windows console / redirected output
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass


# --------------------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------------------
def _in_notebook() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except ImportError:
        return False


def print_html(content: Any, title: str | None = None) -> None:
    """Display text or JSON: an HTML card in Jupyter, plain text in a terminal."""
    if not _in_notebook():
        if title:
            print("\n" + "=" * 80)
            print(title)
            print("=" * 80)
        print(content)
        return

    from IPython.display import HTML, display

    header = (
        f'<div style="font-weight:600;font-size:1.05em;margin-bottom:8px;">{html.escape(title)}</div>'
        if title
        else ""
    )
    display(HTML(
        '<div style="border:1px solid #8884;border-radius:8px;padding:12px 14px;margin:8px 0;">'
        f'{header}<pre style="white-space:pre-wrap;margin:0;">{html.escape(str(content))}</pre></div>'
    ))


# --------------------------------------------------------------------------------------
# Endpoint test helpers
# --------------------------------------------------------------------------------------
def _call(method: str, path: str, title: str, **kwargs):
    response = requests.request(method, f"{BASE_URL}{path}", timeout=10, **kwargs)
    result = response.json()
    status = "" if response.ok else f"  ❌ HTTP {response.status_code}"
    print_html(json.dumps(result, indent=2), title=f"{method} {path}{status} — {title}")
    return result


def test_send_email(recipient: str = "test@example.com", subject: str = "Test email",
                    body: str = "Hello! This is a test email sent through the endpoint helper."):
    return _call("POST", "/send", "send a new email",
                 json={"recipient": recipient, "subject": subject, "body": body})


def test_get_email(email_id: int):
    return _call("GET", f"/emails/{email_id}", f"fetch email {email_id}")


def test_list_emails():
    return _call("GET", "/emails", "list all emails")


def test_filter_emails(recipient: str | None = None, date_from: str | None = None, date_to: str | None = None):
    params = {"recipient": recipient, "date_from": date_from, "date_to": date_to}
    return _call("GET", "/emails/filter", "filter by recipient / date range",
                 params={k: v for k, v in params.items() if v})


def test_search_emails(query: str):
    return _call("GET", "/emails/search", f"search for '{query}'", params={"q": query})


def test_unread_emails():
    return _call("GET", "/emails/unread", "unread emails")


def test_mark_read(email_id: int):
    return _call("PATCH", f"/emails/{email_id}/read", f"mark email {email_id} as read")


def test_mark_unread(email_id: int):
    return _call("PATCH", f"/emails/{email_id}/unread", f"mark email {email_id} as unread")


def test_delete_email(email_id: int):
    return _call("DELETE", f"/emails/{email_id}", f"delete email {email_id}")


def reset_database():
    return _call("GET", "/reset_database", "reset the inbox to its initial state")
