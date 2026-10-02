"""
Email tools for the LLM agent. Each function wraps one route of the simulated email service.

Design (as in the lab's hints):
  - short, imperative docstrings: aisuite turns them into the tool descriptions
  - one clear responsibility per tool (single route, single effect)
  - compact JSON results (dicts / lists), so the model can chain them;
    failures come back as {"error": "..."} instead of raising, so the agent can react
"""

import requests

import email_service

# Make sure the email service is up before any tool is used
BASE_URL = email_service.ensure_running()


def _request(method: str, path: str, **kwargs):
    response = requests.request(method, f"{BASE_URL}{path}", timeout=10, **kwargs)
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return {"error": f"{response.status_code}: {detail}"}
    return response.json()


def list_all_emails():
    """List all emails, newest first."""
    return _request("GET", "/emails")


def list_unread_emails():
    """List unread emails, newest first."""
    return _request("GET", "/emails/unread")


def search_emails(query: str):
    """Search emails by keyword in the subject, body, or sender.

    Args:
        query: Keyword or email address to search for.
    """
    return _request("GET", "/emails/search", params={"q": query})


def filter_emails(recipient: str = "", date_from: str = "", date_to: str = ""):
    """Filter emails by recipient and/or date range.

    Args:
        recipient: Recipient email address to match. Leave empty for any recipient.
        date_from: Earliest date to include, as YYYY-MM-DD. Leave empty for no lower bound.
        date_to: Latest date to include, as YYYY-MM-DD. Leave empty for no upper bound.
    """
    params = {"recipient": recipient, "date_from": date_from, "date_to": date_to}
    return _request("GET", "/emails/filter", params={k: v for k, v in params.items() if v})


def get_email(email_id: int):
    """Get one email by its ID.

    Args:
        email_id: ID of the email.
    """
    return _request("GET", f"/emails/{email_id}")


def mark_email_as_read(email_id: int):
    """Mark an email as read.

    Args:
        email_id: ID of the email.
    """
    return _request("PATCH", f"/emails/{email_id}/read")


def mark_email_as_unread(email_id: int):
    """Mark an email as unread.

    Args:
        email_id: ID of the email.
    """
    return _request("PATCH", f"/emails/{email_id}/unread")


def send_email(recipient: str, subject: str, body: str):
    """Send an email from you@email.com (simulated, nothing is really delivered).

    Args:
        recipient: Recipient email address.
        subject: Subject line.
        body: Email body text.
    """
    return _request("POST", "/send", json={"recipient": recipient, "subject": subject, "body": body})


def delete_email(email_id: int):
    """Delete an email by its ID.

    Args:
        email_id: ID of the email to delete.
    """
    return _request("DELETE", f"/emails/{email_id}")


def search_unread_from_sender(sender: str):
    """List unread emails from a given sender.

    Args:
        sender: Sender email address, e.g. boss@email.com.
    """
    unread = list_unread_emails()
    if isinstance(unread, dict):  # error from the service
        return unread
    return [email for email in unread if email["sender"].lower() == sender.strip().lower()]
