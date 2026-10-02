"""
Helper module for the Reflection Design Pattern demo (M2 - Chart Generation).

Provides:
  - load_and_prepare_data      : load coffee_sales.csv and derive year / quarter / month
  - print_html                 : pretty display (HTML in Jupyter, plain text in a terminal)
  - get_response               : text-only LLM call (OpenAI or Anthropic)
  - encode_image_b64           : read an image file as (media_type, base64)
  - image_openai_call          : multimodal call (prompt + image) to an OpenAI model
  - image_anthropic_call       : multimodal call (prompt + image) to a Claude model
  - ensure_execute_python_tags : make sure code is wrapped in <execute_python> tags
"""

from __future__ import annotations

import base64
import html
import mimetypes
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

# Read OPENAI_API_KEY / ANTHROPIC_API_KEY from a .env in this project folder,
# or the shared one in the parent ReflectDesinPatternDemo folder (project values win)
_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env")
load_dotenv(_HERE.parent / ".env")

# Emojis in the step messages must not crash a Windows console / redirected output
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

MAX_TOKENS = 4096


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load the coffee sales CSV, parse dates and add year / quarter / month columns."""
    df = pd.read_csv(csv_path, dtype={"time": str, "card": str})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)

    df["card"] = df["card"].fillna("")
    df["quarter"] = df["date"].dt.quarter.astype(int)
    df["month"] = df["date"].dt.month.astype(int)
    df["year"] = df["date"].dt.year.astype(int)
    return df


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


def print_html(content: Any, title: str | None = None, is_image: bool = False) -> None:
    """
    Display text, code, a DataFrame or an image.

    In Jupyter it renders a small HTML card; in a terminal it prints plain text
    (for images it prints where the file was saved).
    """
    if _in_notebook():
        _display_notebook(content, title, is_image)
    else:
        _display_console(content, title, is_image)


def _display_notebook(content: Any, title: str | None, is_image: bool) -> None:
    from IPython.display import HTML, display

    if is_image:
        path = Path(str(content))
        if path.exists():
            media_type, b64 = encode_image_b64(str(path))
            body = (
                f'<img src="data:{media_type};base64,{b64}" '
                'style="max-width:100%;height:auto;border-radius:6px;">'
            )
        else:
            body = (
                f"<em>Image not found: {html.escape(str(path))} "
                "(the generated code probably failed to run).</em>"
            )
    elif isinstance(content, pd.DataFrame):
        body = content.to_html(border=0)
    else:
        body = f'<pre style="white-space:pre-wrap;margin:0;">{html.escape(str(content))}</pre>'

    header = (
        f'<div style="font-weight:600;font-size:1.05em;margin-bottom:8px;">{html.escape(title)}</div>'
        if title
        else ""
    )
    display(HTML(
        '<div style="border:1px solid #8884;border-radius:8px;padding:12px 14px;margin:8px 0;">'
        f"{header}{body}</div>"
    ))


def _display_console(content: Any, title: str | None, is_image: bool) -> None:
    if title:
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

    if is_image:
        path = Path(str(content)).resolve()
        if path.exists():
            print(f"[image saved] {path}")
        else:
            print(f"[image not found] {path} (the generated code probably failed to run)")
    elif isinstance(content, pd.DataFrame):
        print(content.to_string())
    else:
        print(content)


# --------------------------------------------------------------------------------------
# LLM clients
# --------------------------------------------------------------------------------------
_clients: dict[str, Any] = {}


def _is_anthropic(model: str) -> bool:
    lower = model.lower()
    return "claude" in lower or "anthropic" in lower


def _require_key(env_var: str, model: str) -> None:
    if not os.getenv(env_var):
        raise RuntimeError(
            f"{env_var} is not set, but model '{model}' needs it. "
            "Add it to the .env file in the ReflectDesinPatternDemo folder (see .env.example)."
        )


def check_api_keys(*models: str) -> None:
    """Fail fast with a clear message if a model's API key is missing."""
    for model in models:
        _require_key("ANTHROPIC_API_KEY" if _is_anthropic(model) else "OPENAI_API_KEY", model)


def _openai_client():
    if "openai" not in _clients:
        from openai import OpenAI

        _clients["openai"] = OpenAI()
    return _clients["openai"]


def _anthropic_client():
    if "anthropic" not in _clients:
        from anthropic import Anthropic

        _clients["anthropic"] = Anthropic()
    return _clients["anthropic"]


def _join_text_blocks(response) -> str:
    """Claude can return several content blocks; keep (and join) only the text ones."""
    return "\n".join(block.text for block in response.content if block.type == "text").strip()


def get_response(model: str, prompt: str) -> str:
    """Send a text-only prompt to an OpenAI or Anthropic model and return the reply text."""
    if _is_anthropic(model):
        _require_key("ANTHROPIC_API_KEY", model)
        response = _anthropic_client().messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return _join_text_blocks(response)

    _require_key("OPENAI_API_KEY", model)
    response = _openai_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return (response.choices[0].message.content or "").strip()


def encode_image_b64(path: str) -> tuple[str, str]:
    """Return (media_type, base64_data) for an image file."""
    media_type = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return media_type, b64


def image_openai_call(model_name: str, prompt: str, media_type: str, b64: str) -> str:
    """Send a prompt plus an image to an OpenAI vision-capable model."""
    _require_key("OPENAI_API_KEY", model_name)
    response = _openai_client().chat.completions.create(
        model=model_name,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
            ],
        }],
    )
    return (response.choices[0].message.content or "").strip()


def image_anthropic_call(model_name: str, prompt: str, media_type: str, b64: str) -> str:
    """Send a prompt plus an image to a Claude model; joins all text blocks of the reply."""
    _require_key("ANTHROPIC_API_KEY", model_name)
    response = _anthropic_client().messages.create(
        model=model_name,
        max_tokens=MAX_TOKENS,
        system=(
            "You are a careful data visualization reviewer. Follow the requested output "
            "format exactly: one line of JSON, then the code inside <execute_python> tags."
        ),
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return _join_text_blocks(response)


# --------------------------------------------------------------------------------------
# Code helpers
# --------------------------------------------------------------------------------------
def ensure_execute_python_tags(code: str) -> str:
    """Wrap code in <execute_python> tags (stripping Markdown fences) if it isn't already."""
    code = code.strip()
    if not code:
        return ""
    if "<execute_python>" in code:
        return code

    fenced = re.match(r"^```(?:python)?\s*\n([\s\S]*?)\n?```$", code)
    if fenced:
        code = fenced.group(1).strip()
    return f"<execute_python>\n{code}\n</execute_python>"
