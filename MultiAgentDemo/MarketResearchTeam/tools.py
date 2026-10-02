"""
Tools for the market research team (M5 lab: multi-agent campaign pipeline).

  - tavily_search_tool(query)     : live web search via Tavily (needs TAVILY_API_KEY)
  - product_catalog_tool()        : the internal sunglasses catalog
  - get_available_tools()         : the tool schemas handed to the LLM
  - handle_tool_call(tool_call)   : run the tool the LLM asked for
  - create_tool_response_message(): wrap a tool result as a "tool" message for the conversation
"""

from __future__ import annotations

import json
import os
from typing import Any

# Same store as CodeAsActionDemo/CustomerServiceAgent
PRODUCT_CATALOG = [
    {"item_id": "SG001", "name": "Aviator",
     "description": "Original aviator style with an iconic teardrop metal frame and polarized lenses.",
     "quantity_in_stock": 23, "price": 80},
    {"item_id": "SG002", "name": "Wayfarer",
     "description": "Bold trapezoid acetate frame, an everyday favourite that suits most face shapes.",
     "quantity_in_stock": 14, "price": 95},
    {"item_id": "SG003", "name": "Cat Eye",
     "description": "Retro cat eye frames with upswept corners and tinted lenses for a glamorous look.",
     "quantity_in_stock": 8, "price": 110},
    {"item_id": "SG004", "name": "Sport",
     "description": "Lightweight wraparound frames for running and cycling, with impact-resistant lenses.",
     "quantity_in_stock": 15, "price": 70},
    {"item_id": "SG005", "name": "Classic",
     "description": "Classic round profile with minimalist metal frames, offering a timeless and versatile "
                    "style that fits both casual and formal wear.",
     "quantity_in_stock": 10, "price": 60},
    {"item_id": "SG006", "name": "Moon",
     "description": "Oversized round frames with gradient lenses for a bold, retro statement.",
     "quantity_in_stock": 6, "price": 120},
    {"item_id": "SG007", "name": "Clubmaster",
     "description": "Vintage browline frames with a half-rim top and metal lower rim.",
     "quantity_in_stock": 2, "price": 130},
    {"item_id": "SG008", "name": "Shield",
     "description": "One-piece shield lens with a futuristic frameless look and full UV protection.",
     "quantity_in_stock": 0, "price": 150},
]


def tavily_search_tool(query: str, max_results: int = 5, include_images: bool = False) -> list[dict] | dict:
    """
    Search the web with Tavily and return a compact list of results
    ({"title", "url", "content"}), or {"error": "..."} if the search can't run.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {"error": "TAVILY_API_KEY is not set, so live web search is unavailable."}
    try:
        from tavily import TavilyClient

        response = TavilyClient(api_key=api_key).search(
            query=query, max_results=max_results, include_images=include_images,
        )
    except Exception as e:  # network / quota / auth problems: let the agent see them
        return {"error": f"Tavily search failed: {type(e).__name__}: {e}"}

    results = [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
        for r in response.get("results", [])
    ]
    if include_images and response.get("images"):
        results.append({"images": response["images"]})
    return results


def product_catalog_tool(max_items: int = 100) -> list[dict]:
    """Return the internal sunglasses catalog (id, name, description, stock, price)."""
    return [dict(item) for item in PRODUCT_CATALOG[:max_items]]


# --------------------------------------------------------------------------------------
# Tool plumbing for the agent loop
# --------------------------------------------------------------------------------------
def get_available_tools() -> list[dict]:
    """Tool schemas in the OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": "tavily_search_tool",
                "description": "Search the web for current information, e.g. sunglasses fashion trends. "
                               "Returns a list of results with title, url and a content snippet.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query."},
                        "max_results": {"type": "integer", "description": "Number of results (default 5)."},
                        "include_images": {"type": "boolean", "description": "Also return image URLs (default false)."},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "product_catalog_tool",
                "description": "Return the internal sunglasses catalog: item_id, name, description, "
                               "quantity_in_stock and price for every product.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "max_items": {"type": "integer", "description": "Maximum number of products (default all)."},
                    },
                },
            },
        },
    ]


_TOOL_NAMES = {"tavily_search_tool", "product_catalog_tool"}


def handle_tool_call(tool_call) -> Any:
    """Execute the tool requested by the LLM and return its (JSON-serializable) result."""
    name = tool_call.function.name
    if name not in _TOOL_NAMES:
        return {"error": f"Unknown tool: {name}"}
    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError as e:
        return {"error": f"Invalid arguments for {name}: {e}"}
    try:
        return globals()[name](**args)  # looked up at call time, so tests can swap a tool
    except TypeError as e:
        return {"error": f"Bad arguments for {name}: {e}"}


def create_tool_response_message(tool_call, result: Any) -> dict:
    """The "tool" message that answers one tool call."""
    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_call.function.name,
        "content": json.dumps(result, default=str),
    }
