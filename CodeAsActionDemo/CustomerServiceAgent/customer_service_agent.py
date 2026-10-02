"""
M5 Agentic AI - Customer Service Agent (planning with code execution / "code as action")

Instead of a JSON plan executed step by step with many small tools, the LLM writes Python
(TinyDB) code that IS the plan: commented steps that filter, compute and update. The code is
then executed in a controlled namespace and its `answer_text` is shown to the customer.

Usage:
    python customer_service_agent.py                      # every lab section in order
    python customer_service_agent.py --step round         # 2.3–2.4: "round sunglasses under $100?" (read-only)
    python customer_service_agent.py --step return        # 2.4: return 2 Aviators (mutation)
    python customer_service_agent.py --step agent         # 4: multi-item purchase via customer_service_agent()
    python customer_service_agent.py --ask "Purchase 3 Wayfarer sunglasses for customer Alice."
    python customer_service_agent.py --ask "How many Aviators are left?"   # state carries over
    python customer_service_agent.py --ask "..." --reseed                   # start from the default store
"""

# ==== Imports ====
from __future__ import annotations
import json
from dotenv import load_dotenv
from openai import OpenAI
import re, io, sys, traceback, json
from typing import Any, Dict, Optional
from tinydb import Query, where

# Utility modules
import utils      # helper functions for prompting/printing
import inv_utils  # functions for inventory, transactions, schema building, and TinyDB seeding

# --- for the command line runner ---
import argparse
import os

load_dotenv()
client = OpenAI() if os.getenv("OPENAI_API_KEY") else None  # created lazily in main() if the key is missing


# ======================================================================================
# 2.1 The plan: prompt that makes the model PLAN BY WRITING CODE
# ======================================================================================
PROMPT = """You are a senior data assistant. PLAN BY WRITING PYTHON CODE USING TINYDB.

Database Schema & Samples (read-only):
{schema_block}

Execution Environment (already imported/provided):
- Variables: db, inventory_tbl, transactions_tbl  # TinyDB Table objects
- Helpers: get_current_balance(tbl) -> float, next_transaction_id(tbl, prefix="TXN") -> str
- Natural language: user_request: str  # the original user message

PLANNING RULES (critical):
- Derive ALL filters/parameters from user_request (shape/keywords, price ranges "under/over/between", stock mentions,
  quantities, buy/return intent). Do NOT hard-code values.
- Build TinyDB queries dynamically with Query(). If a constraint isn't in user_request, don't apply it.
- Be conservative: if intent is ambiguous, do read-only (DRY RUN).

TRANSACTION POLICY (hard):
- Do NOT create aggregated multi-item transactions.
- If the request contains multiple items, create a separate transaction row PER ITEM.
- For each item:
  - compute its own line total (unit_price * qty),
  - insert ONE transaction with that amount,
  - update balance sequentially (balance += line_total),
  - update the item’s stock.
- If any requested item lacks sufficient stock, do NOT mutate anything; reply with STATUS="insufficient_stock".

HUMAN RESPONSE REQUIREMENT (hard):
- You MUST set a variable named `answer_text` (type str) with a short, customer-friendly sentence (1–2 lines).
- This sentence is the only user-facing message. No dataframes/JSON, no boilerplate disclaimers.
- If nothing matches, politely say so and offer a nearby alternative (closest style/price) or a next step.

ACTION POLICY:
- If the request clearly asks to change state (buy/purchase/return/restock/adjust):
    ACTION="mutate"; SHOULD_MUTATE=True; perform the change and write a matching transaction row.
  Otherwise:
    ACTION="read"; SHOULD_MUTATE=False; simulate and explain briefly as a dry run (in logs only).

FAILURE & EDGE-CASE HANDLING (must implement):
- Do not capture outer variables in Query.test. Pass them as explicit args.
- Always set a short `answer_text`. Also set a string `STATUS` to one of:
  "success", "no_match", "insufficient_stock", "invalid_request", "unsupported_intent".
- no_match: No items satisfy the filters → suggest the closest in style/price, or invite a different range.
- insufficient_stock: Item found but stock < requested qty → state available qty and offer the max you can fulfill.
- invalid_request: Unable to parse essential info (e.g., quantity for a purchase/return) → ask for the missing piece succinctly.
- unsupported_intent: The action is outside the store’s capabilities → provide the nearest supported alternative.
- In all cases, keep the tone helpful and concise (1–2 sentences). Put technical details (e.g., ACTION/DRY RUN) only in stdout logs.

OUTPUT CONTRACT:
- Return ONLY executable Python between these tags (no extra text):
  <execute_python>
  # your python
  </execute_python>

CODE CHECKLIST (follow in code):
1) Parse intent & constraints from user_request (regex ok).
2) Build TinyDB condition incrementally; query inventory_tbl.
3) If mutate: validate stock, update inventory, insert a transaction (new id, amount, balance, timestamp).
4) ALWAYS set:
   - `answer_text` (human sentence, required),
   - `STATUS` (see list above).
   Also print a brief log to stdout, e.g., "LOG: ACTION=read DRY_RUN=True STATUS=no_match".
5) Optional: set `answer_rows` or `answer_json` if useful, but `answer_text` is mandatory.

TONE EXAMPLES (for `answer_text`):
- success: "Yes, we have our Classic sunglasses, a round frame, for $60."
- no_match: "We don’t have round frames under $100 in stock right now, but our Moon round frame is available at $120."
- insufficient_stock: "We only have 1 pair of Classic left; I can reserve that for you."
- invalid_request: "I can help with that—how many pairs would you like to purchase?"
- unsupported_intent: "We can’t refurbish frames, but I can suggest similar new models."

Constraints:
- Use TinyDB Query for filtering. Standard library imports only if needed.
- Keep code clear and commented with numbered steps.

User request:
{question}
"""


# ======================================================================================
# 2.2 From Prompt to Code (Planning in Code)
# ======================================================================================
# ---------- 1) Code generation ----------
def generate_llm_code(
    prompt: str,
    *,
    inventory_tbl,
    transactions_tbl,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.2,
) -> str:
    """
    Ask the LLM to produce a plan-with-code response.
    Returns the FULL assistant content (including surrounding text and tags).
    The actual code extraction happens later in execute_generated_code.
    """
    schema_block = inv_utils.build_schema_block(inventory_tbl, transactions_tbl)
    prompt = PROMPT.format(schema_block=schema_block, question=prompt)

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": "You write safe, well-commented TinyDB code to handle data questions and updates."
            },
            {"role": "user", "content": prompt},
        ],
    )
    content = resp.choices[0].message.content or ""

    return content


# ======================================================================================
# 2.4 Define the executor function (run a given plan)
# ======================================================================================
# --- Helper: extract code between <execute_python>...</execute_python> ---
def _extract_execute_block(text: str) -> str:
    """
    Returns the Python code inside <execute_python>...</execute_python>.
    If no tags are found, assumes 'text' is already raw Python code.
    """
    if not text:
        raise RuntimeError("Empty content passed to code executor.")
    m = re.search(r"<execute_python>(.*?)</execute_python>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()


# ---------- 2) Code execution ----------
def execute_generated_code(
    code_or_content: str,
    *,
    db,
    inventory_tbl,
    transactions_tbl,
    user_request: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute code in a controlled namespace.
    Accepts either raw Python code OR full content with <execute_python> tags.
    Returns minimal artifacts: stdout, error, and extracted answer.
    """
    # Extract code here (now centralized)
    code = _extract_execute_block(code_or_content)

    SAFE_GLOBALS = {
        "Query": Query,
        "get_current_balance": inv_utils.get_current_balance,
        "next_transaction_id": inv_utils.next_transaction_id,
        "user_request": user_request or "",
    }
    SAFE_LOCALS = {
        "db": db,
        "inventory_tbl": inventory_tbl,
        "transactions_tbl": transactions_tbl,
    }
    # Change from the lab, which ran exec(code, SAFE_GLOBALS, SAFE_LOCALS): with two separate
    # dicts, functions and lambdas defined in the generated code cannot see its top-level
    # imports/variables (NameError). One shared namespace avoids that.
    namespace = {**SAFE_GLOBALS, **SAFE_LOCALS}

    # Capture stdout from the executed code
    _stdout_buf, _old_stdout = io.StringIO(), sys.stdout
    sys.stdout = _stdout_buf
    err_text = None
    try:
        exec(code, namespace)
    except Exception:
        err_text = traceback.format_exc()
    finally:
        sys.stdout = _old_stdout
    printed = _stdout_buf.getvalue().strip()

    # Extract possible answers set by the generated code
    answer = (
        namespace.get("answer_text")
        or namespace.get("answer_rows")
        or namespace.get("answer_json")
    )


    return {
        "code": code,            # code without the <execute_python> tags
        "stdout": printed,
        "error": err_text,
        "answer": answer,
        "status": namespace.get("STATUS"),  # addition: the STATUS the plan reported
        "transactions_tbl": transactions_tbl.all(),  # For inspection
        "inventory_tbl": inventory_tbl.all(),  # For inspection
    }


def _show_run_details(exec_res: Dict[str, Any]) -> None:
    """Show the plan's STATUS, logs and any error (the lab returns these but doesn't print them)."""
    details = f"STATUS: {exec_res.get('status')}\n\n{exec_res.get('stdout') or '(no logs)'}"
    if exec_res.get("error"):
        details += f"\n\nERROR:\n{exec_res['error']}"
    utils.print_html(details, title="Plan Execution · Status, Logs & Errors")


# ======================================================================================
# 2.3 + 2.4 Read-only example: round sunglasses under $100
# ======================================================================================
def run_round_example(db, inventory_tbl, transactions_tbl, model: str = "o4-mini", temperature: float = 1.0):
    Item = Query()                    # Create a Query object to reference fields (e.g., Item.name, Item.description)

    # Search the inventory table for documents where either the description OR the name
    # contains the word "round" (case-insensitive). The check is done inline:
    # - (v or "") ensures we handle None by converting it to an empty string
    # - .lower() normalizes case
    # - " round " enforces a crude word boundary (won't match "wraparound")
    round_sunglasses = inventory_tbl.search(
        (Item.description.test(lambda v: " round " in ((v or "").lower()))) |
        (Item.name.test(        lambda v: " round " in ((v or "").lower())))
    )

    # Render the results as formatted JSON
    utils.print_html(json.dumps(round_sunglasses, indent=2), title="Inventory Status: Round Sunglasses")

    # Andrew's prompt from the lecture
    prompt_round = "Do you have any round sunglasses in stock that are under $100?"

    # Generate the plan-as-code (FULL content; may include <execute_python> tags)
    full_content_round = generate_llm_code(
        prompt_round,
        inventory_tbl=inventory_tbl,
        transactions_tbl=transactions_tbl,
        model=model,
        temperature=temperature,
    )

    # Inspect the LLM’s plan + code (no execution here)
    utils.print_html(full_content_round, title="Plan with Code (Full Response)")

    # Execute the generated plan for the round-sunglasses question
    result = execute_generated_code(
        full_content_round,          # the full LLM response you generated earlier
        db=db,
        inventory_tbl=inventory_tbl,
        transactions_tbl=transactions_tbl,
        user_request=prompt_round, # e.g., "Do you have any round sunglasses in stock that are under $100?"
    )

    # Peek at the answer the plan produced
    utils.print_html(result["answer"], title="Plan Execution · Extracted Answer")
    _show_run_details(result)
    return result


# ======================================================================================
# 2.4 Mutation example: return two Aviator sunglasses
# ======================================================================================
def run_return_example(db, inventory_tbl, transactions_tbl, model: str = "o4-mini", temperature: float = 1.0):
    Item = Query()                    # Create a Query object to reference fields (e.g., Item.name, Item.description)

    # Query: fetch all inventory rows whose 'name' is exactly "Aviator".
    # Notes:
    # - This is a case-sensitive equality check. "aviator" won't match.
    # - If you need case-insensitive matching, consider a .test(...) or .matches(...) with re.I.
    aviators = inventory_tbl.search(
        (Item.name == "Aviator")
    )

    # Display the matched documents in a readable JSON panel
    utils.print_html(json.dumps(aviators, indent=2), title="Inventory status: Aviator sunglasses before return")

    prompt_aviator = "Return 2 Aviator sunglasses I bought last week."

    # Generate the plan-as-code (FULL content; may include <execute_python> tags)
    full_content_aviator = generate_llm_code(
        prompt_aviator,
        inventory_tbl=inventory_tbl,
        transactions_tbl=transactions_tbl,
        model=model,
        temperature=temperature,
    )

    # Inspect the LLM’s plan + code (no execution here)
    utils.print_html(full_content_aviator, title="Plan with Code (Full Response)")

    utils.print_html(json.dumps(transactions_tbl.all(), indent=2), title="Transactions Table Before Return")

    # Execute the generated plan for the Aviator return
    result = execute_generated_code(
        full_content_aviator,          # the full LLM response you generated earlier
        db=db,
        inventory_tbl=inventory_tbl,
        transactions_tbl=transactions_tbl,
        user_request=prompt_aviator, # e.g., "Return 2 aviator sunglasses I bought last week."
    )

    # Peek at the answer the plan produced
    utils.print_html(result["answer"], title="Plan Execution · Extracted Answer")
    _show_run_details(result)

    utils.print_html(json.dumps(transactions_tbl.all(), indent=2), title="Transactions Table After Return")

    aviators = inventory_tbl.search(
        (Item.name == "Aviator")
    )

    utils.print_html(json.dumps(aviators, indent=2), title="Inventory status: Aviator sunglasses after return")
    return result


# ======================================================================================
# 3. Putting It All Together: Customer Service Agent
# ======================================================================================
def customer_service_agent(
    question: str,
    *,
    db,
    inventory_tbl,
    transactions_tbl,
    model: str = "o4-mini",
    temperature: float = 1.0,
    reseed: bool = False,
) -> dict:
    """
    End-to-end helper:
      1) (Optional) reseed inventory & transactions
      2) Generate plan-as-code from `question`
      3) Execute in a controlled namespace
      4) Render before/after snapshots and return artifacts

    Returns:
      {
        "full_content": <raw LLM response (may include <execute_python> tags)>,
        "exec": {
            "code": <extracted python>,
            "stdout": <plan logs>,
            "error": <traceback or None>,
            "answer": <answer_text/rows/json>,
            "inventory_after": [...],
            "transactions_after": [...]
        }
      }
    """
    # 0) Optional reseed
    if reseed:
        inv_utils.create_inventory()
        inv_utils.create_transactions()

    # 1) Show the question
    utils.print_html(question, title="User Question")

    # 2) Generate plan-as-code (FULL content)
    full_content = generate_llm_code(
        question,
        inventory_tbl=inventory_tbl,
        transactions_tbl=transactions_tbl,
        model=model,
        temperature=temperature,
    )
    utils.print_html(full_content, title="Plan with Code (Full Response)")

    # 3) Before snapshots
    utils.print_html(json.dumps(inventory_tbl.all(), indent=2), title="Inventory Table · Before")
    utils.print_html(json.dumps(transactions_tbl.all(), indent=2), title="Transactions Table · Before")

    # 4) Execute
    exec_res = execute_generated_code(
        full_content,
        db=db,
        inventory_tbl=inventory_tbl,
        transactions_tbl=transactions_tbl,
        user_request=question,
    )

    # 5) After snapshots + final answer
    utils.print_html(exec_res["answer"], title="Plan Execution · Extracted Answer")
    utils.print_html(json.dumps(inventory_tbl.all(), indent=2), title="Inventory Table · After")
    utils.print_html(json.dumps(transactions_tbl.all(), indent=2), title="Transactions Table · After")

    # 6) Return artifacts
    return {
        "full_content": full_content,
        "exec": {
            "code": exec_res["code"],
            "stdout": exec_res["stdout"],
            "error": exec_res["error"],
            "answer": exec_res["answer"],
            "status": exec_res["status"],
            "inventory_after": inventory_tbl.all(),
            "transactions_after": transactions_tbl.all(),
        },
    }


# ======================================================================================
# 4. Try It Out (with the Customer Service Agent)
# ======================================================================================
def run_agent_example(db, inventory_tbl, transactions_tbl, model: str = "o4-mini", temperature: float = 1.0):
    prompt = "I want to buy 3 pairs of classic sunglasses and 1 pair of aviator sunglasses."

    out = customer_service_agent(
        prompt,
        db=db,
        inventory_tbl=inventory_tbl,
        transactions_tbl=transactions_tbl,
        model=model,
        temperature=temperature,
        reseed=True,   # set False to keep current state of the inventory and the transactions
    )
    _show_run_details(out["exec"])
    return out


STEPS = {
    "round": ("2.3–2.4 Read-only: round sunglasses under $100", run_round_example),
    "return": ("2.4 Mutation: return 2 Aviator sunglasses", run_return_example),
    "agent": ("4. Customer service agent: multi-item purchase", run_agent_example),
}


def main():
    global client
    parser = argparse.ArgumentParser(description="Customer service agent that plans by writing TinyDB code")
    parser.add_argument("--step", action="append", choices=list(STEPS),
                        help="lab section(s) to run; repeat the flag for several (default: all)")
    parser.add_argument("--ask", metavar="REQUEST", help="your own customer request")
    parser.add_argument("--reseed", action="store_true", help="with --ask: reset the store before answering")
    parser.add_argument("--model", default="o4-mini")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="o-series models (o4-mini) only accept 1.0; e.g. 0.2 for gpt-4.1-mini")
    args = parser.parse_args()

    if client is None:
        if not os.getenv("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY is not set. Add it to the .env file in the CodeAsActionDemo folder "
                     "(see .env.example).")
        client = OpenAI()

    if args.ask:
        # Keep the store's current state between --ask runs, unless --reseed is given
        db, inventory_tbl, transactions_tbl = inv_utils.seed_db() if args.reseed else inv_utils.load_db()
        out = customer_service_agent(args.ask, db=db, inventory_tbl=inventory_tbl,
                                     transactions_tbl=transactions_tbl, model=args.model,
                                     temperature=args.temperature)
        _show_run_details(out["exec"])
        return

    # 2.1 Create example tables (fresh store)
    db, inventory_tbl, transactions_tbl = inv_utils.seed_db()
    utils.print_html(json.dumps(inventory_tbl.all(), indent=2), title="Inventory Table")
    utils.print_html(json.dumps(transactions_tbl.all(), indent=2), title="Transactions Table")

    failed = []
    for key in args.step or list(STEPS):
        title, run = STEPS[key]
        print(f"\n\n######## {title} ########")
        try:
            run(db, inventory_tbl, transactions_tbl, model=args.model, temperature=args.temperature)
        except Exception as e:  # API problems etc.: report and keep going
            failed.append(key)
            print(f"!! Step '{key}' failed: {type(e).__name__}: {e}")
    if failed:
        sys.exit(f"\nFailed steps: {', '.join(failed)}")


if __name__ == "__main__":
    main()
