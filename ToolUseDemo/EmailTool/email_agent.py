"""
M3 Agentic AI - Email assistant workflow

An LLM email assistant that manages a simulated inbox through tools (aisuite):
it reads natural-language requests, picks the right tools, and chains them
(search → mark as read → send / delete ...) without asking for confirmation.

Usage:
    python email_agent.py                         # every lab section in order (3.3 → 6.5)
    python email_agent.py --step boss             # run only some sections (repeat --step)
    python email_agent.py --ask "Mark the newsletter as read and reply to Bob that noon works"
    python email_agent.py --reset                 # restore the sample inbox first

Steps: endpoints (3.3), tools (4.3), prompt (5), boss (6.3), missing-tool (6.4),
       delete-alice (6.4.1), happy-hour (6.5)
"""

# ================================
# 2. Initialize environment and client
# ================================

# --- Standard library (for the command line runner) ---
import argparse
import os
import sys

# --- Third-party ---
from dotenv import load_dotenv
import aisuite as ai
import json

# --- Local / project ---
import utils
import display_functions
import email_tools


# ================================
# Environment & Client
# ================================
load_dotenv()          # Load environment variables from .env
client = ai.Client()   # Initialize AISuite client

ALL_TOOLS = [
    email_tools.list_all_emails,
    email_tools.list_unread_emails,
    email_tools.search_emails,
    email_tools.filter_emails,
    email_tools.get_email,
    email_tools.mark_email_as_read,
    email_tools.mark_email_as_unread,
    email_tools.send_email,
    email_tools.delete_email,
    email_tools.search_unread_from_sender,
]


# ======================================================================================
# 3.3 Endpoint test helpers (no LLM): sanity-check the simulated email service
# ======================================================================================
def run_endpoint_tests():
    # uncomment the line 'utils.test_*' you want to try
    new_email_id = utils.test_send_email()
    _ = utils.test_get_email(new_email_id['id'])
    #_ = utils.test_list_emails()
    #_ = utils.test_filter_emails(recipient="test@example.com")
    #_ = utils.test_search_emails("lunch")
    #_ = utils.test_unread_emails()
    #_ = utils.test_mark_read(new_email_id['id'])
    #_ = utils.test_mark_unread(new_email_id['id'])
    #_ = utils.test_delete_email(new_email_id['id'])
    #_ = utils.reset_database()


# ======================================================================================
# 4.3 Available tools: try them by hand before the LLM uses them
# ======================================================================================
def run_tool_tests():
    # Test sending a new email and fetch it by ID
    new_email = email_tools.send_email("test@example.com", "Lunch plans", "Shall we meet at noon?")
    content_ = email_tools.get_email(new_email['id'])

    # Uncomment the ones you want to try:
    #content_ = email_tools.list_all_emails()
    #content_ = email_tools.list_unread_emails()
    #content_ = email_tools.search_emails("lunch")
    #content_ = email_tools.filter_emails(recipient="test@example.com")
    #content_ = email_tools.mark_email_as_read(new_email['id'])
    #content_ = email_tools.mark_email_as_unread(new_email['id'])
    #content_ = email_tools.search_unread_from_sender("test@example.com")
    #content_ = email_tools.delete_email(new_email['id'])

    utils.print_html(content=json.dumps(content_, indent=2), title="Testing the email_tools")


# ======================================================================================
# 5. Preparing the agent prompt
# ======================================================================================
def build_prompt(request_: str) -> str:
    return f"""
- You are an AI assistant specialized in managing emails.
- You can perform various actions such as listing, searching, filtering, and manipulating emails.
- Use the provided tools to interact with the email system.
- Never ask the user for confirmation before performing an action.
- If needed, my email address is "you@email.com" so you can use it to send emails or perform actions related to my account.

{request_.strip()}
"""


def run_prompt_example():
    example_prompt = build_prompt("Delete the Happy Hour email")
    utils.print_html(content=example_prompt, title="Example example_prompt")

    # 5.3 Resetting the email service
    utils.reset_database()


# ======================================================================================
# 6.3 LLM + Email tools: unread emails from the boss → mark as read → polite follow-up
# ======================================================================================
def run_boss_followup(model: str = "openai:gpt-4.1"):
    # Try your own requests
    prompt_ = build_prompt("Check for unread emails from boss@email.com, mark them as read, and send a polite follow-up.")

    response = client.chat.completions.create(
        model=model, # LLM
        messages=[{"role": "user", "content": (
            prompt_
        )}],
        tools=[ # list of tools that the LLM can access
            email_tools.search_unread_from_sender,
            email_tools.list_unread_emails,
            email_tools.search_emails,
            email_tools.get_email,
            email_tools.mark_email_as_read,
            email_tools.send_email
        ],
        max_turns=5,
    )

    display_functions.pretty_print_chat_completion(response)

    # Check the result in the inbox (no LLM involved)
    _show_check("Check: unread emails still from boss@email.com",
                email_tools.search_unread_from_sender("boss@email.com"))
    _show_check("Check: emails sent to boss@email.com", email_tools.filter_emails(recipient="boss@email.com"))


# ======================================================================================
# 6.4 Missing tool: delete_email is NOT in the tool list
# ======================================================================================
def run_missing_delete_tool(model: str = "openai:o4-mini"):
    # Try with a request that may call an unavailable tool
    prompt_ = build_prompt("Delete alice@work.com email")

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": (
            prompt_
        )}],
        tools=[ # list of tools that the LLM can access
            email_tools.search_unread_from_sender,
            email_tools.list_unread_emails,
            email_tools.search_emails,
            email_tools.get_email,
            email_tools.mark_email_as_read,
            email_tools.send_email
        ],
        max_turns=5
    )

    display_functions.pretty_print_chat_completion(response)
    _show_check("Check: alice@work.com emails still in the inbox", email_tools.search_emails("alice@work.com"))


# ======================================================================================
# 6.4.1 Try again with delete_email enabled
# ======================================================================================
def run_with_delete_tool(model: str = "openai:o4-mini"):
    prompt_ = build_prompt("Delete alice@work.com email")

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": (
            prompt_
        )}],
        tools=[
            email_tools.search_unread_from_sender,
            email_tools.list_unread_emails,
            email_tools.search_emails,
            email_tools.get_email,
            email_tools.mark_email_as_read,
            email_tools.send_email,
            email_tools.delete_email
        ],
        max_turns=5
    )

    display_functions.pretty_print_chat_completion(response)
    _show_check("Check: alice@work.com emails still in the inbox", email_tools.search_emails("alice@work.com"))


# ======================================================================================
# 6.5 Targeted action: Delete the "Happy Hour" email
# ======================================================================================
def run_delete_happy_hour(model: str = "openai:o4-mini"):
    prompt_ = build_prompt("Delete the happy hour email")

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": (
            prompt_
        )}],
        tools=[
            email_tools.search_unread_from_sender,
            email_tools.list_unread_emails,
            email_tools.search_emails,
            email_tools.get_email,
            email_tools.mark_email_as_read,
            email_tools.send_email,
            email_tools.delete_email
        ],
        max_turns=5
    )

    display_functions.pretty_print_chat_completion(response)
    _show_check("Check: 'Happy Hour' emails still in the inbox", email_tools.search_emails("happy hour"))


# ======================================================================================
# Your own requests (all ten tools available)
# ======================================================================================
def ask(request_: str, model: str = "openai:gpt-4.1", max_turns: int = 10):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": build_prompt(request_)}],
        tools=ALL_TOOLS,
        max_turns=max_turns,
    )
    display_functions.pretty_print_chat_completion(response)
    return response


def _show_check(title: str, result) -> None:
    count = len(result) if isinstance(result, list) else "?"
    utils.print_html(json.dumps(result, indent=2), title=f"{title} → {count}")


STEPS = {
    "endpoints": ("3.3 Endpoint test helpers", run_endpoint_tests, False),
    "tools": ("4.3 Testing the email tools", run_tool_tests, False),
    "prompt": ("5. Preparing the agent prompt + 5.3 reset", run_prompt_example, False),
    "boss": ("6.3 Unread emails from the boss → follow-up", run_boss_followup, True),
    "missing-tool": ("6.4 Missing tool: delete_email", run_missing_delete_tool, True),
    "delete-alice": ("6.4.1 Try again with delete_email enabled", run_with_delete_tool, True),
    "happy-hour": ("6.5 Delete the Happy Hour email", run_delete_happy_hour, True),
}


def main():
    parser = argparse.ArgumentParser(description="LLM email assistant over a simulated email service")
    parser.add_argument("--step", action="append", choices=list(STEPS),
                        help="lab section(s) to run; repeat the flag for several (default: all)")
    parser.add_argument("--ask", metavar="REQUEST", help="your own request; the agent can use all ten tools")
    parser.add_argument("--model", help="override the lab's models (openai:gpt-4.1 for 6.3 and --ask, "
                                        "openai:o4-mini for 6.4–6.5)")
    parser.add_argument("--max-turns", type=int, default=10, help="tool-call turns allowed for --ask")
    parser.add_argument("--reset", action="store_true", help="restore the sample inbox before running")
    args = parser.parse_args()

    selected = [] if args.ask else (args.step or list(STEPS))
    needs_llm = bool(args.ask) or any(STEPS[key][2] for key in selected)
    if needs_llm and not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Add it to the .env file in the ToolUseDemo folder "
                 "(see .env.example).")

    print(f"Simulated email service running at {email_tools.BASE_URL}  (API docs: {email_tools.BASE_URL}/docs)")
    if args.reset:
        utils.reset_database()

    if args.ask:
        ask(args.ask, args.model or "openai:gpt-4.1", args.max_turns)
        return

    failed = []
    for key in selected:
        title, run, uses_llm = STEPS[key]
        print(f"\n\n######## {title} ########")
        try:
            run(args.model) if (uses_llm and args.model) else run()
        except Exception as e:  # keep going so one failure doesn't hide the other sections
            failed.append(key)
            print(f"!! Step '{key}' failed: {type(e).__name__}: {e}")

    if failed:
        sys.exit(f"\nFailed steps: {', '.join(failed)}")


if __name__ == "__main__":
    main()
