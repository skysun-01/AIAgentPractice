# Email Tool (Email Assistant Agent)

An LLM email assistant that manages a **simulated inbox** through tools: you give it plain
instructions ("check unread emails from my boss and send a polite follow-up", "delete the Happy
Hour email") and it picks the tools, chains them, and does the work. Based on the
*M3 Agentic AI – Email assistant workflow* lab.

```
 "Check unread emails from boss@email.com, mark them as read, send a follow-up"
        │
        ▼  LLM (aisuite, max_turns)
 search_unread_from_sender ─► mark_email_as_read ─► send_email ─► "Found 1 unread email, ..."
        │                           │                    │
        ▼                           ▼                    ▼
   GET /emails/unread       PATCH /emails/2/read     POST /send        ← simulated email service
                                                                         (FastAPI + SQLite)
```

## Project files

| File | What it is |
|---|---|
| `email_agent.py` | The lab as a Python script (sections 3.3 → 6.5), plus `--ask` for your own requests |
| `M3_Email_Assistant.ipynb` | The same lab as a notebook, cell by cell |
| `email_service.py` | The simulated email backend: FastAPI endpoints, SQLite + SQLAlchemy storage, Pydantic validation, 10 sample emails |
| `email_tools.py` | The 10 tools the LLM can use, each wrapping one endpoint |
| `utils.py` | `test_*` helpers to call the endpoints directly (no LLM), `reset_database()`, `print_html()` |
| `display_functions.py` | `pretty_print_chat_completion()`: tool sequence → calls → results → final answer |

**You don't need to start the email service.** Importing `utils` or `email_tools` starts it
in the background on `http://127.0.0.1:8765` (or reuses one that's already running). The
inbox is stored in `emails.db`, created on first use. To run the service on its own and click
through the API docs, run `python email_service.py`, then open http://127.0.0.1:8765/docs.

### The tools

| Tool | Endpoint |
|---|---|
| `list_all_emails()` | `GET /emails` |
| `list_unread_emails()` | `GET /emails/unread` |
| `search_emails(query)` | `GET /emails/search?q=` (subject, body, sender) |
| `filter_emails(recipient, date_from, date_to)` | `GET /emails/filter` (dates as `YYYY-MM-DD`, inclusive) |
| `get_email(email_id)` | `GET /emails/{id}` |
| `mark_email_as_read(email_id)` / `mark_email_as_unread(email_id)` | `PATCH /emails/{id}/read` / `/unread` |
| `send_email(recipient, subject, body)` | `POST /send` (stored as sent by `you@email.com`, never delivered) |
| `delete_email(email_id)` | `DELETE /emails/{id}` |
| `search_unread_from_sender(sender)` | unread emails, filtered by sender |

Tools return compact JSON. If something fails (e.g. a wrong id), they return
`{"error": "404: Email 99 not found"}` instead of raising, so the agent can see the problem and react.

## Setup

```powershell
cd C:\Users\SkySun\source\repos\AIAgentPractice\ToolUseDemo
copy .env.example .env        # put your OPENAI_API_KEY in it (shared with FunctionsToTools)
cd EmailTool
pip install -r requirements.txt
```

## Run

```powershell
python email_agent.py                          # every lab section in order
python email_agent.py --step endpoints --step tools    # 3.3 / 4.3: backend checks, no API key needed
python email_agent.py --step boss              # 6.3: unread from boss → mark read → follow-up
python email_agent.py --step missing-tool      # 6.4: delete request WITHOUT delete_email → can't do it
python email_agent.py --step delete-alice      # 6.4.1: same request WITH delete_email
python email_agent.py --step happy-hour        # 6.5: find and delete the Happy Hour email

# your own request: the agent gets all ten tools
python email_agent.py --ask "Mark the IT password email as read and reply to Bob that noon works"
python email_agent.py --reset --step boss      # start from the sample inbox again
```

The lab uses `openai:gpt-4.1` for 6.3 and `openai:o4-mini` for 6.4–6.5. Override them with
`--model`. After each LLM step the script checks the inbox directly (e.g. "'Happy Hour' emails
still in the inbox → 0"), so you can see whether the agent really did the job.

**Notebook:** open `M3_Email_Assistant.ipynb` and run the cells top to bottom.

## Things to know

- **The agent acts without asking.** `build_prompt()` tells it never to ask for confirmation,
  as in the lab. That's fine for a sandbox inbox, but for a real mailbox you'd want a human check
  before sending or deleting.
- **State carries over between runs** because the inbox lives in `emails.db`. Use `--reset`
  or `utils.reset_database()` to get the 10 sample emails back (ids restart at 1, so the
  Happy Hour email is id 1 again).
- **Port in use?** Set `EMAIL_API_PORT` to another port before running.
