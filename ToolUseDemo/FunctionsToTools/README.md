# Functions to Tools (Tool-Use Design Pattern)

Give an LLM controlled access to ordinary Python functions with **aisuite**. Based on the
*M3 Agentic AI – Turning functions into tools* lab.

```
 your prompt ──► LLM ──► "call write_txt_file(file_path=..., content=...)"
                  ▲                    │
                  │                    ▼   aisuite runs the Python function on your machine
                  └──── tool result ◄──┘   and sends the result back (up to max_turns times)
                  │
                  ▼
           final answer
```

## The tools

| Tool | What it does | Arguments |
|---|---|---|
| `get_current_time` | Local time as `HH:MM:SS` | — |
| `get_weather_from_ip` | Current/high/low °F for your location (ipinfo.io → open-meteo.com, no keys needed) | — |
| `write_txt_file` | Writes a text file (overwrites) | `file_path`, `content` |
| `generate_qr_code` | QR code PNG with a logo in the middle | `data`, `filename`, `image_path` |

aisuite builds each tool's schema from the function's **signature and docstring**: type
hints become parameter types, and the `Args:` section becomes parameter descriptions. That's why
the docstrings matter.

## Project files

| File | What it is |
|---|---|
| `tools_agent.py` | The lab as a Python script (sections 2 → 4.3), plus an `--ask` mode for your own prompts |
| `M3_Turning_Functions_Into_Tools.ipynb` | The same lab as a notebook, cell by cell |
| `display_functions.py` | `pretty_print_chat_completion()`: shows tool calls → results → final answer |
| `dl_logo.jpg` | Placeholder logo for the QR code prompts (replace it with your own) |

## Setup

```powershell
cd C:\Users\SkySun\source\repos\AIAgentPractice\ToolUseDemo
copy .env.example .env        # put your OPENAI_API_KEY in it
cd FunctionsToTools
pip install -r requirements.txt
```

## Run

```powershell
python tools_agent.py                            # every lab section in order
python tools_agent.py --step first-tool          # 3.1–3.3: one tool, run automatically (max_turns)
python tools_agent.py --step manual              # 3.4: hand-written schema, you run the tool call
python tools_agent.py --step weather --step note --step qr   # 4.2
python tools_agent.py --step multi               # 4.3: several tools chained in one request

# your own request: the LLM can use all four tools
python tools_agent.py --ask "Write today's weather and the current time into today.txt"
python tools_agent.py --step multi --model openai:gpt-4.1-mini   # try another model
```

The lab uses `openai:gpt-4o` for section 3 and `openai:o4-mini` for section 4. Override both with
`--model`. Files the LLM creates (`reminders.txt`, `dl_qr_code.png`, …) are saved in this folder.

**Notebook:** open `M3_Turning_Functions_Into_Tools.ipynb` and run the cells top to bottom.

## Things to know

- **The LLM chooses the arguments.** `write_txt_file` writes wherever the model says and
  overwrites existing files. That's fine for this demo, but don't hand these tools to
  untrusted prompts.
- **The weather tool sends your public IP to ipinfo.io** to find your location (that's how
  the lab's tool works).
- **Tool errors stop that request.** If a tool raises, e.g. the logo path doesn't exist or
  there's no internet, aisuite passes the exception up. The script reports the failed step
  and moves on to the next one.
