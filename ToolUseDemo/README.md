# ToolUseDemo

Hands-on projects for the **tool-use design pattern** from the *Agentic AI* course (Module 3).
In each one an LLM is given Python functions as tools and decides when and how to call them.

| Project | What it shows |
|---|---|
| [`FunctionsToTools/`](FunctionsToTools/) | Turning functions into aisuite tools: automatic vs manual tool execution, choosing between tools, chaining several tools in one request |
| [`EmailTool/`](EmailTool/) | An email assistant agent with 10 tools over a simulated email service (FastAPI + SQLite): search, read, send, delete — and what happens when a tool is missing |

## Shared setup

Projects in this folder read the API key from one `.env` file here:

```powershell
cd C:\Users\SkySun\source\repos\AIAgentPractice\ToolUseDemo
copy .env.example .env      # then fill in OPENAI_API_KEY
```

Then follow the README inside each project folder.
