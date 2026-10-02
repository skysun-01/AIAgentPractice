# CodeAsActionDemo

Hands-on projects for **planning with code execution** ("code as action") from the *Agentic AI*
course (Module 5). The LLM writes code that *is* the plan: commented steps that are executed
directly, instead of a JSON plan run through many small tools.

| Project | What it shows |
|---|---|
| [`CustomerServiceAgent/`](CustomerServiceAgent/) | A sunglasses store assistant that answers questions and handles purchases/returns by writing and running TinyDB code |

## Shared setup

```powershell
cd C:\Users\SkySun\source\repos\AIAgentPractice\CodeAsActionDemo
copy .env.example .env      # then fill in OPENAI_API_KEY
```

Then follow the README inside each project folder.
