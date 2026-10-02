# ReflectDesinPatternDemo

Hands-on projects for the **reflection design pattern** from the *Agentic AI* course (Module 2).
In each one an agent produces a first draft (V1), critiques it, and produces an improved version (V2).

| Project | What the agent improves | Reflects on |
|---|---|---|
| [`ChartGenerationAgent/`](ChartGenerationAgent/) | matplotlib chart code for coffee sales questions | the rendered chart **image** + its code |
| [`SqlQueryImprover/`](SqlQueryImprover/) | SQL queries for questions about a product database | the SQL **and its real query output** |

## Shared setup

Both projects read API keys from one `.env` file in this folder:

```powershell
cd C:\Users\SkySun\source\repos\AIAgentPractice\ReflectDesinPatternDemo
copy .env.example .env      # then fill in OPENAI_API_KEY (and ANTHROPIC_API_KEY if you use Claude)
```

Then follow the README inside each project folder. Each has its own `requirements.txt`, a
Python script, and a notebook version of the lab.
