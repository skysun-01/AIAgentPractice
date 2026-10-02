# MultiAgentDemo

Hands-on projects for the **multi-agent** design pattern from the *Agentic AI* course (Module 5).
Several specialised agents each do one job and hand their output to the next, together
producing something no single prompt would.

| Project | What it shows |
|---|---|
| [`MarketResearchTeam/`](MarketResearchTeam/) | Research → design → copywriting → packaging agents build a summer sunglasses campaign: web trends (Tavily) + product catalog, a generated campaign image, a quote, and an executive report |

## Shared setup

```powershell
cd C:\Users\SkySun\source\repos\AIAgentPractice\MultiAgentDemo
copy .env.example .env      # then fill in OPENAI_API_KEY and TAVILY_API_KEY
```

Then follow the README inside each project folder.
