# Chart Generation Agent (Reflection Design Pattern)

An agentic workflow that writes, runs, critiques and improves its own matplotlib chart code,
based on the *M2 Agentic AI – Chart Generation* lab.

```
 user instruction
        │
        ▼
 1. Generate V1 code  (LLM, text)          generate_chart_code()
        │
        ▼
 2. Execute V1 → chart_v1.png              exec(<execute_python> block)
        │
        ▼
 3. Reflect: LLM looks at the chart IMAGE  reflect_on_image_and_regenerate()
    + the V1 code → feedback + V2 code
        │
        ▼
 4. Execute V2 → chart_v2.png              exec(<execute_python> block)
```

## Project files

| File | What it is |
|---|---|
| `reflection_agent.py` | The agent as a Python script, following the lab sections 2 → 4 |
| `M2_Chart_Generation_Reflection.ipynb` | The same lab as a notebook, cell by cell (charts render inline) |
| `utils.py` | Helpers: load data, call OpenAI / Claude (text + image), display results |
| `coffee_sales.csv` | Synthetic vending-machine sales, Jan 2024 – Sep 2025 (5,978 rows) |
| `generate_coffee_data.py` | Regenerates `coffee_sales.csv` (fixed seed, reproducible) |

### Dataset

`date, time, cash_type, card, price, coffee_name` — `utils.load_and_prepare_data()` parses
`date` and adds `quarter`, `month`, `year`. The data has a story built in: 2025 has more
customers and ~6% higher prices, Latte/Cortado grow, Espresso declines, and Hot Chocolate/Cocoa
sell more in winter.

## Setup

API keys live in one shared `.env` in the parent `ReflectDesinPatternDemo` folder
(a `.env` in this folder also works and takes priority).

```powershell
cd C:\Users\SkySun\source\repos\AIAgentPractice\ReflectDesinPatternDemo
copy .env.example .env      # then put your OPENAI_API_KEY (and/or ANTHROPIC_API_KEY) in .env
cd ChartGenerationAgent
pip install -r requirements.txt
```

## Run

**Script**

```powershell
python reflection_agent.py                    # section 3 step by step, then the section 4 workflow
python reflection_agent.py --mode steps       # only the step-by-step walkthrough -> chart_v1/v2.png
python reflection_agent.py --mode workflow    # only run_workflow()               -> drink_sales_v1/v2.png

# your own question + a Claude model for the reflection step
python reflection_agent.py --mode workflow `
  --instruction "Plot monthly revenue by coffee type in 2025" `
  --reflection-model claude-sonnet-5-5 `
  --image-basename monthly_revenue
```

Charts are saved in the project folder; open `*_v1.png` and `*_v2.png` side by side to see what
the reflection step changed.

**Notebook** — open `M2_Chart_Generation_Reflection.ipynb` in VS Code / Jupyter and run the cells
top to bottom. Change `user_instructions`, the models and `image_basename` in section 4.2 to experiment.

## Models

Defaults follow the lab: `gpt-4o-mini` generates V1, `o4-mini` reflects (it must accept images).
Any model name containing `claude` is routed to the Anthropic API, everything else to OpenAI.

## Note

The workflow runs LLM-generated Python with `exec()` on your machine, exactly as in the lab.
That's fine for this demo, but don't point it at untrusted instructions or sensitive data.
