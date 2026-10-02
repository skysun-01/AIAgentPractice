# SQL Query Improver (Reflection Design Pattern)

An agentic workflow that turns a plain-English question into SQL, runs it, looks at the
**real query output**, and fixes its own query. Based on the *M2 Agentic AI – Improving SQL
Generation with Reflection* lab.

```
 question + schema
        │
        ▼
 1. Generate SQL V1 (LLM)                         generate_sql()
        │
        ▼
 2. Execute V1 → DataFrame                        utils.execute_sql()
        │
        ▼
 3. Reflect with EXTERNAL feedback:               refine_sql_external_feedback()
    question + V1 SQL + V1 output → feedback + V2
        │
        ▼
 4. Execute V2 → final answer
```

The lab also shows the weaker variant, `refine_sql()`, which reviews only the SQL text. It usually
approves a query that looks right but returns a negative total, which is why the execution output
matters.

## Project files

| File | What it is |
|---|---|
| `sql_reflection_agent.py` | The agent as a Python script, following the lab sections 2 → 3.4 |
| `M2_SQL_Generation_Reflection.ipynb` | The same lab as a notebook, cell by cell |
| `utils.py` | Builds `products.db`, reads its schema, runs SQL, displays results |
| `products.db` | Created on first run by `utils.create_transactions_db()` (not committed) |

### The database

A single SQLite table, `transactions`, where every row is a product **event**:

| action | qty_delta | unit_price |
|---|---|---|
| `insert` (initial stock) | + | launch price |
| `restock` | + | NULL |
| `sale` | **−** | price at that moment |
| `price_update` | 0 | new price |

About 20,000 events for 40 products over 2025 (fixed seed, so every run builds the same data;
pass `seed=None` for fresh random data). Because sales are stored as negative quantities, the
"obvious" `SUM(qty_delta * unit_price)` returns a negative total. That's the trap the reflection
step has to catch. With the default data:

- naive V1 → `Blue  -191062.90` (wrong: the largest negative number sorts last, so the *lowest* seller wins)
- correct  → `White  913782.10`

The database is opened **read-only** when running LLM-written SQL, so a bad query can't modify
or drop data. Errors come back as a one-row DataFrame that the reflection step can see.

## Setup

```powershell
cd C:\Users\SkySun\source\repos\AIAgentPractice\ReflectDesinPatternDemo
copy .env.example .env      # put your OPENAI_API_KEY in it (shared with ChartGenerationAgent)
cd SqlQueryImprover
pip install -r requirements.txt
```

## Run

**Script**

```powershell
python sql_reflection_agent.py                   # sections 3.1–3.2.2 step by step, then the 3.4 workflow
python sql_reflection_agent.py --mode steps      # only the step-by-step walkthrough
python sql_reflection_agent.py --mode workflow   # only run_sql_workflow()

# your own question, with a cheaper model for the first draft
python sql_reflection_agent.py --mode workflow `
  --question "Which category had the biggest revenue drop after a price update?" `
  --model-generation openai:gpt-4.1-mini `
  --model-evaluation openai:gpt-4.1
```

**Notebook:** open `M2_SQL_Generation_Reflection.ipynb` and run the cells top to bottom.

## Models

Models use aisuite's `provider:model` names. The lab suggests `openai:gpt-4o`, `openai:gpt-4.1`,
`openai:gpt-4.1-mini` and `openai:gpt-3.5-turbo`; `openai:gpt-4.1` usually reflects best.
LLMs are stochastic, so results vary between runs.

The `❌` / `✅` titles in the step-by-step section are the lab's narrative. They describe the
typical outcome, not a check the code performs.
