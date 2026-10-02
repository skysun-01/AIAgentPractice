# Eval: component-level evals for the AIAgentPractice projects

Based on *M4 Agentic AI – Adding a component-level eval to the research workflow*. That lab
checked one component of a pipeline (web search) with an **objective, per-example** test that
returns a PASS/FAIL flag and a Markdown summary. This folder applies the same idea to every
project built so far.

| Eval | Project | Component tested | Ground truth per example | Score (default threshold) |
|---|---|---|---|---|
| `sql_eval.py` | ReflectDesinPatternDemo/SqlQueryImprover | `generate_sql` (V1) and `refine_sql_external_feedback` (V2) | hand-written correct SQL query | V2 execution accuracy (80%) |
| `chart_eval.py` | ReflectDesinPatternDemo/ChartGenerationAgent | `generate_chart_code` (V1) and `reflect_on_image_and_regenerate` (V2) | expected numbers computed from `coffee_sales.csv` | V2 charts passing every check (60%) |
| `tool_selection_eval.py` | ToolUseDemo/FunctionsToTools | the LLM's tool choice and arguments | expected tools, call order, argument rules | examples fully correct (80%) |
| `email_agent_eval.py` | ToolUseDemo/EmailTool | the email agent (LLM + 10 tools) | expected change to the inbox | requests handled correctly (75%) |

Each file has an `EVAL_SET`, an `evaluate_*` function for one example (like the lab's
`evaluate_tavily_results`) and a `run_eval(..., min_ratio=...)` that scores the whole set.
No LLM-as-judge is involved; everything is checked in code.

## What each eval checks

**SQL: execution accuracy.** Ten questions, each with a correct SQL query. The agent's SQL is
run on the same database and its result compared with the ground truth. Column names and order,
row order, extra columns and float rounding don't matter; the answer does. Several questions
contain the traps from the M2 lab:

- sales are stored as negative `qty_delta`
- one product's latest event is a restock with a NULL price
- stock levels must include the initial insert

V1 and V2 are scored separately, and the report shows how many queries reflection **fixed**
and how many it **broke**.

**Charts: do the charts show the right numbers?** The generated code is executed, the figure
is captured when it's saved, and the numbers it actually draws are compared with numbers
computed from the CSV: bar heights, line points, pie slices and value labels. The chart passes
if at least 90% of the expected values are found. Ambiguous requests ("Q1 coffee sales":
revenue? number of sales? per coffee?) accept every reasonable reading. A chart also needs to
run, be saved to the requested path, and have a title and axis labels, without `plt.show()` or
seaborn. If V1 produced no image, V2 counts as failed, because reflection needs a chart to
look at.

**Tools: right tool, right arguments, right order.** The four project tools are swapped for
recording fakes. The fakes keep the same name, signature and docstring, so the LLM sees exactly
the same tools, but nothing is written and no network calls are made. The fakes return fixed
values, so the eval can check that results flow into the next call (e.g. the weather ends up
in the note). One prompt needs **no** tool; calling one anyway fails.

**Email: the right inbox change, and nothing else.** For each request the sample inbox is
reset, the agent runs, and the inbox before and after is compared. Wrong deletions, stray
replies and unrelated emails marked as read all count as failures. This eval uses its own
inbox copy (`results/emails_eval.db` on port 8766), so your EmailTool inbox is never touched.

## Setup

The evals use the API key you already set up: the `.env` in `ReflectDesinPatternDemo` or
`ToolUseDemo` is picked up automatically. You can also create `Eval/.env` from
`.env.example`. Everything is already installed in your venv; otherwise run
`pip install -r requirements.txt`.

```powershell
C:\Users\SkySun\venv\Scripts\activate
cd C:\Users\SkySun\source\repos\AIAgentPractice\Eval
```

## Run

```powershell
python run_all_evals.py                    # all four evals + one summary table
python run_all_evals.py --limit 2          # first 2 examples of each: quick, cheap smoke run
python run_all_evals.py --only sql email   # just some of them

python sql_eval.py --model-generation openai:gpt-4.1-mini     # compare models
python chart_eval.py --reflection-model claude-sonnet-5-5     # needs ANTHROPIC_API_KEY
python tool_selection_eval.py --model openai:gpt-4.1 --min-ratio 0.9
python email_agent_eval.py --limit 3
```

Or open **`M4_Component_Evals.ipynb`**, which walks through each eval with its eval set shown as a table.

Every run saves a Markdown report and the raw per-example results (JSON) to `results/`,
with a timestamp, so you can compare runs as you change prompts, models or tool docstrings.
Generated charts go to `results/charts/`.

## Cost

A full run makes about 60 model calls:

| Eval | Calls |
|---|---|
| SQL | 10 questions × 2 |
| Charts | 5 requests × 2, including image input |
| Tools | 8 prompts × 1–3 |
| Email | 8 requests × 2–4 |

Use `--limit` while experimenting.

## Growing the evals

Add examples to an `EVAL_SET`, for example a SQL question with its correct query, or an email
request with the inbox change you expect. About 10–20 examples per component give a much more
reliable signal than a handful. Note that model outputs vary from run to run, so a score that
moves by one example isn't necessarily a real change.
