# Market Research Team (Multi-Agent Campaign Pipeline)

A small team of agents prepares a **summer sunglasses campaign**. They research current
trends, pick matching products from the catalog, design a visual, write a quote, and package
everything into an executive-ready report. Based on the *M5 Agentic AI – Market Research Team*
lab.

```
 🕵️ Market Research Agent ──► 🎨 Graphic Designer Agent ──► ✍️ Copywriter Agent ──► 📦 Packaging Agent
   tavily_search_tool            o4-mini: image prompt         o4-mini sees the image     o4-mini rewrites the
   product_catalog_tool          + caption (JSON)              + trend brief →            brief for a CEO →
   → trend brief + picks         gpt-image-1-mini → image      quote + justification      campaign_summary_*.md
```

## Project files

| File | What it is |
|---|---|
| `market_research_team.py` | The lab as a Python script: the four agents (section 4) and `run_sunglasses_campaign_pipeline()` (section 5) |
| `M5_Market_Research_Team.ipynb` | The same lab as a notebook, cell by cell |
| `tools.py` | `tavily_search_tool`, `product_catalog_tool`, their schemas, and the tool-call dispatcher |
| `utils.py` | Step logging: agent titles, tool calls, results, outputs (styled blocks in Jupyter, plain text in a terminal) |

The product catalog is the same sunglasses store as `CodeAsActionDemo/CustomerServiceAgent`:
8 styles from the $60 round Classic to the $150 Shield.

## Setup

You need two keys:

- `OPENAI_API_KEY`
- `TAVILY_API_KEY`, for live web search. There's a free tier at [tavily.com](https://tavily.com).

```powershell
cd C:\Users\SkySun\source\repos\AIAgentPractice\MultiAgentDemo
copy .env.example .env        # fill in OPENAI_API_KEY and TAVILY_API_KEY
C:\Users\SkySun\venv\Scripts\activate
cd MarketResearchTeam
pip install -r requirements.txt   # already installed in your venv
```

Without a Tavily key the pipeline still runs, but the research agent only has the catalog to
work from (no live trends).

## Run

```powershell
python market_research_team.py                       # full pipeline (section 5) → outputs/
python market_research_team.py --mode walkthrough    # sections 3–4: the tools, then each agent one at a time
python market_research_team.py --mode tools          # just try the two tools (no OpenAI calls)
```

The script saves `generated_image.png` and `campaign_summary_<date_time>.md` in `outputs/`.
Open the `.md` in VS Code's Markdown preview (Ctrl+Shift+V) to see the report with the image.
The notebook saves them next to itself, as in the lab.

**Notebook:** open `M5_Market_Research_Team.ipynb` and run the cells top to bottom. Note that it
runs the agents once in section 4 and again in section 5, which means two images and about twice the cost.

## Changes from the lab

Both changes are in `market_research_agent`.

- **Assistant message appended once per turn.** The lab appended the model's message inside
  the loop, once per tool call. When the model calls both tools in the same turn, which o4-mini
  often does, the conversation becomes `assistant → tool → assistant → tool`. OpenAI rejects
  that, because every `tool_call_id` must be answered right after the assistant message that
  made it. The message is now appended once, followed by all of its tool results.
- **Turn limit.** `while True` became at most `max_turns` (default 10) model calls, so an agent
  that keeps calling tools stops instead of running up a bill.

Everything else, including all prompts, models and the report format, is as in the lab.

## Things to know

- **Image model access:** OpenAI may ask you to verify your organization before your key can use the
  `gpt-image-1` models. If the Graphic Designer Agent fails with a 403, that's the reason
  ([organization settings](https://platform.openai.com/settings/organization/general)).
- **Cost per pipeline run:** about 5–8 `o4-mini` calls, one `gpt-image-1-mini` image
  (medium, 1024×1024), and 1–3 Tavily searches.
- **The research agent decides how to search.** The queries, the number of searches and the
  products it picks vary from run to run.
