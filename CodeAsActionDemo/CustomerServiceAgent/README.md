# Customer Service Agent (Planning with Code Execution)

A customer service agent for a small sunglasses store. Instead of planning in JSON and calling
many small tools, the LLM **writes Python (TinyDB) code that is the plan**: commented steps that
filter, compute and update. The code is then executed. Based on the *M5 Agentic AI – Customer
Service Agent* lab ("code as action").

```
 "I want to buy 3 pairs of classic sunglasses and 1 pair of aviator"
        │
        ▼  generate_llm_code(): prompt = rules + live schema of both tables + request
 <execute_python>
 # 1) parse intent & quantities   # 2) query inventory_tbl   # 3) check stock for EVERY item
 # 4) per item: update stock, insert a transaction, balance += line total
 # 5) answer_text = "…", STATUS = "success"
 </execute_python>
        │
        ▼  execute_generated_code(): run it with db, tables and helpers; capture logs / errors
 answer_text  +  inventory / transactions before → after
```

## Project files

| File | What it is |
|---|---|
| `customer_service_agent.py` | The lab as a Python script (sections 2 → 4), plus `--ask` for your own requests |
| `M5_Customer_Service_Agent.ipynb` | The same lab as a notebook, cell by cell |
| `inv_utils.py` | The store: inventory + transactions in TinyDB (`store_db.json`), the schema block for the prompt, and the `get_current_balance` / `next_transaction_id` helpers |
| `utils.py` | `print_html()` display helper |

### The store

| item_id | name | shape / notes | stock | price |
|---|---|---|---|---|
| SG001 | Aviator | teardrop metal frame | 23 | $80 |
| SG002 | Wayfarer | trapezoid acetate | 14 | $95 |
| SG003 | Cat Eye | upswept corners | 8 | $110 |
| SG004 | Sport | *wraparound* (not "round") | 15 | $70 |
| SG005 | Classic | **round**, the only round frame under $100 | 10 | $60 |
| SG006 | Moon | **round**, oversized | 6 | $120 |
| SG007 | Clubmaster | browline; only 2 left | 2 | $130 |
| SG008 | Shield | out of stock | 0 | $150 |

The transactions table starts with one opening balance of $500 (TXN001).

## Setup

```powershell
cd C:\Users\SkySun\source\repos\AIAgentPractice\CodeAsActionDemo
copy .env.example .env        # put your OPENAI_API_KEY in it
C:\Users\SkySun\venv\Scripts\activate
cd CustomerServiceAgent
pip install -r requirements.txt   # already installed in your venv
```

## Run

```powershell
python customer_service_agent.py                 # every lab section, starting from a fresh store
python customer_service_agent.py --step round    # read-only: "round sunglasses under $100?" → Classic, $60
python customer_service_agent.py --step return   # return 2 Aviators → stock 23 → 25, refund transaction
python customer_service_agent.py --step agent    # buy 3 Classic + 1 Aviator → one transaction per item

# your own requests: the store keeps its state between runs, unless you pass --reseed
python customer_service_agent.py --ask "Purchase 3 Wayfarer sunglasses for customer Alice." --reseed
python customer_service_agent.py --ask "Return 2 Aviator sunglasses."
python customer_service_agent.py --ask "I want 3 Clubmaster sunglasses"      # only 2 left → insufficient_stock
python customer_service_agent.py --ask "Do you have cat eye frames under $100?" --model gpt-4.1-mini --temperature 0.2
```

Each run shows the request, the generated plan-as-code, the tables before and after, the
customer-facing `answer_text`, and the plan's `STATUS` (`success`, `no_match`,
`insufficient_stock`, `invalid_request` or `unsupported_intent`), with its logs and any error.

**Notebook:** open `M5_Customer_Service_Agent.ipynb` and run the cells top to bottom.

## Changes from the lab

- **One namespace for the generated code.** The lab runs `exec(code, SAFE_GLOBALS, SAFE_LOCALS)`.
  With two separate dicts, any function or lambda the model defines can't see the code's own
  top-level imports and variables, and fails with `NameError`. LLM-written plans do this often.
  The executor here runs the code in a single namespace. Everything else in the executor is as
  in the lab.
- **The executor also returns `STATUS`**, and the script prints the plan's logs and errors,
  which the lab returns but doesn't show.

## Things to know

- **Model settings:** `o4-mini` (the lab's model) only accepts `--temperature 1.0`. Use a lower
  temperature only with models like `gpt-4.1-mini`.
- **Not a security sandbox.** The namespace only controls what's handed to the code; the code
  can still import modules. That's fine for this demo store, but production systems run
  generated code in an isolated container with no network or file access.
- **Mutations are real.** The plan decides on its own whether a request changes data
  (buy / return) or only reads it. Use `--reseed` (or `inv_utils.seed_db()`) to reset the store.
