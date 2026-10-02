"""
M2 - Agentic AI - Improving SQL Generation with Reflection

Workflow:
  1. Extract the database schema.
  2. Generate an initial SQL query (V1) from a natural-language question.
  3. Execute V1.
  4. Reflect on V1 using its real execution output (external feedback) -> refined SQL (V2).
  5. Execute V2 -> final answer.

Usage:
    python sql_reflection_agent.py                    # section 3 step by step, then the 3.4 workflow
    python sql_reflection_agent.py --mode steps       # only the step-by-step walkthrough (3.1 - 3.2.2)
    python sql_reflection_agent.py --mode workflow    # only run_sql_workflow()
    python sql_reflection_agent.py --mode workflow --question "Which brand had the most restocks in 2025?"
"""

# ======================================================================================
# 2. Setup: Initialize Environment and Client
# ======================================================================================
import argparse
import json
import os
from pathlib import Path
import Path
import utils
import pandas as pd
from dotenv import load_dotenv

_ = load_dotenv()

# --------------------------------------------------------------------------------------
# 2.1 Getting started with AISuite
# --------------------------------------------------------------------------------------
import aisuite as ai

client = ai.Client()

QUESTION = "Which color of product has the highest total sales?"


# ======================================================================================
# 3.1. Use an LLM to Query a Database
# ======================================================================================
def generate_sql(question: str, schema: str, model: str) -> str:
    prompt = f"""
    You are a SQL assistant. Given the schema and the user's question, write a SQL query for SQLite.

    Schema:
    {schema}

    User question:
    {question}

    Respond with the SQL only.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


# ======================================================================================
# 3.2.1. First Attempt: Refine a SQL query (reviews the SQL text only)
# ======================================================================================
def refine_sql(
    question: str,
    sql_query: str,
    schema: str,
    model: str,
) -> tuple[str, str]:
    """
    Reflect on whether a query's *shown output* answers the question,
    and propose an improved SQL if needed.
    Returns (feedback, refined_sql).
    """
    prompt = f"""
You are a SQL reviewer and refiner.

User asked:
{question}

Original SQL:
{sql_query}

Table Schema:
{schema}

Step 1: Briefly evaluate if the SQL OUTPUT fully answers the user's question.
Step 2: If improvement is needed, provide a refined SQL query for SQLite.
If the original SQL is already correct, return it unchanged.

Return STRICT JSON with two fields:
{{
  "feedback": "<1-3 sentences explaining the gap or confirming correctness>",
  "refined_sql": "<final SQL to run>"
}}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    content = response.choices[0].message.content
    try:
        obj = json.loads(content)
        feedback = str(obj.get("feedback", "")).strip()
        refined_sql = str(obj.get("refined_sql", sql_query)).strip()
        if not refined_sql:
            refined_sql = sql_query
    except Exception:
        # Fallback if model doesn't return valid JSON
        feedback = content.strip()
        refined_sql = sql_query

    return feedback, refined_sql


# ======================================================================================
# 3.2.2. Final Approach: Refine an SQL Query with External Feedback (the real output)
# ======================================================================================
def refine_sql_external_feedback(
    question: str,
    sql_query: str,
    df_feedback: pd.DataFrame,
    schema: str,
    model: str,
) -> tuple[str, str]:
    """
    Evaluate whether the SQL result answers the user's question and,
    if necessary, propose a refined version of the query.
    Returns (feedback, refined_sql).
    """
    prompt = f"""
    You are a SQL reviewer and refiner.

    User asked:
    {question}

    Original SQL:
    {sql_query}

    SQL Output:
    {df_feedback.to_markdown(index=False)}

    Table Schema:
    {schema}

    Step 1: Briefly evaluate if the SQL output answers the user's question.
    Step 2: If the SQL could be improved, provide a refined SQL query.
    If the original SQL is already correct, return it unchanged.

    Return a strict JSON object with two fields:
    - "feedback": brief evaluation and suggestions
    - "refined_sql": the final SQL to run
    """

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    )


    content = response.choices[0].message.content
    try:
        obj = json.loads(content)
        feedback = str(obj.get("feedback", "")).strip()
        refined_sql = str(obj.get("refined_sql", sql_query)).strip()
        if not refined_sql:
            refined_sql = sql_query
    except Exception:
        # Fallback if the model does not return valid JSON:
        # use the raw content as feedback and keep the original SQL
        feedback = content.strip()
        refined_sql = sql_query

    return feedback, refined_sql


# ======================================================================================
# Sections 2.2 → 3.2.2, one step at a time (as in the lab walkthrough)
# ======================================================================================
def run_step_by_step(question: str = QUESTION, model: str = "openai:gpt-4.1"):
    # --- 2.2. Set Up the Database ---
    utils.create_transactions_db()
    utils.print_html(utils.get_schema('products.db'))

    # --- 3.1. Use an LLM to Query a Database ---
    # We provide the schema as a string
    schema = """
Table name: transactions
id (INTEGER)
product_id (INTEGER)
product_name (TEXT)
brand (TEXT)
category (TEXT)
color (TEXT)
action (TEXT)
qty_delta (INTEGER)
unit_price (REAL)
notes (TEXT)
ts (DATETIME)
"""

    utils.print_html(question, title="User Question")

    # Generate the SQL query using the specified model
    sql_V1 = generate_sql(question, schema, model=model)

    # Display the generated SQL query
    utils.print_html(sql_V1, title="SQL Query V1")

    # --- 3.1.1. Query Execution ---
    # Execute the generated SQL query (sql_V1) against the products.db database.
    # The result is returned as a pandas DataFrame.
    df_sql_V1 = utils.execute_sql(sql_V1, db_path='products.db')

    # Render the DataFrame as a table.
    utils.print_html(df_sql_V1, title="Output of SQL Query V1 - ❌ Does NOT fully answer the question")

    # --- 3.2.1. First Attempt: Refine a SQL query (V1 → V2) ---
    feedback, sql_V2 = refine_sql(
        question=question,
        sql_query=sql_V1,   # <- comes from generate_sql() (V1)
        schema=schema, # <- we reuse the schema from section 3.1
        model=model
    )

    # Display the original prompt
    utils.print_html(question, title="User Question")

    # --- V1 ---
    utils.print_html(sql_V1, title="Generated SQL Query (V1)")

    # Execute and show V1 output
    df_sql_V1 = utils.execute_sql(sql_V1, db_path='products.db')
    utils.print_html(df_sql_V1, title="SQL Output of V1 - ❌ Does NOT fully answer the question")

    # --- Feedback + V2 ---
    utils.print_html(feedback, title="Feedback on V1")
    utils.print_html(sql_V2, title="Refined SQL Query (V2)")

    # Execute and show V2 output
    df_sql_V2 = utils.execute_sql(sql_V2, db_path='products.db')
    utils.print_html(df_sql_V2, title="SQL Output of V2 - ❌ Does NOT fully answer the question")

    # --- 3.2.2. Refine SQL with External Feedback (V1 → V2) ---
    # Execute the original SQL (V1)
    df_sql_V1 = utils.execute_sql(sql_V1, db_path='products.db')

    # Use external feedback to evaluate and refine
    feedback, sql_V2 = refine_sql_external_feedback(
        question=question,
        sql_query=sql_V1,   # V1 query
        df_feedback=df_sql_V1,    # Output of V1
        schema=schema,
        model=model
    )

    # --- V1 ---
    utils.print_html(question, title="User Question")
    utils.print_html(sql_V1, title="Generated SQL Query (V1)")
    utils.print_html(df_sql_V1, title="SQL Output of V1 - ❌ Does NOT fully answer the question")

    # --- Feedback & V2 ---
    utils.print_html(feedback, title="Feedback on V1")
    utils.print_html(sql_V2, title="Refined SQL Query (V2)")

    # Execute and display V2 results
    df_sql_V2 = utils.execute_sql(sql_V2, db_path='products.db')
    utils.print_html(df_sql_V2, title="SQL Output of V2 (with External Feedback) - ✅ Fully answers the question")


# ======================================================================================
# 3.3. Putting it all together — Building the Database Query Workflow
# ======================================================================================
def run_sql_workflow(
    db_path: str,
    question: str,
    model_generation: str = "openai:gpt-4.1",
    model_evaluation: str = "openai:gpt-4.1",
):
    """
    End-to-end workflow to generate, execute, evaluate, and refine SQL queries.

    Steps:
      1) Extract database schema
      2) Generate SQL (V1)
      3) Execute V1 → show output
      4) Reflect on V1 with execution feedback → propose refined SQL (V2)
      5) Execute V2 → show final answer
    """

    # 1) Schema
    schema = utils.get_schema(db_path)
    utils.print_html(
        schema,
        title="📘 Step 1 — Extract Database Schema"
    )

    # 2) Generate SQL (V1)
    sql_v1 = generate_sql(question, schema, model_generation)
    utils.print_html(
        sql_v1,
        title="🧠 Step 2 — Generate SQL (V1)"
    )

    # 3) Execute V1
    df_v1 = utils.execute_sql(sql_v1, db_path)
    utils.print_html(
        df_v1,
        title="🧪 Step 3 — Execute V1 (SQL Output)"
    )

    # 4) Reflect on V1 with execution feedback → refine to V2
    feedback, sql_v2 = refine_sql_external_feedback(
        question=question,
        sql_query=sql_v1,
        df_feedback=df_v1,          # external feedback: real output of V1
        schema=schema,
        model=model_evaluation,
    )
    utils.print_html(
        feedback,
        title="🧭 Step 4 — Reflect on V1 (Feedback)"
    )
    utils.print_html(
        sql_v2,
        title="🔁 Step 4 — Refined SQL (V2)"
    )

    # 5) Execute V2
    df_v2 = utils.execute_sql(sql_v2, db_path)
    utils.print_html(
        df_v2,
        title="✅ Step 5 — Execute V2 (Final Answer)"
    )


# ======================================================================================
# 3.4. Run the SQL Workflow
# ======================================================================================
def main():
    parser = argparse.ArgumentParser(description="Reflection design pattern demo: SQL query improver")
    parser.add_argument("--mode", choices=["steps", "workflow", "all"], default="all",
                        help="steps = sections 3.1-3.2.2, workflow = run_sql_workflow(), all = both")
    parser.add_argument("--question", default=QUESTION, help="natural-language question about the data")
    parser.add_argument("--model-generation", default="openai:gpt-4.1",
                        help="aisuite 'provider:model' that writes V1 (e.g. openai:gpt-4o, openai:gpt-4.1-mini)")
    parser.add_argument("--model-evaluation", default="openai:gpt-4.1",
                        help="aisuite 'provider:model' that reflects on V1 and writes V2")
    parser.add_argument("--db", default="products.db")
    args = parser.parse_args()

    # Run from the project folder so products.db is created/read next to this script
    os.chdir(Path(__file__).resolve().parent)
    utils.check_api_keys(args.model_generation, args.model_evaluation)

    if args.mode in ("steps", "all"):
        print("\n### Section 3: Building the SQL generator and reflection step by step ###")
        run_step_by_step(args.question, args.model_generation)

    if args.mode in ("workflow", "all"):
        print("\n### Section 3.4: Running the end-to-end SQL workflow ###")
        if not Path(args.db).exists():
            utils.create_transactions_db(args.db)
        run_sql_workflow(
            args.db,
            args.question,
            model_generation=args.model_generation,
            model_evaluation=args.model_evaluation,
        )


if __name__ == "__main__":
    main()
