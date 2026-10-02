"""
M5 Agentic AI - Market Research Team (multi-agent campaign pipeline)

A small "team" of agents prepares a summer sunglasses campaign:
  1. 🕵️ Market Research Agent  — web trends (Tavily) + internal catalog → trend brief & product picks
  2. 🎨 Graphic Designer Agent — image prompt + caption (o4-mini) → campaign image (gpt-image-1-mini)
  3. ✍️ Copywriter Agent       — image + brief (multimodal) → short campaign quote + justification
  4. 📦 Packaging Agent        — executive-ready Markdown report

Usage:
    python market_research_team.py                      # full pipeline (section 5) → outputs/
    python market_research_team.py --mode walkthrough   # sections 3–4: tools, then each agent one at a time
    python market_research_team.py --mode tools         # section 3: just try the two tools
"""

# =========================
# Imports
# =========================

# --- Standard library ---
import base64
import json
import os
import re
from datetime import datetime
from io import BytesIO

# --- Third-party ---
import requests
import openai
from PIL import Image
from dotenv import load_dotenv
from IPython.display import Markdown, display
import aisuite

# --- Local / project ---
import tools
import utils

# --- for the command line runner ---
import argparse
import sys
from pathlib import Path


# =========================
# Environment & Client
# =========================
load_dotenv()
client = aisuite.Client()


# ======================================================================================
# 4.1. Market Research Agent
# ======================================================================================
def market_research_agent(return_messages: bool = False, max_turns: int = 10):

    utils.log_agent_title_html("Market Research Agent", "🕵️‍♂️")

    prompt_ = f"""
You are a fashion market research agent tasked with preparing a trend analysis for a summer sunglasses campaign.

Your goal:
1. Explore current fashion trends related to sunglasses using web search.
2. Review the internal product catalog to identify items that align with those trends.
3. Recommend one or more products from the catalog that best match emerging trends.
4. If needed, today date is {datetime.now().strftime("%Y-%m-%d")}.

You can call the following tools:
- tavily_search_tool: to discover external web trends.
- product_catalog_tool: to inspect the internal sunglasses catalog.

Once your analysis is complete, summarize:
- The top 2–3 trends you found.
- The product(s) from the catalog that fit these trends.
- A justification of why they are a good fit for the summer campaign.
"""
    messages = [{"role": "user", "content": prompt_}]
    tools_ = tools.get_available_tools()

    # Change from the lab: `while True` → at most `max_turns` model calls, so a model that
    # keeps calling tools can't loop (and bill) forever.
    for _ in range(max_turns):
        response = client.chat.completions.create(
            model="openai:o4-mini",
            messages=messages,
            tools=tools_,
            tool_choice="auto"
        )

        msg = response.choices[0].message

        if msg.content:
            utils.log_final_summary_html(msg.content)
            return (msg.content, messages) if return_messages else msg.content

        if msg.tool_calls:
            # Change from the lab: the assistant message is appended ONCE, before its tool results.
            # The lab appended it inside the loop, once per tool call; when the model calls both
            # tools in one turn, OpenAI rejects that conversation (each tool_call_id must be answered
            # right after the assistant message that made the calls).
            messages.append(msg)
            for tool_call in msg.tool_calls:
                utils.log_tool_call_html(tool_call.function.name, tool_call.function.arguments)
                result = tools.handle_tool_call(tool_call)
                utils.log_tool_result_html(result)

                messages.append(tools.create_tool_response_message(tool_call, result))
        else:
            utils.log_unexpected_html()
            return ("[⚠️ Unexpected: No tool_calls or content returned]", messages) if return_messages else "[⚠️ Unexpected: No tool_calls or content returned]"

    utils.log_unexpected_html()
    warning = f"[⚠️ Stopped after {max_turns} turns without a final summary]"
    return (warning, messages) if return_messages else warning


# ======================================================================================
# 4.2. Graphic Designer Agent
# ======================================================================================
def graphic_designer_agent(trend_insights: str, caption_style: str = "short punchy", size: str = "1024x1024") -> dict:
    utils.log_agent_title_html("Graphic Designer Agent", "🎨")

    # Step 1: Generate prompt and caption using aisuite
    system_message = (
        "You are a visual marketing assistant. Based on the input trend insights, "
        "write a creative and visual prompt for an AI image generation model, and also a short caption."
    )
    user_prompt = f"""
Trend insights:
{trend_insights}
Please output:
1. A vivid, descriptive prompt to guide image generation.
2. A marketing caption in style: {caption_style}.
Respond in this format:
{{"prompt": "...", "caption": "..."}}
"""
    chat_response = client.chat.completions.create(
        model="openai:o4-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]
    )
    content = chat_response.choices[0].message.content.strip()
    match = re.search(r'\{.*\}', content, re.DOTALL)
    parsed = json.loads(match.group(0)) if match else {"error": "No JSON returned", "raw": content}
    prompt = parsed["prompt"]
    caption = parsed["caption"]

    # Step 2: Generate image with gpt-image-1-mini (returns base64, not URL)
    openai_client = openai.OpenAI()
    image_response = openai_client.images.generate(
        model="gpt-image-1-mini",
        prompt=prompt,
        size=size,
        quality="medium",   # gpt-image-1 accepts: low | medium | high | auto
        n=1,
        # no response_format — gpt-image-1 always returns b64_json
    )

    b64 = image_response.data[0].b64_json
    img_bytes = base64.b64decode(b64)
    img = Image.open(BytesIO(img_bytes))

    image_path = "generated_image.png"
    img.save(image_path)

    utils.log_final_summary_html(f"""
        <h3>Generated Image and Caption</h3>
        <p><strong>Image Path:</strong> <code>{image_path}</code></p>
        <p><strong>Generated Image:</strong></p>
        <img src="{image_path}" alt="Generated Image" style="max-width: 100%; height: auto; border: 1px solid #ccc; border-radius: 8px; margin-top: 10px; margin-bottom: 10px;">
        <p><strong>Prompt:</strong> {prompt}</p>
    """)

    return {
        "image_path": image_path,
        "prompt": prompt,
        "caption": caption,
    }


# ======================================================================================
# 4.3. Copywriter Agent
# ======================================================================================
def copywriter_agent(image_path: str, trend_summary: str, model: str = "openai:o4-mini") -> dict:

    """
    Uses aisuite (OpenAI only) to send an image and a trend summary and return a campaign quote.

    Args:
        image_path (str): Path of the local image to be analyzed.
        trend_summary (str): Text from the researcher agent.
        model (str): OpenAI model (e.g., openai:o4-mini, openai:gpt-4o)

    Returns:
        dict: {
            "quote": "...",
            "justification": "...",
            "image_path": "..."
        }
    """

    utils.log_agent_title_html("Copywriter Agent", "✍️")

    # Step 1: Load local image and encode as base64
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    b64_img = base64.b64encode(img_bytes).decode("utf-8")

    # Step 2: Build OpenAI-compliant multimodal message
    messages = [
        {
            "role": "system",
            "content": "You are a copywriter that creates elegant campaign quotes based on an image and a marketing trend summary."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64_img}",
                        "detail": "auto"
                    }
                },
                {
                    "type": "text",
                    "text": f"""
Here is a visual marketing image and a trend analysis:

Trend summary:
\"\"\"{trend_summary}\"\"\"

Please return a JSON object like:
{{
  "quote": "A short, elegant campaign phrase (max 12 words)",
  "justification": "Why this quote matches the image and trend"
}}"""
                }
            ]
        }
    ]

    # Step 3: Send request via aisuite
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    # Step 4: Parse JSON response
    content = response.choices[0].message.content.strip()

    utils.log_final_summary_html(content)

    try:
        match = re.search(r'\{.*\}', content, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {"error": "No valid JSON returned"}
    except Exception as e:
        parsed = {"error": f"Failed to parse: {e}", "raw": content}


    parsed["image_path"] = image_path
    return parsed


# ======================================================================================
# 4.4. Packaging Agent
# ======================================================================================
def packaging_agent(trend_summary: str, image_url: str, quote: str, justification: str, output_path: str = "campaign_summary.md") -> str:

    """
    Packages the campaign assets into a beautifully formatted markdown report for executive review.

    Args:
        trend_summary (str): Summary of the market trends.
        image_url (str): URL of the campaign image.
        quote (str): Marketing quote to overlay.
        justification (str): Explanation for the quote.
        output_path (str): Path to save the markdown report.

    Returns:
        str: Path to the saved markdown file.
    """

    utils.log_agent_title_html("Packaging Agent", "📦")

    # We use this path in the src of the <img>
    styled_image_html = f"""
![Open the generated file to see]({image_url})
    """

    beautified_summary = client.chat.completions.create(
        model="openai:o4-mini",
        messages=[
            {"role": "system", "content": "You are a marketing communication expert writing elegant campaign summaries for executives."},
            {"role": "user", "content": f"""
Please rewrite the following trend summary to be clear, professional, and engaging for a CEO audience:

\"\"\"{trend_summary.strip()}\"\"\"
"""}
        ]
    ).choices[0].message.content.strip()

    utils.log_tool_result_html(beautified_summary)

    # Combine all parts into markdown
    markdown_content = f"""# 🕶️ Summer Sunglasses Campaign – Executive Summary

## 📊 Refined Trend Insights
{beautified_summary}

## 🎯 Campaign Visual
{styled_image_html}

## ✍️ Campaign Quote
{quote.strip()}

## ✅ Why This Works
{justification.strip()}

---

*Report generated on {datetime.now().strftime('%Y-%m-%d')}*
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    return output_path


# ======================================================================================
# 5. Full Campaign Pipeline – run_sunglasses_campaign_pipeline
# ======================================================================================
def run_sunglasses_campaign_pipeline(output_path: str = "campaign_summary.md") -> dict:
    """
    Runs the full summer sunglasses campaign pipeline:
    1. Market research (search trends + match products)
    2. Generate visual + caption
    3. Generate quote based on image + trend
    4. Create executive markdown report

    Returns:
        dict: Dictionary containing all intermediate results + path to final report
    """
    # 1. Run market research agent
    trend_summary = market_research_agent()
    print("✅ Market research completed")

    # 2. Generate image + caption
    visual_result = graphic_designer_agent(trend_insights=trend_summary)
    image_path = visual_result["image_path"]
    print("🖼️ Image generated")

    # 3. Generate quote based on image + trends
    quote_result = copywriter_agent(image_path=image_path, trend_summary=trend_summary)
    quote = quote_result.get("quote", "")
    justification = quote_result.get("justification", "")
    print("💬 Quote created")

    # 4. Generate markdown report
    md_path = packaging_agent(
        trend_summary=trend_summary,
        image_url=image_path,
        quote=quote,
        justification=justification,
        output_path=f"campaign_summary_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.md"
    )

    print(f"📦 Report generated: {md_path}")

    return {
        "trend_summary": trend_summary,
        "visual": visual_result,
        "quote": quote_result,
        "markdown_path": md_path
    }


# ======================================================================================
# Command line runner
# ======================================================================================
def show_report(md_path: str) -> None:
    """Render the Markdown report (Jupyter) or print it (terminal)."""
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
    if utils._in_notebook():
        display(Markdown(md_content))
    else:
        print("\n" + md_content)


def run_tools_demo() -> None:
    """Section 3: try the two tools directly (no LLM involved)."""
    utils.log_agent_title_html("Tool: tavily_search_tool('trends in sunglasses fashion')", "🔎")
    utils.log_tool_result_html(tools.tavily_search_tool('trends in sunglasses fashion'))
    utils.log_agent_title_html("Tool: product_catalog_tool()", "🗂️")
    utils.log_tool_result_html(tools.product_catalog_tool())


def run_walkthrough() -> dict:
    """Sections 3–4: the tools, then each agent one at a time, then the report."""
    run_tools_demo()

    market_research_result = market_research_agent()

    graphic_designer_agent_result = graphic_designer_agent(
        trend_insights=market_research_result,
    )

    copywriter_agent_result = copywriter_agent(
        image_path=graphic_designer_agent_result["image_path"],
        trend_summary=market_research_result,
    )

    packaging_agent_result = packaging_agent(
        trend_summary=market_research_result,
        image_url=graphic_designer_agent_result["image_path"],
        quote=copywriter_agent_result["quote"],
        justification=copywriter_agent_result["justification"],
        output_path=f"campaign_summary_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.md"
    )
    show_report(packaging_agent_result)
    return {"markdown_path": packaging_agent_result}


def main():
    parser = argparse.ArgumentParser(description="Multi-agent summer sunglasses campaign pipeline")
    parser.add_argument("--mode", choices=["pipeline", "walkthrough", "tools"], default="pipeline",
                        help="pipeline = section 5 in one call; walkthrough = sections 3–4 step by step; "
                             "tools = only try the two tools")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "outputs"),
                        help="where the image and the Markdown reports are saved")
    args = parser.parse_args()

    if not os.getenv("TAVILY_API_KEY"):
        print("⚠️  TAVILY_API_KEY is not set: the research agent will work from the catalog only "
              "(no live web trends). Add it to MultiAgentDemo/.env.")
    if args.mode != "tools" and not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Add it to the .env file in the MultiAgentDemo folder "
                 "(see .env.example).")

    # The image and report are written to the current folder (as in the lab); keep them together
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(args.output_dir)

    if args.mode == "tools":
        run_tools_demo()
    elif args.mode == "walkthrough":
        run_walkthrough()
    else:
        results = run_sunglasses_campaign_pipeline()
        show_report(results["markdown_path"])
        print(f"\nOutputs in {Path.cwd()}")


if __name__ == "__main__":
    main()
