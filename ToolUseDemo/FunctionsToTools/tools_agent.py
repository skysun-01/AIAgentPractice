"""
M3 Agentic AI - Turning functions into tools

Give an LLM controlled access to Python functions through aisuite:
  - 3.x  one tool (get_current_time), run automatically by aisuite (max_turns) and manually
  - 4.x  four tools; the LLM picks which one(s) to call, with which arguments, in which order

Usage:
    python tools_agent.py                          # run every lab section in order (3.1 → 4.3)
    python tools_agent.py --step weather --step qr # run only some sections
    python tools_agent.py --ask "Save the current time and weather to today.txt"
    python tools_agent.py --step multi --model openai:gpt-4.1

Steps: first-tool (3.1-3.3), manual (3.4), weather / note / qr (4.2), multi (4.3)
"""

# ======================================================================================
# 2. Setup: Initialize environment and client
# ======================================================================================
import argparse
import json
import os
import sys
from pathlib import Path

import display_functions
from dotenv import load_dotenv
_ = load_dotenv()

# --------------------------------------------------------------------------------------
# 2.1 Getting started with AISuite
# --------------------------------------------------------------------------------------
import aisuite as ai

# Create an instance of the AISuite client
client = ai.Client()


# ======================================================================================
# 3.1 Defining your function
# ======================================================================================
from datetime import datetime

def get_current_time():
    """
    Returns the current time as a string.
    """
    return datetime.now().strftime("%H:%M:%S")


# ======================================================================================
# 4.1 Three new tools
# ======================================================================================
import requests
import qrcode
from qrcode.image.styledpil import StyledPilImage


def get_weather_from_ip():
    """
    Gets the current, high, and low temperature in Fahrenheit for the user's
    location and returns it to the user.
    """
    # Get location coordinates from the IP address
    lat, lon = requests.get('https://ipinfo.io/json').json()['loc'].split(',')

    # Set parameters for the weather API call
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m",
        "daily": "temperature_2m_max,temperature_2m_min",
        "temperature_unit": "fahrenheit",
        "timezone": "auto"
    }

    # Get weather data
    weather_data = requests.get("https://api.open-meteo.com/v1/forecast", params=params).json()

    # Format and return the simplified string
    return (
        f"Current: {weather_data['current']['temperature_2m']}°F, "
        f"High: {weather_data['daily']['temperature_2m_max'][0]}°F, "
        f"Low: {weather_data['daily']['temperature_2m_min'][0]}°F"
    )

# Write a text file
def write_txt_file(file_path: str, content: str):
    """
    Write a string into a .txt file (overwrites if exists).
    Args:
        file_path (str): Destination path.
        content (str): Text to write.
    Returns:
        str: Path to the written file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return file_path


# Create a QR code
def generate_qr_code(data: str, filename: str, image_path: str):
    """Generate a QR code image given data and an image path.

    Args:
        data: Text or URL to encode
        filename: Name for the output PNG file (without extension)
        image_path: Path to the image to be used in the QR code
    """
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H)
    qr.add_data(data)

    img = qr.make_image(image_factory=StyledPilImage, embedded_image_path=image_path)
    output_file = f"{filename}.png"
    img.save(output_file)

    return f"QR code saved as {output_file} containing: {data[:50]}..."


ALL_TOOLS = [get_current_time, get_weather_from_ip, write_txt_file, generate_qr_code]


# ======================================================================================
# 3.1 – 3.3  Your first tool, executed automatically by aisuite
# ======================================================================================
def run_first_tool(model: str = "openai:gpt-4o"):
    # Test out your function to see what exactly it returns
    print("get_current_time() ->", get_current_time())

    # 3.2 Turning your function into an LLM tool
    # Message structure
    prompt = "What time is it?"
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[get_current_time],
        max_turns=5
    )

    # See the LLM response
    print(response.choices[0].message.content)

    # 3.3 Taking a closer look at the response
    display_functions.pretty_print_chat_completion(response)


# ======================================================================================
# 3.4 Manually defining tools (you execute the tool call yourself)
# ======================================================================================
def run_manual_tools(model: str = "openai:gpt-4o"):
    prompt = "What time is it?"
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    tools = [{
        "type": "function",
        "function": {
            "name": "get_current_time", # <--- Your functions name
            "description": "Returns the current time as a string.", # <--- a description for the LLM
            "parameters": {}
        }
    }]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools, # <-- Your list of tools with get_current_time
        # max_turns=5 # <-- When defining tools manually, you must handle calls yourself and cannot use max_turns
    )

    # Look at the raw response: notice tool_calls under message
    print(json.dumps(response.model_dump(), indent=2, default=str))

    response2 = None

    # Create a condition in case tool_calls is in response object
    if response.choices[0].message.tool_calls:
        # Pull out the specific tool metadata from the response
        tool_call = response.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)

        # Run the tool locally
        tool_result = get_current_time()

        # Append the result to the messages list
        messages.append(response.choices[0].message)
        messages.append({
            "role": "tool", "tool_call_id": tool_call.id, "content": str(tool_result)
        })

        # Send the list of messages with the newly appended results back to the LLM
        response2 = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
        )

        print(response2.choices[0].message.content)


# ======================================================================================
# 4.2 Using your new tools  /  4.3 Using multiple tools at once
# ======================================================================================
def ask(prompt: str, model: str = "openai:o4-mini", max_turns: int = 5):
    """Send a prompt with all four tools; aisuite runs whatever tools the LLM asks for."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": (
            prompt
        )}],
        tools=[
            get_current_time,
            get_weather_from_ip,
            write_txt_file,
            generate_qr_code
        ],
        max_turns=max_turns
    )

    display_functions.pretty_print_chat_completion(response)
    return response


def run_weather(model: str = "openai:o4-mini"):
    ask("Can you get the weather for my location?", model)


def run_note(model: str = "openai:o4-mini"):
    ask("Can you make a txt note for me called reminders.txt that reminds me to call Daniel tomorrow at 7PM?", model)

    with open('reminders.txt', 'r') as file:
        contents = file.read()
        print("\nreminders.txt contains:\n" + contents)


def run_qr(model: str = "openai:o4-mini"):
    ask("Can you make a QR code for me using my company's logo that goes to www.deeplearning.ai? "
        "The logo is located at `dl_logo.jpg`. You can call it dl_qr_code.", model)

    # Display image directly
    display_functions.show_image('dl_qr_code.png')


def run_multi(model: str = "openai:o4-mini"):
    ask("Can you help me create a qr code that goes to www.deeplearning.com from the image dl_logo.jpg? "
        "Also write me a txt note with the current weather please.", model, max_turns=10)


STEPS = {
    "first-tool": ("3.1–3.3 Your first tool (automatic execution)", run_first_tool),
    "manual": ("3.4 Manually defining tools", run_manual_tools),
    "weather": ("4.2 Weather tool", run_weather),
    "note": ("4.2 File writing tool", run_note),
    "qr": ("4.2 QR code tool", run_qr),
    "multi": ("4.3 Using multiple tools at once", run_multi),
}


def main():
    parser = argparse.ArgumentParser(description="Tool-calling agent: turning Python functions into LLM tools")
    parser.add_argument("--step", action="append", choices=list(STEPS),
                        help="lab section(s) to run; repeat the flag for several (default: all)")
    parser.add_argument("--ask", metavar="PROMPT", help="your own request; the LLM can use all four tools")
    parser.add_argument("--model", help="override the lab's models (openai:gpt-4o for section 3, "
                                        "openai:o4-mini for section 4), e.g. openai:gpt-4.1-mini")
    parser.add_argument("--max-turns", type=int, default=10, help="tool-call turns allowed for --ask")
    args = parser.parse_args()

    # Run from the project folder so dl_logo.jpg resolves and outputs are saved here
    os.chdir(Path(__file__).resolve().parent)
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Add it to the .env file in the ToolUseDemo folder "
                 "(see .env.example).")

    if args.ask:
        ask(args.ask, args.model or "openai:o4-mini", args.max_turns)
        return

    failed = []
    for key in args.step or list(STEPS):
        title, run = STEPS[key]
        print(f"\n\n######## {title} ########")
        try:
            run(args.model) if args.model else run()
        except Exception as e:  # keep going so one failing tool doesn't hide the other sections
            failed.append(key)
            print(f"!! Step '{key}' failed: {type(e).__name__}: {e}")

    if failed:
        sys.exit(f"\nFailed steps: {', '.join(failed)}")


if __name__ == "__main__":
    main()
