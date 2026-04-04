#!/usr/bin/env python3
# Harness: the loop -- the model's first connection to the real world.
"""
s01_agent_loop_openai.py - The Agent Loop (OpenAI Compatible)

The entire secret of an AI coding agent in one pattern:

    while stop_reason == "tool_use":
        response = LLM(messages, tools)
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)

This is the core loop: feed tool results back to the model
until the model decides to stop. Production agents layer
policy, hooks, and lifecycle controls on top.
"""

import json
import os
import subprocess
from datetime import datetime

from openai import OpenAI
from dotenv import load_dotenv

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def get_log_file():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(LOG_DIR, f"s01_{timestamp}.log")

def log_write(log_file, step: str, content: str):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n=== {step} ===\n")
        f.write(content)
        f.write("\n")

load_dotenv(override=True)

client = OpenAI(
    base_url=os.getenv("NV_BASE_URL"),
    api_key=os.getenv("NV_API_KEY"),
)
MODEL = os.environ["NV_MODEL_ID"]

SYSTEM = f"You are a coding agent on Windows at {os.getcwd()}. Use Windows commands (dir, not ls). Act, don't explain."

TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}]


def run_bash(command: str, log_file: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120,
                           encoding='utf-8', errors='ignore')
        out = ((r.stdout or "") + (r.stderr or "")).strip()
        log_write(log_file, f"BASH_CMD: {command}", out[:50000] if out else "(no output)")
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        log_write(log_file, f"TIMEOUT: {command}", "Error: Timeout (120s)")
        return "Error: Timeout (120s)"


def agent_loop(messages: list, log_file: str):
    while True:
        response = client.chat.completions.create(
            model=MODEL, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        message = response.choices[0].message
        messages.append({"role": "assistant", "content": message.content or "", "tool_calls": message.tool_calls})

        # 记录 LLM 响应
        resp_str = str(message.content) + (str(message.tool_calls) if message.tool_calls else "")
        log_write(log_file, "LLM_RESPONSE", resp_str)

        if not message.tool_calls:
            return

        results = []
        for tool_call in message.tool_calls:
            if tool_call.type == "function":
                args = json.loads(tool_call.function.arguments)
                cmd = args.get("command", "")
                print(f"\033[33m$ {cmd}\033[0m")
                output = run_bash(cmd, log_file)
                print(output[:200])
                results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(output),
                })
        messages.extend(results)


if __name__ == "__main__":
    log_file = get_log_file()
    history = [{"role": "system", "content": SYSTEM}]
    log_write(log_file, "SYSTEM_PROMPT", SYSTEM)
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        log_write(log_file, "USER_INPUT", query)
        agent_loop(history, log_file)
        response_content = history[-1]["content"]
        resp_str = str(response_content)
        log_write(log_file, "FINAL_RESPONSE", resp_str)
        if isinstance(response_content, str):
            print(response_content)
        elif isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "content"):
                    print(block.content)
        print()
