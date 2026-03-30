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

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI(
    base_url=os.getenv("LM_BASE_URL"),
    api_key=os.getenv("LM_API_KEY"),
)
MODEL = os.environ["LM_MODEL_ID"]

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


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120,
                           encoding='utf-8', errors='ignore')
        out = ((r.stdout or "") + (r.stderr or "")).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def agent_loop(messages: list):
    while True:
        response = client.chat.completions.create(
            model=MODEL, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        message = response.choices[0].message
        messages.append({"role": "assistant", "content": message.content or "", "tool_calls": message.tool_calls})

        if not message.tool_calls:
            return

        results = []
        for tool_call in message.tool_calls:
            if tool_call.type == "function":
                args = json.loads(tool_call.function.arguments)
                cmd = args.get("command", "")
                print(f"\033[33m$ {cmd}\033[0m")
                output = run_bash(cmd)
                print(output[:200])
                results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(output),
                })
        messages.extend(results)


if __name__ == "__main__":
    history = [{"role": "system", "content": SYSTEM}]
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, str):
            print(response_content)
        elif isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "content"):
                    print(block.content)
        print()
