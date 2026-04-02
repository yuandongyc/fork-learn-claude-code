#!/usr/bin/env python3
# Harness: context isolation -- protecting the model's clarity of thought.
"""
s04_subagent_NV.py - Subagents (OpenAI Compatible / NVIDIA NIM)

Spawn a child agent with fresh messages=[]. The child works in its own
context, sharing the filesystem, then returns only a summary to the parent.

    Parent agent                     Subagent
    +------------------+             +------------------+
    | messages=[...]   |             | messages=[]      |  <-- fresh
    |                  |  dispatch   |                  |
    | tool: task       | ---------->| while tool_use:  |
    |   prompt="..."   |            |   call tools     |
    |   description="" |            |   append results |
    |                  |  summary   |                  |
    |   result = "..." | <--------- | return last text |
    +------------------+             +------------------+
              |
    Parent context stays clean.
    Subagent context is discarded.

Key insight: "Process isolation gives context isolation for free."
"""

import json
import os
import subprocess
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

WORKDIR = Path.cwd()
client = OpenAI(
    base_url=os.getenv("NV_BASE_URL"),
    api_key=os.getenv("NV_API_KEY", "not-needed"),
)
MODEL = os.environ["NV_MODEL_ID"]

SYSTEM = f"You are a coding agent on Windows at {WORKDIR}. Use the task tool to delegate exploration or subtasks. Use Windows commands (dir, not ls)."
SUBAGENT_SYSTEM = f"You are a coding subagent on Windows at {WORKDIR}. Complete the given task, then summarize your findings. Use Windows commands (dir, not ls)."


def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120,
                           encoding='utf-8', errors='ignore')
        out = ((r.stdout or "") + (r.stderr or "")).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

CHILD_TOOLS = [{
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
}, {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read file contents.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["path"],
        },
    },
}, {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write content to file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
}, {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": "Replace exact text in file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}},
            "required": ["path", "old_text", "new_text"],
        },
    },
}]


def run_subagent(prompt: str) -> str:
    sub_messages = [{"role": "user", "content": prompt}]
    for _ in range(30):
        response = client.chat.completions.create(
            model=MODEL,
            messages=sub_messages,
            tools=CHILD_TOOLS,
            max_tokens=8000,
        )
        message = response.choices[0].message

        if message.tool_calls:
            sub_messages.append({
                "role": "assistant",
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in message.tool_calls
                ]
            })
            results = []
            for tool_call in message.tool_calls:
                handler = TOOL_HANDLERS.get(tool_call.function.name)
                args = json.loads(tool_call.function.arguments)
                output = handler(**args) if handler else f"Unknown tool: {tool_call.function.name}"
                results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(output)[:50000],
                })
            sub_messages.extend(results)
        else:
            sub_messages.append({"role": "assistant", "content": message.content})
            break

    for msg in reversed(sub_messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            return msg["content"]
    return "(no summary)"


PARENT_TOOLS = CHILD_TOOLS + [{
    "type": "function",
    "function": {
        "name": "task",
        "description": "Spawn a subagent with fresh context. It shares the filesystem but not conversation history.",
        "parameters": {
            "type": "object",
            "properties": {"prompt": {"type": "string"}, "description": {"type": "string"}},
            "required": ["prompt"],
        },
    },
}]


def agent_loop(messages: list):
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=PARENT_TOOLS,
            max_tokens=8000,
        )
        message = response.choices[0].message

        if message.tool_calls:
            messages.append({
                "role": "assistant",
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in message.tool_calls
                ]
            })
        else:
            messages.append({"role": "assistant", "content": message.content})
            return

        results = []
        for tool_call in message.tool_calls:
            if tool_call.function.name == "task":
                args = json.loads(tool_call.function.arguments)
                desc = args.get("description", "subtask")
                print(f"> task ({desc}): {args['prompt'][:80]}")
                output = run_subagent(args["prompt"])
            else:
                handler = TOOL_HANDLERS.get(tool_call.function.name)
                args = json.loads(tool_call.function.arguments)
                output = handler(**args) if handler else f"Unknown tool: {tool_call.function.name}"
            print(f"  {str(output)[:200]}")
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
            query = input("\033[36ms04 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_message = history[-1]
        if response_message.get("content"):
            print(response_message["content"])
        print()