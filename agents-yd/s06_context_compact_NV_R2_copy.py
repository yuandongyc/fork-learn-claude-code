#!/usr/bin/env python3
# Harness: compression -- clean memory for infinite sessions.
"""
s06_context_compact.py - Compact

Three-layer compression pipeline so the agent can work forever:

    Every turn:
    +------------------+
    | Tool call result |
    +------------------+
            |
            v
    [Layer 1: micro_compact]        (silent, every turn)
      Replace non-read_file tool_result content older than last 3
      with "[Previous: used {tool_name}]"
            |
            v
    [Check: tokens > 50000?]
       |               |
       no              yes
       |               |
       v               v
    continue    [Layer 2: auto_compact]
                  Save full transcript to .transcripts/
                  Ask LLM to summarize conversation.
                  Replace all messages with [summary].
                        |
                        v
                [Layer 3: compact tool]
                  Model calls compact -> immediate summarization.
                  Same as auto, triggered manually.

Key insight: "The agent can forget strategically and keep working forever."
"""

import json
import os
import platform
import subprocess
import time
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI(
    base_url=os.getenv("NV_BASE_URL"),
    api_key=os.getenv("NV_API_KEY"),
)
MODEL = os.environ["NV_MODEL_ID"]

WORKDIR = Path.cwd()
IS_WINDOWS = platform.system() == "Windows"

ENV_HINT = "Windows environment: prefer read_file/glob over bash commands." if IS_WINDOWS else ""
TASK_PATTERN = (
    "CRITICAL RULE: After calling glob ONE time, you MUST NOT call glob again in the same conversation turn.\n"
    "Task 'Read all files': 1) glob once, 2) read all files from that list sequentially, 3) STOP.\n"
    "Task 'Search code': 1) grep once, 2) read files, 3) STOP."
)
SYSTEM = (
    f"You are a coding agent at {WORKDIR}.\n"
    "Tool usage priority:\n"
    "1. read_file/glob - file reading & discovery (recommended)\n"
    "2. grep - content search in large codebases\n"
    "3. edit_file - text replacement\n"
    "4. bash - only when necessary\n"
    f"{ENV_HINT}\n"
    f"{TASK_PATTERN}\n"
    "Use tools to solve tasks."
)

THRESHOLD = 50000
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
KEEP_RECENT = 3
PRESERVE_RESULT_TOOLS = {"read_file"}


def estimate_tokens(messages: list) -> int:
    """Rough token count: ~4 chars per token."""
    return len(str(messages)) // 4


# -- Layer 1: micro_compact - replace old tool results with placeholders --
def micro_compact(messages: list) -> list:
    tool_results = []
    for msg_idx, msg in enumerate(messages):
        if msg.get("role") == "tool":
            tool_results.append((msg_idx, msg))
    if len(tool_results) <= KEEP_RECENT:
        return messages
    
    tool_name_map = {}
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if isinstance(tc, dict):
                    tool_name_map[tc.get("id", "")] = tc.get("function", {}).get("name", "unknown")
                elif hasattr(tc, "id") and hasattr(tc, "function"):
                    tool_name_map[tc.id] = tc.function.name if hasattr(tc.function, "name") else "unknown"
    
    to_clear = tool_results[:-KEEP_RECENT]
    for msg_idx, result in to_clear:
        if not isinstance(result.get("content"), str) or len(result["content"]) <= 100:
            continue
        tool_id = result.get("tool_call_id", "")
        tool_name = tool_name_map.get(tool_id, "unknown")
        if tool_name in PRESERVE_RESULT_TOOLS:
            continue
        result["content"] = f"[Previous: used {tool_name}]"
    return messages


# -- Layer 2: auto_compact - save transcript, summarize, replace messages --
def auto_compact(messages: list) -> list:
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    print(f"[transcript saved: {transcript_path}]")
    conversation_text = json.dumps(messages, default=str)[-80000:]
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content":
            "Summarize this conversation for continuity. Include: "
            "1) What was accomplished, 2) Current state, 3) Key decisions made. "
            "Be concise but preserve critical details.\n\n" + conversation_text}],
        max_tokens=2000,
    )
    summary = response.choices[0].message.content
    if not summary:
        summary = "No summary generated."
    return [
        {"role": "user", "content": f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}"},
    ]


# -- Tool implementations --
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


def run_glob(pattern: str) -> str:
    try:
        matches = list(WORKDIR.rglob(pattern))
        if not matches:
            return "(no matches)"
        return "\n".join(str(p.relative_to(WORKDIR)) for p in matches[:100])
    except Exception as e:
        return f"Error: {e}"

def run_grep(pattern: str, include: str = None) -> str:
    try:
        import fnmatch
        matches = []
        for p in WORKDIR.rglob(include or "*.py"):
            if p.is_file():
                try:
                    content = p.read_text(encoding='utf-8', errors='ignore')
                    if pattern in content:
                        matches.append(str(p.relative_to(WORKDIR)))
                except:
                    pass
        if not matches:
            return "(no matches)"
        return "\n".join(matches[:50])
    except Exception as e:
        return f"Error: {e}"


TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "glob":       lambda **kw: run_glob(kw["pattern"]),
    "grep":       lambda **kw: run_grep(kw["pattern"], kw.get("include")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "compact":    lambda **kw: "Manual compression requested.",
}

TOOLS = [
    {"type": "function", "function": {
        "name": "glob",
        "description": "ONE-TIME file discovery. After this, use read_file for each file. NEVER call glob again.",
        "parameters": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    }},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read file contents. Recommended for file reading.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["path"],
        },
    }},
    {"type": "function", "function": {
        "name": "glob",
        "description": "Find files by name patterns. Recommended for file discovery.",
        "parameters": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    }},
    {"type": "function", "function": {
        "name": "grep",
        "description": "Fast content search. Use for finding patterns in codebases. "
                       "Windows: works in git bash / WSL.",
        "parameters": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}, "include": {"type": "string"}},
            "required": ["pattern"],
        },
    }},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write content to file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    }},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": "Replace exact text in file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}},
            "required": ["path", "old_text", "new_text"],
        },
    }},
    {"type": "function", "function": {
        "name": "compact",
        "description": "Trigger manual conversation compression.",
        "parameters": {
            "type": "object",
            "properties": {"focus": {"type": "string", "description": "What to preserve in the summary"}},
        },
    }},
]


def agent_loop(messages: list):
    while True:
        # Layer 1: micro_compact before each LLM call
        micro_compact(messages)
        # Layer 2: auto_compact if token estimate exceeds threshold
        if estimate_tokens(messages) > THRESHOLD:
            print("[auto_compact triggered]")
            messages[:] = auto_compact(messages)
        
        system_msg = {"role": "system", "content": SYSTEM}
        response = client.chat.completions.create(
            model=MODEL, messages=[system_msg] + messages,
            tools=TOOLS, max_tokens=8000,
        )
        message = response.choices[0].message
        messages.append({"role": "assistant", "content": message.content or "", "tool_calls": message.tool_calls})
        
        if not message.tool_calls:
            return
        
        results = []
        manual_compact = False
        for tool_call in message.tool_calls:
            func = tool_call.function
            if func.name == "compact":
                manual_compact = True
                output = "Compressing..."
            else:
                handler = TOOL_HANDLERS.get(func.name)
                try:
                    args = json.loads(func.arguments) if isinstance(func.arguments, str) else func.arguments
                    output = handler(**args) if handler else f"Unknown tool: {func.name}"
                except Exception as e:
                    output = f"Error: {e}"
            print(f"> {func.name}: {str(output)[:200]}")
            results.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(output),
            })
        
        messages.extend(results)
        
        # Layer 3: manual compact triggered by the compact tool
        if manual_compact:
            print("[manual compact]")
            messages[:] = auto_compact(messages)


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms06 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, str):
            print(response_content)
        print()
