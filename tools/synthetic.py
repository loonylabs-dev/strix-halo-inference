#!/usr/bin/env python3
"""synthetic — build a Claude-Code-shaped request body without a capture.

Why: real captured bodies carry personal data (e-mail address, device_id,
account_uuid) and Anthropic's system prompt. They do not belong in this repo.
The measurements need the *structure*, not the content — and the structure can
be rebuilt.

What is reproduced is exactly what decides the prompt cache:

    system field       array of text blocks, stable head first
    tools              N schemas, together the largest block
    messages[0] user   two text blocks: <system-reminder> plus the question
    messages[1] system "Available agent types ..." AFTER the question,
                       with the <total_tokens> counter appended

The last point is the decisive one: because that block sits behind the
question, every changed question falls outside the SWA window. See
docs/measurements/cache-hunt-finding.md.

Usage:
    python3 synthetic.py --tools 24 --project /tmp/projA --question "Say alpha."
    python3 synthetic.py --turns 3        # growing tool conversation
"""
import argparse, json, sys

HEAD = (
    "You are a coding agent running in a terminal. You help with software "
    "engineering tasks by reading files, running commands and editing code.\n"
)

def system_text(project, memory):
    """About 6,000 characters, with the working directory in two places —
    just like Claude Code (characters ~2,538 and ~4,670)."""
    filler1 = "\n".join(
        "Guideline %d: prefer the dedicated tools over shell commands where one "
        "fits, and keep changes minimal and reviewable." % i for i in range(1, 31))
    filler2 = "\n".join(
        "Note %d: report outcomes faithfully; if a step was skipped, say so." % i
        for i in range(1, 31))
    return (HEAD + filler1 +
            "\n\n# Memory\n\nYou have a persistent file-based memory at "
            "`%s`. Each memory is one file holding one fact.\n\n" % memory +
            filler2 +
            "\n\n# Environment\n"
            " - Primary working directory: %s\n"
            " - Platform: linux\n - Shell: bash\n" % project)

def tool(i):
    """One schema in the size range of real Claude Code tools."""
    return {
        "name": "Tool%02d" % i,
        "description": ("Description of tool %d. " % i) + (
            "This tool exists for illustration and contributes to the size of "
            "the schema block, so that the prefix reaches the same order of "
            "magnitude as in real requests. " * 13),
        "input_schema": {
            "type": "object",
            "properties": {
                "path":    {"type": "string", "description": "Path to the file"},
                "pattern": {"type": "string", "description": "Search pattern"},
                "depth":   {"type": "integer", "description": "Recursion depth"},
                "mode":    {"type": "string", "enum": ["read", "write", "check"]},
            },
            "required": ["path"],
        },
    }

AGENT_BLOCK = (
    "Available agent types for the Agent tool:\n" +
    "\n".join("- agent-%02d: Description of agent %02d, which handles a certain "
              "class of tasks and whose description brings this block to a "
              "realistic length. It names tools, limits and typical uses of "
              "the agent." % (i, i)
              for i in range(1, 22)))

REMINDER = (
    "<system-reminder>\nAs you answer the user's questions, you can use the "
    "following context:\n# currentDate\nToday's date is 2026-01-01.\n\n"
    "IMPORTANT: this context may or may not be relevant to your tasks.\n"
    "</system-reminder>\n\n")

def body(project="/tmp/projA", memory=None, question="Say alpha.", n_tools=24,
         turns=1, budget_left=15000000):
    memory = memory or ("/home/user/.claude/projects/%s/memory/"
                        % project.strip("/").replace("/", "-"))
    p = {
        "model": "local",
        "max_tokens": 32000,
        "stream": False,
        "system": [
            {"type": "text", "text": "You are a coding agent.",
             "cache_control": {"type": "ephemeral", "ttl": "1h"}},
            {"type": "text", "text": system_text(project, memory),
             "cache_control": {"type": "ephemeral", "ttl": "1h"}},
        ],
        "tools": [tool(i) for i in range(1, n_tools + 1)],
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": REMINDER},
                {"type": "text", "text": question}]},
            # The block that sits behind the question — the actual cause.
            {"role": "system",
             "content": AGENT_BLOCK +
                        "\n\n<total_tokens>%d tokens left</total_tokens>" % budget_left},
        ],
    }
    # Further turns: tool call, result, new counter
    for t in range(1, turns):
        p["messages"] += [
            {"role": "assistant", "content": [
                {"type": "text", "text": "Let me look."},
                {"type": "tool_use", "id": "tu_%03d" % t,
                 "name": "Tool01", "input": {"path": "/tmp/file%d.txt" % t}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_%03d" % t,
                 "content": "line one\nline two\nline three"}]},
            {"role": "system",
             "content": "<total_tokens>%d tokens left</total_tokens>"
                        % (budget_left - t * 127)},
        ]
    return p

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project",  default="/tmp/projA")
    ap.add_argument("--question", default="Say alpha.")
    ap.add_argument("--tools",    type=int, default=24)
    ap.add_argument("--turns",    type=int, default=1)
    ap.add_argument("--out",      default="-")
    a = ap.parse_args()
    p = body(project=a.project, question=a.question, n_tools=a.tools, turns=a.turns)
    s = json.dumps(p, ensure_ascii=False, indent=1)
    if a.out == "-":
        sys.stdout.write(s)
    else:
        open(a.out, "w", encoding="utf-8").write(s)
        print("%s  %d bytes, %d tools, %d messages"
              % (a.out, len(s), len(p["tools"]), len(p["messages"])), file=sys.stderr)
