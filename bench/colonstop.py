#!/usr/bin/env python3
"""colonstop — does the served model end an agent turn it meant to continue?

The failure: an agent turn announces its next step and ends with no tool
call — "Jetzt die Änderungen am Command selbst:" and then <|im_end|>. Claude
Code reads no tool call as "done" and the chat stands still until the user
types "weiter". Found 23.09.2026; the cause was the checkpoint's quality
sidecar (bench/reports/2026-09-23_colonstop/, upstream #89).

This replays a REAL stall from the gateway's text-level trace and asks the
running server, N times, to continue from exactly where the model stopped:

    rendered prompt + its own thinking + '</think>\\n\\n' + its own text

through /v1/completions, so every arm sees the same TOKENS and only the server
differs. A continuation that is empty ended the turn again ('stop'); one that
opens a <tool_call> is what the model should have done ('tool').

    --arms control        the stall as it was (default)
    --arms control,fix    also '\\n\\n' appended: the llama.cpp #19513 workaround
                          (25/25 tool calls on 23.09. — measured, then not built,
                          because turning the sidecar off removed the cause)

The prompt is rendered ONCE, inside the running container, with that
server's own ChatReq / normalize_messages / chat_kwargs and its tokenizer,
and cached next to the output. Rendering on another release would give that
release's template, so --render-from names the container explicitly.

WHAT THIS DOES NOT DO: start, stop or swap servers. An A/B of two server
configurations means production goes down — that is the operator's step
(CLAUDE.md, "Production changes only on the operator's explicit go"), and
sideserver.py cannot start a Halogen container. How the 23.09. A/B was run is
in the report's README.

PRIVACY: the trace holds complete prompts. Nothing of them reaches the output
except counts, timings and the class of each continuation — the rendered
prompts are cached under --cache, which must stay outside the repo.

    python3 bench/colonstop.py --prefix 776999d6717f --at 23,35 \\
        --n 10 --label overlay --out bench/reports/<dir> --cache ~/.cache/colonstop
"""
import argparse
import http.client
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "setup" / "gateway"))
import anthropic_bridge as AB  # noqa: E402

TRACE_DIR = Path(os.path.expanduser("~/.cache/llm-gateway-trace"))
OPENER = "<|im_start|>assistant\n<think>\n"

# Runs INSIDE the container: OpenAI body on stdin, rendered prompt on stdout.
RENDER = r'''
import json, sys
sys.path.insert(0, "/halogen/tools")
import serve_api as S
from tool_parse import normalize_messages
from transformers import AutoTokenizer
req = S.ChatReq(**json.load(sys.stdin))
kw = S.chat_kwargs(req, "auto")
msgs = normalize_messages(req.messages)
tok = AutoTokenizer.from_pretrained("/models/tokenizer")
sys.stdout.write(tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True, **kw))
'''


def conversation(trace, prefix):
    """The longest text-level request body the trace holds for this prefix."""
    body = None
    with open(trace, encoding="utf-8") as f:
        for line in f:
            if '"body_full": {' not in line or f'"prefix": "{prefix}"' not in line:
                continue
            b = json.loads(line)["body_full"]
            if body is None or len(b["messages"]) > len(body["messages"]):
                body = b
    if body is None:
        raise SystemExit(f"{trace}: no text-level request for prefix {prefix} "
                         "(the trace level must have been 'text' while it ran)")
    return body


def stall(body, k):
    """-> (thinking, text) of assistant message k, which must not call a tool."""
    if k >= len(body["messages"]):
        raise SystemExit(f"message {k}: the trace holds only "
                         f"{len(body['messages'])} messages of this conversation")
    msg = body["messages"][k]
    if msg["role"] != "assistant" or not isinstance(msg["content"], list):
        raise SystemExit(f"message {k} is not an assistant turn")
    if any(b.get("type") == "tool_use" for b in msg["content"]):
        raise SystemExit(f"message {k} called a tool — not a stall")
    th = "".join(b.get("thinking") or "" for b in msg["content"] if b.get("type") == "thinking")
    tx = "".join(b.get("text") or "" for b in msg["content"] if b.get("type") == "text")
    return th, tx


def openai_body(body, k):
    """The request the gateway sent for the turn that produced message k."""
    q = dict(body, messages=body["messages"][:k])
    o = AB.anthropic_to_openai_request(q, target_model="halogen")
    # the gateway copies these after translating (gateway.py, dialect bridge)
    for f in ("enable_thinking", "reasoning_effort", "max_thinking_tokens", "stop"):
        if f in q:
            o[f] = q[f]
    return o


def render(container, obody):
    # BYTES, not text=True: text mode folds \r\n into \n, and a Windows client's
    # CLAUDE.md carries \r\n — the replay would no longer be the served prompt.
    r = subprocess.run(["podman", "exec", "-i", container, "python3", "-c", RENDER],
                       input=json.dumps(obody).encode("utf-8"), capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"render in {container} failed:\n"
                         f"{r.stderr.decode('utf-8', 'replace')[-2000:]}")
    text = r.stdout.decode("utf-8")
    if not text.endswith(OPENER):
        raise SystemExit(f"rendered prompt does not end in the assistant opener: "
                         f"{text[-60:]!r}")
    return text


def cached(path, produce):
    """The rendered prompt, from `path` or produced once and kept there.
    newline="" both ways, for the same \\r\\n reason as render()."""
    if not path.exists():
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(produce())
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def classify(out):
    s = out.lstrip()
    if s.startswith("<tool_call>"):
        return "tool"
    return "stop" if s == "" else "other"


def complete(host, port, prompt, max_tokens):
    c = http.client.HTTPConnection(host, port, timeout=None)  # a cold 128k prefill is minutes
    t = time.time()
    c.request("POST", "/v1/completions",
              json.dumps({"prompt": prompt, "max_tokens": max_tokens, "stream": False,
                          "stop": ["<|im_end|>", "<|endoftext|>"]}),
              {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    if r.status != 200:
        raise SystemExit(f"HTTP {r.status}: {data[:300]!r}")
    res = json.loads(data)
    return res["choices"][0], res.get("usage") or {}, time.time() - t


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trace", help="gateway trace file (default: newest)")
    ap.add_argument("--prefix", required=True, help="the conversation's prefix id")
    ap.add_argument("--at", required=True, help="assistant message indices, e.g. 23,35")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--arms", default="control")
    ap.add_argument("--label", required=True, help="names this server configuration")
    ap.add_argument("--out", required=True, help="directory for <label>.jsonl")
    ap.add_argument("--cache", required=True, help="rendered prompts (private — not the repo)")
    ap.add_argument("--render-from", default="halogen", help="container to render in")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--max-tokens", type=int, default=120)
    a = ap.parse_args()

    trace = Path(a.trace) if a.trace else max(TRACE_DIR.glob("trace-*.jsonl"))
    cache, out = Path(a.cache).expanduser(), Path(a.out)
    if REPO in cache.resolve().parents:
        raise SystemExit("--cache is inside the repo; rendered prompts are private")
    cache.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    arms = a.arms.split(",")
    body = conversation(trace, a.prefix)

    for k in (int(x) for x in a.at.split(",")):
        th, tx = stall(body, k)
        prompt = cached(cache / f"{a.prefix}-{k}.txt",
                        lambda: render(a.render_from, openai_body(body, k)))
        base = prompt + th.strip() + "\n</think>\n\n" + tx.rstrip()
        prompts = {"control": base, "fix": base + "\n\n"}
        for _ in range(a.n):  # arms interleaved, so drift over time hits both
            for arm in arms:
                ch, usage, took = complete(a.host, a.port, prompts[arm], a.max_tokens)
                row = {"prefix": a.prefix, "k": k, "arm": arm, "label": a.label,
                       "cls": classify(ch.get("text") or ""),
                       "finish": ch.get("finish_reason"), "took_s": round(took, 2),
                       "prompt_tokens": usage.get("prompt_tokens"),
                       "completion_tokens": usage.get("completion_tokens"),
                       "last_char": tx.rstrip()[-1:], "t": time.strftime("%F %T")}
                with open(out / f"{a.label}.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
