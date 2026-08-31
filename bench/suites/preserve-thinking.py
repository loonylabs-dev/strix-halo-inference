#!/usr/bin/env python3
"""preserve-thinking — does stripping past thinking from the history stop the
model from thinking more and more each turn?

    python3 bench/suites/preserve-thinking.py                    full run
    python3 bench/suites/preserve-thinking.py --sessions 1 --turns 2   smoke
    python3 bench/suites/preserve-thinking.py --effort medium

Why this exists
---------------
kyuz0/amd-strix-halo-toolboxes#127 (28.08.2026): a user of the SAME model on
the same hardware runs thinking permanently ON at reasoning_effort medium and
reports that `preserve_thinking: false` in --chat-template-kwargs is what
made that bearable — without it the model drowns in "Wait, Actually, After
thinking more…" loops. That is exactly why this stack's think modes went
unused: even low and medium overthink.

The mechanism, measured 31.08.2026 before anything ran on a GPU (template out
of Qwen3.8-27B-UD-Q4_K_XL.gguf, rendered with test-chat-template b10665):

    multi-turn, thinking on, no kwarg          aa20f5a4  past <think> IN prompt
    multi-turn, preserve_thinking true         aa20f5a4  same — true is default
    multi-turn, preserve_thinking false        a9d81d4d  past <think> STRIPPED
    single-turn, with and without the kwarg    df8913bb  identical either way

So the template KEEPS every past turn's thinking in the prompt by default,
and the kwarg only touches history — it cannot shorten thinking directly.
The chain that makes this reach production: Claude Code resends thinking
blocks in the history (the Anthropic contract), llama-server's Anthropic
endpoint maps them onto message.reasoning_content (server-chat.cpp), and the
template renders them all back. Each turn the model reads its own
accumulated rumination and continues the style. The hypothesis this suite
tests is that the loops are that feedback, not the effort level:

    strip the history  ->  per-turn thinking stays flat instead of growing

What it measures
----------------
Two arms, identical fixed questions, sessions interleaved so neither arm owns
a warmer half of the run. Both arms resend reasoning_content in the history —
that is what the production path does — and only the kwarg differs:

    default   {enable_thinking, reasoning_effort}
    strip     the same + {"preserve_thinking": false}

Per turn: thinking and answer tokens (counted via /tokenize, not estimated),
prompt_n and cached_tokens (the strip arm makes the slot no longer a prefix
at the previous turn's <think>, and this model is a HYBRID that cannot be
trimmed — see qwen38.env; whatever that re-prefill costs shows up here),
finish_reason (a truncated think is data, not an error), and seconds.

Talks to llama-server DIRECTLY, not the gateway: the gateway offers no
preserve_thinking mode yet, and whether it should is this suite's result.

Caveats stated up front: thinking length under production sampling is noisy,
and a few sessions per arm resolve a large effect only. Read trends, not
single-digit token differences. Decode speed on this machine has a 19 % CoV
(qwen38.env), so the seconds column is context, never a finding.
"""
import argparse, json, os, sys, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# Four turns that build on each other and invite reasoning at every step —
# review, extend, test, refine. Fixed on purpose: reruns stay comparable.
QUESTIONS = [
    "Here is a Python function meant to deduplicate a list while preserving "
    "the order of first occurrence:\n\n"
    "```python\n"
    "def dedupe(items):\n"
    "    return sorted(set(items), key=items.index)\n"
    "```\n\n"
    "Is it correct, and is it efficient? If anything is wrong, fix it.",

    "Now make it handle unhashable items (like dicts) as well, staying O(n) "
    "for the hashable ones.",

    "Write three edge-case tests for the final version, as plain asserts.",

    "One of your tests should avoid comparing floats with ==. Adjust the "
    "tests accordingly and briefly say why.",
]


def post(url, path, body, timeout):
    req = urllib.request.Request(
        url + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as x:
        return json.load(x)


def n_tokens(url, text, timeout):
    if not text:
        return 0
    return len(post(url, "/tokenize", {"content": text}, timeout)["tokens"])


def run_session(url, alias, kwargs, turns, max_tokens, timeout):
    """One conversation, resending reasoning_content like production does."""
    messages, rows = [], []
    for i, q in enumerate(QUESTIONS[:turns]):
        messages.append({"role": "user", "content": q})
        body = {"model": alias, "messages": messages,
                "chat_template_kwargs": kwargs,
                "max_tokens": max_tokens, "stream": False}
        t0 = time.time()
        r = post(url, "/v1/chat/completions", body, timeout)
        secs = time.time() - t0
        choice = r["choices"][0]
        msg = choice["message"]
        think = msg.get("reasoning_content") or ""
        answer = msg.get("content") or ""
        t = r.get("timings") or {}
        rows.append({
            "turn": i + 1,
            "think_tokens": n_tokens(url, think, timeout),
            "answer_tokens": n_tokens(url, answer, timeout),
            "finish": choice.get("finish_reason"),
            "prompt_n": t.get("prompt_n"),
            "cached": (r.get("usage") or {}).get(
                "prompt_tokens_details", {}).get("cached_tokens"),
            "seconds": round(secs, 1),
        })
        # The resend below is the production shape: the history carries the
        # thinking, and the TEMPLATE decides whether it reaches the prompt.
        messages.append({"role": "assistant", "content": answer,
                         "reasoning_content": think})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--alias", default="qwen38")
    ap.add_argument("--effort", default="low",
                    help="template level for both arms (default low — the "
                         "mode whose overthinking made think modes unusable)")
    ap.add_argument("--sessions", type=int, default=3, help="per arm")
    ap.add_argument("--turns", type=int, default=len(QUESTIONS))
    ap.add_argument("--max-tokens", type=int, default=6144,
                    help="per turn. A think that hits this is reported as "
                         "finish=length, which is itself a result")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", help="report directory")
    a = ap.parse_args()

    arms = {
        "default": {"enable_thinking": True, "reasoning_effort": a.effort},
        "strip": {"enable_thinking": True, "reasoning_effort": a.effort,
                  "preserve_thinking": False},
    }

    # Interleaved: default, strip, default, strip … so slow drift of the
    # server (thermals, cache state) cannot favour one arm.
    plan = [(arm, s) for s in range(a.sessions) for arm in arms]

    all_rows = []
    for arm, s in plan:
        print("=== %s  session %d/%d" % (arm, s + 1, a.sessions))
        rows = run_session(a.url, a.alias, arms[arm], a.turns,
                           a.max_tokens, a.timeout)
        for r in rows:
            r.update({"arm": arm, "session": s + 1})
            all_rows.append(r)
            print("    turn %d  think %5d  answer %4d  prompt_n %6s  "
                  "cached %6s  %5.1fs  %s"
                  % (r["turn"], r["think_tokens"], r["answer_tokens"],
                     r["prompt_n"], r["cached"], r["seconds"],
                     "" if r["finish"] == "stop" else "FINISH=" + str(r["finish"])))

    print("\n  mean think tokens per turn (n=%d sessions per arm)" % a.sessions)
    print("  %-8s %s" % ("arm", "  ".join("turn%d" % (i + 1)
                                          for i in range(a.turns))))
    for arm in arms:
        means = []
        for i in range(a.turns):
            vals = [r["think_tokens"] for r in all_rows
                    if r["arm"] == arm and r["turn"] == i + 1]
            means.append(sum(vals) / len(vals) if vals else 0)
        print("  %-8s %s" % (arm, "  ".join("%5.0f" % m for m in means)))
    print("\n  What to read: if the default arm grows turn over turn and the")
    print("  strip arm stays flat, the overthinking is the history feedback.")
    print("  If both arms look alike, it is the effort level itself, and the")
    print("  kwarg buys only context space, not calm.")

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "rows.jsonl"), "w") as fh:
            for r in all_rows:
                fh.write(json.dumps(r) + "\n")
        with open(os.path.join(a.out, "result.json"), "w") as fh:
            json.dump({"url": a.url, "alias": a.alias, "effort": a.effort,
                       "sessions": a.sessions, "turns": a.turns,
                       "max_tokens": a.max_tokens, "arms": arms,
                       "rows": all_rows}, fh, indent=1)
        print("\n  written: %s" % a.out)


if __name__ == "__main__":
    main()
