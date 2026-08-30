#!/usr/bin/env python3
"""Why is the model's own answer worthless to the next turn?

Measured 30.08.2026 in live Claude Code traffic: every turn re-prefilled
EXACTLY the previous answer. 16,787 tokens on one turn, 12,041 on the next —
about 190 s and 140 s of pure re-reading, on top of generating. The history
itself was a clean append (5 of 5 messages unchanged), and the client does
send the thinking back: the assistant message carried 54,973 characters of it.

So the tokens are there and they still do not match. Something about the way a
turn is RE-RENDERED differs from the way it was GENERATED, and because the
difference is at the front of the answer, everything behind it is lost.

This finds the exact character where they part.

    1  send a prompt, thinking on, and keep the whole answer
    2  ask llama-server to render prompt + that answer as history
       (/apply-template — no inference, no slot)
    3  compare that rendering against what the model actually produced
    4  send the follow-up for real and read `timings.cache_n`, so the
       character-level finding and the token-level cost stand side by side

Step 3 is the point. A divergence in a delimiter or a newline is fixable in
the gateway; a re-render that drops or reorders content is not, and the two
look identical from the outside.

    python3 bench/suites/answer-reuse.py

IT TAKES THE ONE SLOT for two short generations. Keep the prompt small — the
question is where the rendering differs, and that does not need a long answer.
"""
import argparse, json, sys, time, urllib.request

DEFAULT_URL = "http://127.0.0.1:8080"


def post(url, path, body, timeout=1800):
    req = urllib.request.Request(
        url + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def first_difference(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default="qwen38")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--ask", default="Nenne drei Primzahlen und begruende kurz.")
    a = ap.parse_args()

    kwargs = {"enable_thinking": True, "reasoning_effort": a.effort}
    first = {"model": a.model, "max_tokens": 900, "chat_template_kwargs": kwargs,
             "messages": [{"role": "user", "content": a.ask}]}

    print(__doc__.split("\n")[0])
    print("  1/4 generating an answer (thinking %s)" % a.effort)
    t0 = time.time()
    d = post(a.url, "/v1/chat/completions", dict(
        first, messages=first["messages"]))
    took = time.time() - t0
    msg = d["choices"][0]["message"]
    answer = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    print("      %.1f s, %d characters of answer, %d of reasoning"
          % (took, len(answer), len(reasoning)))

    print("  2/4 rendering prompt + that answer as history")
    hist = list(first["messages"]) + [dict(msg)] + [
        {"role": "user", "content": "Danke."}]
    rendered = post(a.url, "/apply-template",
                    {"messages": hist, "chat_template_kwargs": kwargs})["prompt"]
    base = post(a.url, "/apply-template",
                {"messages": first["messages"], "chat_template_kwargs": kwargs,
                 "add_generation_prompt": True})["prompt"]

    print("  3/4 where the rendering parts from what was generated")
    produced = base + reasoning + answer
    n = first_difference(rendered, produced)
    print("      common prefix: %d characters" % n)
    print("      base prompt was %d, so the answer starts at %d" % (len(base), len(base)))
    if n >= len(base) + len(reasoning) + len(answer) - 2:
        print("      -> they agree. The cost measured in production is NOT the "
              "rendering; look at the tokeniser or at what the slot kept.")
    else:
        where = "inside the reasoning" if n < len(base) + len(reasoning) else "in the answer text"
        print("      -> they part %d characters into the assistant turn, %s"
              % (n - len(base), where))
        print("      rendered : %r" % rendered[max(0, n - 60):n + 60])
        print("      generated: %r" % produced[max(0, n - 60):n + 60])

    print("  4/4 what it costs in tokens")
    d2 = post(a.url, "/v1/chat/completions",
              {"model": a.model, "max_tokens": 8, "chat_template_kwargs": kwargs,
               "messages": hist})
    tm = d2.get("timings") or {}
    print("      cache_n=%s prompt_n=%s" % (tm.get("cache_n"), tm.get("prompt_n")))
    print("      (cache_n at roughly the size of the FIRST prompt means the "
          "answer was worthless to this turn)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
