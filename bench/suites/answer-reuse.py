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


def one_round(a, quiet=False):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default="qwen38")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--ask", default="Nenne drei Primzahlen und begruende kurz.")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run the pair N times. SINGLE SHOTS OF THIS ARE "
                         "WORTHLESS: two runs of the same configuration gave "
                         "opposite answers on 30.08.2026, because the watchdog "
                         "probe can take the slot between the two requests. "
                         "The pattern across repetitions is the measurement, "
                         "not any one number.")
    ap.add_argument("--with-tool", action="store_true",
                    help="offer a tool, so the answer ends in a tool_use block. "
                         "EVERY answer lost in production had one, and the "
                         "first run of this suite — without a tool — reused 525 "
                         "of 526 tokens. That contrast is the experiment.")
    a = ap.parse_args()

    # THE ANTHROPIC ROUTE, because that is where the loss was measured. The
    # OpenAI route returns `reasoning_content` as a flat string; Claude Code
    # gets `thinking` BLOCKS and sends them back as blocks, and the whole
    # question is what happens to those on the way back in.
    kwargs = {"enable_thinking": True, "reasoning_effort": a.effort}
    first = {"model": a.model, "max_tokens": 900, "chat_template_kwargs": kwargs,
             "messages": [{"role": "user", "content": a.ask}]}
    if a.with_tool:
        first["tools"] = [{"name": "note", "description": "Write a short note.",
                           "input_schema": {"type": "object",
                                            "properties": {"text": {"type": "string"}},
                                            "required": ["text"]}}]
        first["messages"][0]["content"] += " Halte das Ergebnis mit `note` fest."

    if not quiet:
        print("  1/4 generating an answer (thinking %s)" % a.effort)
    t0 = time.time()
    d = post(a.url, "/v1/messages", first)
    took = time.time() - t0
    blocks = d.get("content") or []
    reasoning = "".join(b.get("thinking", "") for b in blocks
                        if b.get("type") == "thinking")
    answer = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    print("      %.1f s, blocks: %s"
          % (took, "  ".join("%s:%d" % (b.get("type"),
                                        len(json.dumps(b, ensure_ascii=False)))
                             for b in blocks)))
    print("      usage: %s" % json.dumps(d.get("usage") or {}))
    print("      %d characters of thinking, %d of answer text"
          % (len(reasoning), len(answer)))

    print("  2/4 rendering prompt + that answer as history")
    # /apply-template REFUSES Anthropic blocks — "unsupported content[].type"
    # for `thinking`. It takes the OpenAI shape, where the reasoning is a flat
    # `reasoning_content` beside the text, and wraps it itself. Measured
    # 30.08.2026 against the running server:
    #
    #   without reasoning  …assistant\n<think>\n\n</think>\n\nHi
    #   with it            …assistant\n<think>\nDENKEN\n</think>\n\nHi
    #
    # So the wrapper is the template's, not the model's, and an EMPTY think
    # block is emitted even when there was no reasoning at all.
    assistant = {"role": "assistant", "content": answer}
    if reasoning:
        assistant["reasoning_content"] = reasoning
    hist_oai = list(first["messages"]) + [assistant] + [
        {"role": "user", "content": "Danke."}]
    # Step 4 replays the ANTHROPIC shape — the blocks exactly as the route
    # returned them, which is what Claude Code sends back.
    # A tool_use MUST be answered by a tool_result, or the conversation is
    # malformed and the block is dropped — measured 30.08.2026: the follow-up
    # prompt came back SHORTER than the first (269 against 312 tokens), which
    # is the tool call vanishing, not a cache miss. A test that sends "Danke."
    # after a tool call measures its own mistake.
    follow = [{"type": "text", "text": "Danke."}]
    calls = [b for b in blocks if b.get("type") == "tool_use"]
    if calls:
        follow = [{"type": "tool_result", "tool_use_id": c.get("id"),
                   "content": "ok"} for c in calls] + follow
    hist_anthropic = list(first["messages"]) + [
        {"role": "assistant", "content": blocks}] + [
        {"role": "user", "content": follow}]
    rendered = post(a.url, "/apply-template",
                    {"messages": hist_oai, "chat_template_kwargs": kwargs})["prompt"]
    base = post(a.url, "/apply-template",
                {"messages": first["messages"], "chat_template_kwargs": kwargs,
                 "add_generation_prompt": True})["prompt"]

    print("  3/4 where the rendering parts from what was generated")
    # What the model actually produced. NOT prefixed with <think> here: with
    # enable_thinking the GENERATION PROMPT already ends in
    # `<|im_start|>assistant\n<think>\n`, so the model's first token comes
    # after it. Adding it again produced `<think>\n<think>\n` and pointed the
    # first difference at the base prompt — a bug in the reconstruction that
    # looked exactly like a finding.
    # rstrip, because the template does: measured 30.08.2026, a reasoning
    # string ending in a newline renders as `…\n</think>`, not `…\n\n</think>`.
    # Without this the reconstruction reports a divergence that step 4 then
    # contradicts — and a step 3 that disagrees with the measurement beside it
    # is worse than no step 3.
    produced = base + reasoning.rstrip() + "\n</think>\n\n" + answer
    n = first_difference(rendered, produced)
    print("      base prompt %d characters, rendered history %d, reconstruction %d"
          % (len(base), len(rendered), len(produced)))
    print("      common prefix: %d" % n)
    if n <= len(base):
        print("      -> they part BEFORE the answer even starts (%d into the "
              "base prompt). The history rendering changed something ahead of "
              "the assistant turn." % n)
    else:
        print("      -> they part %d characters into the assistant turn"
              % (n - len(base)))
    print("      rendered : %r" % rendered[max(0, n - 70):n + 70])
    print("      generated: %r" % produced[max(0, n - 70):n + 70])

    print("  4/4 what it costs in tokens")
    # THE FOLLOW-UP IS THE FIRST REQUEST WITH MESSAGES APPENDED, built by
    # copying it. Assembling it by hand dropped `tools` on 30.08.2026, which
    # changes the prompt at its FRONT — reuse came back as 30 tokens and looked
    # like the finding this suite is hunting. Three bugs in this file have now
    # had the same shape: a follow-up that is not what a client would send.
    follow_body = dict(first)
    follow_body["max_tokens"] = 8
    follow_body["messages"] = hist_anthropic
    d2 = post(a.url, "/v1/messages", follow_body)
    u = d2.get("usage") or {}
    print("      usage: %s" % json.dumps(u))
    generated = (d.get("usage") or {}).get("output_tokens")
    print("      the first answer was %s tokens; if the reuse below is about "
          "the size of the FIRST prompt, none of them helped" % generated)
    first_prompt = ((d.get("usage") or {}).get("cache_read_input_tokens", 0)
                    + (d.get("usage") or {}).get("input_tokens", 0))
    slot = first_prompt + (generated or 0)
    got = u.get("cache_read_input_tokens") or 0
    return {"first_prompt": first_prompt, "generated": generated or 0,
            "slot": slot, "reused": got,
            "kept_the_answer": got >= slot - 40}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default="qwen38")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--ask", default="Nenne drei Primzahlen und begruende kurz.")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--with-tool", action="store_true")
    a = ap.parse_args()
    print(__doc__.split("\n")[0])
    print("  tools: %s, thinking %s, %d round(s)"
          % ("yes" if a.with_tool else "no", a.effort, a.repeat))
    rows = []
    for i in range(a.repeat):
        print("  --- round %d" % (i + 1))
        rows.append(one_round(a, quiet=(i > 0)))
    print("\n  %8s %10s %8s %10s  %s"
          % ("prompt", "generated", "slot", "reused", "the answer"))
    for r in rows:
        print("  %8d %10d %8d %10d  %s"
              % (r["first_prompt"], r["generated"], r["slot"], r["reused"],
                 "KEPT" if r["kept_the_answer"] else "lost"))
    kept = sum(1 for r in rows if r["kept_the_answer"])
    print("\n  the answer survived %d of %d rounds" % (kept, len(rows)))
    if 0 < kept < len(rows):
        print("  -> NOT DETERMINISTIC. Something between the two requests "
              "decides it; the watchdog probe is the first suspect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
