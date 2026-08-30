#!/usr/bin/env python3
"""What does the slot hold that the re-rendered history does not?

The companion to bench/suites/answer-reuse.py, and the question that survived
it. That suite established the COST and, on 30.08.2026, the CAUSE: a hybrid
model cannot be trimmed to a common prefix, so llama.cpp falls back to a
context checkpoint — and checkpoints exist only for PROMPTS, never for
answers. The fallback therefore lands before the previous answer and the turn
pays for it again.

What it did not establish is the TRIGGER. Something makes the follow-up's
tokens part from the slot's at the very end of the assistant turn. With
LLAMA_SERVER_SLOTS_DEBUG=1 the server printed two tokens on each side:

    old: ...  | <|endoftext|><|im_start|>      248044  248045
    new: ...  | <|im_start|>user               248045     846

That switch also makes /slots serve complete prompts, which docs/SECURITY.md
calls the worst finding this project has had. THIS SUITE NEEDS NEITHER IT NOR
A RESTART. It measures two separate things and keeps them apart, because they
disagree and the disagreement IS the finding:

  THE RENDERING, from /apply-template and /tokenize. Does re-rendering the
  history reproduce what was generated? Measured 30.08.2026 over 13 rounds:
  YES, every time — the reconstruction is a COMPLETE prefix of the follow-up.
  Nothing is lost, reordered or re-delimited on the way back in.

  THE SLOT, from f_keep in the journal. Is the server's own copy a prefix of
  the new prompt? Measured the same evening: NO — it carries 2 tokens more,
  in 9 of 9 rounds across all three routes.

So the tail is the SERVER's and it is invisible to the client. That is not a
figure of speech: `/completion` with `return_tokens` returns a sequence
ending cleanly at <|im_end|>, and the usage counts agree with it. Only
f_keep shows the two tokens behind it. A measurement that trusts the tokens
the server hands back concludes TAIL 0 and is wrong — this file did, for one
round of the evening, before f_keep contradicted it.

    python3 bench/suites/slot-tail.py --repeat 4
    python3 bench/suites/slot-tail.py --repeat 4 --effort off
    python3 bench/suites/slot-tail.py --repeat 3 --route openai

IT TAKES THE ONE SLOT for one short generation per round, and it reads /slots
BETWEEN requests — never during one. A round whose slot is busy, or whose
id_task moved under it, is DISCARDED rather than reported: a slot state read
mid-prefill looking like a finished prompt is a mistake this project has
already made once, and it cost an evening.
"""
import argparse, json, re, subprocess, sys, time, urllib.request

DEFAULT_URL = "http://127.0.0.1:8080"

# WHY THE JOURNAL AND NOT ARITHMETIC ON TEXT. The obvious way to size the tail
# is to tokenize the reconstruction and subtract, and it is off by one or two:
# re-tokenizing a text does not have to reproduce the tokens that GENERATED it,
# because the merges are chosen over the whole string rather than one token at
# a time. Measured 30.08.2026 — three rounds gave overhangs of +4, +4 and -2
# against a slot whose real tail was 2. A negative overhang is the instrument,
# not the server.
#
# llama.cpp has already done the comparison exactly, on the tokens themselves,
# and prints it at INFO level in EVERY server — no debug switch, no restart:
#
#   slot get_availabl: id 0 | task -1 | selected slot by LCP similarity,
#                      f_sim_best = 0.956 (> 0.100 thold), f_keep = 0.993
#
#   f_sim_best = lcp / len(new prompt)      lcp  = f_sim_best * len(new)
#   f_keep     = lcp / len(slot)            tail = len(slot) - lcp
#
# Three decimals is plenty at these lengths: 0.0005 * 250 tokens is an eighth
# of a token, so the rounding cannot move the answer by one.
SEL = re.compile(r"f_sim_best = ([0-9.]+).*f_keep = ([0-9.]+)")


def post(url, path, body, timeout=1800):
    req = urllib.request.Request(
        url + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def get(url, path, timeout=30):
    with urllib.request.urlopen(url + path, timeout=timeout) as r:
        return json.loads(r.read().decode())


def tokenize(url, text):
    return post(url, "/tokenize", {"content": text})["tokens"]


def piece(url, tok):
    """One token as text. /detokenize, so the name comes from the server's own
    vocabulary rather than from a table written here that could drift."""
    try:
        return post(url, "/detokenize", {"tokens": [tok]})["content"]
    except Exception:
        return "?"


def common_prefix(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def journal_since(unit, since):
    """The unit's log lines since a timestamp, or None if it cannot be read.

    None is not a failure of the measurement — it is one column of it. The
    reconstruction half stands on its own and says so."""
    try:
        p = subprocess.run(
            ["journalctl", "--user", "-u", unit, "--since", since,
             "--no-pager", "-o", "cat"],
            capture_output=True, text=True, timeout=30)
        return p.stdout if p.returncode == 0 else None
    except Exception:
        return None


def last_selection(unit, since):
    """(f_sim_best, f_keep) of the most recent LCP slot selection, or None."""
    out = journal_since(unit, since)
    if out is None:
        return None
    hits = SEL.findall(out)
    return (float(hits[-1][0]), float(hits[-1][1])) if hits else None


def read_slot(url):
    """(n_prompt_tokens, id_task) of slot 0, or (None, None) while it is busy.

    Busy is not an error and not a retry: it means another caller owns the
    slot, and every number that could be read in that state belongs to their
    request rather than to this one."""
    slots = get(url, "/slots")
    if not slots:
        return None, None
    s = slots[0]
    if s.get("is_processing"):
        return None, None
    return s.get("n_prompt_tokens"), s.get("id_task")


def one_round(a, url):
    kwargs = {"enable_thinking": a.effort != "off"}
    if a.effort != "off":
        kwargs["reasoning_effort"] = a.effort

    body = {"model": a.model, "max_tokens": 900, "chat_template_kwargs": kwargs,
            "messages": [{"role": "user", "content": a.ask}]}
    if a.with_tool:
        body["tools"] = [{"name": "note", "description": "Write a short note.",
                          "input_schema": {"type": "object",
                                           "properties": {"text": {"type": "string"}},
                                           "required": ["text"]}}]
        body["messages"][0]["content"] += " Halte das Ergebnis mit `note` fest."

    t0 = time.time()
    if a.route == "openai":
        # The OpenAI route answers with a flat message, and the reasoning
        # arrives beside the text as `reasoning_content` rather than as
        # blocks. Normalised to blocks here so the rest of the file has one
        # shape to handle.
        oai = dict(body)
        oai.pop("tools", None)
        raw = post(url, "/v1/chat/completions", oai)
        m = raw["choices"][0]["message"]
        d = {"content": [], "usage": raw.get("usage") or {},
             "stop_reason": raw["choices"][0].get("finish_reason")}
        if m.get("reasoning_content"):
            d["content"].append({"type": "thinking",
                                 "thinking": m["reasoning_content"]})
        d["content"].append({"type": "text", "text": m.get("content") or ""})
    else:
        d = post(url, "/v1/messages", body)
    took = time.time() - t0

    # THE SLOT, read immediately and before anything else touches it.
    slot_len, id_task = read_slot(url)

    blocks = d.get("content") or []
    reasoning = "".join(b.get("thinking", "") for b in blocks
                        if b.get("type") == "thinking")
    answer = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    usage = d.get("usage") or {}
    stop = d.get("stop_reason")

    # The base prompt, as the server renders it for GENERATION.
    base = post(url, "/apply-template",
                {"messages": body["messages"], "chat_template_kwargs": kwargs,
                 "add_generation_prompt": True})["prompt"]

    # What the slot should hold. The reasoning is NOT prefixed with <think>:
    # with enable_thinking the generation prompt already ends in one, so
    # adding it again produces `<think>\n<think>\n` and points the divergence
    # at the base prompt — a bug in the reconstruction that reads exactly like
    # a finding. rstrip, because the template rstrips too.
    if a.effort == "off":
        produced = base + answer
    else:
        produced = base + reasoning.rstrip() + "\n</think>\n\n" + answer

    # The history as the NEXT turn will send it. /apply-template refuses
    # Anthropic blocks ("unsupported content[].type" for `thinking`), so the
    # assistant turn goes in the OpenAI shape it does accept.
    assistant = {"role": "assistant", "content": answer}
    if reasoning:
        assistant["reasoning_content"] = reasoning
    hist = list(body["messages"]) + [assistant] + [
        {"role": "user", "content": "Danke."}]
    rendered = post(url, "/apply-template",
                    {"messages": hist, "chat_template_kwargs": kwargs})["prompt"]

    t_prod = tokenize(url, produced)
    t_next = tokenize(url, rendered)
    t_base = tokenize(url, base)

    lcp = common_prefix(t_prod, t_next)

    # The slot was read before the three template calls, and none of them
    # touches it — but say so out loud rather than assume it.
    slot_len2, id_task2 = read_slot(url)
    moved = (slot_len2 != slot_len) or (id_task2 != id_task)

    # NOW send the follow-up for real, and let llama.cpp compare the tokens.
    # A tool_use MUST be answered by a tool_result or the conversation is
    # malformed and the block is silently dropped — a follow-up that is not
    # what a client would send measures its own mistake.
    since = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - 1))
    follow = [{"type": "text", "text": "Danke."}]
    calls = [b for b in blocks if b.get("type") == "tool_use"]
    if calls:
        follow = [{"type": "tool_result", "tool_use_id": c.get("id"),
                   "content": "ok"} for c in calls] + follow
    follow_body = dict(body)
    follow_body["max_tokens"] = 8
    follow_body["messages"] = list(body["messages"]) + [
        {"role": "assistant", "content": blocks}, {"role": "user", "content": follow}]
    if a.route == "openai":
        oai = dict(follow_body)
        oai.pop("tools", None)
        oai["messages"] = list(body["messages"]) + [assistant] + [
            {"role": "user", "content": "Danke."}]
        u2 = (post(url, "/v1/chat/completions", oai).get("usage") or {})
        reused = ((u2.get("prompt_tokens_details") or {}).get("cached_tokens")
                  or u2.get("cache_read_input_tokens") or 0)
    else:
        d2 = post(url, "/v1/messages", follow_body)
        reused = (d2.get("usage") or {}).get("cache_read_input_tokens") or 0

    # THE TAIL, and it deliberately does NOT go through t_next. f_keep is
    # lcp/len(slot) — llama.cpp's own comparison of the tokens themselves —
    # and the slot's length comes from /slots. Neither number passes through a
    # rendering of this file's making, which matters because one of them did:
    # with --with-tool the follow-up carries the tool definitions at its FRONT
    # and /apply-template was asked without them, so t_next was ~430 tokens
    # short and every figure derived from it was wrong (measured 30.08.2026 —
    # it reported a 182-token slot that /slots said was 602). The tail survives
    # that mistake because it never touches t_next.
    sel = last_selection(a.unit, since)
    lcp_exact = tail = None
    if sel and slot_len:
        f_keep = sel[1]
        tail = round(slot_len * (1.0 - f_keep))
        lcp_exact = slot_len - tail

    # A TAIL IS SMALL BY CONSTRUCTION — it is the last token or two of a turn.
    # A two-digit one means the slot held something else entirely when the
    # follow-up arrived, which with -np 1 is what ordinary traffic looks like:
    # one round on 30.08.2026 came back f_keep 0.709, TAIL 69, between two
    # rounds of 2. Such a round measures the interruption, not the tail, so it
    # is reported and then left out of the statistic rather than averaged in.
    # THE BOUND IS A HEURISTIC, not a derived figure: every tail measured so
    # far is 0, 1 or 2, and 16 is simply far above them and far below a
    # foreign prompt.
    foreign = tail is not None and tail > 16

    return {
        "took": took, "stop": stop, "usage": usage,
        "base": len(t_base), "produced": len(t_prod), "next": len(t_next),
        "slot": slot_len, "id_task": id_task, "moved": moved,
        "overhang": None if slot_len is None else slot_len - len(t_prod),
        "lcp": lcp,
        "tail_prod": t_prod[lcp:lcp + 3],
        "tail_next": t_next[lcp:lcp + 3],
        "sel": sel, "lcp_exact": lcp_exact,
        "tail_exact": tail, "reused": reused, "with_tool": a.with_tool,
        "foreign": foreign,
        "kept": slot_len is not None and reused >= slot_len - 40,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default="qwen38")
    ap.add_argument("--effort", default="low",
                    help="thinking level, or `off` for none at all. The "
                         "measured losses were all on a thinking turn, and "
                         "`off` is the control.")
    ap.add_argument("--ask", default="Nenne drei Primzahlen und begruende kurz.")
    ap.add_argument("--repeat", type=int, default=1,
                    help="SINGLE SHOTS ARE WORTHLESS here: the defect this "
                         "measures is not deterministic, and one round that "
                         "goes either way says nothing.")
    ap.add_argument("--with-tool", action="store_true",
                    help="offer a tool, so the answer ends in a tool_use block")
    ap.add_argument("--route", default="anthropic",
                    choices=["anthropic", "openai"],
                    help="which chat route generates and replays. Measured "
                         "30.08.2026: the tail is 2 on BOTH, and on the raw "
                         "/completion route as well, so it is not the "
                         "Anthropic conversion and not the chat template.")
    ap.add_argument("--unit", default="llama-user@qwen38",
                    help="the systemd user unit whose journal carries the "
                         "`selected slot by LCP similarity` line. That line is "
                         "INFO level and needs no debug switch. Without a "
                         "readable journal the exact tail column stays empty "
                         "and the rest of the measurement still stands.")
    a = ap.parse_args()

    print(__doc__.split("\n")[0])
    print("  model %s, route %s, thinking %s, tools %s, %d round(s)"
          % (a.model, a.route, a.effort, "yes" if a.with_tool else "no", a.repeat))

    rows = []
    for i in range(a.repeat):
        r = one_round(a, a.url)
        rows.append(r)
        print("  --- round %d: %.1f s, stop=%s, usage=%s"
              % (i + 1, r["took"], r["stop"], json.dumps(r["usage"])))
        if r["slot"] is None:
            print("      DISCARDED: the slot was busy — another caller owns it")
            continue
        if r["moved"]:
            print("      DISCARDED: the slot moved under the measurement")
            continue
        if a.with_tool:
            print("      slot %d; the rendering half is NOT measured with "
                  "--with-tool: /apply-template is asked without the tool "
                  "definitions the real follow-up carries at its front"
                  % r["slot"])
        else:
            print("      base %d, reconstruction %d, slot %d (re-tokenized, so "
                  "+-2)" % (r["base"], r["produced"], r["slot"]))
            print("      follow-up %d tokens, parts from the reconstruction at %d"
                  % (r["next"], r["lcp"]))
            print("      follow-up continues: %s"
                  % "  ".join("%d %r" % (t, piece(a.url, t)) for t in r["tail_next"]))
        if r["sel"]:
            print("      llama.cpp: f_keep = %.3f of a %d-token slot -> TAIL %d, "
                  "reused %d  %s%s"
                  % (r["sel"][1], r["slot"], r["tail_exact"], r["reused"],
                     "KEPT" if r["kept"] else "lost",
                     "   <- FOREIGN: the slot held another prompt, not a tail"
                     if r["foreign"] else ""))
        else:
            print("      llama.cpp: journal not readable — no exact tail")

    good = [r for r in rows if r["slot"] is not None and not r["moved"]]
    print("\n  %6s %6s %6s %8s %6s  %s"
          % ("base", "slot", "lcp", "f_keep", "TAIL", "the answer"))
    for r in good:
        print("  %6d %6d %6s %8s %6s  %s"
              % (r["base"], r["slot"],
                 "-" if r["lcp_exact"] is None else r["lcp_exact"],
                 "-" if not r["sel"] else "%.3f" % r["sel"][1],
                 "-" if r["tail_exact"] is None else r["tail_exact"],
                 "foreign slot" if r["foreign"] else
                 ("KEPT" if r["kept"] else "lost")))

    if not good:
        print("\n  NOTHING MEASURED — every round was discarded.")
        return 1

    # THE RENDERING VERDICT, and it needs no journal. If the reconstruction is
    # a COMPLETE prefix of the follow-up, then re-rendering the history loses
    # nothing: everything the first turn produced is still there, in order, at
    # the same position. Any f_keep below 1.000 after that belongs to the slot.
    if a.with_tool:
        print("\n  the rendering half is not measured with --with-tool")
    else:
        whole = [r for r in good if r["lcp"] == r["produced"]]
        print("\n  the reconstruction is a complete prefix of the follow-up in "
              "%d of %d rounds" % (len(whole), len(good)))

    usable = [r for r in good
              if r["tail_exact"] is not None and not r["foreign"]]
    n_foreign = len([r for r in good if r["foreign"]])
    if n_foreign:
        print("  %d round(s) left out: the slot held a foreign prompt, so their "
              "f_keep is not a tail" % n_foreign)
    tails = sorted({r["tail_exact"] for r in usable})
    if not tails:
        print("  no exact tail measured — the journal said nothing")
        return 0
    print("  the slot's tail beyond that prefix, over %d rounds: %s"
          % (len(usable), ", ".join(str(t) for t in tails)))
    if tails == [0]:
        print("  -> the slot is EXACTLY what was rendered. The trigger is not here.")
    else:
        print("  -> the slot carries %s token(s) the rendered history does not, "
              "and a client sending messages cannot be asked to match them."
              % "/".join(str(t) for t in tails))
    return 0


if __name__ == "__main__":
    sys.exit(main())
