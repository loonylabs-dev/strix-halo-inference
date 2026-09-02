#!/usr/bin/env python3
"""restore-determinism — is a restored state the SAME state, byte for byte?

    D=~/.cache/slot-save-cost && mkdir -p $D
    python3 bench/sideserver.py --env setup/env/flashnext.env --port 8081 \\
        --stop "llama-user@$(bash setup/lib/models.sh serving)" --deadline 45 \\
        --extra "--slot-save-path $D/ -cram 0" -- \\
        python3 bench/suites/restore-determinism.py --url http://127.0.0.1:8081 \\
            --dir $D --out $D/restore-determinism.json

WHY THIS EXISTS, when restore-safety already ran green.

bench/reports/2026-09-02_2247_restore-safety-… found the idle cell CLEAN for
flashnext, with and without speculation: after a restore the model still
answers its arithmetic probes. That answers "is it broken", which is the
question the 25.08. incident posed — output degenerating into '////'.

It does not answer the question setup/env/flashnext.env actually raises. Two
upstream reviewers objected to save/restore on #27742 because the QSA indexer
carries state the restore path does not know about; ngxson's wording is
"a context save/load will corrupt it". Corruption of a learned INDEX does not
look like '////'. It looks like fluent, plausible output that picked the wrong
2048 tokens to attend to — and three sums cannot see that, because 391 is 391
however the index was chosen.

SO THIS ASKS THE STRICTER QUESTION: not "does it still work" but "is it the
same". The same prompt is answered from a freshly computed state and from a
restored one, and the two answers are compared byte for byte.

WHAT MAKES THE COMPARISON ADMISSIBLE. Byte equality is only evidence if the
server is byte-deterministic for this prompt in the first place. bench/README
records that it is for fixed prompts (5 of 5 identical at 140 and 1449 tokens)
and that it is NOT once an answer is long enough to hit near-ties in reasoning.
So the run computes the fresh answer TWICE and compares those first. If the two
fresh arms already differ, the instrument cannot resolve anything here and the
run says so instead of reporting a difference against the restored arm.

AND THE PROMPT HAS TO MAKE THE INDEXER WORK. flashnext gives 12 of its 48
layers a learned index that picks 2048 tokens (setup/env/flashnext.env). Below
that budget there is nothing to pick and the mechanism under suspicion never
runs. The context is therefore ~15k tokens of numbered facts (measured:
1200 facts tokenise to 28,986, not the 15,600 a chars/4 estimate suggests —
the numbers tokenise one by one), and the question
asks for one of them from the middle — a needle the index has to find. A wrong
index shows up twice over: as different bytes, and as a wrong number.

Written 02.09.2026.
"""
import argparse, json, os, sys, time, urllib.request

SRV = None
STATE = "restore-determinism.bin"


def req(path, payload=None, t=7200):
    d = json.dumps(payload).encode("utf-8") if payload is not None else None
    r = urllib.request.Request(SRV + path, data=d,
                               headers={"content-type": "application/json"})
    with urllib.request.urlopen(r, timeout=t) as x:
        return json.loads(x.read().decode("utf-8"))


def haystack(n_facts, needle_at, needle_value):
    """Numbered facts, one of which carries the value the question asks for.

    Deterministic text rather than random tokens: the answer has to DEPEND on
    the context, or a wrong index would produce the same reply as a right one
    and the comparison would pass on a broken state.
    """
    lines = []
    for i in range(1, n_facts + 1):
        val = needle_value if i == needle_at else (10000 + (i * 7919) % 80000)
        lines.append("Fact %d: the reference code for unit %d is %d."
                     % (i, i, val))
    return "\n".join(lines)


# The chat markers this model's template uses. /completion takes RAW tokens, so
# the template has to be written here — measured 02.09.2026, first run: without
# it every arm returned an EMPTY answer, the two fresh ones were trivially
# "identical", and the run would have read as a pass on a test that asked the
# model nothing. The suite caught it only because the needle check failed too.
# The context part must stay a strict PREFIX of the full prompt, or the saved
# state is not a prefix of what is asked later and a restore is discarded whole
# (bench/reports/2026-08-29_restore-semantics/).
CHAT_HEAD = "<|im_start|>user\n"
# The empty think block is not cosmetic. WITH thinking the model writes a
# paragraph of reasoning before the answer, and bench/README records what that
# costs an equality test: a reasoning step is full of two-way near-ties, a
# last-bit difference in the logits decides one of them, and the two runs
# diverge from there. Measured 02.09.2026 on this very prompt: one run had the
# two FRESH arms byte-identical, the next run had them differing - so a
# fresh-vs-restored difference could not be told from ordinary variance.
# Suppressing the reasoning leaves a short answer that sits on no near-tie,
# which is what makes byte equality mean anything here.
CHAT_TAIL = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def ask(prompt_tokens, tag, npredict=1):
    t0 = time.time()
    r = req("/completion", {"prompt": prompt_tokens, "n_predict": npredict,
                            "cache_prompt": True, "stream": False,
                            "temperature": 0, "seed": 7})
    tm = r.get("timings") or {}
    out = {"wall_s": round(time.time() - t0, 2), "text": r.get("content") or "",
           "cache_n": tm.get("cache_n"), "prompt_n": tm.get("prompt_n")}
    print("  %-34s %7.2f s   cache_n=%-8s %r"
          % (tag, out["wall_s"], out["cache_n"], out["text"][:60]), flush=True)
    return out


def main():
    global SRV
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--dir", default=os.path.expanduser("~/.cache/slot-save-cost"))
    ap.add_argument("--facts", type=int, default=600,
                    help="numbered facts in the context; 600 is ~15k tokens, "
                         "well past the 2048-token index budget")
    ap.add_argument("--needle", type=int, default=0,
                    help="which fact the question asks for (0 = the middle)")
    ap.add_argument("--predict", type=int, default=64)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    SRV = a.url
    if "8080" in a.url:
        raise SystemExit("refusing: 8080 is production.")

    needle_at = a.needle or (a.facts // 2)
    needle_value = 424242
    text = CHAT_HEAD + haystack(a.facts, needle_at, needle_value)
    question = ("\n\nQuestion: what is the reference code for unit %d? "
                "Answer with the number only." % needle_at) + CHAT_TAIL

    ctx = req("/tokenize", {"content": text})["tokens"]
    full = req("/tokenize", {"content": text + question})["tokens"]
    if full[:len(ctx)] != ctx:
        raise SystemExit("the context is not a prefix of the full prompt — a "
                         "restore of it would be discarded whole and the run "
                         "would measure that instead of the index")
    junk = req("/tokenize", {"content": "unrelated filler about weather"})["tokens"]
    print("context %d tokens, question makes it %d, needle at fact %d = %d\n"
          % (len(ctx), len(full), needle_at, needle_value))
    R = {"url": a.url, "facts": a.facts, "needle_at": needle_at,
         "needle_value": needle_value, "n_ctx_tokens": len(ctx),
         "n_full_tokens": len(full), "cells": {}}

    print("=== 1. build the context and save it")
    ask(ctx, "prefill context", 1)
    d = req("/slots/0?action=save", {"filename": STATE})
    n_saved = d.get("n_saved")
    print("  %-34s n_saved=%s" % ("save", n_saved))
    R["n_saved"] = n_saved

    print("\n=== 2. fresh arm, twice — this is the instrument check")
    ask(junk, "displace")
    a1 = ask(full, "fresh answer 1", a.predict)
    ask(junk, "displace")
    a2 = ask(full, "fresh answer 2", a.predict)
    R["cells"]["fresh_1"], R["cells"]["fresh_2"] = a1, a2

    if not (a1["text"] or "").strip():
        R["verdict"] = "inadmissible: the fresh arm generated nothing"
        print("\n  INADMISSIBLE — the fresh arm returned an EMPTY answer, so")
        print("  the model was never asked anything and two empty strings")
        print("  would compare equal against any state at all. Check that the")
        print("  prompt carries this model's chat markers (CHAT_HEAD/TAIL).")
        if a.out:
            json.dump(R, open(a.out, "w"), indent=1)
        return

    deterministic = a1["text"] == a2["text"]
    print("  the two fresh answers are %s"
          % ("IDENTICAL" if deterministic else "DIFFERENT"))
    if not deterministic:
        R["verdict"] = "inadmissible: server not byte-deterministic here"
        print("\n  INADMISSIBLE — the server is not byte-deterministic for this")
        print("  prompt, so a difference against the restored arm would prove")
        print("  nothing. Shorten --predict or pick a prompt whose answer does")
        print("  not sit on a near-tie; see bench/README on decode variance.")
        if a.out:
            json.dump(R, open(a.out, "w"), indent=1)
        return

    # THE CONTROL THAT DECIDES WHAT A DIFFERENCE MEANS. The fresh arm computes
    # all ~14k tokens in one go; the restored arm computes 26 and takes the rest
    # off disk. Different batch shapes give bit-different logits, and bench/README
    # records that a last-bit difference is enough to send two runs down different
    # near-ties. So a fresh-vs-restored difference has TWO candidate causes, and
    # without this cell they cannot be told apart.
    #
    # This arm has the same SHAPE as the restored one - context already in the
    # slot, only the question computed - but no file anywhere. If it matches the
    # restored arm, the difference is about warm-vs-cold arithmetic. If it matches
    # the fresh arm, the file is what changed the state.
    print("\n=== 2b. warm arm — context in the slot, no file involved")
    ask(junk, "displace")
    ask(ctx, "prefill context again")
    warm = ask(full, "answer while warm", a.predict)
    R["cells"]["warm"] = warm

    print("\n=== 3. restored arm — same question, state from the file")
    ask(junk, "displace")
    r = req("/slots/0?action=restore", {"filename": STATE})
    print("  %-34s n_restored=%s" % ("restore", r.get("n_restored")))
    b = ask(full, "answer after restore", a.predict)
    R["cells"]["restored"] = b
    R["n_restored"] = r.get("n_restored")

    print("\n=== verdict")
    same = a1["text"] == b["text"]
    warm_eq_fresh = warm["text"] == a1["text"]
    warm_eq_rest  = warm["text"] == b["text"]
    correct_fresh = str(needle_value) in (a1["text"] or "")
    correct_rest = str(needle_value) in (b["text"] or "")
    print("  fresh vs fresh      : IDENTICAL   (instrument can resolve)")
    print("  fresh vs warm       : %s" % ("IDENTICAL" if warm_eq_fresh else "DIFFERENT"))
    print("  warm  vs restored   : %s" % ("IDENTICAL" if warm_eq_rest else "DIFFERENT"))
    print("  fresh vs restored   : %s" % ("IDENTICAL" if same else "DIFFERENT"))
    print("  needle found fresh  : %s" % correct_fresh)
    print("  needle found restored: %s" % correct_rest)
    R["verdict"] = {"deterministic": True, "fresh_eq_restored": same,
                    "warm_eq_fresh": warm_eq_fresh, "warm_eq_restored": warm_eq_rest,
                    "needle_fresh": correct_fresh, "needle_restored": correct_rest}

    if same and correct_fresh:
        print("\n  THE RESTORED STATE IS THE SAME STATE. Byte-identical output "
              "from a state read off disk and one computed in place, on a "
              "prompt long enough that the QSA index had to choose. The "
              "#27742 objection does not reach this path on this build.")
    elif same and not correct_fresh:
        print("\n  Identical, but the FRESH arm already missed the needle — so "
              "the prompt does not test what it was meant to. The equality "
              "still says the states match; the index claim is untested.")
    elif not same and warm_eq_rest:
        print("\n  THE DIFFERENCE IS WARM-VS-COLD ARITHMETIC, NOT THE FILE. The "
              "warm arm - context in the slot, no file anywhere - produced the "
              "SAME text as the restored one, and both differ from the arm that "
              "recomputed everything. So what changed the output is how many "
              "tokens were evaluated in this pass, not where the state came "
              "from. The #27742 objection is NOT visible here.")
        for k, v in (("fresh   ", a1["text"]), ("warm    ", warm["text"]),
                     ("restored", b["text"])):
            print("    %s %r" % (k, (v or "")[:160]))
    elif not same and warm_eq_fresh:
        print("\n  THE FILE CHANGES THE STATE. The warm arm computed exactly as "
              "few tokens as the restored one and still matched the FRESH text, "
              "so batch shape is ruled out - what differs is that the state came "
              "off disk. That is the #27742 objection, and it does NOT raise an "
              "error.")
        for k, v in (("fresh   ", a1["text"]), ("warm    ", warm["text"]),
                     ("restored", b["text"])):
            print("    %s %r" % (k, (v or "")[:160]))
    else:
        print("\n  THE RESTORE CHANGES THE OUTPUT, and the warm control matches "
              "NEITHER arm - so this run cannot attribute it. Same prompt, same seed, "
              "same temperature, and a different answer depending on whether "
              "the state came off disk. That is the #27742 objection showing "
              "up, and it does NOT raise an error — exactly the failure mode "
              "flashnext.env removed --slot-save-path for.")
        for k, v in (("fresh   ", a1["text"]), ("restored", b["text"])):
            print("    %s %r" % (k, (v or "")[:200]))

    if a.out:
        json.dump(R, open(a.out, "w"), indent=1)
        print("\n  rows -> %s" % a.out)


if __name__ == "__main__":
    main()
