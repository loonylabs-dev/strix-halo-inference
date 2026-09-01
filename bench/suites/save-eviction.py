#!/usr/bin/env python3
"""save-eviction — does saving one prefix throw another out of the slot?

    AUTO_SAVE=0 in ~/.config/llm-gateway.env, restart llm-gateway, then:
    python3 bench/suites/save-eviction.py --label controlled

It does. Measured 28.08.2026, qwen38, a 24-tool synthetic prefix:

    A, warm                                0.7 s      4 tokens evaluated
    save of B (a DIFFERENT prefix)        12.7 s
    A again                               13.6 s   1950 tokens evaluated
    save of A (the SAME prefix)           12.8 s
    A again                                1.5 s

19x, and the token counts say why: after saving B the server had to evaluate
1950 tokens it had already had.

The mechanism, and why it is not obvious
----------------------------------------
To save a prefix, prewarm has to put that prefix INTO the slot — that is what
a slot save is. With `-np 1` there is one slot, so saving B replaces A.

That looks harmless, because the gateway saves a prefix straight after the
cold request that created it: by then the slot holds exactly that prefix and
saving it changes nothing. The measurement above shows the harmless case as
step 5-6.

What makes it bite is that the save is ASYNCHRONOUS —
`asyncio.create_task(auto_save(...))` in gateway.py — and slow. Measured on
this machine: **101.9 s** for one automatic save. The user does not wait 102
seconds between requests. So the sequence that actually happens is:

    request A, cold          -> save of A queued, runs in the background
    request B, cold          -> B is prefilled into the slot
    ...save of A completes   -> A is put back into the slot, B is gone
    request B again          -> reported WARM by the gateway, and cold in fact

The gateway's warm/cold verdict is its own bookkeeping: it means "I have seen
this prefix", not "llama.cpp still holds it". Nothing lies; the two simply
answer different questions, and only one of them is what the clock measures.

Why this suite exists rather than a paragraph
---------------------------------------------
The first attempt to explain a 19x slowdown blamed the save and was WRONG in
its test: it saved the prefix that was already in the slot, saw no eviction,
and concluded the save was innocent. The difference between saving the same
prefix and a different one is the whole finding, so both are steps here.
"""
import argparse, json, os, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tools"))
import synthetic as SYN                                       # noqa: E402

# The INSTALLED path, not REPO/tools, and unlike every other suite here that
# is deliberate: the gateway spawns exactly this file (gateway.py, auto_save)
# and the defect being reproduced is what that spawn does to the serving slot.
# Measuring the repo copy would measure a different process than production
# runs. It is a symlink into the repo, so the code is the same either way —
# only the path the gateway uses is reproduced.
#
# Resolved rather than written out, the way gateway.py resolves it: the hard
# path this used to carry (~/.claude/bin) survived the 09/2026 move by three
# weeks, and it would have failed at the worst possible moment — after
# production has been stopped for the measurement.
#
# Falling back to the repo copy keeps the suite runnable uninstalled, but it
# is then no longer reproducing the spawn production performs. That changes
# what the numbers mean, so it says so instead of quietly measuring something
# else.
PREWARM = os.path.expanduser("~/.local/lib/llm-stack/prewarm.py")
if not os.path.exists(PREWARM):
    PREWARM = os.path.join(REPO, "tools", "prewarm.py")
    print("NOTE: %s is not installed — running the repo copy, which is NOT "
          "the file the gateway spawns" % "~/.local/lib/llm-stack/prewarm.py",
          file=sys.stderr)


def ask(url, body, model, timeout=900):
    b = json.loads(json.dumps(body))
    b.update({"model": model, "max_tokens": 8, "stream": False})
    req = urllib.request.Request(
        url + "/v1/messages", data=json.dumps(b).encode(),
        headers={"Content-Type": "application/json",
                 "anthropic-version": "2023-06-01"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as x:
        x.read()
    return time.time() - t0


def save(body, name, model, workdir):
    """Exactly what the gateway's auto_save does, synchronously."""
    tmp = os.path.join(workdir, name + ".json")
    b = json.loads(json.dumps(body))
    b["model"] = model
    with open(tmp, "w") as fh:
        json.dump(b, fh)
    t0 = time.time()
    p = subprocess.run([sys.executable, PREWARM, "save", "--body", tmp,
                        "--name", name, "--gateway-id", name,
                        "--dialect", "anthropic"],
                       capture_output=True, text=True)
    return time.time() - t0, p.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--model", default="qwen38")
    ap.add_argument("--tools", type=int, default=24)
    ap.add_argument("--label", default="unlabelled")
    ap.add_argument("--out")
    a = ap.parse_args()

    wd = os.path.join("/tmp", "save-eviction-%d" % os.getpid())
    os.makedirs(wd, exist_ok=True)
    A = SYN.body(project="/tmp/evictA", n_tools=a.tools, question="Say alpha.")
    B = SYN.body(project="/tmp/evictB", n_tools=a.tools, question="Say beta.")

    rows = []

    def step(name, seconds, extra=""):
        rows.append({"step": name, "seconds": round(seconds, 2)})
        print("  %-38s %7.1f s %s" % (name, seconds, extra))

    print("  %-38s %9s" % ("step", "seconds"))
    print("  " + "-" * 58)
    step("1) A, cold", ask(a.url, A, a.model))
    warm = ask(a.url, A, a.model)
    step("2) A again — the warm reference", warm)
    d, rc = save(B, "evict-b", a.model, wd)
    step("3) save of B (a DIFFERENT prefix)", d, "rc=%d" % rc)
    after_other = ask(a.url, A, a.model)
    step("4) A again — evicted?", after_other)
    d, rc = save(A, "evict-a", a.model, wd)
    step("5) save of A (the SAME prefix)", d, "rc=%d" % rc)
    after_same = ask(a.url, A, a.model)
    step("6) A again — evicted?", after_same)

    ratio_other = after_other / warm if warm else 0
    ratio_same = after_same / warm if warm else 0
    print("\n  after saving a DIFFERENT prefix: %5.1fx the warm time" % ratio_other)
    print("  after saving the SAME prefix:    %5.1fx the warm time" % ratio_same)
    print("\n  The first number is the finding. The second is the control that")
    print("  the first attempt at this measurement ran BY ITSELF, which is how")
    print("  it concluded the save was innocent.")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump({"label": a.label, "steps": rows,
                       "ratio_after_other": round(ratio_other, 2),
                       "ratio_after_same": round(ratio_same, 2)}, fh, indent=1)
        print("\n  written: %s" % a.out)


if __name__ == "__main__":
    main()
