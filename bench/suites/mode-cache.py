#!/usr/bin/env python3
"""mode-cache — does a thinking mode get its own prompt cache, and is the
gateway's warm/cold verdict TRUE?

    python3 bench/suites/mode-cache.py --label after
    python3 bench/suites/mode-cache.py --url http://127.0.0.1:8091 --label before

Why this exists
---------------
Until 28.08.2026 the prefix id was `system_head + tools` and nothing else, and
the gateway wrote `chat_template_kwargs` into the body AFTER computing it. So
every thinking mode of one model shared ONE cache key while rendering
different prompts. Measured against the served Qwen template, one message and
one tool:

    thinking off        sha 1ad7792b   1090 chars
    reasoning low       sha 938681af   1217 chars
    reasoning medium    sha e3ca72a6

They diverge at CHARACTER 19 — the template puts `Reasoning effort is set to
…` at the very front, before the tools. So a request in one mode could be
answered "warm" from a slot prefilled in another, diverge at token ~5, and be
re-prefilled in full while the gateway logged RESTORED and counted it warm.
The warm percentages this repo reasons from were measuring the wrong thing.

What is asserted here is therefore not "is it fast" but a HONESTY property:

    reported warm  <=>  actually fast

A label that says warm while the wall clock says cold is the defect. That is
checkable without knowing what "fast" means in absolute terms, because the
same prefix in the same mode gives the reference.

Four conditions, each isolating one claim
-----------------------------------------
    same-mode     the same prefix twice in one mode. Must be warm and fast —
                  the control. If this fails nothing else means anything.
    mode-switch   the same prefix in a DIFFERENT mode. Must be COLD, and must
                  take cold time. Before 28.08. it was reported warm and took
                  cold time anyway, which is the bug in one line.
    stale-name    a model name that no longer exists (`<alias>-think`, from
                  before the vocabulary was fixed). Falls through to the bare
                  alias, so it must share the BARE alias's prefix id — not get
                  one of its own, and not error.
    back-again    return to the first mode. Must be warm again: the modes must
                  not evict each other, which they would if they shared a key.

Talks through the GATEWAY, deliberately — the prefix id and the warm/cold
verdict are its work, and they are the subject.
"""
import argparse, json, os, statistics, sys, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tools"))
import synthetic as SYN                                       # noqa: E402


def ask(url, model, body, timeout=600):
    """(seconds, ok). The answer's content does not matter here; the clock
    does."""
    b = json.loads(json.dumps(body))
    b["model"] = model
    b["max_tokens"] = 24
    b["stream"] = False
    req = urllib.request.Request(
        url + "/v1/messages", data=json.dumps(b).encode(),
        headers={"Content-Type": "application/json",
                 "anthropic-version": "2023-06-01"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as x:
            x.read()
        return time.time() - t0, True
    except urllib.error.HTTPError as e:
        return time.time() - t0, "HTTP %s" % e.code
    except Exception as e:
        return time.time() - t0, repr(e)


def prefixes(url):
    """The gateway's own bookkeeping: {prefix id: {...}}. Its warm/cold verdict
    comes from here, so reading it is reading the thing under test."""
    try:
        with urllib.request.urlopen(url + "/gateway/status", timeout=20) as x:
            return json.load(x)
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--alias", default="qwen38")
    ap.add_argument("--mode-a", default="low")
    ap.add_argument("--mode-b", default="medium")
    ap.add_argument("--tools", type=int, default=24,
                    help="a big prefix, so cold and warm are far apart")
    ap.add_argument("--project", default="/tmp/mode-cache",
                    help="what the synthetic body calls itself. It decides the "
                         "prefix ids, so a value used by an EARLIER run can be "
                         "answered from the saved-prefix store on disk instead "
                         "of being prefilled — measured 28.08.2026: the first "
                         "step restored 14957 tokens from a .bin written eight "
                         "hours before, was reported warm, and took 75.3 s "
                         "anyway. Pass a fresh value for a run that owes "
                         "nothing to what is lying around.")
    ap.add_argument("--label", default="unlabelled")
    ap.add_argument("--out")
    a = ap.parse_args()

    body = SYN.body(project=a.project, n_tools=a.tools,
                    question="Say alpha.")
    A = "%s-%s" % (a.alias, a.mode_a)
    B = "%s-%s" % (a.alias, a.mode_b)
    STALE = "%s-think" % a.alias          # a name from before the vocabulary

    steps = [("same-mode  1st  %s" % A, A),
             ("same-mode  2nd  %s" % A, A),
             ("mode-switch     %s" % B, B),
             ("mode-switch 2nd %s" % B, B),
             ("stale-name      %s" % STALE, STALE),
             ("bare            %s" % a.alias, a.alias),
             ("back-again      %s" % A, A)]

    print("  %-28s %8s  %s" % ("step", "seconds", "result"))
    print("  " + "-" * 60)
    out = []
    for name, model in steps:
        secs, ok = ask(a.url, model, body)
        out.append({"step": name.strip(), "model": model,
                    "seconds": round(secs, 2), "ok": ok is True})
        print("  %-28s %8.2f  %s" % (name, secs, "ok" if ok is True else ok))

    st = prefixes(a.url)
    ids = st.get("prefixes") if isinstance(st, dict) else None
    if isinstance(ids, dict):
        print("\n  prefix ids the gateway now holds: %d" % len(ids))
        for pid, info in list(ids.items())[:10]:
            print("    %s  %s" % (pid, json.dumps(info)[:80]))

    # The honesty property, stated as numbers rather than asserted: a warm
    # request must be far faster than the cold one that preceded it. Both
    # switch steps are compared against their own first call, so no absolute
    # threshold is needed.
    def sec(i):
        return out[i]["seconds"]
    print("\n  %-34s %6.2f s -> %6.2f s" % ("same mode, 1st -> 2nd", sec(0), sec(1)))
    print("  %-34s %6.2f s -> %6.2f s" % ("switched mode, 1st -> 2nd", sec(2), sec(3)))
    print("  %-34s %6.2f s" % ("returning to the first mode", sec(6)))
    print("\n  What to read: the 2nd call of each mode must be much faster than")
    print("  its 1st — that is the mode having its own warm slot. And the 1st")
    print("  call after a SWITCH must be slow: if it were fast, the modes would")
    print("  be sharing a slot they cannot share, which is the defect.")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump({"label": a.label, "url": a.url, "steps": out}, fh, indent=1)
        print("\n  written: %s" % a.out)


if __name__ == "__main__":
    main()
