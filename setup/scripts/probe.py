#!/usr/bin/env python3
"""probe — ask the running model one question whose answer is known.

    python3 setup/scripts/probe.py            once, exit 1 if the answer is wrong
    python3 setup/scripts/probe.py --json     machine-readable
    python3 setup/scripts/probe.py --restart  restart the unit on failure

Why this exists, and why it is not in the gateway.

The dangerous defects on this hardware do not raise. The gfx1151 HIP race and
a poisoned slot restore both end the same way: the server keeps answering, and
every answer degenerates to '////' until it is restarted. setup/defects.json
classifies that as `silent`, and silent is the expensive kind — nothing in a
normal stack notices, and the consumer just gets worse answers.

The obvious place to catch it is the gateway, which sees every response. It is
the wrong place. `forward()` streams chunks through with no buffering so that
SSE stays intact, and a detector there would have to parse SSE frames in the
hot path of every request to reassemble the text. That is real work in the one
code path that must not get slower or more fragile.

A probe outside the request path costs nothing per request, needs no dialect
knowledge, and catches the fault whatever produced it — because the fault is a
property of the SERVER, not of one response. A poisoned server stays poisoned
until it restarts, so a probe every few minutes catches it with a bounded
delay rather than never.

Two independent verdicts, because they fail differently:

  correctness   the answer must contain 391. A server that answers something
                else is wrong in a way arithmetic can see.
  degeneracy    the answer must not be dominated by a single character. This
                is the '////' signature, and it is checked separately because
                a degenerate answer is a KNOWN failure with a known cause,
                while a merely wrong one is not.
"""
import argparse, collections, json, os, subprocess, sys, time
import urllib.error, urllib.request

URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")
UNIT = os.environ.get("LLAMA_UNIT", "")
QUESTION = "What is 17*23? Answer with the number only."
EXPECT = "391"


def looks_degenerate(text, min_len=24, share=0.6):
    """Is this answer dominated by one character?

    The signature of both two-slot defects here is a response made almost
    entirely of one character. A RATIO, not a run length: a run test would
    fire on a markdown rule or a line of dashes inside an otherwise fine
    answer, and a false alarm in a watchdog is how a real one gets ignored.

    Whitespace is excluded — an answer that is mostly spaces is not this
    fault, and counting them would make long, well-formatted answers look
    suspicious.
    """
    body = "".join(text.split())
    if len(body) < min_len:
        return False, ""
    char, n = collections.Counter(body).most_common(1)[0]
    if n / len(body) >= share:
        return True, "%.0f%% of %d characters are %r" % (100.0 * n / len(body),
                                                         len(body), char)
    return False, ""


def ask(url=URL, timeout=180):
    r = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps({"model": "probe", "max_tokens": 16, "stream": False,
                         "messages": [{"role": "user", "content": QUESTION}]}).encode(),
        headers={"content-type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as x:
        d = json.loads(x.read().decode())
    return (d["choices"][0]["message"].get("content") or "").strip()


def judge(text):
    """(ok, verdict, detail). Degeneracy outranks wrongness: it names a cause."""
    bad, why = looks_degenerate(text)
    if bad:
        return False, "DEGENERATE", why
    if EXPECT not in text:
        return False, "WRONG", "expected %s in %r" % (EXPECT, text[:60])
    return True, "ok", text[:60]


# How long a server that refuses connections is given to become one.
#
# This watchdog exists for the SILENT fault: a server that answers, and answers
# wrongly. A server that refuses a connection is DOWN, which is a different
# fault with three other detectors already on it — systemd, check.sh, and the
# gateway. Conflating them made the probe cry wolf every time production was
# restarted, and a detector that cries wolf is one people stop reading.
#
# Measured 27.08.: sideserver restarted production and this timer in the same
# second, at 10:55:46. The timer's interval had elapsed while it was stopped,
# so it fired at 10:55:47 into a server whose model finished loading at
# 10:55:55 — nine seconds later. Two runs, two false alarms, two red lines in
# check.sh, nothing wrong either time.
#
# 90 s covers a 16.7 GiB model coming up. A server still refusing after that is
# genuinely worth a failure.
GRACE_S = 90
RETRY_EVERY_S = 10


def ask_with_patience(url, grace=GRACE_S, sleep=time.sleep):
    """(text, None) once it answers, or (None, reason) after `grace` seconds.

    Only CONNECTION failures are retried. A server that answers is judged on
    what it said, immediately — that is the fault this watchdog is for, and
    retrying it would let a poisoned server look healthy for another minute.
    """
    deadline = time.monotonic() + max(0, grace)
    last = ""
    while True:
        try:
            return ask(url), None
        except urllib.error.HTTPError as e:
            return None, repr(e)          # it answered, with a status. Real.
        except Exception as e:
            last = repr(e)
        if time.monotonic() >= deadline:
            return None, "%s (still refusing after %ds)" % (last, grace)
        sleep(RETRY_EVERY_S)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--restart", action="store_true",
                    help="restart $LLAMA_UNIT when the verdict is DEGENERATE")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--grace", type=int, default=GRACE_S,
                    help="seconds to keep retrying a server that will not "
                         "connect, before calling it unreachable")
    a = ap.parse_args()

    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    text, err = ask_with_patience(a.url, a.grace)
    if err is not None:
        ok, verdict, detail, text = False, "UNREACHABLE", err, ""
    else:
        ok, verdict, detail = judge(text)

    if a.json:
        print(json.dumps({"at": stamp, "verdict": verdict, "ok": ok,
                          "detail": detail, "answer": text}))
    else:
        print("%s  %-12s %s" % (stamp, verdict, detail))

    # Restart only for DEGENERATE. UNREACHABLE is usually a server that is
    # still loading, and restarting it then would turn a slow start into a
    # loop. WRONG is not a known fault of this machine and might be the model
    # having a bad day — a watchdog that acts on everything acts on noise.
    if not ok and a.restart and verdict == "DEGENERATE" and UNIT:
        print("  restarting %s — this is the known gfx1151 signature" % UNIT)
        subprocess.run(["systemctl", "--user", "restart", UNIT], check=False)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
