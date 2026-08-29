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

And a third question, which is not about the answer at all: was there one to
judge? Production serves ONE slot, so this probe queues behind whatever is in
flight. A round that timed out in that queue is BUSY — not a fault — while a
run of them is BLIND, which is. See classify_stall().
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


def ask(url=URL, timeout=None):
    r = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps({"model": "probe", "max_tokens": 16, "stream": False,
                         "messages": [{"role": "user", "content": QUESTION}]}).encode(),
        headers={"content-type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout or ANSWER_TIMEOUT_S) as x:
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

# How long ONE attempt may take before it is given up on.
#
# This is the deadline that actually fires. GRACE_S only governs retries of a
# connection that was refused; once the server accepts and then computes,
# nothing but this number ends the wait. Measured 29.08. over 277 runs behind
# production's single slot: median 1 s, p90 8 s, twelve runs over a minute,
# and the longest that still answered took 160 s. So 180 is not generous —
# it is just past the tail, and the runs that hit it were queued, not down.
ANSWER_TIMEOUT_S = 180


def ask_with_patience(url, grace=GRACE_S, sleep=time.sleep, now=time.monotonic,
                      timeout=None):
    """(text, None) once it answers, or (None, reason) after `grace` seconds.

    A server that ANSWERS is judged on what it said, immediately — that is the
    fault this watchdog is for, and retrying it would let a poisoned server
    look healthy for another minute.

    Retried: a connection failure, and **503**. Those are the two ways of
    saying "not ready yet", and the grace window exists for exactly that. 503
    is not an answer about the model — llama-server returns it while the
    weights load and the gateway passes it through — so treating it as a
    verdict made the patience cover only half the case it was written for.

    Measured 27.08. 23:19:06 and 23:42:36, both `UNREACHABLE
    <HTTPError 503: 'Service Unavailable'>`: two red lines in check.sh from a
    probe that fired into a model still coming up after a measurement had
    restored production. Nothing was wrong either time, which is the whole
    problem — a watchdog that cries wolf after every measurement is a watchdog
    that gets ignored, and this one exists to make silent faults loud.

    A 503 that PERSISTS past the window still fails, so nothing is lost.
    """
    started = now()
    deadline = started + max(0, grace)
    last = ""
    while True:
        try:
            return ask(url, timeout or ANSWER_TIMEOUT_S), None
        except urllib.error.HTTPError as e:
            if e.code != 503:
                return None, repr(e)      # it answered, with a status. Real.
            last = repr(e)                # "not ready", not a verdict
        except Exception as e:
            last = repr(e)
        if now() >= deadline:
            # The ELAPSED time, not `grace`. It said "after 90s" while 180 had
            # passed — one read timeout in ask() outlasts the whole connect
            # window, so the number in the message was never the number that
            # mattered, and on 29.08. it sent the investigation looking at the
            # connect path for a fault that was in the queue.
            return None, "%s (gave up after %ds)" % (last, round(now() - started))
        sleep(RETRY_EVERY_S)


# ---------------------------------------------------- busy is not down -------
#
# Production serves ONE slot — the mitigation for gfx1151-two-slots — so every
# request queues behind the one in flight, this probe included. Measured
# 29.08. over 277 runs: median 1 s, p90 8 s, but twelve runs took over a
# minute and the longest that still answered took 160 s against a 180 s read
# timeout. The three UNREACHABLE verdicts on record are not a different kind
# of event from that 160 s success — they are the same queue, one request
# longer, and a red `failed` unit for a server that was working perfectly.
#
# What makes the two separable, measured the same day with the slot occupied
# for 112 s: /health answered 200 in 0.001 s on 11 of 11 samples while
# is_processing stayed true, and a probe-shaped request queued behind it came
# back correct after 109 s. So the distinction costs one cheap GET, and only
# in the case that was going to be a failure anyway.
BUSY_LIMIT = 3
STREAK_PATH = os.path.join(
    os.environ.get("XDG_RUNTIME_DIR") or os.path.expanduser("~/.cache"),
    "llama-probe-busy-streak")


def server_state(url, timeout=5):
    """(alive, processing, detail) — is it there, and is it working?

    /health says the HTTP server is up; /slots says whether anything is being
    computed. Both answer in a millisecond under load, which is the whole
    reason this can be asked at the moment the chat request has given up.
    """
    try:
        with urllib.request.urlopen(url + "/health", timeout=timeout) as x:
            x.read()
    except Exception as e:
        return False, None, "no /health (%r)" % e
    try:
        with urllib.request.urlopen(url + "/slots", timeout=timeout) as x:
            slots = json.loads(x.read().decode())
        busy = sum(1 for sl in slots if sl.get("is_processing"))
        return True, busy > 0, "%d of %d slots busy" % (busy, len(slots))
    except Exception as e:
        # /health answered, so the server IS there. Whether it is working is
        # then unknown rather than false — and unknown must not be filed as a
        # stall, which is a finding.
        return True, None, "/health ok, no /slots (%r)" % e


def busy_streak(path, busy):
    """How many runs in a row have now ended BUSY. 0 once a real verdict
    lands. State on disk, because each run is its own process — and losing
    the file must never take the watchdog down with it."""
    n = 0
    if busy:
        try:
            with open(path, encoding="utf-8") as f:
                n = int(f.read().strip() or 0)
        except Exception:
            n = 0
        n += 1
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(n))
    except Exception:
        pass
    return n


def classify_stall(url, err, streak_path=STREAK_PATH, state=None):
    """(verdict, ok, detail) for a probe that got no answer.

    Four outcomes, and the three that are not BUSY all stay failures:

      UNREACHABLE  nothing answers. The original meaning, untouched.
      BUSY         the server answers /health and a slot is computing. Someone
                   else's request is in front of ours — not a fault.
      UNKNOWN      /health is fine but /slots could not be read, so whether it
                   computes is unknown. Not a finding — `--no-slots` alone
                   would otherwise manufacture one on every busy minute.
      STALLED      /health is fine, NOTHING is computing, and a chat request
                   still timed out. Nothing explains that.
      BLIND        so many rounds in a row without a look at the model that
                   the watchdog has stopped watching. Silence from a
                   silent-failure detector reads exactly like good news,
                   which is why it is counted.
    """
    state = state or server_state
    # A status code IS an answer. Only a request that never came back can be
    # explained by the queue, so everything else keeps its old verdict.
    if "TimeoutError" not in err and "timed out" not in err:
        busy_streak(streak_path, False)
        return "UNREACHABLE", False, err
    alive, processing, detail = state(url)
    if not alive:
        busy_streak(streak_path, False)
        return "UNREACHABLE", False, "%s; %s" % (err, detail)
    if processing is False:
        busy_streak(streak_path, False)
        return "STALLED", False, "%s; %s" % (err, detail)
    # True, or None for "/slots did not say". Neither is a look at the model,
    # so both count towards BLIND.
    n = busy_streak(streak_path, True)
    if n > BUSY_LIMIT:
        return ("BLIND", False,
                "%s; %s — %d rounds in a row without a look at the model"
                % (err, detail, n))
    return ("BUSY" if processing else "UNKNOWN", True,
            "%s; %s (streak %d)" % (err, detail, n))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--restart", action="store_true",
                    help="restart $LLAMA_UNIT when the verdict is DEGENERATE")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--timeout", type=int, default=ANSWER_TIMEOUT_S,
                    help="seconds one attempt may take before it is given up "
                         "on (default: %(default)s)")
    ap.add_argument("--grace", type=int, default=GRACE_S,
                    help="seconds to keep retrying a server that will not "
                         "connect, before calling it unreachable")
    a = ap.parse_args()

    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    text, err = ask_with_patience(a.url, a.grace, timeout=a.timeout)
    if err is not None:
        verdict, ok, detail = classify_stall(a.url, err)
        text = ""
    else:
        busy_streak(STREAK_PATH, False)
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
