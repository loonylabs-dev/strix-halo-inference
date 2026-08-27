#!/usr/bin/env python3
"""spec-determinism — is speculative decoding what makes a run unrepeatable?

The chain that leads here, all measured on 26.08.2026:

  1. `hard-count-numbers` failed once in a battery run and passed three times
     after, at 161 / 243 / 284 s. A single run is one sample and a
     model decision was once made on a one-task margin.
  2. Sampling was eliminated: three reps at `temperature: 0`, where greedy
     decoding should give byte-identical output, produced 5700 / 8192 / 7212
     tokens.
  3. The server is NOT generally non-deterministic. bench/suites/determinism.py
     asked a 4759-token TRANSCRIPTION three times and got byte-identical
     answers, warm and after erasing every slot.
  4. The three reasoning answers are byte-identical for their first 1107
     characters and then split — at a semantically neutral phrasing choice
     ("the digit 0 can be part of the SET" against "part of the NUMBER").

So the effect is real but only bites at NEAR-TIES: a transcription's next token
wins by a mile, a reasoning step is full of two-way coin flips, and one flipped
coin cascades. A difference in the logits' last bits is enough, and that points
at the forward pass, not the sampler.

What is left to test is where that difference comes from. The production
profile runs `--spec-type draft-mtp,ngram-mod`: the number of tokens verified
per batch depends on how many drafts are accepted, a different batch shape
means a different reduction order in the matmuls, and that changes the logits
in their last bits. This suite starts TWO side servers that differ in exactly
that one flag and runs the same reasoning task three times against each.

    python3 bench/suites/spec-determinism.py
    python3 bench/suites/spec-determinism.py --reps 5 --task hard-sql-window

Both servers are side servers on port 8081; production is not touched. Neither
is the production prefix store — the side server gets its own.

What the outcomes mean
----------------------
  spec off identical, spec on not   speculation is the cause. A DECISION run
                                    should turn it off; production keeps it,
                                    because it buys real decode speed and the
                                    price is only reproducibility.
  both not identical                the non-determinism is in the GPU
                                    reductions themselves and no flag helps.
                                    Then repetitions are the only answer and
                                    bench/README.md's --reps 3 rule stands.
  both identical                    something else in the battery differs
                                    between reps. Look there.
"""
import argparse, hashlib, json, os, signal, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
REPO = os.path.dirname(BENCH)
sys.path.insert(0, BENCH)
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "setup", "lib"))
import run as runlib                                          # noqa: E402
import tasklib                                                # noqa: E402
import systemdfile                                            # noqa: E402

PORT = 8081
URL = "http://127.0.0.1:%d" % PORT
SIDE_SLOTS = "/tmp/claude-1000/spec-det-slots"


def side_args(with_spec):
    """The production profile, with three things changed and nothing else.

    Reading the profile instead of restating it is the point: a hand-copied
    argument list is a second source of truth that drifts, and the whole
    question here is what ONE flag does.
    """
    args = systemdfile.llama_args(os.path.join(REPO, "setup/env/qwen38.env"))
    out, skip = [], 0
    for i, x in enumerate(args):
        if skip:
            skip -= 1
            continue
        if x == "--port":
            out += ["--port", str(PORT)]; skip = 1; continue
        if x == "--alias":
            out += ["--alias", "specdet"]; skip = 1; continue
        if x == "--slot-save-path":
            out += ["--slot-save-path", SIDE_SLOTS]; skip = 1; continue
        if not with_spec and x.startswith("--spec-"):
            # every --spec-* flag takes a value
            skip = 1; continue
        out.append(x)
    return out


def ready(timeout=600):
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(URL + "/slots", timeout=5)
            return True
        except Exception:
            time.sleep(3)
    return False


def ask(task, timeout=1800):
    body = {"model": "specdet", "stream": False,
            "max_tokens": task.get("max_tokens", 2048), "seed": 7,
            "temperature": 0,
            "messages": [{"role": "system", "content": task["system"]},
                         {"role": "user", "content": task["user"]}]}
    r = urllib.request.Request(URL + "/v1/chat/completions",
                               data=json.dumps(body).encode(),
                               headers={"content-type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(r, timeout=timeout) as x:
        resp = json.loads(x.read().decode())
    msg = (resp.get("choices") or [{}])[0].get("message") or {}
    tm = resp.get("timings") or {}
    text = msg.get("content") or ""
    # Reproducible and WRONG is a different result from reproducible and
    # right, and the first run of this suite could not tell them apart.
    #
    # The checker returns a TUPLE (passed, reason). bool() on that is True for
    # every non-empty tuple, so the first version of these three lines would
    # have printed a confident PASS on every row including the wrong ones.
    ok, why = None, None
    try:
        ok, why = tasklib.get_checker(task)(text, task)
    except Exception as e:
        why = "checker raised: %s" % str(e)[:80]
    return {"text": text, "sha": hashlib.sha256(text.encode()).hexdigest()[:12],
            "passed": ok, "reason": why,
            "tokens": tm.get("predicted_n"), "wall": round(time.time() - t0, 1)}


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def one_case(with_spec, task, reps):
    label = "spec-on" if with_spec else "spec-off"
    args = side_args(with_spec)
    log = "/tmp/claude-1000/side-specdet-%s.log" % label
    os.makedirs(SIDE_SLOTS, exist_ok=True)
    print("\n== %s   (%s)" % (label,
                              " ".join(x for x in args if x.startswith("--spec"))
                              or "no speculation"))
    proc = runlib.start_server(args, log,
                               os.path.expanduser("~/llama.cpp/build-rocm-patched/bin/llama-server"))
    try:
        if not ready():
            print("   server never served /slots — see %s" % log)
            return None
        runs = []
        for i in range(reps):
            r = ask(task)
            runs.append(r)
            print("   %d  sha=%s  %-5s %5s tokens  %6.1f s   %s" %
                  (i + 1, r["sha"],
                   {True: "PASS", False: "FAIL", None: "?"}[r["passed"]],
                   r["tokens"], r["wall"], (r["reason"] or "")[:44]))
        shas = {r["sha"] for r in runs}
        identical = len(shas) == 1
        div = (None if identical else
               min(first_divergence(runs[0]["text"], r["text"])
                   for r in runs[1:] if r["sha"] != runs[0]["sha"]))
        print("   -> %s" % ("IDENTICAL" if identical else
                            "%d distinct, first divergence at character %d"
                            % (len(shas), div)))
        return {"label": label, "identical": identical, "distinct": len(shas),
                "first_divergence": div,
                "runs": [{k: v for k, v in r.items() if k != "text"} for r in runs]}
    finally:
        # Wait for the MEMORY, not just the port — see bench/run.py's guard.
        before = runlib._gtt("used")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        for _ in range(60):
            try:
                urllib.request.urlopen(URL + "/health", timeout=2)
                time.sleep(1)
            except Exception:
                break
        if before is not None:
            runlib.wait_for_gtt_release(
                max(0.0, before - (runlib._model_size_gib(args) or 0) * 0.8), 180)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="hard-count-numbers")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cases", default="on,off",
                    help="which side servers to start: on, off, or both")
    ap.add_argument("--out")
    a = ap.parse_args()
    task = tasklib.TASK_INDEX.get(a.task)
    if not task:
        print("unknown task %r — known: %s"
              % (a.task, ", ".join(sorted(tasklib.TASK_INDEX))), file=sys.stderr)
        return 2
    import sweep
    sweep.reexec_with_inhibit()

    stamp = time.strftime("%Y-%m-%d_%H%M")
    dest = a.out or os.path.join(BENCH, "reports", "%s_spec-determinism" % stamp)
    os.makedirs(dest, exist_ok=True)
    print("spec-determinism · task=%s · %d reps each · %s" % (a.task, a.reps, URL))

    want = [c.strip() for c in a.cases.split(",")]
    result = {"stamp": stamp, "task": a.task, "reps": a.reps, "cases": []}
    for with_spec in [x for x in (True, False)
                      if ("on" if x else "off") in want]:
        c = one_case(with_spec, task, a.reps)
        if c:
            result["cases"].append(c)
    with open(os.path.join(dest, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)

    print("\nRESULT")
    for c in result["cases"]:
        print("  %-9s %s" % (c["label"],
                             "reproducible" if c["identical"]
                             else "NOT reproducible (%d distinct, split at char %d)"
                                  % (c["distinct"], c["first_divergence"])))
    on = next((c for c in result["cases"] if c["label"] == "spec-on"), None)
    off = next((c for c in result["cases"] if c["label"] == "spec-off"), None)
    if on and off:
        if off["identical"] and not on["identical"]:
            print("\n-> Speculation is what makes a run unrepeatable. A DECISION run\n"
                  "   should turn it off; production keeps it — it buys decode speed\n"
                  "   and the price is only reproducibility.")
        elif not off["identical"]:
            print("\n-> No flag helps: the non-determinism is in the GPU reductions\n"
                  "   themselves. Repetitions are the only answer (bench/README.md).")
        else:
            print("\n-> Both reproducible. Then something in the BATTERY differs\n"
                  "   between reps, and that is where to look next.")
    print("\nreport: %s" % dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
