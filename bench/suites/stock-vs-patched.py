#!/usr/bin/env python3
"""stock-vs-patched — must a user COMPILE llama.cpp to run this stack safely?

The stack has always required a self-built, patched llama.cpp. That is the
single largest barrier to anybody else using it: ROCm toolchain, a git
checkout, `GPU_TARGETS=gfx1151`, a compile. And the matrix in
setup/patches/README.md never contained the cell that decides whether it is
necessary:

    stock,   2 slots   CORRUPT 6/6
    patched, 2 slots   clean, then CORRUPT 3/6 on the next start
    patched, 1 slot    clean in every cell
    stock,   1 slot    <- never measured, and it is what production runs

Since 26.08. the comparison can be made honestly, because llama.cpp ships
OFFICIAL prebuilt Linux binaries per build — including ROCm with gfx1151
among its gpu_targets — and one of them is build b10631, which is exactly
the upstream commit (5d5cb4c3a) our patched build sits on. Same commit,
patch on and off, no compiling required to find out.

    OFFICIAL_ROCM=/path/to/official/llama-server \\
    python3 bench/suites/stock-vs-patched.py official-rocm-np1 patched-rocm-np1 official-rocm-np2

Get an official binary:

    B=b10631
    curl -L -o /tmp/l.tgz https://github.com/ggml-org/llama.cpp/releases/download/$B/llama-$B-bin-ubuntu-rocm-7.14-x64.tar.gz
    mkdir -p /tmp/official-rocm && tar xzf /tmp/l.tgz -C /tmp/official-rocm
    export OFFICIAL_ROCM=/tmp/official-rocm/llama-$B/llama-server

WHAT A CLEAN RESULT DOES NOT MEAN. This defect is intermittent: the patched
two-slot cell was clean on one server start and corrupt 3/6 on the next, and
on b10631 it stopped reproducing altogether (24 of 24 clean over four starts,
26.08.). So no number of clean runs proves safety. The `-np 2` cell is here
for exactly that reason — it is the CONTROL: if the harness cannot make the
defect appear where it is known to live, then a clean `-np 1` says nothing
about the binary and only something about the day.

The decision-relevant question is therefore not "is stock safe" but "is stock
at -np 1 measurably WORSE than patched at -np 1". If it is not, requiring a
compile buys nothing that can be measured — and per the project's own rule,
what is complicated and avoidable goes.

MAY -np 2 COME BACK? The decision rule, written down BEFORE the run so it is
not argued about afterwards. `-np 2` would end the eviction between Claude
Code's two prompt types — measured as ~4 s per title generation instead of a
~50 s re-prefill — so it is worth something, but not worth a silent defect.

All three must hold. Any one failing and -np 1 stays:

  (a) QUANTITY. 30 fresh starts, zero corrupt answers, on the binary we would
      actually ship, with the faithful recipe. The unit of risk is the START,
      not the answer: when it did corrupt, it corrupted 3 of 6 within one
      start. Zero in 30 starts bounds the per-start rate at about 10 % with
      95 % confidence — and the honest reading of that sentence is that it is
      an UPPER BOUND, not a clean bill of health.

  (b) RESTORE. bench/suites/restore-safety.py clean in every cell at -np 2.
      The second two-slot defect was the poisoned restore, and it is a
      different mechanism from this one. Passing here says nothing there.

  (c) A DETECTOR IN PRODUCTION. The cause is unchanged in master — `git log
      b10577..b10631` shows no commit touching the integrated path — so any
      number of clean starts is a statement about the trigger, not the cause.
      Without something that notices degenerate output and says so loudly,
      returning to -np 2 is a bet on statistics against live code. With one,
      the failure mode moves from `silent` to `loud`, which is the whole axis
      setup/defects.json sorts by.

THE ANSWER, 26.08. — -np 2 STAYS CLOSED. Rule (a) was met and the other two
decided it, which is the entire reason the rule was written down first:

  (a) MET.      30 fresh starts, 180 answers, zero corruption, zero failed
                starts, on the patched b10631 build with the faithful recipe.
  (b) FAILED.   restore-safety cell busy-nospec — a slot restore into a busy
                server, speculation off — ran into its 300 s timeout, and
                REPRODUCED on a re-run. Two of two.

                RE-RUN 27.08.2026, same build b10631, and it is three of
                three. Four cells clean — idle-spec, busy-spec, prefill-spec,
                idle-nospec, all 391 391 391 — and busy-nospec into the same
                300 s wall. Rule (b) still fails, so -np 2 stays closed, and
                the reason is now a reproduction rather than a pair.

                (c) has since been DONE differently than it reads below:
                llama-probe.timer was `linked` and never `enabled` when that
                was written, so the detector existed and had never fired. It
                is armed since 27.08. That changes nothing here — (b) decides
                — but it means the next attempt starts with two of the three
                conditions met rather than one.

                WHAT THE RE-RUN ALSO SHOWED, about the instrument rather than
                the defect: the timeout ABORTS the whole run instead of
                recording the cell as DIRTY and continuing, so prefill-nospec
                was never measured. A suite whose first hanging cell costs it
                the remaining cells reports less than it found. Four other cells were
                clean, including prefill-spec, which is the cell that matched
                the production incident of 25.08. exactly. So the trigger has
                MOVED again rather than gone: what used to poison now passes,
                and what used to pass now hangs.
  (c) DONE.     setup/scripts/probe.py plus llama-probe.timer.

Thirty clean starts had made two slots look tempting. A rule agreed in
advance is what stops a tempting number from becoming a decision — written
afterwards, "it hung, but it did not corrupt" would have been an easy thing
to say.

Everything runs on a SIDE server (port 8081); production is untouched.

CAUTION: a corrupting run leaves that side server poisoned until it is
stopped. Each case gets a fresh one.
"""
import json, os, signal, subprocess, sys, time, urllib.request

# The repo, derived from this file rather than written down. It used to be
# assigned TWICE here — once derived and once as the absolute path of one
# clone, with the hard-coded line winning because it came second.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO + "/bench")
import run as runlib                                          # noqa: E402
import systemdfile                                            # noqa: E402

# Where the .gguf live: $LLAMA_MODELS, then ~/.config/llm-stack.env, then
# the conventional locations. No absolute path written down here.
MODELS = systemdfile.models_dir()

PORT = 8081
URL = "http://127.0.0.1:%d" % PORT
MODEL = os.path.join(MODELS, "Qwen3.8-27B-UD-Q4_K_XL.gguf")

BIN = {
    "official-rocm": os.environ.get("OFFICIAL_ROCM", ""),
    "official-vulkan": os.environ.get("OFFICIAL_VULKAN", ""),
    "patched-rocm": os.path.expanduser("~/llama.cpp/build-rocm-patched/bin/llama-server"),
}

# Deliberately the production shape minus what is under test. -cram stays
# because production has it and it was already exonerated as a cause (26.08.);
# speculation stays out because it makes runs unrepeatable, which is the last
# thing a corruption hunt needs.
def args_for(np_, faithful=False):
    """`faithful` reproduces bench/suites/np2-candidates.py's BASE exactly.

    This distinction cost a conclusion. The first run of this suite used a
    tidied recipe — no --no-kv-unified, no speculation — and the -np 2 CONTROL
    came back clean, which would have read as "the defect is gone". But
    --no-kv-unified gives each slot its own KV cache, which is a different
    memory layout and a plausible trigger condition, and the original finding
    was made WITH it. A control that fails because the recipe changed is not
    a control. Reproduce first, tidy afterwards.
    """
    a = ["--alias", "sidetest", "-m", MODEL,
         "-ngl", "999", "-fa", "on", "-c", "32768", "-np", str(np_),
         "-b", "2048", "-ub", "2048", "-cram", "32768"]
    if faithful and np_ > 1:
        a += ["--no-kv-unified"]
    if faithful:
        a += ["--spec-type", "draft-mtp,ngram-mod",
              "--spec-draft-n-max", "12", "--spec-ngram-mod-n-min", "24",
              "--mmproj", os.path.join(MODELS, "mmproj-F16.gguf")]
    return a + ["--chat-template-kwargs", '{"enable_thinking":false}',
                "--jinja", "--host", "127.0.0.1", "--port", str(PORT)]


CASES = {                  # name -> (binary key, -np, faithful-to-the-original)
    "official-rocm-np1":        ("official-rocm", 1, False),
    "patched-rocm-np1":         ("patched-rocm", 1, False),
    "official-rocm-np2":        ("official-rocm", 2, False),
    "patched-rocm-np2":         ("patched-rocm", 2, False),
    "official-vulkan-np1":      ("official-vulkan", 1, False),
    # the control, in the exact shape the defect was found in
    "official-rocm-np2-orig":   ("official-rocm", 2, True),
    "patched-rocm-np2-orig":    ("patched-rocm", 2, True),
    "official-rocm-np1-orig":   ("official-rocm", 1, True),
}


def body(which, nonce):
    """Two distinct long prefixes. It is the SECOND prefix that triggers it —
    not concurrency: serialising every request in the gateway did not help."""
    system = ("You are assistant %s. " % which) + ("Directive %s. " % which) * 200
    bulk = ("Handbook %s.\n" % which) + ("Rule %s here. " % which) * 400
    return {"model": "sidetest", "stream": False, "max_tokens": 60,
            "messages": [{"role": "system", "content": system},
                         {"role": "user",
                          "content": "Reply with exactly this word and nothing "
                                     "else: " + nonce},
                         {"role": "user", "content": bulk}]}


def ask(which, nonce):
    r = urllib.request.Request(URL + "/v1/chat/completions",
                               data=json.dumps(body(which, nonce)).encode(),
                               headers={"content-type": "application/json"})
    with urllib.request.urlopen(r, timeout=900) as x:
        return (json.loads(x.read().decode())["choices"][0]["message"]
                .get("content") or "").strip()


def ready(timeout=420):
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(URL + "/slots", timeout=5)
            return True
        except Exception:
            time.sleep(3)
    return False


def one_start(case, run_no):
    key, np_, faithful = CASES[case]
    binary = BIN[key]
    if not binary or not os.path.exists(binary):
        print("   %s: no binary (set OFFICIAL_ROCM / OFFICIAL_VULKAN)" % key)
        return None
    # The official tarballs ship their own shared libraries next to the binary.
    os.environ["LD_LIBRARY_PATH"] = os.path.dirname(binary)
    log = "/tmp/claude-1000/stock-vs-patched-%s-%d.log" % (case, run_no)
    proc = runlib.start_server(args_for(np_, faithful), log, binary)
    try:
        if not ready():
            print("   start %d: server never served /slots (%s)" % (run_no, log))
            return None
        bad = 0
        for i in range(3):
            for w in ("A", "B"):
                n = "%s-%d-%d" % (w, int(time.time()) % 10000, i)
                t = ask(w, n)
                corrupt = "////" in t or t.count("/") > 8
                bad += corrupt
                print("      %-14s %-8s %r" % (n, "CORRUPT" if corrupt else
                                               ("ok" if n in t else "other"), t[:34]))
        return bad
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        for _ in range(40):
            try:
                urllib.request.urlopen(URL + "/health", timeout=2)
                time.sleep(1)
            except Exception:
                break


def main():
    cases = sys.argv[1:] or ["official-rocm-np1", "patched-rocm-np1",
                             "official-rocm-np2"]
    starts = int(os.environ.get("STARTS", "3"))
    results = {}
    for case in cases:
        if case not in CASES:
            print("unknown case %r; known: %s" % (case, ", ".join(CASES)))
            return 2
        print("\n== %s  (%d fresh starts, port %d)" % (case, starts, PORT))
        runs = []
        for r in range(1, starts + 1):
            print("   start %d/%d" % (r, starts))
            runs.append(one_start(case, r))
        results[case] = runs

    print("\nRESULT — corrupt answers of 6 per start")
    for case, runs in results.items():
        shown = ["-" if r is None else str(r) for r in runs]
        done = [r for r in runs if r is not None]
        verdict = ("no result" if not done else
                   "CORRUPT in %d of %d starts" % (sum(1 for r in done if r), len(done))
                   if any(done) else "no corruption in %d starts" % len(done))
        print("  %-22s %-12s %s" % (case, " ".join(shown), verdict))
    print("\n  A clean -np 2 control means the harness could not make the defect")
    print("  appear at all — then a clean -np 1 says nothing about the binary.")

    # Rule (a), evaluated here so nobody has to remember the threshold.
    for case, runs in results.items():
        if not case.endswith("np2-orig"):
            continue
        done = [r for r in runs if r is not None]
        if any(done):
            print("\n  RULE (a) FAILED for %s — corruption seen. -np 1 stays." % case)
        elif len(done) >= 30:
            print("\n  RULE (a) met for %s: %d starts, zero corruption." % (case, len(done)))
            print("  That is an upper bound of ~10 % per start at 95 %, not a clean")
            print("  bill of health. Rules (b) restore-safety and (c) a production")
            print("  detector are still outstanding before -np 2 may return.")
        else:
            print("\n  RULE (a) NOT YET for %s: %d of 30 starts." % (case, len(done)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
