#!/usr/bin/env python3
"""slot-corruption — which ingredient makes this build produce '////'?

A consumer on a second machine received another session's answer, and the
server here was found returning endless slashes to everyone. Four cases,
each against a freshly started server, one variable at a time:

    seq-one-prefix     three requests, one prefix, sequential
    seq-two-prefixes   two prefixes alternating, sequential
    par-two-prefixes   two prefixes, concurrent
    seq-no-tools       one prefix, no tool block

Measured 26.08.2026 on qwen38 / ROCm / gfx1151, build b10577:

    -np 2   one prefix           clean
    -np 2   two prefixes seq     CORRUPT on the 2nd request
    -np 2   two prefixes par     CORRUPT 8/8
    -np 1   both two-prefix runs clean

So it is the SECOND SLOT, not concurrency — serialising every request in
the gateway did not help, and neither did disabling the RAM prompt cache.
That is the gfx1151 HIP race (llama.cpp #27579, root cause #27572), and
the mitigation is -np 1 in the profile.

    python3 bench/suites/slot-corruption.py --binary rocm
    python3 bench/suites/slot-corruption.py --binary rocm --starts 5 \
        --tools 24 --ctx 65536

IT COULD NOT RUN AT ALL UNTIL 28.08.2026, and that is worth reading twice.
It drove the production GATEWAY on port 8090 and restarted
`llama-user@qwen38` before every case — and production has served `-np 1`
since 26.08. Every case here needs a SECOND SLOT. So the suite would have
restarted production, measured a one-slot server, and reported `clean` for
a configuration in which the defect cannot appear: a check that cannot
fail, in the instrument for the defect the patch exists for.

It runs against a SIDE SERVER on port 8081 now, started through
bench/run.py like every other suite here, so it touches neither production
nor the gateway. That also removes the gateway from the experiment, which
is right for a hunt: the original finding already established that
serialising there changed nothing.

WHAT MAKES THIS SUITE DIFFERENT from bench/suites/np2-candidates.py, and
the reason the hunt of 28.08. needed it: the bodies carry a TOOLS BLOCK.
np2-candidates' do not, and neither did the 30-start control that failed
to reproduce anything. `seq-no-tools` is the discriminator for exactly
that, and it has been in this file since the original finding.

CAUTION: a corrupting run leaves that server poisoned until it restarts.
Every case gets a fresh one.
"""
import argparse, json, os, sys, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
REPO = os.path.dirname(BENCH)
sys.path.insert(0, BENCH)
sys.path.insert(0, os.path.join(REPO, "setup", "lib"))
import run as runlib                                          # noqa: E402
import sweep                                                  # noqa: E402
import systemdfile                                            # noqa: E402

MODELS = systemdfile.models_dir()
MODEL = os.path.join(MODELS, "Qwen3.8-27B-UD-Q4_K_XL.gguf")
PORT = 8081
URL = "http://127.0.0.1:%d" % PORT

# Set by main(): the binary under test, and the shape of the server.
BINARY = None
ARGS = {"np": 2, "ctx": 32768, "tools": 10, "spec": True, "cram": True,
        "mmproj": False, "kv_unified": False}


def server_argv():
    """The production shape minus what is under test.

    Every one of these is a HUNT VARIABLE, because the defect stopped
    reproducing between 26.08. and 28.08. and nothing here knows which of
    them moved. They are flags rather than constants for that reason.
    """
    a = ["--alias", "sidetest", "-m", MODEL,
         "-ngl", "999", "-fa", "on",
         "-c", str(ARGS["ctx"]), "-np", str(ARGS["np"]),
         "-b", "2048", "-ub", "2048"]
    if ARGS["cram"]:
        a += ["-cram", "32768"]
    if not ARGS["kv_unified"] and ARGS["np"] > 1:
        a += ["--no-kv-unified"]
    if ARGS["spec"]:
        a += ["--spec-type", "draft-mtp,ngram-mod",
              "--spec-draft-n-max", "12", "--spec-ngram-mod-n-min", "24"]
    if ARGS["mmproj"]:
        a += ["--mmproj", os.path.join(MODELS, "mmproj-F16.gguf")]
    return a + ["--chat-template-kwargs", '{"enable_thinking":false}',
                "--jinja", "--host", "127.0.0.1", "--port", str(PORT)]


def make_body(which, nonce, n_tools=None, bulk_reps=700, sys_reps=300):
    n_tools = ARGS["tools"] if n_tools is None else n_tools
    system = ("You are assistant %s. " % which) + ("Directive %s. " % which) * sys_reps
    bulk = ("Handbook for workspace %s.\n" % which) + \
           ("Rule %s about this repository. " % which) * bulk_reps
    b = {"model": "sidetest", "stream": False, "max_tokens": 120,
         "messages": [{"role": "system", "content": system},
                      {"role": "user",
                       "content": "Reply with exactly this word and nothing "
                                  "else: " + nonce},
                      {"role": "user", "content": bulk}]}
    if n_tools:
        b["tools"] = [{"type": "function",
                       "function": {"name": "T%02d_%s" % (i, which),
                                    "description": "d " * 50,
                                    "parameters": {"type": "object"}}}
                      for i in range(n_tools)]
    return b


def ask(body_, timeout=900):
    r = urllib.request.Request(URL + "/v1/chat/completions",
                               data=json.dumps(body_).encode(),
                               headers={"content-type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as x:
        resp = json.loads(x.read().decode())
    return (resp["choices"][0]["message"].get("content") or "").strip()


def verdict(text, nonce):
    if "////" in text or text.count("/") > 8:
        return "CORRUPT"
    return "ok" if nonce in text else ("empty" if not text else "other")


def seq_one_prefix():
    out = []
    for i in range(3):
        n = "SEQ-%d-%d" % (int(time.time()) % 10000, i)
        out.append((n, ask(make_body("A", n))))
    return out


def seq_two_prefixes():
    out = []
    for i in range(2):
        for w in ("A", "B"):
            n = "%s-%d-%d" % (w, int(time.time()) % 10000, i)
            out.append((n, ask(make_body(w, n))))
    return out


def par_two_prefixes():
    out, lock = [], threading.Lock()

    def run(w):
        for i in range(2):
            n = "%s-%d-%d" % (w, int(time.time()) % 10000, i)
            t = ask(make_body(w, n))
            with lock:
                out.append((n, t))
    ts = [threading.Thread(target=run, args=(w,), daemon=True) for w in ("A", "B")]
    for t in ts:
        t.start()
    for t in ts:
        t.join(1800)
    return out


def seq_no_tools():
    out = []
    for i in range(3):
        n = "NOTOOL-%d-%d" % (int(time.time()) % 10000, i)
        out.append((n, ask(make_body("A", n, n_tools=0))))
    return out


CASES = {"seq-one-prefix": seq_one_prefix,
         "seq-two-prefixes": seq_two_prefixes,
         "par-two-prefixes": par_two_prefixes,
         "seq-no-tools": seq_no_tools}


def one_run(name, run_no, dest):
    """One case against one fresh server. Errors are RECORDED, not fatal.

    The unit of risk is the START — one start clean and the next CORRUPT 3/6
    is on record for this defect — so a hunt runs the same case many times and
    must not lose the rest of them to one bad start.
    """
    log = os.path.join(dest, "%s-%d.log" % (name, run_no))
    before = runlib._gtt("used")
    proc = None
    rec = {"case": name, "run": run_no}
    try:
        proc = runlib.start_server(server_argv(), log, BINARY)
        if not sweep.slots_ready(URL, 300):
            rec["error"] = "/slots never answered"
            return rec
        outs = CASES[name]()
        rec["answers"] = [{"nonce": n, "verdict": verdict(t, n),
                           "text": t[:120]} for n, t in outs]
        rec["corrupt"] = sum(1 for n, t in outs if verdict(t, n) == "CORRUPT")
        rec["other"] = sum(1 for n, t in outs
                           if verdict(t, n) in ("other", "empty"))
        for a in rec["answers"]:
            print("   %-18s %-8s %r" % (a["nonce"], a["verdict"], a["text"][:36]))
    except (Exception, SystemExit) as e:
        rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
        print("   ABORTED %s" % rec["error"])
    finally:
        if proc is not None:
            sweep.stop_server(proc)
            try:
                runlib.wait_for_gtt_release(before)
            except Exception as e:                            # noqa: BLE001
                print("   (gtt wait: %s)" % e)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cases", nargs="*", help="one or more of: "
                    + ", ".join(CASES))
    ap.add_argument("--binary", required=True,
                    help="a path, a build directory name, or a build id")
    ap.add_argument("--starts", type=int, default=1,
                    help="fresh servers per case (default: %(default)s)")
    ap.add_argument("--np", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--tools", type=int, default=10)
    ap.add_argument("--no-spec", action="store_true")
    ap.add_argument("--no-cram", action="store_true")
    ap.add_argument("--mmproj", action="store_true")
    ap.add_argument("--kv-unified", action="store_true",
                    help="drop --no-kv-unified, which the original recipe had")
    ap.add_argument("--label", default="", help="goes into the report name")
    a = ap.parse_args()

    global BINARY
    BINARY = runlib.resolve_binary(a.binary)
    ARGS.update(np=a.np, ctx=a.ctx, tools=a.tools, spec=not a.no_spec,
                cram=not a.no_cram, mmproj=a.mmproj, kv_unified=a.kv_unified)
    meta = runlib.provenance(BINARY)
    meta["shape"] = dict(ARGS)
    meta["argv"] = server_argv()
    meta["starts"] = a.starts
    # The ENVIRONMENT is part of the configuration and was missing from the
    # record. llama.cpp reads GGML_SCHED_UMA_RING, LLAMA_SET_ROWS and friends
    # at runtime, so two runs of the same binary with the same argv can be two
    # different experiments and the report could not tell them apart. Found
    # 28.08. while using GGML_SCHED_UMA_RING to switch the candidate fix off
    # on a binary that contains it.
    meta["env"] = {k: v for k, v in sorted(os.environ.items())
                   if k.startswith(("GGML_", "LLAMA_")) and k != "LLAMA_SRC"}

    stamp = time.strftime("%Y-%m-%d_%H%M")
    # The BUILD DIRECTORY names the report, not the build id. `git describe`
    # gives b10631 for the patched AND the unpatched build of the same
    # upstream commit, so two runs of two different binaries landed in
    # directories differing only by the minute they started — found 28.08. an
    # hour after the same defect was fixed in restore-safety.py, in a file
    # that had not been touched yet.
    #
    # The directory rather than the stamp's `family`, because stamps written
    # before 28.08. do not carry that field and the patched build's is from
    # the 26th. A directory name is weaker evidence than a stamp and it is
    # always there; the stamp still decides what the JSON says.
    tag = os.path.basename(os.path.dirname(os.path.dirname(BINARY)))
    dest = os.path.join(BENCH, "reports", "%s_slot-corruption_%s%s"
                        % (stamp, tag, ("_" + a.label) if a.label else ""))
    os.makedirs(dest, exist_ok=True)
    print("binary: %s" % meta["binary"])
    print("build:  %s   shape: %s" % (meta["build_id"], ARGS))
    print("report: %s" % dest)

    names = a.cases or list(CASES)
    runs = []
    for name in names:
        if name not in CASES:
            raise SystemExit("unknown case %r; known: %s"
                             % (name, ", ".join(CASES)))
        for r in range(1, a.starts + 1):
            print("== %s  start %d/%d" % (name, r, a.starts))
            runs.append(one_run(name, r, dest))
            with open(os.path.join(dest, "result.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"_meta": meta, "runs": runs}, f, indent=1,
                          ensure_ascii=False)

    print("\nRESULT")
    for name in names:
        mine = [r for r in runs if r["case"] == name]
        corrupt = sum(1 for r in mine if r.get("corrupt"))
        errs = sum(1 for r in mine if r.get("error"))
        print("  %-18s %d of %d starts CORRUPT%s"
              % (name, corrupt, len(mine),
                 ", %d failed" % errs if errs else ""))
    total = sum(1 for r in runs if r.get("corrupt"))
    print("\n  %s" % ("REPRODUCED — %d of %d starts" % (total, len(runs))
                      if total else
                      "nothing reproduced in %d starts. A clean run here is "
                      "an ABSENCE, not a\n  finding: this defect is a "
                      "per-start gamble and the recipe is a guess." % len(runs)))
    print("  report: %s" % dest)


if __name__ == "__main__":
    main()
