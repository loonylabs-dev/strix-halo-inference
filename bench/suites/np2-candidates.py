#!/usr/bin/env python3
"""Can we get -np 2 back? Test the candidates on a SIDE server (port 8081),
so production keeps running untouched.

    python3 bench/suites/np2-candidates.py rocm
    python3 bench/suites/np2-candidates.py nopatch --binary rocm-unpatched-b10631

This is the suite for DEFECT 1: two slots, an EMPTY prefix store, and no
restore anywhere. That is what setup/patches/hip-integrated-off.patch was
written for — CORRUPT 6/6 on the stock build — and it is a different code path
from the restore-during-prefill corruption that bench/suites/restore-safety.py
measures. Answering one says nothing about the other, which is exactly the
mistake of 27.08.: llama.cpp PR #27311 was shown to remove the restore
corruption, and that was briefly read as "we do not need the patch".

`--binary` takes a path, a build directory name, or a build id, and measures
that build without moving the symlink production starts from. With it, the
first word of a case is just a LABEL for the report.
"""
import argparse, json, os, signal, subprocess, sys, time, urllib.request

# The repo, derived from this file rather than written down. It used to be
# the absolute path of one clone, which made every suite here unusable
# anywhere else — including from a second checkout on the same machine.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO + "/bench")
import run as runlib                                          # noqa: E402
import systemdfile                                            # noqa: E402

# Where the .gguf live: $LLAMA_MODELS, then ~/.config/llm-stack.env, then
# the conventional locations. No absolute path written down here.
MODELS = systemdfile.models_dir()

PORT = 8081
URL = "http://127.0.0.1:%d" % PORT
BIN = {"rocm": os.path.expanduser("~/llama.cpp/build-rocm/bin/llama-server"),
       "rocm-patched": os.path.expanduser("~/llama.cpp/build-rocm-patched/bin/llama-server"),
       "vulkan": os.path.expanduser("~/llama.cpp/build-vulkan/bin/llama-server")}
BASE = ["--alias", "sidetest",
        "-m", os.path.join(MODELS, "Qwen3.8-27B-UD-Q4_K_XL.gguf"),
        "-ngl", "999", "-fa", "on", "-c", "32768", "-np", "2",
        "--no-kv-unified", "-b", "2048", "-ub", "2048",
        "--spec-type", "draft-mtp,ngram-mod",
        "--spec-draft-n-max", "12", "--spec-ngram-mod-n-min", "24",
        "--chat-template-kwargs", '{"enable_thinking":false}',
        "--jinja", "--host", "127.0.0.1", "--port", str(PORT)]


def body(which, nonce):
    system = ("You are assistant %s. " % which) + ("Directive %s. " % which) * 200
    bulk = ("Handbook %s.\n" % which) + ("Rule %s here. " % which) * 400
    return {"model": "sidetest", "stream": False, "max_tokens": 60,
            "messages": [{"role": "system", "content": system},
                         {"role": "user",
                          "content": "Reply with exactly this word and "
                                     "nothing else: " + nonce},
                         {"role": "user", "content": bulk}]}


def ask(which, nonce):
    r = urllib.request.Request(URL + "/v1/chat/completions",
                               data=json.dumps(body(which, nonce)).encode(),
                               headers={"content-type": "application/json"})
    with urllib.request.urlopen(r, timeout=900) as x:
        resp = json.loads(x.read().decode())
    return (resp["choices"][0]["message"].get("content") or "").strip()


def ready(timeout=300):
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(URL + "/slots", timeout=5)
            return True
        except Exception:
            time.sleep(3)
    return False


EXTRA = {"plain": [],
         "cram": ["-cram", "32768"],
         "mmproj": ["--mmproj", os.path.join(MODELS, "mmproj-F16.gguf")],
         "bigctx": [],
         "slotsave": ["--slot-save-path", "/tmp/claude-1000/sideslots"]}


BINARY_OVERRIDE = None


def run_case(backend):
    log = "/tmp/claude-1000/side-%s.log" % backend
    parts = backend.split("+")
    # With an override the first word is a label, not a key — so a build that
    # is not one of the three roles can be measured without inventing a role
    # for it.
    bin_ = BINARY_OVERRIDE or BIN[parts[0]]
    args = list(BASE)
    for extra in parts[1:]:
        if extra == "bigctx":
            args[args.index("-c") + 1] = "409600"
        args += EXTRA[extra]
    print("== %s (np 2, port %d)" % (backend, PORT))
    proc = runlib.start_server(args, log, bin_)
    try:
        if not ready():
            print("   server never served /slots"); return None
        bad = 0
        for i in range(3):
            for w in ("A", "B"):
                n = "%s-%d-%d" % (w, int(time.time()) % 10000, i)
                t = ask(w, n)
                corrupt = "////" in t or t.count("/") > 8
                if corrupt:
                    bad += 1
                print("   %-14s %-9s %r" % (n, "CORRUPT" if corrupt else
                                            ("ok" if n in t else "other"), t[:30]))
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
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cases", nargs="*", default=["vulkan"],
                    help="backend[+extra…]; with --binary the first word is "
                         "only a label")
    ap.add_argument("--binary",
                    help="a path, a build directory name, or a build id — "
                         "measured WITHOUT moving the production symlink")
    a = ap.parse_args()
    global BINARY_OVERRIDE
    BINARY_OVERRIDE = runlib.resolve_binary(a.binary) if a.binary else None

    meta = (runlib.provenance(BINARY_OVERRIDE) if BINARY_OVERRIDE
            else {"binary": "per case, from BIN"})
    if BINARY_OVERRIDE:
        print("binary: %s" % meta["binary"])
        print("build:  %s  (the binary itself reports %s)"
              % (meta["build_id"], meta["build_from_binary"]))

    results = {}
    for backend in (a.cases or ["vulkan"]):
        results[backend] = run_case(backend)

    # A report, because a printed number cannot be cited. Same shape as every
    # other suite here: one directory per run, the build in its name, and the
    # provenance in _meta — a reader must never have to ask which binary a
    # verdict is about.
    stamp = time.strftime("%Y-%m-%d_%H%M")
    tag = meta.get("build_id", "per-case")
    dest = os.path.join(REPO, "bench", "reports",
                        "%s_np2-candidates_%s" % (stamp, tag))
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, "result.json"), "w", encoding="utf-8") as f:
        json.dump({"_meta": meta, "cases": results}, f, indent=1,
                  ensure_ascii=False)

    print("\nRESULT")
    for b, bad in results.items():
        print("  %-16s np2 -> %s" % (b, "CORRUPT (%d of 6)" % bad if bad else
                                     ("clean 6/6" if bad == 0 else "no result")))
    print("report: %s" % dest)


if __name__ == "__main__":
    main()
