#!/usr/bin/env python3
"""Can we get -np 2 back? Test the candidates on a SIDE server (port 8081),
so production keeps running untouched.

    vulkan-np2   the Vulkan backend with two slots
    rocm-np2     the same on ROCm, as the control that must fail
"""
import json, os, signal, subprocess, sys, time, urllib.request

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


def run_case(backend):
    log = "/tmp/claude-1000/side-%s.log" % backend
    parts = backend.split("+")
    bin_ = BIN[parts[0]]
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


results = {}
for backend in (sys.argv[1:] or ["vulkan"]):
    results[backend] = run_case(backend)
print("\nRESULT")
for b, bad in results.items():
    print("  %-8s np2 -> %s" % (b, "CORRUPT (%d)" % bad if bad else
                                ("clean" if bad == 0 else "no result")))
