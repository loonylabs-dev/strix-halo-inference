#!/usr/bin/env python3
"""restore-safety — which ingredient makes slot restore poison the server?

Production incident 25.08.: the gateway restored a saved prefix while both
slots were busy ("all slots busy — evicts slot 0", the other slot mid-
generation with MTP/ngram speculation). From then on the server produced
degenerate output ('////…') until a fresh start. Laguna ran the same
mechanism for months — without speculation.

Four cells isolate the ingredient:

    idle-spec     restore into a fresh, idle server, speculation ON
    busy-spec     restore WHILE two generations run, speculation ON
    prefill-spec  restore WHILE a large prefill runs, speculation ON <- prod
    idle-nospec   fresh idle server, speculation OFF
    busy-nospec   two generations running, speculation OFF

(prefill-spec matches the production timeline exactly: the poisonous
restore of 25.08. hit 0.6 s after two requests had STARTED — the slots
were mid-prompt-processing, not mid-decode. First run of the other four
cells: all CLEAN, which is what pointed at the prefill path.)

Verdict per cell: three arithmetic probes after the restore (plus the
surviving generation's tail) — '391' or garbage. The decision matrix is in
The result decides one thing: idle clean -> a boot restore may come back;
busy dirty only
with spec -> the gateway needs an only-restore-when-idle guard.

    python3 bench/suites/restore-safety.py

Starts its own servers (sweep-style, service restored afterwards), saves
under exp-* names without sidecars — invisible to the gateway — and deletes
them at the end.
"""
import argparse, json, os, sys, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
REPO = os.path.dirname(BENCH)
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, BENCH)
import speed, sweep                                          # noqa: E402
import run as runlib                                         # noqa: E402
from measure import request_body                             # noqa: E402
sys.path.insert(0, os.path.join(REPO, "setup", "lib"))
import systemdfile                                           # noqa: E402

# Where the .gguf live: $LLAMA_MODELS, then ~/.config/llm-stack.env, then the
# conventional locations. No absolute path written down here.
MODELS = systemdfile.models_dir()

URL = sweep.URL
# --backend vulkan checks the HIP-race theory from llama.cpp #27572/#27579:
# if the poisonous prefill cell stays clean on Vulkan, the restore is only a
# trigger writing into an open ROCm H2D race window, not the cause.
# The DEFAULT is the binary production actually runs. It used to be
# build-rocm — the stock build — which no profile has ever pointed at:
# setup/env/qwen38.env has said build-rocm-patched since the patch existed.
# So this suite was answering "does a restore poison THAT build", about a
# binary nobody serves, and by 26.08. that stock directory was 52 upstream
# builds behind as well. Stock is kept as a deliberate comparison, not as
# the thing you get by typing nothing.
BINARIES = {"rocm-patched": os.path.expanduser("~/llama.cpp/build-rocm-patched/bin/llama-server"),
            "rocm": os.path.expanduser("~/llama.cpp/build-rocm/bin/llama-server"),
            "vulkan": os.path.expanduser("~/llama.cpp/build-vulkan/bin/llama-server")}
BINARY = BINARIES["rocm-patched"]
SLOT_DIR = os.path.expanduser("~/.cache/llama-slots")
BASE = ["--alias", "qwen38-bench",
        "-m", os.path.join(MODELS, "Qwen3.8-27B-UD-Q4_K_XL.gguf"),
        "-ngl", "999", "-fa", "on", "-c", "65536", "-np", "2",
        "--no-kv-unified", "-b", "2048", "-ub", "2048",
        "--slot-save-path", SLOT_DIR,
        "--chat-template-kwargs", '{"enable_thinking":false}',
        "--jinja", "--host", "127.0.0.1", "--port", "8080"]
SPEC = ["--spec-type", "draft-mtp,ngram-mod",
        "--spec-draft-n-max", "12", "--spec-ngram-mod-n-min", "24"]


def post(path, payload, timeout=900):
    r = urllib.request.Request(URL + path, data=json.dumps(payload).encode(),
                               headers={"content-type": "application/json",
                                        "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(r, timeout=timeout) as x:
        return json.loads(x.read().decode())


def probe():
    r = post("/v1/chat/completions", {
        "model": "exp", "max_tokens": 16,
        "messages": [{"role": "user",
                      "content": "What is 17*23? Answer with the number only."}]})
    return (r["choices"][0]["message"].get("content") or "").strip()


def probes_clean(n=3):
    texts = [probe() for _ in range(n)]
    return all("391" in t for t in texts), texts


def warm_prefix():
    """A ~7k-token state in a slot, via the production-shaped body."""
    p = speed._system_mid_conversation_remap(
        request_body(project="/tmp/restore-exp", n_tools=8,
                     question="Say alpha.", max_tokens=1))
    post("/v1/messages", p)


def slot_action(sid, action, filename):
    return post("/slots/%d?action=%s" % (sid, action),
                {"filename": filename}, timeout=300)


def big_prefill(box, project):
    """A cold ~14k-token prefill, the way Claude Code sends one."""
    try:
        p = speed._system_mid_conversation_remap(
            request_body(project=project, n_tools=24,
                         question="Say alpha.", max_tokens=8))
        box["resp"] = post("/v1/messages", p)
    except Exception as e:
        box["error"] = str(e)[:200]


def long_generation(box):
    try:
        r = post("/v1/chat/completions", {
            "model": "exp", "max_tokens": 2500,
            "messages": [{"role": "user",
                          "content": "Count from 1 to 2000, one number per "
                                     "line, no other text."}]})
        box["text"] = r["choices"][0]["message"].get("content") or ""
    except Exception as e:
        box["error"] = str(e)[:200]


def start(argv, log):
    proc = runlib.start_server(argv, log, BINARY)
    if not sweep.slots_ready(URL, 240):
        raise RuntimeError("/slots never answered")
    return proc


def gen_tail_sane(text):
    """The surviving generation must still look like counting."""
    tail = (text or "").strip().splitlines()[-5:]
    digits = sum(1 for line in tail if line.strip().rstrip(".,").isdigit())
    return digits >= 3, tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("rocm-patched", "rocm", "vulkan"),
                    default="rocm-patched")
    ap.add_argument("--cells", default="idle,busy,prefill",
                    help="comma list out of idle,busy,prefill,parallel — "
                         "'parallel' is two concurrent big prefills WITHOUT "
                         "any restore (the llama.cpp #27579 trigger)")
    ap.add_argument("--spec", choices=("spec", "nospec", "both"),
                    default="both")
    a = ap.parse_args()
    global BINARY
    BINARY = BINARIES[a.backend]
    cells = {c.strip() for c in a.cells.split(",") if c.strip()}

    sweep.reexec_with_inhibit()
    stamp = time.strftime("%Y-%m-%d_%H%M")
    dest = os.path.join(BENCH, "reports", "%s_restore-safety-%s"
                        % (stamp, a.backend))
    os.makedirs(dest, exist_ok=True)
    was_active = sweep.active_llama_unit()
    results, proc = {}, None
    configs = [c for c in (("spec", SPEC), ("nospec", []))
               if a.spec in ("both", c[0])]
    try:
        if was_active:
            sweep.stop_production(was_active)
            for _ in range(60):
                if sweep.port_free(URL):
                    break
                time.sleep(1)

        for label, extra in configs:
            fname = "exp-%s.bin" % label
            argv = BASE + extra

            if cells & {"idle", "busy", "prefill"}:
                print("\n=== %s · build state and save" % label)
                proc = start(argv, os.path.join(dest, "%s-save.log" % label))
                warm_prefix()
                slot_action(0, "save", fname)
                sweep.stop_server(proc); proc = None

            if "idle" in cells:
                print("=== %s · cell idle: restore into a fresh server" % label)
                proc = start(argv, os.path.join(dest, "%s-idle.log" % label))
                slot_action(0, "restore", fname)
                ok, texts = probes_clean()
                results["idle-%s" % label] = {"clean": ok, "probes": texts}
                print("    %s  %r" % ("CLEAN" if ok else "DIRTY", texts))
                sweep.stop_server(proc); proc = None

            if "busy" in cells:
                print("=== %s · cell busy: restore while two generations run"
                      % label)
                proc = start(argv, os.path.join(dest, "%s-busy.log" % label))
                boxes = [{}, {}]
                threads = [threading.Thread(target=long_generation, args=(b,))
                           for b in boxes]
                for t in threads:
                    t.start()
                    time.sleep(2)
                time.sleep(4)                    # both mid-generation
                slot_action(0, "restore", fname)
                ok, texts = probes_clean()
                for t in threads:
                    t.join(600)
                tails = [gen_tail_sane(b.get("text")) for b in boxes]
                gen_ok = all(s for s, _ in tails) and not any(
                    "error" in b for b in boxes)
                results["busy-%s" % label] = {
                    "clean": ok and gen_ok, "probes": texts,
                    "probes_clean": ok, "generations_sane": gen_ok,
                    "gen_tails": [t for _, t in tails],
                    "gen_errors": [b.get("error") for b in boxes]}
                print("    probes %s · generations %s"
                      % ("CLEAN" if ok else "DIRTY",
                         "SANE" if gen_ok else "DAMAGED"))
                sweep.stop_server(proc); proc = None

            if "prefill" in cells:
                print("=== %s · cell prefill: restore while a large prefill "
                      "runs" % label)
                proc = start(argv, os.path.join(dest, "%s-prefill.log" % label))
                box = {}
                th = threading.Thread(target=big_prefill, args=(
                    box, "/tmp/restore-exp-p-%s" % label))
                th.start()
                time.sleep(3)                # ~14k tokens: mid-way through
                slot_action(0, "restore", fname)
                ok, texts = probes_clean()
                th.join(600)
                prefill_ok = "error" not in box
                results["prefill-%s" % label] = {
                    "clean": ok and prefill_ok, "probes": texts,
                    "prefill_survived": prefill_ok,
                    "prefill_error": box.get("error")}
                print("    probes %s · prefill %s"
                      % ("CLEAN" if ok else "DIRTY",
                         "survived" if prefill_ok else "FAILED: %s"
                         % box.get("error")))
                sweep.stop_server(proc); proc = None

            if "parallel" in cells:
                # NO restore anywhere: two concurrent cold prefills are the
                # bare trigger condition from llama.cpp #27579 (-np >= 2,
                # long prefill). Dirty here = the production config corrupts
                # on its own and the restore was never required.
                print("=== %s · cell parallel: two concurrent big prefills, "
                      "no restore" % label)
                proc = start(argv, os.path.join(dest, "%s-parallel.log" % label))
                boxes = [{}, {}]
                threads = [threading.Thread(target=big_prefill, args=(
                    b, "/tmp/restore-exp-par-%s-%d" % (label, i)))
                    for i, b in enumerate(boxes)]
                for t in threads:
                    t.start()
                    time.sleep(1)
                for t in threads:
                    t.join(900)
                ok, texts = probes_clean()
                answers = [((b.get("resp") or {}).get("content") or [{}])
                           for b in boxes]
                errs = [b.get("error") for b in boxes]
                results["parallel-%s" % label] = {
                    "clean": ok and not any(errs), "probes": texts,
                    "prefill_errors": errs}
                print("    probes %s · prefill errors %s"
                      % ("CLEAN" if ok else "DIRTY", errs))
                sweep.stop_server(proc); proc = None
    finally:
        if proc is not None:
            sweep.stop_server(proc)
        for f in os.listdir(SLOT_DIR):
            if f.startswith("exp-"):
                os.unlink(os.path.join(SLOT_DIR, f))
        if was_active:
            print("\nrestoring %s ..." % was_active)
            sweep.start_production(was_active)
            sweep.slots_ready(URL, 900)
        with open(os.path.join(dest, "result.json"), "w") as f:
            json.dump(results, f, indent=1, ensure_ascii=False)
        print("report: %s" % dest)

    print("\nVERDICT")
    for cell in ("idle-spec", "busy-spec", "prefill-spec",
                 "idle-nospec", "busy-nospec", "prefill-nospec"):
        r = results.get(cell)
        print("  %-12s %s" % (cell, "?" if r is None else
                              ("CLEAN" if r["clean"] else "DIRTY")))


if __name__ == "__main__":
    main()
