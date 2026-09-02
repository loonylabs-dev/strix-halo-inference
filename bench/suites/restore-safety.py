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

A CELL THAT FAILS IS A RESULT, NOT THE END OF THE RUN — added 27.08., and
it is the reason this docstring says "four cells" above while the verdict
lists six. Until then the 300 s timeout of the restore in `busy-nospec`
propagated out of main(), the run ended in its finally, and `prefill-nospec`
was never measured. Three times: 26.08. twice and 27.08. once. Every one of
those reports looked complete, because the verdict prints `?` for a cell it
never reached and `?` for a cell nobody asked for.

The cell that dies may be the one carrying information, and the cells after
it are the ones the NEXT question needs. This suite exists to compare BUILDS,
and a comparison missing the same cell on both sides compares nothing there.

What the timeout in that cell actually was is worth keeping, because it was
filed as a defect (`slot-restore-hangs-busy`, now WITHDRAWN in
setup/defects.json) and blocked a decision for three sessions: a restore
queues behind the slot it targets, and the cell had put a 325-341 s
generation in front of a 300 s bound. See RESTORE_TIMEOUT below.
"""
import argparse, json, os, sys, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
REPO = os.path.dirname(BENCH)
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, BENCH)
import speed, sweep                                          # noqa: E402
import run as runlib                                         # noqa: E402
from run import provenance, resolve_binary                   # noqa: E402
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
LLAMA_SRC = runlib.LLAMA_SRC
BINARIES = {"rocm-patched": os.path.join(LLAMA_SRC, "build-rocm-patched/bin/llama-server"),
            "rocm": os.path.join(LLAMA_SRC, "build-rocm/bin/llama-server"),
            "vulkan": os.path.join(LLAMA_SRC, "build-vulkan/bin/llama-server")}
BINARY = BINARIES["rocm-patched"]
# The profile a run was given, or None. Reaches the memory guard through
# start(); see there for what leaving it out costs.
PROFILE = None
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


def base_from_profile(env_path, port="8080"):
    """BASE and SPEC for a model this suite was not written for.

    The list above is qwen38's, down to `-c 65536` and `-np 2`. Asking whether
    a restore poisons THIS server means asking it about the flags THIS server
    runs, so they are read from the profile rather than retyped — the same
    source systemd starts the unit from, via the same reader bench/sideserver.py
    uses.

    WHY THIS EXISTS AT ALL, 02.09.2026: setup/env/flashnext.env removed
    `--slot-save-path` on two upstream reviews of #27742, the QSA indexer
    carrying state the restore path does not know about, and recorded that the
    flag stays out "until `restore-safety.py` has run against THIS model.
    Nobody has run it." Nobody could: the suite could only load qwen38.

    `--slot-save-path` is ADDED here even though the profile omits it — the
    omission is what is under test, and the whole point of the cell is to find
    out whether it may come back.

    The spec flags are returned separately because the caller runs a
    spec/nospec pair; a profile that carries none gets an empty list, and the
    pair then measures the same configuration twice, which the report says.
    """
    argv = systemdfile.llama_args(env_path)
    base, spec, i = [], [], 0
    while i < len(argv):
        tok = argv[i]
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        if tok.startswith("--spec-"):
            spec.append(tok)
            if nxt is not None and not nxt.startswith("-"):
                spec.append(nxt)
                i += 1
        elif tok in ("--port", "--host", "--slot-save-path"):
            i += 1 if nxt is not None and not nxt.startswith("-") else 0
        else:
            base.append(tok)
        i += 1
    base += ["--slot-save-path", SLOT_DIR,
             "--host", "127.0.0.1", "--port", port]
    return base, spec

# How long a probe may take. The healthy value is the one that has always
# stood here: a probe queues behind two 2,500-token generations, and without
# speculation those are minutes rather than seconds, so a short bound would
# turn a slow server into a dirty one. The short value applies ONLY after the
# restore has already failed — the cell is decided by then, and the remaining
# question is merely whether the server answers at all, which a 16-token sum
# does in seconds or not at all. Nothing about a CLEAN cell changes.
PROBE_TIMEOUT_OK = 900
PROBE_TIMEOUT_AFTER_FAILURE = 120

# How long the client waits for a restore. A RESTORE QUEUES BEHIND THE SLOT
# IT TARGETS, so this bound is not a property of the restore — it is a bound
# on the work the cell itself put in front of it, and the two were never
# compared. Measured 27.08. on b10631, cell busy-nospec:
#
#     the generation occupying slot 0   335.5 s   2,500 tokens at 7.45 t/s
#     this timeout                      300   s
#
# So the restore was cut off 35 s before the slot could free, three runs
# running, and the result was filed as the defect `slot-restore-hangs-busy`.
# The same cell WITH speculation ran the same generation in 73.7 s and the
# restore returned at 68.8 — which is the asymmetry the defect entry recorded
# as unexplained, and it is the drafter, not the restore.
#
# The default stays 300 so that no recorded report changes meaning. What is
# new is that a cell now says when the bound was the shorter of the two.
RESTORE_TIMEOUT_DEFAULT = 300
RESTORE_TIMEOUT = RESTORE_TIMEOUT_DEFAULT


def post(path, payload, timeout=900):
    r = urllib.request.Request(URL + path, data=json.dumps(payload).encode(),
                               headers={"content-type": "application/json",
                                        "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(r, timeout=timeout) as x:
        return json.loads(x.read().decode())


def probe(timeout=PROBE_TIMEOUT_OK):
    r = post("/v1/chat/completions", {
        "model": "exp", "max_tokens": 16,
        "messages": [{"role": "user",
                      "content": "What is 17*23? Answer with the number only."}]},
        timeout=timeout)
    return (r["choices"][0]["message"].get("content") or "").strip()


def probes_clean(n=3, timeout=PROBE_TIMEOUT_OK):
    """Three probes — unless one of them does not come back.

    A probe that raises has already answered the question the suite asks, so
    it is recorded verbatim and the remaining probes are skipped: two more
    waits of the same length only add to how long production is down. And
    recorded VERBATIM matters — a cell whose probes read
    '<TimeoutError: ...>' is dirty for a different reason than one answering
    '////', and a boolean cannot tell those apart.
    """
    texts = []
    for _ in range(n):
        try:
            texts.append(probe(timeout))
        except Exception as e:
            texts.append("<%s: %s>" % (type(e).__name__, str(e)[:120]))
            break
    return (len(texts) == n and all("391" in t for t in texts)), texts


def warm_prefix():
    """A ~7k-token state in a slot, via the production-shaped body."""
    p = speed._system_mid_conversation_remap(
        request_body(project="/tmp/restore-exp", n_tools=8,
                     question="Say alpha.", max_tokens=1))
    post("/v1/messages", p)


def slot_action(sid, action, filename, timeout=None):
    return post("/slots/%d?action=%s" % (sid, action),
                {"filename": filename},
                timeout=RESTORE_TIMEOUT if timeout is None else timeout)


def timed_restore(r, sid, fname):
    """Restore, and record whether it came back rather than raising.

    This is the one call in the suite that is KNOWN to hang: defect
    `slot-restore-hangs-busy`. Letting it raise past the cell threw away
    three things that were already paid for — how long it took, whether the
    server still answers afterwards, and every cell that came later.
    """
    r["phase"] = "restore"
    r["restore_timeout"] = RESTORE_TIMEOUT
    t0 = time.time()
    try:
        slot_action(sid, "restore", fname)
        r["restore_returned"] = True
    except Exception as e:
        r["restore_returned"] = False
        r["restore_error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
    r["restore_seconds"] = round(time.time() - t0, 1)
    if not r["restore_returned"]:
        print("    restore did NOT return after %.0f s — %s"
              % (r["restore_seconds"], r["restore_error"]))
    return r["restore_returned"]


def probe_timeout_for(r):
    return (PROBE_TIMEOUT_OK if r.get("restore_returned")
            else PROBE_TIMEOUT_AFTER_FAILURE)


def big_prefill(box, project):
    """A cold ~14k-token prefill, the way Claude Code sends one."""
    box["seconds"] = None
    t0 = time.time()
    try:
        p = speed._system_mid_conversation_remap(
            request_body(project=project, n_tools=24,
                         question="Say alpha.", max_tokens=8))
        box["resp"] = post("/v1/messages", p)
    except Exception as e:
        box["error"] = str(e)[:200]
    finally:
        box["seconds"] = round(time.time() - t0, 1)


def long_generation(box):
    box["seconds"] = None
    t0 = time.time()
    try:
        r = post("/v1/chat/completions", {
            "model": "exp", "max_tokens": 2500,
            "messages": [{"role": "user",
                          "content": "Count from 1 to 2000, one number per "
                                     "line, no other text."}]})
        box["text"] = r["choices"][0]["message"].get("content") or ""
    except Exception as e:
        box["error"] = str(e)[:200]
    finally:
        box["seconds"] = round(time.time() - t0, 1)


def spawn(target, args):
    """Daemon threads on purpose.

    A worker whose request is stuck against a wedged server never returns.
    As a non-daemon thread it would hold the interpreter open at exit, so
    the run that already recorded its finding would hang on the way out —
    the abort one level up.
    """
    t = threading.Thread(target=target, args=args, daemon=True)
    t.start()
    return t


def join_all(threads, budget):
    """Join with a bound; report what was still running rather than wait."""
    alive = 0
    for t in threads:
        t.join(budget)
        if t.is_alive():
            alive += 1
    return alive


def start(argv, log):
    # PROFILE reaches the memory guard, and leaving it out is not a smaller
    # check — check_room_for's own docstring calls it "a DIFFERENT and wronger
    # one", and names Qwen3.8-Flash-Next as the model the profile figures were
    # written for. Measured 02.09.2026 on exactly that model: without the
    # profile the guard asked for 155.6 GiB and refused a server that
    # sideserver.py starts at 89.9 GiB of GTT with the same argv.
    proc = runlib.start_server(argv, log, BINARY, env=PROFILE)
    if not sweep.slots_ready(URL, 240):
        raise RuntimeError("/slots never answered")
    return proc


def gen_tail_sane(text):
    """The surviving generation must still look like counting."""
    tail = (text or "").strip().splitlines()[-5:]
    digits = sum(1 for line in tail if line.strip().rstrip(".,").isdigit())
    return digits >= 3, tail


# ---------------------------------------------------------------- the cells --
def cell_idle(r):
    ok, texts = probes_clean(timeout=probe_timeout_for(r))
    r.update(clean=ok, probes=texts)
    print("    %s  %r" % ("CLEAN" if ok else "DIRTY", texts))


def note_what_blocked(r, boxes):
    """What the restore had to wait for, beside its own bound.

    Without this the report says "the restore did not return within 300 s"
    and a reader concludes the server wedged. What it actually says is that
    the cell asked for a restore into a slot it had just filled with 2,500
    tokens of work, and then gave up before that work could finish. A bound
    that is shorter than the thing it bounds is not a measurement of the
    thing — it is a measurement of the bound.
    """
    held = [b.get("seconds") for b in boxes if b.get("seconds") is not None]
    r["blocking_seconds"] = max(held) if held else None
    if (r.get("restore_returned") is False and r["blocking_seconds"]
            and r["blocking_seconds"] > r.get("restore_timeout", 0)):
        r["timeout_was_shorter_than_the_work"] = True
        print("    the slot was occupied for %.0f s — LONGER than the %.0f s "
              "the restore was given" % (r["blocking_seconds"],
                                         r["restore_timeout"]))


def cell_busy(r, fname):
    boxes = [{}, {}]
    threads = []
    for b in boxes:
        threads.append(spawn(long_generation, (b,)))
        time.sleep(2)
    time.sleep(4)                                    # both mid-generation
    restored = timed_restore(r, 0, fname)
    r["phase"] = "probes"
    ok, texts = probes_clean(timeout=probe_timeout_for(r))
    r["phase"] = "generations"
    r["generations_still_running"] = join_all(threads, 600 if restored else 60)
    tails = [gen_tail_sane(b.get("text")) for b in boxes]
    gen_ok = (all(s for s, _ in tails) and not any("error" in b for b in boxes)
              and r["generations_still_running"] == 0)
    r.update(clean=restored and ok and gen_ok, probes=texts, probes_clean=ok,
             generations_sane=gen_ok, gen_tails=[t for _, t in tails],
             gen_seconds=[b.get("seconds") for b in boxes],
             gen_errors=[b.get("error") for b in boxes], phase="done")
    note_what_blocked(r, boxes)
    print("    probes %s · generations %s"
          % ("CLEAN" if ok else "DIRTY", "SANE" if gen_ok else "DAMAGED"))


def cell_prefill(r, fname, project):
    box = {}
    th = spawn(big_prefill, (box, project))
    time.sleep(3)                                # ~14k tokens: mid-way through
    restored = timed_restore(r, 0, fname)
    r["phase"] = "probes"
    ok, texts = probes_clean(timeout=probe_timeout_for(r))
    r["phase"] = "prefill"
    r["prefill_still_running"] = join_all([th], 600 if restored else 60)
    prefill_ok = "error" not in box and r["prefill_still_running"] == 0
    r.update(clean=restored and ok and prefill_ok, probes=texts,
             prefill_survived=prefill_ok, prefill_error=box.get("error"),
             prefill_seconds=box.get("seconds"), phase="done")
    note_what_blocked(r, [box])
    print("    probes %s · prefill %s"
          % ("CLEAN" if ok else "DIRTY",
             "survived" if prefill_ok else "FAILED: %s" % box.get("error")))


def cell_parallel(r, label):
    # NO restore anywhere: two concurrent cold prefills are the bare trigger
    # condition from llama.cpp #27579 (-np >= 2, long prefill). Dirty here =
    # the production config corrupts on its own and the restore was never
    # required.
    boxes = [{}, {}]
    threads = []
    for i, b in enumerate(boxes):
        threads.append(spawn(big_prefill,
                             (b, "/tmp/restore-exp-par-%s-%d" % (label, i))))
        time.sleep(1)
    r["phase"] = "prefills"
    r["prefills_still_running"] = join_all(threads, 900)
    r["phase"] = "probes"
    ok, texts = probes_clean()
    errs = [b.get("error") for b in boxes]
    r.update(clean=ok and not any(errs) and r["prefills_still_running"] == 0,
             probes=texts, prefill_errors=errs, phase="done")
    print("    probes %s · prefill errors %s"
          % ("CLEAN" if ok else "DIRTY", errs))


# --------------------------------------------------------------- the runner --
def run_cell(name, results, save, argv, logfile, body):
    """One cell: start a server, run the body, always stop the server.

    Every failure is caught HERE and written into the cell, so a cell can
    only ever end the cell. `phase` says where it died, which is the
    difference between "the restore never returned" and "the probes after it
    did not answer" — two findings that used to arrive as the same traceback.

    SystemExit is caught alongside Exception on purpose: runlib.start_server
    raises it when the server does not come up or the memory guard refuses,
    and that is a per-cell fact worth recording. If it is a standing
    condition, every cell records it and the report says so six times, which
    is more useful than one traceback and five cells nobody ran.
    """
    r = {"phase": "start"}
    results[name] = r
    proc = None
    try:
        proc = start(argv, logfile)
        r["phase"] = "body"
        body(r)
    except (Exception, SystemExit) as e:
        r["clean"] = False
        r["aborted"] = True
        r["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
        print("    ABORTED in phase %r — %s" % (r.get("phase"), r["error"]))
    finally:
        if proc is not None:
            try:
                sweep.stop_server(proc)
            except Exception as e:                            # noqa: BLE001
                print("    (teardown: %s)" % e)
        save()
    return r


def verdict_line(r):
    """A cell that was not measured must never read like one that passed —
    and must never read like one that failed either. Three states, three
    words: the reports of 26./27.08. had two, and the cells the aborted run
    never reached printed the same `?` as the cells nobody asked for."""
    if r is None:
        return "?         not run"
    if r.get("skipped"):
        return "SKIPPED   %s" % r.get("error", "")
    if r.get("clean"):
        return "CLEAN"
    why = []
    if r.get("aborted"):
        why.append("aborted in %s: %s" % (r.get("phase"), r.get("error")))
    if r.get("restore_returned") is False:
        why.append("restore did not return within %.0f s"
                   % r.get("restore_seconds", 0))
        if r.get("timeout_was_shorter_than_the_work"):
            why.append("BUT the slot it targets was occupied for %.0f s — "
                       "the bound was the shorter of the two"
                       % r["blocking_seconds"])
    if r.get("probes") and not all("391" in t for t in r["probes"]):
        why.append("probes %r" % (r["probes"],))
    if r.get("generations_sane") is False:
        why.append("generations damaged")
    if r.get("prefill_survived") is False:
        why.append("prefill failed")
    return "DIRTY     " + " · ".join(why or ["see result.json"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("rocm-patched", "rocm", "vulkan"),
                    default="rocm-patched")
    ap.add_argument("--binary",
                    help="measure a specific build without moving the "
                         "production symlink: a path to llama-server, a build "
                         "directory name under ~/llama.cpp, or a build id")
    ap.add_argument("--cells", default="idle,busy,prefill",
                    help="comma list out of idle,busy,prefill,parallel — "
                         "'parallel' is two concurrent big prefills WITHOUT "
                         "any restore (the llama.cpp #27579 trigger)")
    ap.add_argument("--spec", choices=("spec", "nospec", "both"),
                    default="both")
    ap.add_argument("--restore-timeout", type=float, default=RESTORE_TIMEOUT_DEFAULT,
                    help="seconds the client waits for a restore. A restore "
                         "queues behind the slot it targets, so a bound "
                         "below the cell's own workload measures the bound "
                         "(default: %(default)s)")
    ap.add_argument("--env",
                    help="a setup/env/*.env profile: take the model, its flags "
                         "and LLAMA_BIN from there instead of the qwen38 list "
                         "built into this file. Required to answer the "
                         "question for any other model")
    a = ap.parse_args()
    global BINARY, RESTORE_TIMEOUT, BASE, SPEC, PROFILE
    RESTORE_TIMEOUT = a.restore_timeout
    profile = None
    if a.env:
        if not os.path.exists(a.env):
            raise SystemExit("no such profile: %s" % a.env)
        PROFILE = a.env
        BASE, SPEC = base_from_profile(a.env)
        if a.spec == "both" and not SPEC:
            print("NOTE %s carries no --spec-* flags, so the spec/nospec pair "
                  "measures the same configuration twice." % a.env)
        # A profile that pins LLAMA_BIN pins it for a reason (setup/README.md,
        # family table). Honour it unless the caller names a binary outright —
        # measuring a profile's model on some other build answers about
        # neither.
        if not a.binary:
            pinned = systemdfile.variable(a.env, "LLAMA_BIN")
            if pinned:
                a.binary = os.path.expanduser("~/" + pinned)
    BINARY = resolve_binary(a.binary, BINARIES[a.backend])
    cells = {c.strip() for c in a.cells.split(",") if c.strip()}

    sweep.reexec_with_inhibit()
    meta = provenance(BINARY)
    meta["restore_timeout"] = RESTORE_TIMEOUT
    stamp = time.strftime("%Y-%m-%d_%H%M")
    # The build belongs in the NAME, not only in the file. Two runs of two
    # builds in one session used to land in directories distinguished by the
    # minute they started.
    #
    # And the label is the build's FAMILY, not --backend. --backend is a role
    # that --binary overrides, so with both in play the directory name said
    # `rocm-patched` for a build stamped `patched=no` — the file was honest and
    # its name was not, which is the worse half. Believed only when the stamp
    # has been checked against the binary; otherwise the role, which is all
    # that is known.
    label = a.backend
    if meta.get("stamp_matches_binary") and meta.get("stamp", {}).get("family"):
        label = meta["stamp"]["family"]
    meta["label"] = label
    dest = os.path.join(BENCH, "reports", "%s_restore-safety-%s_%s"
                        % (stamp, label, meta["build_id"]))
    os.makedirs(dest, exist_ok=True)
    print("binary: %s" % BINARY)
    print("build:  %s  (the binary itself reports %s)"
          % (meta["build_id"], meta["build_from_binary"]))
    was_active = sweep.active_llama_unit()
    results = {"_meta": meta}
    configs = [c for c in (("spec", SPEC), ("nospec", []))
               if a.spec in ("both", c[0])]

    def save():
        """After every cell, not only at the end. A run that is killed keeps
        what it had measured — which the three truncated runs of 26./27.08.
        did not."""
        with open(os.path.join(dest, "result.json"), "w",
                  encoding="utf-8") as f:
            json.dump(results, f, indent=1, ensure_ascii=False)

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
            def log(n, label=label):
                return os.path.join(dest, "%s-%s.log" % (label, n))

            saved = True
            if cells & {"idle", "busy", "prefill"}:
                print("\n=== %s · build state and save" % label)

                def do_save(r):
                    warm_prefix()
                    slot_action(0, "save", fname)
                    r.update(clean=True, phase="done")

                saved = run_cell("save-%s" % label, results, save, argv,
                                 log("save"), do_save).get("clean", False)
                if not saved:
                    # Without the saved state there is nothing to restore, so
                    # the three cells below would measure the harness rather
                    # than the build. Say which cells were skipped and why —
                    # a missing key in the verdict must never read as CLEAN.
                    print("    no saved state — skipping idle/busy/prefill "
                          "for %s" % label)
                    for c in ("idle", "busy", "prefill"):
                        if c in cells:
                            results["%s-%s" % (c, label)] = {
                                "clean": None, "skipped": True,
                                "error": "the save cell failed, so there was "
                                         "no state to restore"}
                    save()

            if "idle" in cells and saved:
                print("=== %s · cell idle: restore into a fresh server" % label)

                def body(r, fname=fname):
                    timed_restore(r, 0, fname)
                    cell_idle(r)

                run_cell("idle-%s" % label, results, save, argv, log("idle"),
                         body)

            if "busy" in cells and saved:
                print("=== %s · cell busy: restore while two generations run"
                      % label)
                run_cell("busy-%s" % label, results, save, argv, log("busy"),
                         lambda r, fname=fname: cell_busy(r, fname))

            if "prefill" in cells and saved:
                print("=== %s · cell prefill: restore while a large prefill "
                      "runs" % label)
                proj = "/tmp/restore-exp-p-%s" % label
                run_cell("prefill-%s" % label, results, save, argv,
                         log("prefill"),
                         lambda r, fname=fname, p=proj: cell_prefill(r, fname, p))

            if "parallel" in cells:
                print("=== %s · cell parallel: two concurrent big prefills, "
                      "no restore" % label)
                run_cell("parallel-%s" % label, results, save, argv,
                         log("parallel"),
                         lambda r, label=label: cell_parallel(r, label))
    finally:
        for f in os.listdir(SLOT_DIR):
            if f.startswith("exp-"):
                os.unlink(os.path.join(SLOT_DIR, f))
        if was_active:
            print("\nrestoring %s ..." % was_active)
            sweep.start_production(was_active)
            sweep.slots_ready(URL, 900)
        save()
        print("report: %s" % dest)

    print("\nVERDICT  (%s)" % meta["build_id"])
    for cell in ("idle-spec", "busy-spec", "prefill-spec",
                 "idle-nospec", "busy-nospec", "prefill-nospec"):
        print("  %-14s %s" % (cell, verdict_line(results.get(cell))))


if __name__ == "__main__":
    main()
