#!/usr/bin/env python3
"""run — carry out the cache measurements against a given configuration.

The point: the findings in docs/ hold for *one* state — Laguna S 2.1,
llama.cpp b10577, Vulkan, particular flags. As soon as model, build or flags
change, they have to be proven again. This script turns that into one call.

    python3 bench/run.py --env setup/env/laguna.env
    python3 bench/run.py --env setup/env/gemma26.env --suites basic,tools
    python3 bench/run.py --running                  # against a running server

Every run writes to bench/reports/<date>_<model>_<build>/ with the full
context, so that results stay comparable across states.
"""
import argparse, json, os, re, signal, subprocess, sys, time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "setup", "lib"))
sys.path.insert(0, HERE)
from synthetic import body                      # noqa: E402
from measure import evaluate, gtt_gib             # noqa: E402
from systemdfile import llama_args               # noqa: E402
import systemdfile                               # noqa: E402
import budget                                    # noqa: E402  the one memory guard

SRV = "http://127.0.0.1:8080"

# ------------------------------------------------------------ Hilfsmittel ---
def post(path, payload, t=1800):
    r = urllib.request.Request(SRV + path, data=json.dumps(payload).encode(),
                               headers={"content-type": "application/json",
                                        "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(r, timeout=t) as x:
        return json.loads(x.read().decode())

def get(path, t=60):
    with urllib.request.urlopen(SRV + path, timeout=t) as x:
        return json.loads(x.read().decode())

def measure_one(p, t=1800):
    p = dict(p); p["stream"] = False; p["max_tokens"] = 1
    t0 = time.time()
    r = post("/v1/messages", p, t)
    # evaluate() aborts rather than computing a rate of -0.0 % out of an
    # answer that carries no usage. See measure.py.
    return evaluate(r, time.time() - t0)

def row(label, m):
    print("   %-34s new=%6d cached=%6d (%5.1f%%) %7.1fs"
          % (label, m["new"], m["cached"], m["rate"], m["seconds"]))
    sys.stdout.flush()

def gtt():
    return gtt_gib()

def clear_slots():
    # Ask for the slot count instead of guessing: with more than eight slots
    # the rest stayed filled, and the next "cold" measurement would be none.
    try:
        n = max(len(get("/slots", t=10)), 8)
    except Exception:
        n = 8
    for i in range(n):
        try:
            urllib.request.urlopen(urllib.request.Request(
                SRV + "/slots/%d?action=erase" % i, data=b"{}", method="POST",
                headers={"content-type": "application/json"}), timeout=30).read()
        except Exception:
            pass

# ----------------------------------------------------------------- Suiten ---
def suite_basic(_):
    """The core case: the same request, only the question changed."""
    out = {}
    clear_slots()
    out["cold"] = measure_one(body(question="Say alpha.")); row("cold (fuellt Slot)", out["cold"])
    out["identical"] = measure_one(body(question="Say alpha.")); row("identical repeated", out["identical"])
    out["changed"] = measure_one(body(question="Say beta.")); row("changed question", out["changed"])
    return out

def suite_tools(_):
    """Tool conversation over four turns — pure appending."""
    out = {}
    clear_slots()
    for t in range(1, 5):
        m = measure_one(body(question="Read the file.", turns=t))
        out["turn%d" % t] = m; row("Turn %d" % t, m)
    return out

def suite_multiproject(arg):
    """N project prefixes in rotation — exercises slots and the RAM cache."""
    n = int(arg or 4)
    out = {}
    clear_slots()
    for i in range(1, n + 1):
        m = measure_one(body(project="/tmp/proj%d" % i, question="Say alpha."))
        out["warm_p%d" % i] = m; row("P%d warm up" % i, m)
    for round_ in (1, 2):
        for i in range(1, n + 1):
            m = measure_one(body(project="/tmp/proj%d" % i, question="Say alpha."))
            out["r%d_p%d" % (round_, i)] = m; row("R%d P%d same question" % (round_, i), m)
    return out

def suite_similar(_):
    """The pathological case: same project, slightly different tool set."""
    out = {}
    clear_slots()
    for round_ in (1, 2, 3):
        a = measure_one(body(project="/tmp/projA", n_tools=24, question="Say alpha."))
        out["r%d_voll" % round_] = a; row("R%d projA full" % round_, a)
        b = measure_one(body(project="/tmp/projA", n_tools=23, question="Say alpha."))
        out["r%d_minus1" % round_] = b; row("R%d projA -1 tool" % round_, b)
    return out

def suite_swa(_):
    """SWA metadata of the GGUFs under the model path — pure inventory."""
    # MODELLPFAD was the pre-rename name and kept working by accident; it is
    # an alias now, and the answer itself comes from the one resolver.
    models = (os.environ.get("LLAMA_MODELS") or os.environ.get("MODELLPFAD")
              or systemdfile.models_dir())
    out = {}
    try:
        sys.path.insert(0, os.path.expanduser("~/llama.cpp/gguf-py"))
        from gguf import GGUFReader
    except Exception as e:
        print("   (gguf-py not available: %s)" % e)
        return {"error": str(e)}
    import glob
    for path in sorted(glob.glob(os.path.join(models, "*.gguf"))):
        if re.search(r"-0000[2-9]-of-", path):
            continue
        try:
            r = GGUFReader(path)
            arch = str(r.fields["general.architecture"].contents())
            sw = None
            for k, f in r.fields.items():
                if k.endswith("attention.sliding_window"):
                    sw = int(f.contents())
            out[os.path.basename(path)] = {"arch": arch, "sliding_window": sw}
            print("   %-46s %-10s sliding_window=%s"
                  % (os.path.basename(path)[:46], arch, sw if sw is not None else "none"))
        except Exception as e:
            out[os.path.basename(path)] = {"error": str(e)}
    return out

SUITES = {
    "basic":       ("core case: changed question",          suite_basic),
    "tools":       ("tool conversation over four turns",    suite_tools),
    "multiproject":("N projects in rotation",               suite_multiproject),
    "similar":     ("similar prefixes in the same project", suite_similar),
    "swa":         ("SWA inventory of the models",          suite_swa),
}

# ------------------------------------------------------------------ server ---
def args_from_env(path):
    """LLAMA_ARGS from a profile, split the way systemd splits it.

    The implementation moved to setup/lib/systemdfile.py on 26.08. and this is
    now a name kept for the callers. It had to move: three parsers existed,
    they disagreed, and a bench harness that reads the profile differently
    from the service is not measuring the service. The two failures that
    cost the most here — shlex.split stripping the quotes out of
    --chat-template-kwargs, and a regex running past the end of the
    assignment into the comment lines — are pinned in
    tests/test_models.py::TestArgsReader.
    """
    return llama_args(path)

# --------------------------------------------------- the memory guard ------
#
# The formula is NOT here. It lives in setup/lib/budget.py, because it lived
# in three places at once until 27.08. and the three had drifted: this file
# charged weights x 1.10 with a host reserve of 10, sideserver.py used 12 for
# the same machine, and tests/test_models.py added -cram but no KV at all.
# None of the three sat where a model actually gets started.
#
# What stays here is the ADAPTER: the readers below are module-level on
# purpose so a test can replace them, and check_room_for() hands their values
# to the pure functions rather than letting the core read the world twice.
#
# The history the guard exists for is in budget.py's docstring. The short
# version: GTT comes out of system RAM, is not swappable, and is not charged
# to any process in a way `free` makes obvious — so a model that does not fit
# does not page and does not get OOM-killed. It hangs the machine. That
# happened three times on 26.08.2026.
HOST_RESERVE_GIB = budget.host_reserve_gib()


def _gtt(which):
    return budget._gtt_gib(which)


def _mem_available_gib():
    return budget._meminfo_gib("MemAvailable")


def _model_size_gib(argv):
    """The weights a profile will load, in GiB, or None.

    A sharded GGUF names part one and finds the rest, so the siblings are
    counted too — the whole point is not to underestimate.
    """
    return budget.weights_gib(argv)


def _who_holds_gtt():
    out = []
    try:
        pids = subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True,
                              text=True, timeout=10).stdout.split()
    except Exception:
        return out
    for pid in pids:
        try:
            with open("/proc/%s/cmdline" % pid) as f:
                argv = f.read().split("\0")
            alias = argv[argv.index("--alias") + 1] if "--alias" in argv else "?"
            out.append("pid %s (%s)" % (pid, alias))
        except Exception:
            pass
    return out


def check_room_for(argv, what="this server", env=None, binary=None):
    """Refuse to start a server that does not fit in what is left.

    Checked against BOTH limits, because either one alone lets the machine
    down: the GTT cap bounds what amdgpu may hold, and MemAvailable bounds
    everything — and GTT allocations cannot be swapped out to make room.

    The readers are looked up as module globals at call time, so the tests can
    substitute a machine. Everything else is budget.py's.

    `env` is the profile whose numbers this server was measured with, and
    leaving it out is not a smaller check — it is a DIFFERENT and wronger one.
    budget.plan() takes three measurements a profile may carry, and its
    docstring names Qwen3.8-Flash-Next as the reason they exist; this call site
    passed none of them, so the one model they were written for was judged on
    the estimate they were meant to replace. Measured 28.08.: 122.9 GiB of GTT
    and 126.9 resident against the profile's 80.7 and 111.8, an overstatement
    of 42 GiB, and the KV charged at an estimated 8.0 GiB where the profile
    declares 2.3.
    """
    if budget.guard_disabled():
        return
    on_disk = _model_size_gib(argv)
    plan = budget.plan(argv, on_disk, what=what,
                       declared=budget.declared_kv(env),
                       gtt_base=budget.declared_gtt(env),
                       host_anon=budget.declared_anon(env),
                       lazy=budget.lazy_relief(env, argv, binary))
    machine = budget.Machine(mem_total=budget._meminfo_gib("MemTotal"),
                             mem_available=_mem_available_gib(),
                             gtt_total=_gtt("total"), gtt_used=_gtt("used"))
    v = budget.verdict(plan, machine, reserve=HOST_RESERVE_GIB)
    if v.fits:
        return
    holders = _who_holds_gtt()
    msg = budget.refusal(plan, machine, v)
    if holders:
        msg += "\n  holding GTT right now: %s\n" % ", ".join(holders)
    raise SystemExit(msg)



def wait_for_gtt_release(before_gib, timeout=120):
    """After stopping a server, wait for the MEMORY to come back.

    Polling /health only says the port is closed. The process can be gone from
    the port while its GTT allocation is still being torn down, and the next
    server then loads on top of it. That transition is what took the machine
    down on 26.08.
    """
    end = time.time() + timeout
    while time.time() < end:
        now = _gtt("used")
        if now is None or now <= before_gib + 1.0:
            return True
        time.sleep(2)
    return False


def wait_for_gtt_to_settle(timeout=180, quiet_s=6.0, tol=0.3):
    """Wait until GTT STOPS FALLING — not until it reaches a number.

    wait_for_gtt_release() above answers "is it back to what it was before I
    started something", and that needs a baseline the caller measured itself.
    After stopping a server SOMEBODY ELSE started there is no such baseline,
    and bench/sideserver.py passed 0.0 for it — which with the +1.0 GiB
    tolerance means "wait until GTT is below 1 GiB".

    On this machine the DESKTOP holds 1.5 GiB and never gives it back, so that
    condition cannot be met while anyone is logged in. Measured 28.08.2026:
    production stopped, GTT already down to 1.5, and the wait still ran its
    full 180 s and refused the start. The guard was not wrong about the
    memory; it was asking a question with no answer on this machine, and the
    answer it defaulted to was "refuse".

    The question that CAN be answered without a baseline is whether the
    teardown has finished: GTT falls while amdgpu unmaps and then stops. So
    this waits for the reading to hold still — within `tol` for `quiet_s` —
    and returns the settled value, or None if it was still moving when the
    timeout ran out.

    It deliberately does not judge whether that value is SMALL ENOUGH.
    check_room_for() does, against the profile and the machine, and refuses
    with the arithmetic in hand. Two questions, two functions: one asks
    whether the teardown is over, the other whether the next model fits.
    """
    end = time.time() + timeout
    last, stable_since = None, None
    while time.time() < end:
        now = _gtt("used")
        if now is None:
            # No GTT to read — a machine without amdgpu, or the file gone.
            # Same answer as wait_for_gtt_release gives there: do not block a
            # measurement over a question this machine cannot be asked.
            return 0.0
        if last is not None and abs(now - last) <= tol:
            stable_since = stable_since or time.time()
            if time.time() - stable_since >= quiet_s:
                return now
        else:
            stable_since = None
        last = now
        time.sleep(2)
    return None


def start_server(argv, logfile, binary, env=None):
    # LLAMA_SERVER_SLOTS_DEBUG stand hier fest auf "1". Damit gibt /slots den
    # LLAMA_SERVER_SLOTS_DEBUG used to be hard-wired to "1" here. That makes
    # /slots hand out the complete rendered prompt of every slot — per
    # docs/SECURITY.md the single worst finding of this project. None of the
    # five suites here reads prompts; it is needed only by
    # bench/suites/replay.py, which runs standalone and can set it itself.
    # Side effect before: during a measurement run setup/smoketest.sh reported
    # an exposure that was no regression at all.
    # env_ was referenced here without ever being defined — every --env run
    # died with a NameError before the server came up; only --running worked.
    check_room_for(argv, os.path.basename(binary), env=env, binary=binary)
    env_ = dict(os.environ)
    if os.environ.get("SLOTS_DEBUG") == "1":
        env_["LLAMA_SERVER_SLOTS_DEBUG"] = "1"
    # Closed straight after the fork: the child holds its own descriptor for
    # the same file, so keeping the parent's open leaks one per server start
    # and does nothing. Found 27.08. as the last ResourceWarning left in the
    # suite, once the other 560 had been cleared out of the CI log.
    log = open(logfile, "w")
    p = subprocess.Popen([binary] + argv, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True, env=env_)
    log.close()
    for _ in range(600):
        time.sleep(1)
        if p.poll() is not None:
            raise SystemExit("server exits immediately, see %s" % logfile)
        try:
            with open(logfile, encoding="utf-8", errors="replace") as f:
                if "model loaded" in f.read():
                    return p
        except Exception:
            pass
    # Do not abort without cleaning up: otherwise the process keeps running,
    # holds the GPU and blocks the port for the next run.
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except Exception:
        pass
    raise SystemExit("server did not finish in time, see %s" % logfile)
    log.close()

# ---------------------------------------------------------------- builds ---
# Which llama-server a measurement is about, and what produced its numbers.
#
# These live here rather than in one suite because two suites now need them:
# bench/suites/restore-safety.py, which compares builds, and
# bench/suites/np2-candidates.py, which asks the same question of the other
# gfx1151 defect. The second copy of anything is where this repository's bugs
# live — three parsers for LLAMA_ARGS, three copies of the memory arithmetic,
# a convention list in three places within three hours.
LLAMA_SRC = os.path.expanduser(os.environ.get("LLAMA_SRC", "~/llama.cpp"))


def resolve_binary(spec, default=None):
    """Which llama-server to measure.

    `--backend` names a ROLE: `rocm-patched` is whatever the production
    symlink points at today. That is the right default and the wrong handle
    for a BUILD COMPARISON, which is what this suite is used for — "does
    llama.cpp PR #27311 fix this" means running one build against another,
    and through --backend alone the only way to reach a second build is to
    move the symlink that production starts from. A measurement must not
    require a production change first, and a rollback must not be the thing
    standing between a report and a serving machine.

    So `--binary` takes a path, a build directory name, or a build id.
    """
    if not spec:
        if default is None:
            raise SystemExit("no binary: pass --binary, or give a default")
        return default
    if "/" in spec or spec.startswith("~"):
        cands = [os.path.expanduser(spec)]
    else:
        # Three shapes, because the help text promises "a build directory
        # name, or a build id" and the first version honoured only two of
        # them: `--binary rocm` — the stock build, and the obvious thing to
        # type — resolved to nothing. Found by typing it, which is the only
        # way this kind of gap is ever found.
        cands = [os.path.join(LLAMA_SRC, spec, "bin", "llama-server"),
                 os.path.join(LLAMA_SRC, "build-" + spec, "bin", "llama-server"),
                 os.path.join(LLAMA_SRC, "build-rocm-patched-" + spec,
                              "bin", "llama-server")]
    for c in cands:
        if os.access(c, os.X_OK):
            return c
    raise SystemExit("no executable llama-server for --binary %r. Tried:\n  %s"
                     % (spec, "\n  ".join(cands)))


def provenance(binary):
    """What produced these numbers.

    result.json recorded the cells and nothing about the build until 27.08.
    For a suite whose entire output is "clean or dirty ON THIS BINARY" that
    is the one field a reader cannot reconstruct afterwards: the report
    directory carried the backend LABEL, and a label is a role, not a build.
    Two runs a day apart under the same name were two different binaries and
    said so nowhere.
    """
    reported = build_id(binary)
    # RECORDED unexpanded. A report lives in the repository and is read on
    # other machines: "/home/<someone>/llama.cpp/build-.../bin/llama-server"
    # names a person and tells a reader nothing they can use. What identifies
    # the binary is the stamp and the commit, both recorded beside it. Three
    # reports were written with the raw path before tests/test_localenv.py
    # said so — the same rule bench/sweep.py learned earlier the same day.
    meta = {"binary": systemdfile.unexpand(binary),
            "build_from_binary": reported, "build_id": reported,
            # The ENVIRONMENT is part of the configuration. llama.cpp reads
            # GGML_SCHED_UMA_RING, LLAMA_SET_ROWS and friends at runtime, so
            # two runs of the same binary with the same argv can be two
            # different experiments. Added to one suite on 28.08. while
            # exactly that variable WAS the independent variable — and not
            # carried across, so the restore-safety run that carries the
            # single-cause finding recorded nothing. It lives in the shared
            # reader now, where a suite cannot forget it.
            "env": {k: v for k, v in sorted(os.environ.items())
                    if k.startswith(("GGML_", "LLAMA_")) and k != "LLAMA_SRC"}}
    stamp = os.path.join(os.path.dirname(os.path.dirname(binary)),
                         ".build-stamp")
    if not os.path.exists(stamp):
        return meta
    fields = {}
    with open(stamp, encoding="utf-8", errors="replace") as f:
        for line in f:
            k, sep, v = line.partition("=")
            if sep:
                fields[k.strip()] = v.strip()
    meta["stamp"] = fields
    # A stamp is a FILE BESIDE the binary, not a property of it. Nothing makes
    # the two agree once anything has renamed a directory or rebuilt in place,
    # and a build comparison that reads the wrong stamp attributes a
    # measurement to the wrong commit — which is the one error this whole
    # report exists not to make. So the stamp is believed only when the commit
    # in it is the commit the binary itself prints.
    #
    # WHICH commit that is depends on what the build is. A patched build is
    # built from the patch branch's tip; an unpatched one is built from the
    # upstream commit and carries `patch_commit=none`. The first version
    # compared `patch_commit` in both cases, so every unpatched build failed
    # the check against the literal string "none" — a FALSE NEGATIVE that then
    # made the report fall back to the --backend label and name three
    # directories `rocm-patched` for builds stamped `patched=no`.
    #
    # It failed safe, which is the right direction and not an excuse: a check
    # that refuses a correct stamp teaches its reader to ignore the warning.
    if fields.get("patched") == "no":
        commit = (fields.get("upstream_commit") or "")[:9]
    else:
        commit = (fields.get("patch_commit") or "")[:9]
    meta["stamp_matches_binary"] = bool(commit and commit[:7] in reported)
    if meta["stamp_matches_binary"]:
        meta["build_id"] = fields.get("build_id") or reported
    else:
        print("  ! .build-stamp says %s, the binary says %s — using the "
              "binary" % (commit or "nothing", reported))
    return meta


def build_id(binary):
    try:
        r = subprocess.run([binary, "--version"], capture_output=True, text=True,
                           timeout=30)
        o = (r.stderr or "") + (r.stdout or "")
        m = re.search(r"build\s+(\d+),\s*commit\s+([0-9a-f]+)", o)
        if m:
            return "b%s-%s" % (m.group(1), m.group(2)[:7])
        m = re.search(r"commit\s+([0-9a-f]+)", o)
        if m:
            return m.group(1)[:7]
    except Exception:
        pass
    return "unbekannt"

# -------------------------------------------------------------------- Main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", help="setup/env/<model>.env")
    ap.add_argument("--binary", default=os.path.expanduser(
        "~/llama.cpp/build-vulkan/bin/llama-server"))
    ap.add_argument("--suites", default="basic,tools,multiproject,similar")
    ap.add_argument("--running", action="store_true",
                    help="do not start a server — measure against the one already running")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    name = os.path.basename(a.env).replace(".env", "") if a.env else "laufend"
    build = build_id(a.binary)
    stamp = time.strftime("%Y-%m-%d_%H%M")
    dest = os.path.join(HERE, "reports", "%s_%s_%s" % (stamp, name, build))
    os.makedirs(dest, exist_ok=True)

    argv = args_from_env(a.env) if a.env else []
    proc = None
    report = {"timestamp": stamp, "model": name, "build": build,
               "flags": argv, "note": a.note, "suites": {}}
    print("=" * 92)
    print("measurement run  model=%s  build=%s" % (name, build))
    if argv:
        print("flags: %s" % " ".join(argv))
    print("=" * 92)

    try:
        if not a.running:
            proc = start_server(argv, os.path.join(dest, "server.log"),
                                a.binary, env=a.env)
        props = {}
        try:
            props = {"slots": len(get("/slots"))}
        except Exception:
            pass

        report["slots"] = props.get("slots")
        report["gtt_gib_start"] = gtt()
        for s in [x.strip() for x in a.suites.split(",") if x.strip()]:
            if s not in SUITES:
                # Previously just one line and on we went — a typo in a suite
                # name left the report silently incomplete.
                raise SystemExit("unknown suite: %s (known: %s)"
                                 % (s, ", ".join(SUITES)))
            title, fn = SUITES[s]
            print("\n%s · %s" % (s, title))
            arg = None
            report["suites"][s] = fn(arg)

    finally:
        # ALWAYS write the report, even if a suite aborts halfway through:
        # half a measurement run is worth more than a lost one.
        try:
            report["gtt_gib_ende"] = gtt()
            with open(os.path.join(dest, "report.json"), "w") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print("\nreport: %s" % os.path.join(dest, "report.json"))
        except Exception as e:
            print("report not writable: %r" % (e,))
        if proc is not None:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            time.sleep(4)

if __name__ == "__main__":
    main()
