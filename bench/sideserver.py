#!/usr/bin/env python3
"""sideserver — start a model beside production, safely, and put it back.

    python3 bench/sideserver.py --env setup/env/flashnext.env --port 8081 \\
        --stop llama-user@qwen38 -- \\
        python3 bench/speed.py --url http://127.0.0.1:8081 --label flashnext

    python3 bench/sideserver.py --workload setup/workloads/sdxl.env \\
        --stop llama-user@qwen38          # a fenced batch job (see below)

Why this exists, and it is not a convenience.

On 26.08.2026 a measurement took this machine down TWICE. The morning one
produced bench/run.py's memory guard — `check_room_for()`, which refuses a
server whose weights do not fit, and `wait_for_gtt_release()`, whose docstring
says in as many words: after stopping a server, wait for the MEMORY to come
back, not for the port to close.

The evening one happened anyway, because the throwaway shell scripts written
to run the Flash-Next cells started `llama-server` directly. They called
neither function. They did `kill; sleep 5` and started the next 87 GiB model
on top of a GTT allocation that was still being torn down. `user.slice` peaked
at 114.8 GiB with 7.9 of 8 GiB of swap consumed, the desktop stopped
responding, and the machine had to be power-cycled.

The guard was not missing. It was skippable. So this file exists to make the
safe path the only convenient one — anything that wants a second model beside
production goes through here, and gets, in order:

  1. a DEAD MAN'S SWITCH armed before anything is stopped — a transient
     systemd timer that starts production again after --deadline minutes no
     matter what happens to this process;
  2. the production unit stopped, if asked, and then a WAIT until GTT
     actually falls — the step the first two incidents skipped;
  3. `check_room_for()`, which refuses rather than swaps, because GTT is
     pinned and an over-large start does not page, it hangs the machine;
  4. the server started as a TRANSIENT SYSTEMD UNIT with its own memory
     limits, and waited for on /slots, never /health;
  5. the command run;
  6. teardown: unit stopped, GTT waited for AGAIN, production restarted,
     dead man's switch disarmed.

Points 1 and 4 exist because of the third incident, at 23:11 on 26.08., and
each fixes something the first two did not reveal:

  * The limits on `llama-user@.service` did not apply. A server started from
    here inherits the CALLER's cgroup — the kernel log names it:
    `task_memcg=/user.slice/.../app-com.anthropic.Claude-13190.scope`. So
    `MemoryMax=108G` guarded the service and nothing else, and what actually
    fired was a global OOM. A transient unit gets its own cgroup and its own
    ceiling.
  * The teardown did not run. The OOM killer took this process with SIGKILL,
    `finally` never executed, and production stayed down until somebody
    noticed. A `finally` is not a guarantee; a timer that systemd owns is.

Note for this machine specifically: swap is partly zram, which lives in RAM.
Under this kind of pressure it cannot free anything — it compresses in place
while 87 GiB sits pinned in GTT beside it. That is why the box froze instead
of killing one process.

What this does NOT do (measured 31.08.2026): the transient unit is started
via systemd-run WITHOUT the profile as EnvironmentFile, so a profile's env
vars (GGML_*, LLAMA_*) never reach the side server. A measurement that needs
them wraps the binary in a script passed via --bin and exports them there.
The production unit is unaffected — it reads the profile as EnvironmentFile.
"""
import argparse, os, signal, subprocess, sys, time
from collections import namedtuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "setup", "lib"))
import run as runlib                                          # noqa: E402
import systemdfile                                            # noqa: E402
import budget                                                 # noqa: E402


def gtt_used():
    return runlib._gtt("used")


def say(msg):
    print("  %s" % msg, flush=True)


def systemctl(action, unit, run=subprocess.run):
    """One systemctl call — AND ITS RESULT, which this used to throw away.

    Measured 04.09.2026: a seven-point memory sweep tripped
    `StartLimitBurst=3` / `StartLimitIntervalSec=120` on the production
    unit, so `systemctl start` was REFUSED — and because the exit code was
    discarded here, restore_production could not tell that from a slow
    load. It waited 180 s and then 600 s for a unit systemd had already
    given up on, thirteen minutes per later point, with production down the
    whole time. A wait that looks exactly like progress.

    `run` is injectable for tests/test_sideworkload.py.
    """
    return run(["systemctl", "--user", action, unit], check=False,
               capture_output=True)


def wait_for_slots(url, timeout=420):
    import urllib.request
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(url + "/slots", timeout=5)
            return True
        except Exception:
            time.sleep(2)
    return False


# --- the ceiling, and why a fixed number was never one -----------------------

# Desktop, gateway, this process, page cache. Read from setup/lib/budget.py
# rather than declared here: this file said 12.0 and bench/run.py said 10.0
# about the same machine, which is what one number in two places always
# becomes.
HOST_RESERVE_GIB = budget.host_reserve_gib()

# The watchdog that asks production one question every ten minutes. It has to
# go down with production and come back with it — see the teardown.
PROBE_TIMER = "llama-probe.timer"
PRODUCTION_URL = "http://127.0.0.1:8080"

# Where the transient units' stdout/stderr land. Was /tmp/claude-1000 — a
# Claude-harness path with uid 1000 baked in, never created by this tool:
# after a reboot without a Claude session the unit failed at START and the
# error pointed at a log that had never existed (review, 01.09.2026).
LOG_DIR = os.path.expanduser("~/.cache/llm-stack/logs")


def release_baseline(before, settled):
    """Which GTT reading the teardown waits back down to.

    The pre-stop reading includes production (~36 GiB here), so with --stop
    the wait 'GTT <= before + 1' is true the moment the workload BEGINS
    tearing down — and production then restarts onto a teardown in flight,
    which is the step the 26.08. incidents skipped, on the way back
    (review, 01.09.2026). The settled post-stop floor is the honest target;
    without a stop, the pre-start reading is.
    """
    if settled is not None:
        return settled
    return before if before is not None else 0.0


def _memtotal_gib():
    """MemTotal, BY NAME. /proc/meminfo's first token is the label, not the
    number — reading it positionally returned None and the ceiling silently
    fell back to its conservative default. One reader, in budget.py."""
    return budget._meminfo_gib("MemTotal")


def machine_gib():
    """(total RAM, GTT currently pinned), both in GiB, None where unreadable."""
    return _memtotal_gib(), budget._gtt_gib("used")


def expected_gtt_gib(argv, env=None):
    """What the model ABOUT to start will pin, not what is pinned now.

    The ceiling has to be computed before the server exists, and at that moment
    sideserver has just waited for GTT to fall — so reading GTT live would give
    ~0.6 GiB and derive a ceiling of 112G, which is the same useless number the
    flat default was.

    Weights AND KV, since 27.08. The KV term is why the ceiling could sit above
    the cliff: a window is not free, and the old estimate counted only the file.
    No slack is added here on purpose — this figure is subtracted to find what
    is LEFT for the host, so overestimating it would strangle the process the
    ceiling is meant to protect.

    `env` since 28.08., and it is that last sentence coming true. A profile
    that has MEASURED what it pins says so, and reading the file size instead
    strangles exactly the model the measurement was taken on: for Flash-Next
    the file gives 103.7 + an estimated 8.0 of KV against a measured 78.1 +
    2.3, and `room` then falls under the 8 GiB clamp — MemoryMax=8G for a
    server whose host side is 31 GiB. The cgroup would have killed it on
    start, and the ceiling meant to protect the machine would have been the
    thing that broke the measurement.
    """
    override = budget._num("MODEL_GIB")
    gtt_base = budget.declared_gtt(env)
    kv, _ = budget.kv_gib(argv, budget.declared_kv(env))
    if override is None and gtt_base is not None:
        # An observation of GTT already contains the compute buffers; see
        # budget.plan(). Adding the KV to it is the same sum plan() makes.
        return gtt_base + kv
    weights = override if override is not None else (budget.weights_gib(argv) or 0.0)
    return weights + kv


def ceiling(want_max=None, want_high=None, argv=None, total_gib=None,
            env=None, gtt_override=None, live_gtt_gib=None):
    """The transient unit's MemoryMax, derived rather than assumed.

    The default was a flat `100G`, and on 27.08. a measurement showed what that
    is worth. Flash-Next pins 80 GiB in GTT and holds 29.6 GiB of ANONYMOUS
    host memory beside it — measured, `RssAnon 27.1 GiB`, `Private_Dirty
    28.1 GiB`, no file mapping at all. Total 109.6 of 124.9, and the machine
    ran at 10 GiB available.

    Now put the profile's own `-cram 32768` back: 32 GiB of RAM prompt cache on
    top, so the host side may reach ~57 GiB and the total ~137 — more than the
    machine has. **And `MemoryMax=100G` would not have stopped it**, because 57
    is less than 100. The ceiling sat above the cliff.

    GTT is not charged to the cgroup (verified: qwen38 shows MemoryCurrent
    32.9 GiB while holding 35.7 GiB of GTT), so the limit governs exactly the
    host side — which is the half that can take the machine down, since GTT is
    pinned and cannot be swapped to make room for it.

    So the ceiling is what is LEFT: total RAM, minus what GTT already holds,
    minus a share for the desktop and everything else. An explicit --memory-max
    still wins, because a caller who has measured knows more than this does.
    """
    if want_max:
        return want_max, (want_high or want_max), "given on the command line"
    total, live_gtt = machine_gib()
    # `total_gib` states the machine instead of reading it. It exists for
    # tests: what this function derives is a fraction of the RAM present, so a
    # test that asserts a number is asserting something about the machine it
    # runs on. That went unnoticed until the first CI run, on a 7.8 GiB runner.
    # Production never passes it — a guard that can be told how much memory
    # there is, is not a guard.
    if total_gib is not None:
        total = total_gib
    # `live_gtt_gib` exists for the same reason as total_gib: tests.
    if live_gtt_gib is not None:
        live_gtt = live_gtt_gib
    if total is None:
        return "64G", "58G", "could not read MemTotal — falling back low"
    # `gtt_override` is for callers whose expectation does not come from a
    # llama argv — a workload plan already carries its own GTT figure, and
    # a plan's 0.0 is a MEASURED zero (chatterbox pins nothing), not
    # absence. `if not gtt:` conflated the two and substituted the live
    # desktop reading, printing a fabricated rationale (review,
    # 01.09.2026). The estimator's 0.0 below, by contrast, really does
    # mean "could not estimate" (weights_gib falls back to 0.0).
    if gtt_override is not None:
        gtt = gtt_override
    else:
        gtt = expected_gtt_gib(argv or [], env)
        if not gtt:
            gtt = live_gtt or 0.0
    room = total - gtt - HOST_RESERVE_GIB
    if room < 8:
        room = 8.0
    return ("%dG" % int(room), want_high or "%dG" % int(room * 0.9),
            "%.0f GiB RAM - %.0f the model will pin in GTT - %.0f for the host"
            % (total, gtt, HOST_RESERVE_GIB))


# --- the production dance, shared ---------------------------------------
#
# Arm, stop, settle / restore, disarm. ONE implementation for the llama path
# and the workload path — the second copy of this dance is exactly where the
# 26.08. incidents came from (a throwaway script that did `kill; sleep 5`).

def install_sigterm_handler():
    """SIGTERM must run the finally blocks — by default it ends Python
    WITHOUT unwinding (only SIGINT becomes an exception), so a SIGTERM
    mid-metering skipped the whole teardown: unit kept pinning GTT,
    production stayed down until the dead man's switch started qwen38
    INTO the still-running workload (review, 01.09.2026). Converted to
    SystemExit so both paths' finally teardowns run; 128+15 is the
    conventional exit code for a SIGTERM death."""
    def to_exit(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, to_exit)


def arm_deadman(deadman, production_unit, deadline_min, run=subprocess.run):
    """True when the switch is ARMED — verified, not assumed.

    The result used to be discarded and 'armed' announced unconditionally;
    a stale deadman unit from a SIGKILLed fence makes systemd-run fail on
    the name collision, and the run then proceeded with no working switch —
    or with the STALE timer firing mid-measurement on its older clock
    (review, 01.09.2026). Callers refuse before anything is stopped.
    `run` is injectable for tests/test_sideworkload.py.
    """
    # ARM FIRST, before anything is stopped. systemd owns this timer, so it
    # fires even if this process is SIGKILLed — which is exactly what
    # happened at 23:11 on 26.08., leaving production down.
    # IT CLEARS THE START LIMITER BEFORE IT STARTS, and that is not a
    # flourish: this switch exists for the case where this process died and
    # production is down, which is exactly the case in which
    # `StartLimitBurst=3` / `StartLimitIntervalSec=120` may already be
    # tripped (measured 04.09.2026). A bare `start` there fires into
    # `Start request repeated too quickly` and leaves the machine with no
    # model on it — and `--collect` means this unit's own verdict cannot be
    # read back afterwards, because a collected transient unit answers
    # Result=success. The refusal would land in the journal under the
    # PRODUCTION unit and nowhere else.
    #
    # `reset-failed` on a healthy unit is a no-op, so clearing first costs
    # nothing; `|| true` keeps a no-op from ending the shell under -e-like
    # conditions and makes the start unconditional.
    revive = ("systemctl --user reset-failed %s || true; "
              "exec systemctl --user start %s %s"
              % (production_unit, production_unit, PROBE_TIMER))
    r = run(["systemd-run", "--user", "--quiet", "--collect",
             "--unit", deadman,
             "--on-active=%dmin" % deadline_min,
             "/bin/sh", "-c", revive],
            check=False, capture_output=True)
    if r.returncode != 0:
        say("REFUSING: the dead man's switch did not arm (%s). A stale "
            "timer from an earlier run is the usual cause — inspect with "
            "`systemctl --user list-timers %s*` and stop it with "
            "`systemctl --user stop %s.timer` before retrying. Stopping "
            "production without a working switch is how 26.08. ended."
            % (str(getattr(r, "stderr", "") or
                   getattr(r, "stdout", ""))[:200].strip(),
               deadman, deadman))
        return False
    say("dead man's switch armed: %s restarts in %d min whatever "
        "happens here" % (production_unit, deadline_min))
    return True


def stop_production_and_settle(production_unit):
    """Stop the unit, wait for its GTT teardown to FINISH. Settled GiB, or
    None if GTT was still moving — the caller must then refuse to start."""
    # The watchdog goes down WITH production, and comes back with it. It asks
    # the production server one question every ten minutes and reports a
    # failure when nothing answers — so every measurement that stops
    # production used to leave a failed unit behind and a red line in
    # check.sh. Measured 27.08.: the 10:09 run produced "UNREACHABLE
    # ConnectionRefused" for exactly that reason.
    #
    # A detector that cries wolf on every measurement is a detector people
    # learn to ignore, which is the one failure mode it cannot afford: it
    # exists for the SILENT faults. It is stopped here and restarted in the
    # teardown — and, more importantly, by the dead man's switch, so a killed
    # sideserver cannot leave the watchdog off. That is the worse half of the
    # trade and it is the half systemd owns rather than this process.
    say("stopping %s (and %s with it)" % (production_unit, PROBE_TIMER))
    systemctl("stop", PROBE_TIMER)
    systemctl("stop", production_unit)
    # THE step both incidents skipped. Not sleep(5): the process can be gone
    # while amdgpu is still tearing its GTT down, and the next model then
    # loads on top of it.
    before = gtt_used()
    say("waiting for GTT to fall (now %.1f GiB) ..." % (before or -1))
    # Wait for the teardown to be OVER, not for a number. This asked
    # wait_for_gtt_release(0.0) until 28.08.2026, which means "under 1 GiB" —
    # and the desktop on this machine holds 1.5 and keeps it. So the wait
    # could never succeed with anyone logged in: three runs that evening,
    # production already stopped, GTT already at 1.5, all three refused after
    # the full 180 s. Whether the remaining GTT leaves ROOM is the room
    # check's question, and it answers with the arithmetic in hand.
    settled = runlib.wait_for_gtt_to_settle(timeout=GTT_SETTLE_TIMEOUT_S)
    if settled is None:
        say("GTT was still moving after 180 s — refusing to stack a "
            "second model on a teardown that has not finished")
        return None
    say("GTT settled at %.1f GiB" % settled)
    return settled


def start_production(production_unit, sc=None, props=None):
    """Start production and know WHY it did not start, if it did not.

    Two refusals are not the same thing and this is where they part:

      * `start-limit-hit` — systemd refused because the unit was started
        too often (StartLimitBurst=3 inside StartLimitIntervalSec=120). The
        unit is fine; the COUNTER is full. `reset-failed` clears it and one
        retry brings production back. Measured 04.09.2026, when a
        seven-point memory sweep hit it on the fourth point: the limiter is
        reachable from any campaign that loads a fast model repeatedly,
        because only a fast model fits four starts into two minutes.
      * anything else — the unit tried and failed. Retrying is how a
        crash-loop is built, which is what the limiter exists to prevent,
        so this says what systemd said and stops.

    The limiter itself is NOT relaxed, and that is deliberate: the unit
    carries Restart=on-failure with RestartSec=5, so without it a server
    that dies during load would restart every five seconds forever.

    `sc` and `props` are injectable for tests/test_sideworkload.py.
    """
    sc = sc or systemctl
    props = props or unit_props
    if getattr(sc("start", production_unit), "returncode", 0) == 0:
        return True

    p = props(production_unit, ["LoadState", "ActiveState", "Result"]) or {}
    result = p.get("Result", "")
    if result != "start-limit-hit":
        say("%s did not start and systemd is not going to take it back: "
            "LoadState=%s ActiveState=%s Result=%s. Not retrying — that is "
            "how a crash-loop is built. `journalctl --user -u %s` has the "
            "reason." % (production_unit, p.get("LoadState", "?"),
                         p.get("ActiveState", "?"), result or "?",
                         production_unit))
        return False

    say("%s hit systemd's start limiter (Result=start-limit-hit) — the unit "
        "is fine, the counter is full. Clearing it and starting once more."
        % production_unit)
    sc("reset-failed", production_unit)
    if getattr(sc("start", production_unit), "returncode", 0) == 0:
        return True
    say("%s still refused after reset-failed. Production is DOWN; the dead "
        "man's switch stays armed." % production_unit)
    return False


def restore_production(production_unit, deadman, sc=None, props=None,
                       wait=None):
    sc = sc or systemctl
    wait = wait or wait_for_slots
    say("restarting %s and %s" % (production_unit, PROBE_TIMER))
    if not start_production(production_unit, sc=sc, props=props):
        # DO NOT WAIT, AND DO NOT DISARM. Waiting 180 s and then 600 s on a
        # unit systemd has refused is the thirteen minutes this whole
        # function was rewritten for on 04.09.2026 — and it is worse than
        # the wasted time, because the operator reads a wait as progress.
        # The probe timer goes back so the absence is LOUD; the dead man's
        # switch stays armed, because production really is down and that
        # timer is now the only thing that will bring it back.
        say("NOT waiting on a unit systemd refused to start, and NOT "
            "disarming the dead man's switch — it fires in the remaining "
            "deadline and is the only thing left that will restore "
            "production. Fix the unit, or run "
            "`systemctl --user reset-failed %s && systemctl --user start %s`."
            % (production_unit, production_unit))
        sc("start", PROBE_TIMER)
        return
    # The watchdog goes back LAST, and after production actually answers.
    # Its interval elapsed while it was stopped, so starting it alongside
    # production makes it fire at once — measured 27.08.: both started at
    # 10:55:46, the probe failed at 10:55:47, the model finished loading at
    # 10:55:55. Nine seconds, two false alarms.
    #
    # probe.py is patient about this now as well, and both halves are
    # wanted: this one keeps the normal path quiet, and the patience covers
    # the dead man's switch, which cannot wait for anything.
    if not wait(PRODUCTION_URL, timeout=180):
        say("%s did not come back within 180 s — starting %s anyway, "
            "it is better loud than absent" % (production_unit, PROBE_TIMER))
    sc("start", PROBE_TIMER)
    wait(PRODUCTION_URL, 600)
    # Disarm LAST: until production actually answers, the timer is still the
    # thing standing between a failure here and a machine with no model on it.
    subprocess.run(["systemctl", "--user", "stop", deadman + ".timer"],
                   check=False, capture_output=True)
    say("done · dead man's switch disarmed")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--env", help="a setup/env/*.env profile")
    ap.add_argument("--workload",
                    help="a setup/workloads/*.env profile — a fenced job "
                         "that is not llama-server (exactly one of --env "
                         "and --workload)")
    ap.add_argument("--job-timeout", type=int, default=1800,
                    help="seconds a batch workload may run before it is "
                         "stopped. A bound shorter than the work it waits "
                         "for measures the bound, not the work — the output "
                         "names this value so a timeout is readable as one")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--stop", default=None,
                    help="systemd --user unit to stop first and restart after")
    ap.add_argument("--bin", default=None, help="override LLAMA_BIN")
    ap.add_argument("--extra", default="", help="extra llama-server arguments")
    ap.add_argument("--memory-max", default=None,
                    help="MemoryMax for the transient unit. Default: DERIVED "
                         "from what the machine has left once GTT is pinned — "
                         "a fixed number is meaningless here, see ceiling()")
    ap.add_argument("--memory-high", default=None,
                    help="MemoryHigh; default 90 %% of the derived MemoryMax")
    ap.add_argument("--slots-timeout", type=int, default=420,
                    help="seconds to wait for the server to answer /slots. "
                         "420 was hard-wired and on 28.08. it decided a "
                         "measurement: forcing --load-mode mmap on this iGPU "
                         "made Flash-Next load through the mapping, which got "
                         "as far as allocating its buffers in 33 s and was "
                         "still reading at 7 min. The run was torn down as a "
                         "failure when the only thing that had failed was a "
                         "constant nobody chose for that case")
    ap.add_argument("--deadline", type=int, default=45,
                    help="minutes after which production is restarted no "
                         "matter what happens to this process")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="-- command to run while the server is up")
    a = ap.parse_args(argv)

    if bool(a.env) == bool(a.workload):
        ap.error("exactly one of --env and --workload")
    if a.workload and (a.bin or a.extra or a.port != 8081
                       or a.slots_timeout != 420):
        # Silently ignoring them is worse than refusing: --extra
        # "--steps 5" beside --workload ran the profile's full 30 steps
        # and the timing was read as a 5-step figure (review, 01.09.2026).
        ap.error("--bin/--extra/--port/--slots-timeout apply to the llama "
                 "path only — a workload runs exactly its profile's "
                 "WORKLOAD_CMD (edit the profile, or pass a full command "
                 "after --)")
    # Armed before EITHER path touches production: SIGTERM otherwise skips
    # the finally teardowns entirely (see install_sigterm_handler).
    install_sigterm_handler()
    if a.workload:
        return workload_main(a)

    argv = systemdfile.llama_args(a.env)
    for i, tok in enumerate(argv):
        if tok == "--port" and i + 1 < len(argv):
            argv[i + 1] = str(a.port)
    argv += a.extra.split()
    binary = a.bin or os.path.expanduser(
        "~/" + systemdfile.variable(a.env, "LLAMA_BIN",
                                    "llama.cpp/build-vulkan/bin/llama-server"))
    url = "http://127.0.0.1:%d" % a.port
    cmd = [c for c in a.cmd if c != "--"]

    baseline = gtt_used()
    settled = None
    unit = "sideserver-%d" % a.port
    deadman = "sideserver-deadman-%d" % a.port
    proc = None
    try:
        if a.stop:
            if not arm_deadman(deadman, a.stop, a.deadline):
                return 2
            settled = stop_production_and_settle(a.stop)
            if settled is None:
                return 2

        # Refuses rather than swaps. GTT is pinned; an over-large start does
        # not page, it hangs the machine.
        runlib.check_room_for(argv, os.path.basename(binary),
                              env=a.env, binary=binary)

        # A TRANSIENT UNIT, not a bare child process. A child inherits the
        # caller's cgroup, so the limits on llama-user@.service never touched
        # it and a global OOM was the only thing left to stop it. This gets
        # its own cgroup and its own ceiling — and a name, so it can be
        # stopped even by somebody who did not start it.
        mem_max, mem_high, why = ceiling(a.memory_max, a.memory_high, argv,
                                         env=a.env)
        say("starting %s on port %d as %s (MemoryMax=%s — %s)"
            % (os.path.basename(a.env), a.port, unit, mem_max, why))
        subprocess.run(["systemctl", "--user", "stop", unit],
                       check=False, capture_output=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        log = os.path.join(LOG_DIR, "sideserver-%d.log" % a.port)
        r = subprocess.run(
            ["systemd-run", "--user", "--quiet", "--collect", "--unit", unit,
             "-p", "MemoryMax=%s" % mem_max,
             "-p", "MemoryHigh=%s" % mem_high,
             "-p", "StandardOutput=append:%s" % log,
             "-p", "StandardError=append:%s" % log,
             "--", binary] + argv,
            check=False, capture_output=True, text=True)
        if r.returncode != 0:
            say("systemd-run refused: %s" % (r.stderr or r.stdout)[:200])
            return 1
        proc = unit
        if not wait_for_slots(url, a.slots_timeout):
            say("the server never served /slots")
            return 1
        say("up · GTT %.1f GiB" % (gtt_used() or -1))
        if not cmd:
            say("no command given — nothing to do")
            return 0
        say("running: %s" % " ".join(cmd))
        return subprocess.run(cmd).returncode
    finally:
        if proc is not None:
            say("stopping %s" % unit)
            systemctl("stop", unit)
            say("waiting for GTT to be given back ...")
            runlib.wait_for_gtt_release(release_baseline(baseline, settled),
                                        timeout=GTT_RELEASE_TIMEOUT_S)
            say("GTT now %.1f GiB" % (gtt_used() or -1))
        if a.stop:
            restore_production(a.stop, deadman)


# --- foreign workloads ---------------------------------------------------

# 5 s is generous for a local dbus round trip (normally milliseconds) and
# short against the 1 Hz cadence's purpose — heuristic, not derived.
SYSTEMCTL_TIMEOUT_S = 5


def unit_props(unit, names, run=None):
    """The named systemd properties of one unit, as a dict — ONE systemctl
    call, so the answers are a single snapshot.

    LoadState is the honesty bit and is why this exists: `systemctl show`
    of a unit systemd no longer knows answers Result=success,
    ExecMainStatus=0, ActiveState=inactive (measured on systemd 259,
    01.09.2026) — a fence reading those without LoadState FABRICATES a
    verdict. `run` is injectable so that exact shape is pinned in
    tests/test_sideworkload.py without a systemd.
    """
    if run is None:
        def run(unit, names):
            cmd = ["systemctl", "--user", "show", unit]
            for n in names:
                cmd += ["-p", n]
            # timeout: meter_until_exit polls this at 1 Hz inside the
            # fenced window, and an untimed call let a dbus stall block
            # the loop IN the syscall — the job-timeout was never
            # re-checked, the teardown queued behind the block, and the
            # dead man's switch then started production onto the
            # still-pinning workload (ultrareview, 01.09.2026). A
            # timeout is the dbus-hiccup shape the two-consecutive-reads
            # guard already tolerates once and acts on twice.
            try:
                r = subprocess.run(cmd, check=False, capture_output=True,
                                   text=True, timeout=SYSTEMCTL_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                return ""
            return r.stdout or ""
    out = {}
    for line in run(unit, names).splitlines():
        k, sep, v = line.partition("=")
        if sep:
            out[k.strip()] = v.strip()
    return out


def job_outcome(props):
    """(ok, line) — the verdict a batch job earned, or 'unknown'.

    Success requires the unit to still be LOADED. A vanished unit reads as
    success/0 (see unit_props), and the convenient answer is exactly the
    one a fence must not invent — its peaks feed ready-to-paste
    declarations.
    """
    if props.get("LoadState") != "loaded":
        return False, ("unknown — the unit is gone (LoadState=%s), and "
                       "systemd fabricates success for units it no longer "
                       "knows" % (props.get("LoadState") or "?"))
    ok = (props.get("Result") == "success"
          and props.get("ExecMainStatus") == "0")
    return ok, ("Result=%s ExecMainStatus=%s"
                % (props.get("Result"), props.get("ExecMainStatus")))


# Model load + restore around the job — heuristic, not derived; generous
# beats a dead man's switch firing into a measurement. The two waits below
# are NOT heuristic: they are the coded worst cases of the fence's own
# settle and release phases, which run on the SAME deadline clock (it
# starts at arming) and which the first version of this arithmetic forgot —
# a boundary-passing job then had the switch fire into the GTT-release
# wait, production loading onto a teardown in flight (review, 01.09.2026).
DEADLINE_SLACK_S = 300
GTT_SETTLE_TIMEOUT_S = 180   # stop_production_and_settle's wait bound
GTT_RELEASE_TIMEOUT_S = 180  # both teardowns' wait_for_gtt_release bound


def deadline_covers(job_timeout_s, deadline_min, slack_s=DEADLINE_SLACK_S):
    """False when the dead man's switch could fire INTO the running job
    or its teardown.

    It would start production mid-metering: the system-wide GTT peak then
    swallows qwen38's ~36 GiB and the contaminated number is offered as a
    declaration — plus an unplanned co-residency (review, 01.09.2026).
    Equality is not coverage; the boundary belongs to the refusal side.
    """
    return (job_timeout_s + GTT_SETTLE_TIMEOUT_S + GTT_RELEASE_TIMEOUT_S
            + slack_s < deadline_min * 60)


def _rss_anon_of(pid):
    try:
        with open("/proc/%s/status" % pid) as fh:
            for line in fh:
                if line.startswith("RssAnon:"):
                    return float(line.split()[1]) / 1048576.0
    except (OSError, ValueError, IndexError):
        pass
    return None


def _cgroup_rss_anon(cgroup):
    """Summed RssAnon over every process in the unit's cgroup, in GiB.

    The unit's MainPID alone is wrong the moment the job is a WRAPPER —
    bench/imagebench.py spawning sd-cli reads as 0.05 GiB of Python while
    the child holds five. The cgroup is the unit's own bookkeeping of who
    belongs to it, so it needs no pgrep pattern (which would find this very
    tool's command line — the global CLAUDE.md lesson).
    """
    if not cgroup:
        return None
    procs = os.path.join("/sys/fs/cgroup", cgroup.lstrip("/"), "cgroup.procs")
    try:
        with open(procs) as fh:
            pids = [line.strip() for line in fh if line.strip()]
    except OSError:
        return None
    total, seen = 0.0, False
    for pid in pids:
        r = _rss_anon_of(pid)
        if r is not None:
            total += r
            seen = True
    return total if seen else None


Meter = namedtuple("Meter",
                   "peak_gtt peak_rss gtt_seen rss_seen timed_out vanished")
METER_PROPS = ("LoadState", "ActiveState", "SubState", "MainPID",
               "ControlGroup")


def meter_until_exit(unit, base_gtt, timeout,
                     props_of=None, gtt=None, rss_of=None, sleep=None):
    """Watch a transient unit until it exits; return what it PEAKED at.

    Sampled at 1 Hz, so a spike shorter than a second can hide between
    samples — the figures are lower bounds, which the declaration hint
    says. GTT is the SYSTEM-WIDE reading minus the settled baseline:
    production is stopped while this runs, so the delta belongs to the job
    (plus whatever the desktop does meanwhile — the safe direction).
    RssAnon is summed over the unit's cgroup, so a wrapper's children are
    counted.

    A quantity never sampled is UNSEEN, not 0.0 — the gtt_seen/rss_seen
    flags keep a measured zero (chatterbox pins no GTT: true) distinct
    from a blind instrument (no amdgpu, cgroup unreadable, unit gone
    before the first poll). Printing both as "+0.00" was the
    silent-wrongness the review named (01.09.2026).

    The unit runs with RemainAfterExit=yes, so completion shows as
    SubState=exited while ActiveState stays active; inactive/failed still
    end the watch, and LoadState leaving "loaded" means somebody collected
    the unit under us — reported as `vanished`, never as success. The four
    readers are injectable for tests/test_sideworkload.py.
    """
    props_of = props_of or (lambda u, names: unit_props(u, list(names)))
    gtt = gtt or gtt_used
    rss_of = rss_of or _cgroup_rss_anon
    sleep = sleep or time.sleep
    peak_gtt, peak_rss = 0.0, 0.0
    gtt_seen = rss_seen = False
    cgroup = None
    not_loaded_reads = 0
    end = time.time() + timeout
    while time.time() < end:
        props = props_of(unit, METER_PROPS)
        if not cgroup:
            cgroup = props.get("ControlGroup") or None
        g = gtt()
        if g is not None:
            gtt_seen = True
            peak_gtt = max(peak_gtt, g - (base_gtt or 0.0))
        r = rss_of(cgroup)
        if r is not None:
            rss_seen = True
            peak_rss = max(peak_rss, r)
        if props.get("LoadState") != "loaded":
            # TWO consecutive reads before the verdict: a single
            # transiently empty `systemctl show` answer (dbus hiccup) used
            # to read as vanished and tear down a RUNNING job — loud, but
            # a full fenced cycle lost (re-review, 01.09.2026).
            not_loaded_reads += 1
            if not_loaded_reads >= 2:
                return Meter(peak_gtt, peak_rss, gtt_seen, rss_seen,
                             False, True)
            sleep(1)
            continue
        not_loaded_reads = 0
        if (props.get("ActiveState") in ("inactive", "failed")
                or props.get("SubState") in ("exited", "failed")):
            return Meter(peak_gtt, peak_rss, gtt_seen, rss_seen,
                         False, False)
        sleep(1)
    return Meter(peak_gtt, peak_rss, gtt_seen, rss_seen, True, False)


def workload_main(a):
    """The fence for a job that is not llama-server. Same dance, different
    tenant: dead man's switch, stop production, settle, WEIGH, run the job
    as a transient unit with its own ceiling, meter its peaks, put
    production back. The peaks are printed as ready-to-paste declarations —
    the measurement the profile's empty fields are waiting for.
    """
    path = budget._workload_path(a.workload)
    name = os.path.basename(path).rsplit(".env", 1)[0]
    mode = systemdfile.variable(path, "WORKLOAD_MODE", "batch")
    if mode != "batch":
        # Refused rather than pretended: a code path nobody has run is a
        # claim, and the first consumer (sdxl) is a batch job. Server-mode
        # workloads land together with their first real consumer.
        say("WORKLOAD_MODE=%s is not implemented yet — batch only" % mode)
        return 2
    argv = systemdfile.args_of(path, "WORKLOAD_CMD")
    if not argv:
        say("no WORKLOAD_CMD in %s" % path)
        return 2
    cmd = [c for c in a.cmd if c != "--"]
    if cmd:
        # A caller's command REPLACES the profile's job inside the fence —
        # this is how a bench runs N repetitions under one production cycle
        # instead of paying a stop/start per image.
        job = cmd
    else:
        prompt = systemdfile.variable(path, "WORKLOAD_PROMPT")
        # webm, not mp4, for video: sd-cli's single-file video formats are
        # .avi/.webm/.webp — an .mp4 target is silently rewritten to .avi
        # upstream, and a smoke that asks for what the tool cannot write is
        # a smoke of the wrong thing.
        ext = {"image": "png", "audio": "wav", "video": "webm"}.get(
            systemdfile.variable(path, "WORKLOAD_KIND", ""), "out")
        out = os.path.expanduser("~/.cache/llm-stack/%s-smoke.%s"
                                 % (name, ext))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        job = argv + (["-p", prompt] if prompt else []) + ["-o", out]
        say("no command given — smoke run of the profile's own job -> %s"
            % out)

    if a.stop and not deadline_covers(a.job_timeout, a.deadline):
        say("REFUSING: --job-timeout %d s plus %d s settle, %d s release "
            "and %d s of slack does not fit inside --deadline %d min. The "
            "dead man's switch would start production INTO the running "
            "measurement or its teardown, the peaks would swallow its "
            "~36 GiB, and the contaminated number would be offered as a "
            "declaration. Raise --deadline or lower --job-timeout."
            % (a.job_timeout, GTT_SETTLE_TIMEOUT_S, GTT_RELEASE_TIMEOUT_S,
               DEADLINE_SLACK_S, a.deadline))
        return 2

    baseline = gtt_used()
    settled = None
    unit = "sideworkload-%s" % name
    deadman = "sideworkload-deadman-%s" % name
    started = False
    try:
        if a.stop:
            if not arm_deadman(deadman, a.stop, a.deadline):
                return 2
            settled = stop_production_and_settle(a.stop)
            if settled is None:
                return 2

        # Refuses rather than swaps — the same guard, fed by the workload's
        # own declarations (or its file sizes, announced as an estimate).
        plan = budget.workload_plan(path)
        machine = budget.read_machine()
        v = budget.workload_verdict(plan, machine)
        for note in v.notes:
            say("note: %s" % note)
        if not v.fits:
            if budget.guard_disabled():
                say("LLM_NO_MEMORY_GUARD=1 — starting anyway")
            else:
                print(budget.refusal(plan, machine, v,
                                     advice=budget.WORKLOAD_ADVICE),
                      file=sys.stderr)
                return 1

        mem_max, mem_high, why = ceiling(a.memory_max, a.memory_high,
                                         gtt_override=plan.gtt_gib)
        os.makedirs(LOG_DIR, exist_ok=True)
        log = os.path.join(LOG_DIR, "%s.log" % unit)
        say("starting %s as %s (MemoryMax=%s — %s)" % (name, unit, mem_max, why))
        say("job log: %s" % log)
        subprocess.run(["systemctl", "--user", "stop", unit],
                       check=False, capture_output=True)
        subprocess.run(["systemctl", "--user", "reset-failed", unit],
                       check=False, capture_output=True)
        # RemainAfterExit=yes: a SUCCESSFUL transient unit is collected the
        # moment it exits, and `systemctl show` then fabricates
        # success-shaped answers for it (see unit_props) — with this
        # property the unit stays until the teardown stops it, so Result
        # and ExecMainStatus are read from a unit that still exists.
        r = subprocess.run(
            ["systemd-run", "--user", "--quiet", "--unit", unit,
             "-p", "MemoryMax=%s" % mem_max,
             "-p", "MemoryHigh=%s" % mem_high,
             "-p", "RemainAfterExit=yes",
             # The caller typed the command from THEIR directory; a transient
             # unit starts in $HOME, and `python3 bench/imagebench.py` then
             # dies with ENOENT (cost one fenced production cycle,
             # 01.09.2026). The unit runs where the caller stood.
             "-p", "WorkingDirectory=%s" % os.getcwd(),
             "-p", "StandardOutput=append:%s" % log,
             "-p", "StandardError=append:%s" % log,
             "--"] + job,
            check=False, capture_output=True, text=True)
        if r.returncode != 0:
            say("systemd-run refused: %s" % (r.stderr or r.stdout)[:200])
            return 1
        started = True
        base = settled if settled is not None else (baseline or 0.0)
        m = meter_until_exit(unit, base, a.job_timeout)
        if m.timed_out:
            say("the job is still running after --job-timeout %d s — "
                "stopping it. If the work is simply longer than the bound, "
                "raise the bound; do not read this as a hang without a "
                "second look at %s" % (a.job_timeout, log))
            systemctl("stop", unit)
            return 1
        if m.vanished:
            say("the unit disappeared mid-meter (collected or stopped from "
                "outside) — no verdict and no declarations can come from "
                "this run")
        ok, line = job_outcome(unit_props(
            unit, ["LoadState", "Result", "ExecMainStatus"]))
        say("job finished: %s" % line)
        gtt_txt = ("+%.2f GiB over the %.1f GiB baseline"
                   % (m.peak_gtt, base)) if m.gtt_seen else \
            "unmeasurable — not one sample (no amdgpu reading)"
        rss_txt = ("%.2f GiB" % m.peak_rss) if m.rss_seen else \
            "unmeasurable — not one sample (cgroup unreadable or job too fast)"
        say("peaks (1 Hz samples, lower bounds): GTT %s · RssAnon %s"
            % (gtt_txt, rss_txt))
        if ok and (m.gtt_seen or m.rss_seen):
            say("declare in %s, each with date + method + machine:" % path)
            if m.gtt_seen:
                say("  WORKLOAD_GTT_GIB=%.1f" % m.peak_gtt)
            if m.rss_seen:
                say("  WORKLOAD_HOST_RSS_GIB=%.1f" % m.peak_rss)
        elif ok:
            say("nothing to declare — the meter never got a sample; a "
                "declaration of 0.0 here would be blindness wearing a "
                "measurement's clothes")
        else:
            say("a failed job's peaks are not a footprint — fix the job "
                "first, see %s" % log)
        return 0 if ok else 1
    finally:
        if started:
            systemctl("stop", unit)
            subprocess.run(["systemctl", "--user", "reset-failed", unit],
                           check=False, capture_output=True)
            say("waiting for GTT to be given back ...")
            runlib.wait_for_gtt_release(release_baseline(baseline, settled),
                                        timeout=GTT_RELEASE_TIMEOUT_S)
            say("GTT now %.1f GiB" % (gtt_used() or -1))
        if a.stop:
            restore_production(a.stop, deadman)


if __name__ == "__main__":
    sys.exit(main())
