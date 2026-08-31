#!/usr/bin/env python3
"""sideserver — start a model beside production, safely, and put it back.

    python3 bench/sideserver.py --env setup/env/flashnext.env --port 8081 \\
        --stop llama-user@qwen38 -- \\
        python3 bench/speed.py --url http://127.0.0.1:8081 --label flashnext

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
import argparse, os, subprocess, sys, time

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


def systemctl(action, unit):
    subprocess.run(["systemctl", "--user", action, unit], check=False,
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
            env=None):
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
    if total is None:
        return "64G", "58G", "could not read MemTotal — falling back low"
    gtt = expected_gtt_gib(argv or [], env)
    if not gtt:
        gtt = live_gtt or 0.0
    room = total - gtt - HOST_RESERVE_GIB
    if room < 8:
        room = 8.0
    return ("%dG" % int(room), want_high or "%dG" % int(room * 0.9),
            "%.0f GiB RAM - %.0f the model will pin in GTT - %.0f for the host"
            % (total, gtt, HOST_RESERVE_GIB))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--env", required=True, help="a setup/env/*.env profile")
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
    a = ap.parse_args()

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
    unit = "sideserver-%d" % a.port
    deadman = "sideserver-deadman-%d" % a.port
    proc = None
    try:
        if a.stop:
            # ARM FIRST, before anything is stopped. systemd owns this timer,
            # so it fires even if this process is SIGKILLed — which is exactly
            # what happened at 23:11 on 26.08., leaving production down.
            subprocess.run(["systemd-run", "--user", "--quiet", "--collect",
                            "--unit", deadman,
                            "--on-active=%dmin" % a.deadline,
                            "systemctl", "--user", "start", a.stop,
                            PROBE_TIMER],
                           check=False, capture_output=True)
            say("dead man's switch armed: %s restarts in %d min whatever "
                "happens here" % (a.stop, a.deadline))
        if a.stop:
            # The watchdog goes down WITH production, and comes back with it.
            # It asks the production server one question every ten minutes and
            # reports a failure when nothing answers — so every measurement
            # that stops production used to leave a failed unit behind and a
            # red line in check.sh. Measured 27.08.: the 10:09 run produced
            # "UNREACHABLE ConnectionRefused" for exactly that reason.
            #
            # A detector that cries wolf on every measurement is a detector
            # people learn to ignore, which is the one failure mode it cannot
            # afford: it exists for the SILENT faults. It is stopped here and
            # restarted in the teardown — and, more importantly, by the dead
            # man's switch above, so a killed sideserver cannot leave the
            # watchdog off. That is the worse half of the trade and it is the
            # half systemd owns rather than this process.
            say("stopping %s (and %s with it)" % (a.stop, PROBE_TIMER))
            systemctl("stop", PROBE_TIMER)
            systemctl("stop", a.stop)
            # THE step both incidents skipped. Not sleep(5): the process can be
            # gone while amdgpu is still tearing its GTT down, and the next
            # model then loads on top of it.
            before = gtt_used()
            say("waiting for GTT to fall (now %.1f GiB) ..." % (before or -1))
            # Wait for the teardown to be OVER, not for a number. This asked
            # wait_for_gtt_release(0.0) until 28.08.2026, which means "under 1
            # GiB" — and the desktop on this machine holds 1.5 and keeps it.
            # So the wait could never succeed with anyone logged in: three
            # runs that evening, production already stopped, GTT already at
            # 1.5, all three refused after the full 180 s. Whether the
            # remaining GTT leaves ROOM is check_room_for's question, three
            # lines down, and it answers with the arithmetic in hand.
            settled = runlib.wait_for_gtt_to_settle(timeout=180)
            if settled is None:
                say("GTT was still moving after 180 s — refusing to stack a "
                    "second model on a teardown that has not finished")
                return 2
            say("GTT settled at %.1f GiB" % settled)

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
        log = "/tmp/claude-1000/sideserver-%d.log" % a.port
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
            runlib.wait_for_gtt_release(baseline or 0.0, timeout=180)
            say("GTT now %.1f GiB" % (gtt_used() or -1))
        if a.stop:
            say("restarting %s and %s" % (a.stop, PROBE_TIMER))
            systemctl("start", a.stop)
            # The watchdog goes back LAST, and after production actually
            # answers. Its interval elapsed while it was stopped, so starting
            # it alongside production makes it fire at once — measured 27.08.:
            # both started at 10:55:46, the probe failed at 10:55:47, the model
            # finished loading at 10:55:55. Nine seconds, two false alarms.
            #
            # probe.py is patient about this now as well, and both halves are
            # wanted: this one keeps the normal path quiet, and the patience
            # covers the dead man's switch, which cannot wait for anything.
            if not wait_for_slots(PRODUCTION_URL, timeout=180):
                say("%s did not come back within 180 s — starting %s anyway, "
                    "it is better loud than absent" % (a.stop, PROBE_TIMER))
            systemctl("start", PROBE_TIMER)
            wait_for_slots("http://127.0.0.1:8080", 600)
            # Disarm LAST: until production actually answers, the timer is
            # still the thing standing between a failure here and a machine
            # with no model on it.
            subprocess.run(["systemctl", "--user", "stop", deadman + ".timer"],
                           check=False, capture_output=True)
            say("done · dead man's switch disarmed")


if __name__ == "__main__":
    sys.exit(main())
