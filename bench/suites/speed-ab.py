#!/usr/bin/env python3
"""speed-ab — two builds, one difference, interleaved.

    python3 bench/suites/speed-ab.py                     the newest unroll build
    python3 bench/suites/speed-ab.py --variant-family rocm-altsdk
    python3 bench/suites/speed-ab.py --depths 0,16384 --reps 2
    python3 bench/suites/speed-ab.py --dry-run           what it would run

Compares the build that SHIPS against one that differs from it in exactly one
way — a compiler flag, or the ROCm SDK it was built against — and refuses the
pair when it cannot name that one way. Two variables produce a table that
looks like an answer and is not, which is how llama.cpp#19984 came to
attribute a build-configuration difference to a compiler flag.

WHAT IT WAS WRITTEN FOR. llama.cpp#19984 reports, on this exact hardware —
gfx1151, 128 GB Strix Halo — a prefill collapse it attributes to a
loop-unrolling regression in ROCm 7+, and works around it with

    -mllvm --amdgpu-unroll-threshold-local=600

    pp512 @ d32768   343.99 t/s (self-built, WITH the flag)  vs   93.26 t/s
    pp512 @ d65536   251.15 t/s (self-built, WITH the flag)  vs   49.72 t/s

WHAT THAT COMPARISON DOES NOT CONTAIN, and the reason this suite exists: the
fast side is self-built AND carries the flag, the slow side is an official
prebuilt binary. Two variables. Where a self-built build WITHOUT the flag
lands is the cell nobody has — and it is the cell this stack actually ships,
because setup/scripts/build-llama.sh has never passed the flag.

So the arms here differ in ONE thing. Same commit, same two patches, same
cmake line, one `-DCMAKE_HIP_FLAGS`. build-llama.sh --unroll builds the
second one into a family of its own precisely so it cannot overwrite the
first.

INTERLEAVED, and that is not fussiness. Running three of A and then three of
B measures the flag AND whatever the machine did between them — this SoC
clocks down as it warms, so the second arm would be slower for a reason that
has nothing to do with unrolling. A,B,A,B,A,B puts that drift into both arms
instead of into the difference.

WHY llama-bench AND NOT ONE OF OUR SUITES. Because the number being checked
is somebody else's, and reproducing it needs their instrument. What a user
FEELS is a different question, measured through bench/speed.py behind
bench/sideserver.py, and worth the machine time only if this screening shows
something.

SAFETY. The GPU takes system RAM through GTT and that allocation is PINNED:
a model that does not fit does not page, it stops the machine — three times
on 26.08.2026, once needing a power cycle. So production is stopped first,
GTT is waited for until it stops FALLING rather than until a port closes, and
a transient systemd timer restarts production no matter what happens to this
process. The teardown in `finally` is not enough on its own: an OOM kill
takes it with SIGKILL and `finally` never runs, which is exactly how the
third incident left the machine without a model server.
"""
import argparse, json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "bench"))
import run as runlib                                          # noqa: E402

# The FILE is the profile's; WHERE it lives is the machine's, and only the
# one resolver knows that. A path written out here would be this machine's
# path, which tests/test_localenv.py refuses on sight — correctly, since the
# repository is meant to run on somebody else's disk layout too.
MODEL_FILE = "Qwen3.8-27B-UD-Q4_K_XL.gguf"      # what qwen38.env serves
DEADMAN = "speed-ab-deadman"

_UNIT = None


def unit():
    """Which llama-user@ instance is ACTUALLY serving — asked, not assumed.

    `UNIT = "llama-user@qwen38"` stood here as a constant until 04.09.2026,
    and it fired: a flag-ab run measuring a different model stopped
    llama-user@flashnext at the start and started llama-user@qwen38 at the
    end, so the machine served a model nobody had switched to. `is-enabled`
    still said flashnext; only the process holding port 8080 disagreed.
    Production was restored by hand. The dead man's switch armed the same
    wrong start, so a crash would have done it too.

    CLAUDE.md carries the rule and names this exact defect being fixed in the
    determinism lane on 01.09.2026 — "no script hard-wires a production unit;
    derive it from `models.sh serving`". This copy survived that review
    because it hard-wires the unit rather than the profile, and nobody
    grepped for the second spelling.

    ONE reader, and it is setup/lib/models.sh: it takes `--alias` off the
    command line of the process that holds the port. Deliberately not
    `is-active`, which cannot say which of two started instances won the race
    for 8080 — that distinction is the whole reason models.sh has a `serving`
    verb separate from `active`.

    Resolved ONCE and cached, because it is asked again in the `finally` that
    restarts production, and by then nothing is serving.

    Returns None when nothing is serving. Then there is nothing to stop and
    nothing to restart — and inventing a unit to start is precisely how this
    defect did its damage.
    """
    global _UNIT
    if _UNIT is None:
        r = subprocess.run(
            ["bash", os.path.join(REPO, "setup", "lib", "models.sh"),
             "serving"], capture_output=True, text=True)
        names = [n for n in (r.stdout or "").split() if n]
        _UNIT = "llama-user@%s" % names[0] if len(names) == 1 else False
        if len(names) > 1:
            say("  MORE THAN ONE llama-server is serving (%s) — refusing to "
                "guess which one to put back" % " ".join(names))
    return _UNIT or None


def default_model():
    """Same resolution order bench/run.py uses, and the same one resolver."""
    explicit = os.environ.get("SPEEDAB_MODEL")
    if explicit:
        return explicit
    models = (os.environ.get("LLAMA_MODELS")
              or runlib.systemdfile.models_dir())
    return os.path.join(models, MODEL_FILE)


def say(msg):
    print(msg, flush=True)


def systemctl(*args):
    return subprocess.run(["systemctl", "--user", *args],
                          capture_output=True, text=True, timeout=120)


def arm_deadman(minutes):
    """Restart production in `minutes`, whatever happens to this process.

    Armed BEFORE anything is stopped. A `finally` block cannot survive
    SIGKILL, and the failure mode it guards against is the machine going down
    hard with the model server still stopped.
    """
    subprocess.run(["systemctl", "--user", "stop", DEADMAN + ".timer"],
                   capture_output=True, text=True)
    target = unit()
    if target is None:
        # Nothing is serving, so there is nothing for the switch to put back.
        # Arming it anyway would mean choosing a model, which is the mistake
        # unit() exists to prevent.
        say("  nothing is serving — no dead man's switch to arm")
        return
    r = subprocess.run(
        ["systemd-run", "--user", "--quiet", "--collect",
         "--unit", DEADMAN, "--on-active=%dmin" % minutes,
         "--timer-property=AccuracySec=10s",
         "systemctl", "--user", "start", target],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("could not arm the dead man's switch, refusing to "
                         "stop production: %s" % (r.stderr or r.stdout)[:300])
    say("  dead man's switch armed: %s starts again in %d min"
        % (target, minutes))


def disarm_deadman():
    subprocess.run(["systemctl", "--user", "stop", DEADMAN + ".timer"],
                   capture_output=True, text=True)


# The serving profile's batch geometry, so a build comparison runs at the
# operating point rather than at llama-bench's default. A DECLARED copy of
# qwen38.env — it sat here hardcoded as 2048 after the profile moved to 512,
# which is exactly how the comparison would have been run off the operating
# point; tests/test_speedab.py holds it against the profile's LLAMA_ARGS now.
UB = "512"
BATCH = "2048"


def bench(binary, model, depths, prompt, gen, extra, dry=False,
          ub=UB, batch=BATCH):
    """One llama-bench run. Returns its parsed JSON rows, or [] on failure."""
    argv = [binary, "-m", model, "-p", str(prompt), "-n", str(gen),
            "-d", ",".join(str(d) for d in depths),
            # -fa takes on|off|auto, NOT 1. With `1` llama-bench does not
            # fail — it parses it as a mode it does not know and the run is
            # not the one the profile serves.
            "-ngl", "999", "-fa", "on", "-ub", str(ub), "-b", str(batch),
            "-r", "1", "-o", "json"] + extra
    if dry:
        say("  would run: %s" % " ".join(argv))
        return []
    r = subprocess.run(argv, capture_output=True, text=True, timeout=3600,
                       env=env_for(binary))
    if r.returncode != 0:
        say("  FAILED (%d): %s" % (r.returncode, (r.stderr or "")[-400:]))
        return []
    try:
        return json.loads(r.stdout)
    except Exception as e:
        say("  unparseable output (%s): %s" % (e, r.stdout[:300]))
        return []


ROCM_LIBS = ("libamdhip64", "libhsa-runtime64", "librocblas", "libhipblas")


def env_for(binary):
    """The environment this build has to run in.

    A build made against another ROCm SDK needs that SDK's libraries at RUN
    time too, and RUNPATH is only `$ORIGIN:`. Without this the binary loads
    /lib64 — the system ROCm — and the run measures a compiler difference
    while reporting an SDK one. The stamp is what says which SDK, so a build
    that names none keeps the system's.
    """
    libs = [os.path.dirname(binary)]
    sdk = stamp_beside(binary).get("rocm_path", "")
    if sdk:
        libs = [os.path.join(sdk, "lib"), os.path.join(sdk, "lib64")] + libs
    prev = os.environ.get("LD_LIBRARY_PATH", "")
    if prev:
        libs.append(prev)
    return dict(os.environ, LD_LIBRARY_PATH=os.pathsep.join(libs))


def rocm_libs_of(binary, env=None):
    """Where this binary's ROCm libraries actually come from.

    Not which SDK it was BUILT against — which one it will LOAD. The two are
    the same only by accident here: RUNPATH is `$ORIGIN:`, so anything not
    beside the binary is resolved through the system search path, and a build
    made against another ROCm silently picks up /lib64 when the sonames match.
    That failure produces numbers rather than an error, which is the only
    reason this function exists.

    Returns {soname: resolved path}, empty if ldd could not be run.
    """
    target = binary
    if os.path.basename(binary).startswith("llama-"):
        # The executables are 12 KB stubs; the HIP backend is in the .so
        # beside them, and that is what links against ROCm.
        for cand in ("libggml-hip.so", "libggml-hip.so.0"):
            p = os.path.join(os.path.dirname(binary), cand)
            if os.path.exists(p):
                target = p
                break
    try:
        r = subprocess.run(["ldd", target], capture_output=True, text=True,
                           timeout=60, env=env or os.environ)
    except Exception:
        return {}
    out = {}
    for line in r.stdout.splitlines():
        m = re.match(r"\s*(\S+)\s*=>\s*(\S+)", line)
        if m and any(m.group(1).startswith(x) for x in ROCM_LIBS):
            out[m.group(1)] = m.group(2)
    return out


def rec(text):
    """A path as it should be RECORDED rather than run.

    Expanded to run, unexpanded to record. Reports are published; a home
    directory in one names the machine it was measured on. systemdfile's
    docstring says this already happened once, to three reports, before
    tests/test_localenv.py caught it — and that test does not read
    bench/reports/, so here it has to be got right rather than caught.
    """
    return runlib.systemdfile.unexpand(str(text))


def order_for(rnd, arms):
    """Which arm runs first in round `rnd`. Alternates, so that over an even
    number of rounds each arm is first exactly half the time and the
    second-place thermal penalty cancels instead of accumulating on one."""
    return list(arms) if rnd % 2 == 0 else list(reversed(arms))


def gpu_state():
    """Temperature and power draw, for the record rather than for a decision.

    The operator watched package power fall from 51 W to 49 W during a run and
    asked whether the arms were being compared fairly. They were — but the
    answer had to be reconstructed from a previous run's numbers because
    nothing here recorded the machine's own state. Now it does.
    """
    import glob
    for hw in glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*/"):
        try:
            with open(hw + "temp1_input") as f:
                t = int(f.read()) / 1000.0
        except OSError:
            continue
        w = None
        try:
            with open(hw + "power1_average") as f:
                w = int(f.read()) / 1e6
        except OSError:
            pass
        return {"temp_c": round(t, 1), "power_w": round(w, 1) if w else None}
    return {}


def key_of(row):
    """(test-name, depth) — what makes two rows comparable across arms."""
    return (row.get("n_prompt"), row.get("n_gen"), row.get("n_depth", 0))


def label_of(k):
    n_prompt, n_gen, depth = k
    what = "pp%s" % n_prompt if n_prompt else "tg%s" % n_gen
    return "%s @ d%s" % (what, depth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="rocm-patched",
                    help="the build that ships (path, dir name or build id)")
    ap.add_argument("--variant", "--unroll", dest="variant", default=None,
                    help="the build to compare against; default: the newest "
                         "one in --variant-family")
    ap.add_argument("--variant-family", default="rocm-unroll",
                    help="which build family the variant comes from "
                         "(rocm-unroll, rocm-altsdk, ...)")
    ap.add_argument("--model", default=None,
                    help="default: the profile's model under the resolved "
                         "model directory")
    ap.add_argument("--depths", default="0,16384,32768,65536",
                    help="prefix depths, as llama-bench -d")
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3,
                    help="interleaved rounds; each round runs BOTH arms")
    ap.add_argument("--deadline", type=int, default=90,
                    help="minutes after which production restarts regardless")
    ap.add_argument("--keep-production", action="store_true",
                    help="do not stop the model server (only safe for a "
                         "depth-0 run on a small model, and not checked)")
    ap.add_argument("--no-warmup", action="store_true",
                    help="skip the discarded first pass (faster, and the "
                         "first round then carries the warm-up drift)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    a.model = a.model or default_model()
    depths = [int(d) for d in a.depths.split(",") if d.strip()]
    ref = runlib.resolve_binary(a.reference)
    variant = a.variant or newest_of_family(a.variant_family)
    if not variant:
        raise SystemExit(
            "no %s build found. Build one, for example:\n"
            "    bash setup/scripts/build-llama.sh --unroll --with-bench\n"
            "    bash setup/scripts/build-llama.sh --rocm-path DIR --with-bench"
            % a.variant_family)
    var = runlib.resolve_binary(variant)

    arms = [("reference", bench_binary(ref)), ("variant", bench_binary(var))]
    for name, b in arms:
        if not os.path.exists(b):
            raise SystemExit(
                "%s: no llama-bench beside %s.\n"
                "    That build predates --with-bench. Rebuild it with\n"
                "        bash setup/scripts/build-llama.sh --with-bench ...\n"
                "    or build the target into the existing directory." % (name, b))

    say("model:     %s" % a.model)
    say("depths:    %s" % depths)
    say("rounds:    %d, interleaved" % a.reps)
    hips = {}
    for name, b in arms:
        say("%-10s %s" % (name + ":", rec(b)))
        st = stamp_beside(b)
        say("           build %s, cmake carries the flag: %s"
            % (st.get("build_id", "?"),
               "yes" if "amdgpu-unroll" in st.get("cmake", "") else "NO"))
        libs = rocm_libs_of(b, env_for(b))
        hip = libs.get("libamdhip64.so.7") or next(
            (v for k, v in libs.items() if k.startswith("libamdhip64")), None)
        say("           ROCm it will LOAD: %s" % (hip or "could not tell"))
        hips[name] = hip

    # WHAT DIFFERS, named rather than assumed. A pair with no difference
    # measures nothing; a pair with two measures neither of them. The issue
    # this suite was written for made exactly that mistake, so the check is
    # the point of the file and not a formality.
    sr, su = stamp_beside(arms[0][1]), stamp_beside(arms[1][1])
    axes = []
    if ("amdgpu-unroll" in sr.get("cmake", "")) != \
            ("amdgpu-unroll" in su.get("cmake", "")):
        axes.append("the unroll flag")
    if hips["reference"] != hips["variant"]:
        axes.append("the ROCm they load (%s vs %s)"
                    % (hips["reference"], hips["variant"]))
    if sr.get("upstream_commit") != su.get("upstream_commit"):
        axes.append("the llama.cpp commit (%s vs %s)"
                    % ((sr.get("upstream_commit") or "?")[:9],
                       (su.get("upstream_commit") or "?")[:9]))

    if not axes:
        raise SystemExit(
            "the two builds do not differ in anything this suite can name — "
            "same flag, same ROCm, same commit. There is nothing to compare.")
    say("\ndiffers in: %s" % "; ".join(axes))
    if len(axes) > 1:
        raise SystemExit(
            "MORE THAN ONE difference. Whatever the table showed could not be "
            "attributed to any of them. Build a pair that differs in one.")

    if a.dry_run:
        for name, b in arms:
            bench(b, a.model, depths, a.prompt, a.gen, [], dry=True)
        return 0

    stopped = False
    results = {name: {} for name, _ in arms}
    try:
        if not a.keep_production:
            before = runlib.gtt()
            say("\nGTT now: %.1f GiB" % (before or 0))
            arm_deadman(a.deadline)
            if unit() is None:
                say("nothing is serving — nothing to stop")
            else:
                say("stopping %s" % unit())
                systemctl("stop", unit())
                stopped = True
            runlib.wait_for_gtt_to_settle()
            say("GTT after stop: %.1f GiB" % (runlib.gtt() or 0))

        if not a.no_warmup:
            # A DISCARDED FIRST PASS. Package power settles from ~51 W to ~49 W
            # as the SoC warms, and the previous run lost 3-4 % between its
            # first and last round because of it. One short pass per arm puts
            # every measured round in the same thermal state.
            say("\nwarm-up (discarded): one shallow pass per arm")
            for name, b in arms:
                bench(b, a.model, [0], a.prompt, min(a.gen, 32), [])
            say("  GPU after warm-up: %s" % gpu_state())

        for rnd in range(a.reps):
            # COUNTERBALANCED. Straight A,B,A,B still runs B second in every
            # round, on a machine one pass warmer — worth -0.5 to -1.2 % to
            # whichever arm is second, measured on a pair that differed in
            # nothing. Alternating makes each arm first exactly half the time,
            # so the offset cancels instead of merely being small.
            order = order_for(rnd, arms)
            for name, b in order:
                say("\n== round %d/%d — %s   %s"
                    % (rnd + 1, a.reps, name, gpu_state()))
                t0 = time.time()
                for row in bench(b, a.model, depths, a.prompt, a.gen, []):
                    k = key_of(row)
                    results[name].setdefault(k, []).append(
                        row.get("avg_ts", 0.0))
                    say("   %-16s %8.2f t/s" % (label_of(k), row.get("avg_ts", 0)))
                say("   (%.0f s)" % (time.time() - t0))
    finally:
        if stopped:
            say("\nrestarting %s" % unit())
            systemctl("start", unit())
        disarm_deadman()

    report(results, arms, a, depths)
    return 0


def bench_binary(server_path):
    """llama-bench sits beside llama-server in the same build's bin/."""
    return os.path.join(os.path.dirname(server_path), "llama-bench")


def stamp_beside(binary):
    """The .build-stamp two directories up from bin/llama-bench."""
    p = os.path.join(os.path.dirname(os.path.dirname(binary)), ".build-stamp")
    out = {}
    try:
        with open(p) as f:
            for line in f:
                k, sep, v = line.partition("=")
                if sep:
                    out[k.strip()] = v.strip()
    except OSError:
        pass
    return out


def newest_of_family(family="rocm-unroll"):
    src = os.environ.get("LLAMA_SRC", os.path.expanduser("~/llama.cpp"))
    best, best_at = None, ""
    prefix = "build-%s-" % family
    try:
        names = os.listdir(src)
    except OSError:
        return None
    for d in names:
        if not d.startswith(prefix):
            continue
        st = stamp_beside(os.path.join(src, d, "bin", "x"))
        at = st.get("built_at", "")
        if best is None or at > best_at:
            # the DIRECTORY NAME, which is one of the three things
            # runlib.resolve_binary() accepts. A full path would be the
            # fourth, and it resolves it against $LLAMA_SRC instead.
            best, best_at = d, at
    return best


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return 0.0
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def report(results, arms, a, depths):
    ref, unr = results["reference"], results["variant"]
    keys = [k for k in ref if k in unr]
    keys.sort(key=lambda k: (k[2], k[0] or 0))

    lines = []
    lines.append("")
    lines.append("%-18s %12s %12s %10s" % ("", "reference", "variant", "change"))
    for k in keys:
        r, u = median(ref[k]), median(unr[k])
        pct = ((u - r) / r * 100.0) if r else 0.0
        lines.append("%-18s %9.2f t/s %9.2f t/s %+9.1f %%"
                     % (label_of(k), r, u, pct))
    lines.append("")
    lines.append("medians of %d interleaved rounds; every round ran both arms."
                 % a.reps)
    lines.append("A difference smaller than the spread between rounds is not a")
    lines.append("difference. The per-round values are in the JSON beside this.")
    text = "\n".join(lines)
    say(text)

    d = os.path.join(REPO, "bench", "reports",
                     time.strftime("%Y-%m-%d_%H%M") + "_speed-ab")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "RESULT.md"), "w") as f:
        # The title names the PAIR, not the suite's first use case. It said
        # "# unroll-flag" verbatim for every comparison until 31.08. — the
        # ROCm-10 report shipped under that title and one of the two copies
        # was corrected by hand, the other was not.
        sr, sv = stamp_beside(arms[0][1]), stamp_beside(arms[1][1])
        f.write("# speed-ab — %s vs %s\n\n"
                % (sr.get("build_id", "reference"),
                   sv.get("build_id", "variant")))
        f.write("model: `%s`\n\n" % rec(a.model))
        for name, b in arms:
            st = stamp_beside(b)
            f.write("- **%s**: `%s`, build `%s`\n"
                    % (name, rec(b), st.get("build_id", "?")))
            f.write("  - cmake: `%s`\n" % rec(st.get("cmake", "?")))
        f.write("\n```%s\n```\n" % text)
    with open(os.path.join(d, "rounds.json"), "w") as f:
        json.dump({name: {label_of(k): v for k, v in res.items()}
                   for name, res in results.items()}, f, indent=2)
    say("\nreport: %s" % d)


if __name__ == "__main__":
    sys.exit(main())
