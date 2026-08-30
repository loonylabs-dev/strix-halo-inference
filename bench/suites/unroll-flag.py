#!/usr/bin/env python3
"""unroll-flag — does ROCm's loop-unrolling workaround do anything HERE?

    python3 bench/suites/unroll-flag.py
    python3 bench/suites/unroll-flag.py --depths 0,16384 --reps 2
    python3 bench/suites/unroll-flag.py --dry-run       what it would run

THE QUESTION. llama.cpp#19984 reports, on this exact hardware — gfx1151,
128 GB Strix Halo — a prefill collapse it attributes to a loop-unrolling
regression in ROCm 7+, and works around it with

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

MODEL = os.environ.get(
    "UNROLL_MODEL", "/mnt/shared/LLM/Qwen3.8-27B-UD-Q4_K_XL.gguf")
UNIT = "llama-user@qwen38"
DEADMAN = "unroll-flag-deadman"


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
    r = subprocess.run(
        ["systemd-run", "--user", "--quiet", "--collect",
         "--unit", DEADMAN, "--on-active=%dmin" % minutes,
         "--timer-property=AccuracySec=10s",
         "systemctl", "--user", "start", UNIT],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("could not arm the dead man's switch, refusing to "
                         "stop production: %s" % (r.stderr or r.stdout)[:300])
    say("  dead man's switch armed: %s starts again in %d min" % (UNIT, minutes))


def disarm_deadman():
    subprocess.run(["systemctl", "--user", "stop", DEADMAN + ".timer"],
                   capture_output=True, text=True)


def bench(binary, model, depths, prompt, gen, extra, dry=False):
    """One llama-bench run. Returns its parsed JSON rows, or [] on failure."""
    argv = [binary, "-m", model, "-p", str(prompt), "-n", str(gen),
            "-d", ",".join(str(d) for d in depths),
            # -fa takes on|off|auto, NOT 1. With `1` llama-bench does not
            # fail — it parses it as a mode it does not know and the run is
            # not the one the profile serves.
            "-ngl", "999", "-fa", "on", "-ub", "2048", "-b", "2048",
            "-r", "1", "-o", "json"] + extra
    if dry:
        say("  would run: %s" % " ".join(argv))
        return []
    env = dict(os.environ, LD_LIBRARY_PATH=os.path.dirname(binary))
    r = subprocess.run(argv, capture_output=True, text=True, timeout=3600,
                       env=env)
    if r.returncode != 0:
        say("  FAILED (%d): %s" % (r.returncode, (r.stderr or "")[-400:]))
        return []
    try:
        return json.loads(r.stdout)
    except Exception as e:
        say("  unparseable output (%s): %s" % (e, r.stdout[:300]))
        return []


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
    ap.add_argument("--unroll", default=None,
                    help="the build carrying the flag; default: the newest "
                         "rocm-unroll build")
    ap.add_argument("--model", default=MODEL)
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
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    depths = [int(d) for d in a.depths.split(",") if d.strip()]
    ref = runlib.resolve_binary(a.reference)
    unroll = a.unroll or newest_unroll()
    if not unroll:
        raise SystemExit(
            "no rocm-unroll build found. Build one:\n"
            "    bash setup/scripts/build-llama.sh --unroll --with-bench")
    unr = runlib.resolve_binary(unroll)

    arms = [("reference", bench_binary(ref)), ("unroll", bench_binary(unr))]
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
    for name, b in arms:
        say("%-10s %s" % (name + ":", runlib.systemdfile.unexpand(b)
                          if hasattr(runlib, "systemdfile") else b))
        st = stamp_beside(b)
        say("           build %s, cmake carries the flag: %s"
            % (st.get("build_id", "?"),
               "yes" if "amdgpu-unroll" in st.get("cmake", "") else "NO"))

    # The one thing that must be true of the pair, asserted rather than
    # assumed: same commit, and exactly one of them carrying the flag.
    sr, su = stamp_beside(arms[0][1]), stamp_beside(arms[1][1])
    if sr.get("upstream_commit") != su.get("upstream_commit"):
        say("\n  ! the two builds are NOT the same commit:\n"
            "      reference %s\n      unroll    %s\n"
            "    the difference would carry more than the flag."
            % (sr.get("upstream_commit"), su.get("upstream_commit")))
    if "amdgpu-unroll" in sr.get("cmake", ""):
        raise SystemExit("the REFERENCE build already carries the flag — "
                         "there is nothing to compare.")
    if "amdgpu-unroll" not in su.get("cmake", ""):
        raise SystemExit("the unroll build does NOT carry the flag. Its "
                         "stamp says: %s" % su.get("cmake", "")[:200])

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
            say("stopping %s" % UNIT)
            systemctl("stop", UNIT)
            stopped = True
            runlib.wait_for_gtt_to_settle()
            say("GTT after stop: %.1f GiB" % (runlib.gtt() or 0))

        for rnd in range(a.reps):
            for name, b in arms:
                say("\n== round %d/%d — %s" % (rnd + 1, a.reps, name))
                t0 = time.time()
                for row in bench(b, a.model, depths, a.prompt, a.gen, []):
                    k = key_of(row)
                    results[name].setdefault(k, []).append(
                        row.get("avg_ts", 0.0))
                    say("   %-16s %8.2f t/s" % (label_of(k), row.get("avg_ts", 0)))
                say("   (%.0f s)" % (time.time() - t0))
    finally:
        if stopped:
            say("\nrestarting %s" % UNIT)
            systemctl("start", UNIT)
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


def newest_unroll():
    src = os.environ.get("LLAMA_SRC", os.path.expanduser("~/llama.cpp"))
    best, best_at = None, ""
    try:
        names = os.listdir(src)
    except OSError:
        return None
    for d in names:
        if not d.startswith("build-rocm-unroll-"):
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
    ref, unr = results["reference"], results["unroll"]
    keys = [k for k in ref if k in unr]
    keys.sort(key=lambda k: (k[2], k[0] or 0))

    lines = []
    lines.append("")
    lines.append("%-18s %12s %12s %10s" % ("", "reference", "unroll", "change"))
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
                     time.strftime("%Y-%m-%d_%H%M") + "_unroll-flag")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "RESULT.md"), "w") as f:
        f.write("# unroll-flag\n\n")
        f.write("model: `%s`\n\n" % a.model)
        for name, b in arms:
            st = stamp_beside(b)
            f.write("- **%s**: `%s`, build `%s`\n" % (name, b, st.get("build_id", "?")))
            f.write("  - cmake: `%s`\n" % st.get("cmake", "?"))
        f.write("\n```%s\n```\n" % text)
    with open(os.path.join(d, "rounds.json"), "w") as f:
        json.dump({name: {label_of(k): v for k, v in res.items()}
                   for name, res in results.items()}, f, indent=2)
    say("\nreport: %s" % d)


if __name__ == "__main__":
    sys.exit(main())
