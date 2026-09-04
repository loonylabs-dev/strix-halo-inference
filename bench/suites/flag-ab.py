#!/usr/bin/env python3
"""flag-ab — one build, N arms, one runtime difference, interleaved.

    python3 bench/suites/flag-ab.py --name ub-sweep \
        --arm "ub512:-ub 512" --arm "ub1024:-ub 1024" --arm "ub2048:-ub 2048" \
        --depths 0,32768 --prompt 2048 --reps 2
    python3 bench/suites/flag-ab.py --name hipblaslt \
        --arm "off:ROCBLAS_USE_HIPBLASLT=0" --arm "on:ROCBLAS_USE_HIPBLASLT=1" \
        --depths 0,32768 --prompt 2048 --reps 2
    python3 bench/suites/flag-ab.py ... --dry-run      what it would run

The sibling of speed-ab.py, for the other half of the question. speed-ab
compares two BUILDS that differ in one way; this compares one build under N
runtime settings that differ in one way — a llama-bench flag, or an
environment variable the ROCm runtime reads. Everything speed-ab learned the
hard way is reused rather than re-earned: interleaved counterbalanced rounds,
a discarded warm-up pass, the dead man's switch, GTT settling, reports that
do not name this machine.

WHY THE ONE-DIFFERENCE RULE IS ENFORCED HERE TOO. An arm spec is free text,
so nothing stops "--arm fast:-ub 512 -fa off" — a pair of variables wearing
one label. The table such a run prints could not attribute its difference to
either, which is the mistake llama.cpp#19984 made with two build variables
and this repo has now twice paid to unlearn. So every arm must name the SAME
set of variables, and exactly one of them may vary across arms. An arm that
omits the variable would silently mean "llama-bench's default", which is a
value nobody wrote down — refused for the same reason.

WHY -b STAYS FIXED IN A -ub SWEEP. The serving profile pairs them equal, but
llama-bench's -b is the logical batch and -ub the physical chunk; sweeping
both at once would be two variables. Sweep -ub with -b at the profile's 2048
and the chunking matches what a server with that -ub would do.

WHAT THIS SUITE CANNOT SEE, stated up front: llama-bench runs without
speculation, without the gateway, without saved prefixes. A winner here is a
SCREENING result; before it touches a profile it has to survive
bench/speed.py behind bench/sideserver.py with the full serving
configuration — draft-mtp included, which comparable prefill tuning
elsewhere regressed by a measured 1.4-5.5 %.
"""
import argparse, importlib.util, json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "bench"))
import run as runlib                                          # noqa: E402


def _load_speed_ab():
    """speed-ab.py carries the shared machinery; the hyphen keeps it from
    being imported by name, so it is loaded the way the tests load suites."""
    spec = importlib.util.spec_from_file_location(
        "speed_ab", os.path.join(HERE, "speed-ab.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sab = _load_speed_ab()

# What every invocation carries unless an arm overrides it — the serving
# profile's values, so an arm that does not vary a knob measures production's
# setting rather than llama-bench's default. A DECLARED copy of qwen38.env,
# and tests/test_flagab.py holds it against the profile's own LLAMA_ARGS so
# the two cannot drift apart in silence — the fate of every second reader of
# one file in this repo.
BASE = {"-ngl": "999", "-fa": "on", "-ub": "512", "-b": "2048", "-r": "1"}

ENV_TOKEN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def parse_arm(text):
    """'label:SPEC' -> (label, {flag: value}, {env: value}).

    SPEC is whitespace-separated. A token shaped NAME=VALUE is an environment
    variable; anything else must be a llama-bench flag followed by exactly one
    value. Free text that is neither is refused rather than passed through —
    llama-bench ACCUMULATES repeated flags into a sweep, so a stray token
    would not fail, it would quietly multiply the run matrix.
    """
    label, sep, spec = text.partition(":")
    if not sep or not label.strip():
        raise SystemExit("--arm needs 'label:SPEC', got %r" % text)
    flags, env = {}, {}
    toks = spec.split()
    i = 0
    while i < len(toks):
        t = toks[i]
        if ENV_TOKEN.match(t):
            k, _, v = t.partition("=")
            env[k] = v
            i += 1
        elif t.startswith("-"):
            if i + 1 >= len(toks) or toks[i + 1].startswith("-") \
                    or ENV_TOKEN.match(toks[i + 1]):
                raise SystemExit(
                    "arm %r: flag %s has no value. Every flag here takes "
                    "exactly one." % (label, t))
            flags[t] = toks[i + 1]
            i += 2
        else:
            raise SystemExit(
                "arm %r: %r is neither NAME=VALUE nor a flag" % (label, t))
    if not flags and not env:
        raise SystemExit("arm %r varies nothing" % label)
    return label.strip(), flags, env


def the_one_axis(arms):
    """The single variable the arms differ in, or a refusal.

    arms: [(label, flags, env)]. Returns (kind, name) with kind 'flag'|'env'.
    """
    def names(a):
        _, flags, env = a
        return {("flag", k) for k in flags} | {("env", k) for k in env}

    first = names(arms[0])
    for a in arms[1:]:
        if names(a) != first:
            raise SystemExit(
                "the arms do not name the same variables (%s vs %s). An "
                "omitted variable would mean an unstated default — write it "
                "into every arm." % (sorted(first), sorted(names(a))))

    def value(a, key):
        kind, name = key
        return a[1][name] if kind == "flag" else a[2][name]

    varying = [k for k in sorted(first)
               if len({value(a, k) for a in arms}) > 1]
    if not varying:
        raise SystemExit(
            "the arms do not differ in anything — every variable carries the "
            "same value in every arm. There is nothing to compare.")
    if len(varying) > 1:
        raise SystemExit(
            "MORE THAN ONE difference: %s. Whatever the table showed could "
            "not be attributed to any of them. Vary one."
            % ", ".join("%s %s" % k for k in varying))
    return varying[0]


def bench(binary, model, depths, prompt, gen, flags, env, dry=False):
    """One llama-bench run under an arm's flags and environment."""
    merged = dict(BASE)
    merged.update(flags)
    argv = [binary, "-m", model, "-p", str(prompt), "-n", str(gen),
            "-d", ",".join(str(d) for d in depths), "-o", "json"]
    for k in sorted(merged):
        argv += [k, merged[k]]
    if dry:
        sab.say("  would run: %s" % " ".join(argv)
                + ("   with %s" % " ".join("%s=%s" % kv
                                           for kv in sorted(env.items()))
                   if env else ""))
        return []
    import subprocess
    full_env = sab.env_for(binary)
    full_env.update(env)
    r = subprocess.run(argv, capture_output=True, text=True, timeout=3600,
                       env=full_env)
    if r.returncode != 0:
        sab.say("  FAILED (%d): %s" % (r.returncode, (r.stderr or "")[-400:]))
        return []
    try:
        return json.loads(r.stdout)
    except Exception as e:
        sab.say("  unparseable output (%s): %s" % (e, r.stdout[:300]))
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="rocm-patched",
                    help="the ONE build every arm runs (path, dir name or id)")
    ap.add_argument("--arm", action="append", default=[],
                    help="'label:SPEC', at least twice. SPEC mixes llama-bench "
                         "flags ('-ub 512') and env vars ('NAME=1')")
    ap.add_argument("--name", default="flag-ab",
                    help="suffix for the report directory")
    ap.add_argument("--model", default=None)
    ap.add_argument("--depths", default="0,32768",
                    help="prefix depths, as llama-bench -d. Two points by "
                         "design: this machine's findings inverted with depth "
                         "twice (ROCm 10.1, the unroll claim), so a depth-0 "
                         "screen alone does not carry")
    ap.add_argument("--prompt", type=int, default=2048,
                    help="pp size. 2048, not 512: a -ub above the prompt "
                         "cannot show itself on a shorter one")
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--reps", type=int, default=2,
                    help="interleaved rounds; each round runs every arm")
    ap.add_argument("--deadline", type=int, default=90,
                    help="minutes after which production restarts regardless")
    ap.add_argument("--keep-production", action="store_true")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if len(a.arm) < 2:
        raise SystemExit("need at least two --arm")
    arms = [parse_arm(t) for t in a.arm]
    if len({label for label, _, _ in arms}) != len(arms):
        raise SystemExit("arm labels repeat")
    kind, varied = the_one_axis(arms)

    a.model = a.model or sab.default_model()
    depths = [int(d) for d in a.depths.split(",") if d.strip()]
    binary = sab.bench_binary(runlib.resolve_binary(a.build))
    if not os.path.exists(binary):
        raise SystemExit("no llama-bench beside the build: %s\n"
                         "    rebuild with build-llama.sh --with-bench"
                         % sab.rec(binary))

    st = sab.stamp_beside(binary)
    sab.say("build:     %s (build %s)" % (sab.rec(binary),
                                          st.get("build_id", "?")))
    sab.say("model:     %s" % sab.rec(a.model))
    sab.say("depths:    %s   prompt: %d" % (depths, a.prompt))
    sab.say("rounds:    %d, interleaved, %d arms" % (a.reps, len(arms)))
    sab.say("differs in: the %s %s" % (kind, varied))
    for label, flags, env in arms:
        sab.say("  %-10s %s" % (label + ":", " ".join(
            ["%s %s" % kv for kv in sorted(flags.items())]
            + ["%s=%s" % kv for kv in sorted(env.items())])))
    # The ambient value reaches EVERY arm equally — but if the varied
    # variable itself is already set outside, the reader deserves to know.
    if kind == "env" and varied in os.environ:
        sab.say("NOTE: %s=%s is set in the ambient environment; arms override "
                "it explicitly, so the table is unaffected — recorded for the "
                "record." % (varied, os.environ[varied]))

    if a.dry_run:
        for label, flags, env in arms:
            bench(binary, a.model, depths, a.prompt, a.gen, flags, env,
                  dry=True)
        return 0

    stopped = False
    results = {label: {} for label, _, _ in arms}
    try:
        if not a.keep_production:
            sab.say("\nGTT now: %.1f GiB" % (runlib.gtt() or 0))
            sab.arm_deadman(a.deadline)
            if sab.unit() is None:
                sab.say("nothing is serving — nothing to stop")
            else:
                sab.say("stopping %s" % sab.unit())
                sab.systemctl("stop", sab.unit())
                stopped = True
            runlib.wait_for_gtt_to_settle()
            sab.say("GTT after stop: %.1f GiB" % (runlib.gtt() or 0))

        if not a.no_warmup:
            sab.say("\nwarm-up (discarded): one shallow pass per arm")
            for label, flags, env in arms:
                bench(binary, a.model, [0], min(a.prompt, 512),
                      min(a.gen, 32), flags, env)
            sab.say("  GPU after warm-up: %s" % sab.gpu_state())

        for rnd in range(a.reps):
            # Counterbalanced by reversal: positions i and n-1-i swap between
            # rounds, so over an even number every arm's mean position is the
            # middle. With three arms the middle one always sits middle —
            # thermally the average, fine for a screening.
            order = sab.order_for(rnd, arms)
            for label, flags, env in order:
                sab.say("\n== round %d/%d — %s   %s"
                        % (rnd + 1, a.reps, label, sab.gpu_state()))
                t0 = time.time()
                for row in bench(binary, a.model, depths, a.prompt, a.gen,
                                 flags, env):
                    k = sab.key_of(row)
                    results[label].setdefault(k, []).append(
                        row.get("avg_ts", 0.0))
                    sab.say("   %-16s %8.2f t/s"
                            % (sab.label_of(k), row.get("avg_ts", 0)))
                sab.say("   (%.0f s)" % (time.time() - t0))
    finally:
        if stopped:
            sab.say("\nrestarting %s" % sab.unit())
            sab.systemctl("start", sab.unit())
        sab.disarm_deadman()

    report(results, arms, a, st, (kind, varied))
    return 0


def report(results, arms, a, stamp, axis):
    labels = [label for label, _, _ in arms]
    ref = results[labels[0]]
    keys = [k for k in ref if all(k in results[l] for l in labels)]
    keys.sort(key=lambda k: (k[2], k[0] or 0))

    lines = [""]
    head = "%-18s" % "" + "".join("%14s" % l for l in labels)
    lines.append(head)
    for k in keys:
        cells = "%-18s" % sab.label_of(k)
        r = sab.median(ref[k])
        for l in labels:
            v = sab.median(results[l][k])
            if l == labels[0]:
                cells += "%10.2f t/s" % v
            else:
                pct = ((v - r) / r * 100.0) if r else 0.0
                cells += "%8.2f %+4.0f%%" % (v, pct)
        lines.append(cells)
    lines.append("")
    lines.append("medians of %d interleaved rounds; every round ran every arm."
                 % a.reps)
    lines.append("Change is against the FIRST arm. A difference smaller than")
    lines.append("the spread between rounds is not a difference — the")
    lines.append("per-round values are in the JSON beside this.")
    text = "\n".join(lines)
    sab.say(text)

    d = os.path.join(REPO, "bench", "reports",
                     time.strftime("%Y-%m-%d_%H%M") + "_" + a.name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "RESULT.md"), "w") as f:
        f.write("# %s — one build, the %s %s varied\n\n"
                % (a.name, axis[0], axis[1]))
        f.write("model: `%s`\nbuild: `%s` (`%s`)\n\n"
                % (sab.rec(a.model), stamp.get("build_id", "?"),
                   sab.rec(stamp.get("upstream_commit", "?"))))
        f.write("Screening via llama-bench: no speculation, no gateway, no\n"
                "saved prefixes. A winner still has to survive the serving\n"
                "profile (bench/speed.py behind bench/sideserver.py).\n\n")
        for label, flags, env in arms:
            f.write("- **%s**: `%s`\n" % (label, " ".join(
                ["%s %s" % kv for kv in sorted(flags.items())]
                + ["%s=%s" % kv for kv in sorted(env.items())])))
        f.write("\n```%s\n```\n" % text)
    with open(os.path.join(d, "rounds.json"), "w") as f:
        json.dump({"_meta": {"argv": [sab.rec(x) for x in sys.argv],
                             "build": stamp,
                             "axis": {"kind": axis[0], "name": axis[1]}},
                   **{label: {sab.label_of(k): v for k, v in res.items()}
                      for label, res in results.items()}}, f, indent=2)
    sab.say("\nreport: %s" % d)


if __name__ == "__main__":
    sys.exit(main())
