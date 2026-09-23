#!/usr/bin/env python3
"""logstats.py — what the SERVED turns say, read off the container's own log.

Why this exists. The sweeps in this directory measure a server under a shape
this repo chooses; this reads the shape the OPERATOR actually ran. Both are
needed and neither substitutes for the other: a sweep is reproducible and
answers "how fast can it go", a log answers "how fast was it, on the work that
happened" — and only the second can say whether a tuning change held up over
days.

It was written on 21.09.2026 for the 0.8.1 -> 0.12.3 bump, when the 17.09.
KV-pool finding turned out to live in `podman logs halogen` and nowhere else.
`halogenexec` runs `podman rm -f halogen` before every start, so that log is
DELETED by the next restart. Anything read from it has to be written down
first; this tool is the writing-down.

    python3 bench/logstats.py --log <file> [--json <out>] [--label 0.8.1]
    podman logs halogen 2>&1 | python3 bench/logstats.py --log -

WHAT IT REFUSES TO DO is as important as what it reports:

  * It does not print a max/min "spread". The 0.8.1 baseline's flat band reads
    98.7x that way and 1.54 s at p99 — the spread was ONE outlier against a
    0.06 s minimum, and reporting it invites exactly the wrong conclusion.
    Percentiles only.
  * It does not merge prompt depths. Decode falls with KV size and prefill
    rises with it, so one median over every turn describes no turn at all.
  * It says n per band and prints the parallel share, because a band of 11
    turns and a band of 3,979 do not carry the same weight and a log cannot
    tell you what was NOT exercised. On the 0.8.1 baseline 90% of turns had
    prompts under 2,000 tokens: whatever that log proves, it does not prove
    anything about deep concurrent sessions.

The line it parses is the front end's own finish line, e.g.

  serve_api: mtp 487 tok in 15.04s = 32.38 t/s | 256 rounds, commit 1.41/round
  | prompt 17747 (13308 cached), prefill 5.83s | detok 52us/tok | 124 tok
  beside other streams

Fields after `prefill` vary by release and by what ran (pld, pool occupancy,
clamps, `think on|off` since 0.12.2), so the pattern anchors on the stable
head and treats the rest as optional. A line it cannot parse is COUNTED and
reported, never dropped silently: a release that renames a field would
otherwise read as a quiet server.
"""

import argparse
import json
import re
import sys

# The stable head of the finish line. Everything after `prefill Xs` is
# release-dependent and is read by separate optional patterns below.
#
# `= n/a` is a real and correct value, not a broken line: a turn that produced
# one token in 0.00 s has no rate to report. Such a turn is counted and its
# PREFILL is used (that number is sound); only its decode rate is dropped.
# Treating it as a parse failure the way the first version did points the
# reader at a renamed field that never moved.
LINE = re.compile(
    r"serve_api: (?P<drafter>[a-z]+) (?P<gen>\d+) tok in (?P<secs>[\d.]+)s"
    r" = (?:(?P<tps>[\d.]+) t/s|n/a)"
    r".*?prompt (?P<prompt>\d+)"
    r"(?: \((?P<cached>\d+) cached(?:, (?P<cached_pct>[\d.]+)%)?\))?"
    r", prefill (?P<prefill>[\d.]+)s")

# 0.11.5 added the pool occupancy and the clamp; 0.12.2 added `think on|off`.
# Absent on 0.8.1, so every one of these is optional and None means "this
# release did not say", never 0.
POOL = re.compile(r"pool (?P<used>\d+)/(?P<size>\d+) (?P<pct>\d+)%")
CLAMP = re.compile(r"max_tokens clamped (?P<from>\d+) -> (?P<to>\d+)")
THINK = re.compile(r"think (?P<state>on|off)")

# Prompt-depth bands. Chosen for what this stack does rather than round
# numbers: under 2k is a Claude Code tool turn, 10k-50k a working session,
# 50k-150k a deep one, and above 150k is where the 262,144 window starts to
# matter.
BANDS = ((0, 2_000), (2_000, 10_000), (10_000, 50_000),
         (50_000, 150_000), (150_000, None))

PCTS = ((50, "p50"), (90, "p90"), (99, "p99"))


def pct(sorted_values, p):
    """Nearest-rank percentile. No interpolation: with n=11 an interpolated
    p99 invents a value between two samples and reads as more precision than
    eleven turns can carry."""
    if not sorted_values:
        return None
    i = min(int(len(sorted_values) * p / 100), len(sorted_values) - 1)
    return sorted_values[i]


def parse(text):
    rows, unparsed = [], []
    for line in text.splitlines():
        if "serve_api:" not in line:
            continue
        m = LINE.search(line)
        if not m:
            # Only lines that LOOK like a finish line are a parse failure;
            # startup and defaults lines legitimately do not match.
            if " tok in " in line or " t/s" in line:
                unparsed.append(line)
            continue
        g = m.groupdict()
        pool = POOL.search(line)
        clamp = CLAMP.search(line)
        think = THINK.search(line)
        rows.append({
            "drafter": g["drafter"],
            "gen": int(g["gen"]),
            "secs": float(g["secs"]),
            "tps": float(g["tps"]) if g["tps"] else None,
            "prompt": int(g["prompt"]),
            "cached": int(g["cached"]) if g["cached"] else 0,
            "prefill": float(g["prefill"]),
            "parallel": "beside other streams" in line,
            "pool_pct": int(pool.group("pct")) if pool else None,
            "clamped": bool(clamp),
            "think": think.group("state") if think else None,
        })
    return rows, unparsed


def summarise(rows):
    out = {"n": len(rows), "bands": []}
    for lo, hi in BANDS:
        band = [r for r in rows
                if r["prompt"] >= lo and (hi is None or r["prompt"] < hi)]
        if not band:
            continue
        pf = sorted(r["prefill"] for r in band)
        tp = sorted(r["tps"] for r in band if r["tps"] is not None)
        out["bands"].append({
            "from": lo, "to": hi, "n": len(band),
            "n_rated": len(tp),
            "prefill_s": {name: pct(pf, p) for p, name in PCTS},
            "decode_tps": {name: pct(tp, p) for p, name in PCTS},
            # The floor matters more than the ceiling here: the 768k era's
            # symptom was a decode MEDIAN of 24.8 t/s, so "how bad did the
            # worst turns get" is the question this answers.
            # None, not 0.0, when every turn in the band was a single token:
            # a floor of zero would read as a stalled server.
            "decode_tps_min": tp[0] if tp else None,
            "prefill_s_max": pf[-1],
            "parallel": sum(1 for r in band if r["parallel"]),
            "clamped": sum(1 for r in band if r["clamped"]),
            "cached_share": (
                sum(r["cached"] for r in band) / sum(r["prompt"] for r in band)
                if sum(r["prompt"] for r in band) else None),
        })
    deep = [r for r in rows if r["prompt"] > 100_000]
    out["coverage"] = {
        "deepest_prompt": max((r["prompt"] for r in rows), default=None),
        "over_100k": len(deep),
        "parallel_total": sum(1 for r in rows if r["parallel"]),
        "pool_pct_max": max((r["pool_pct"] for r in rows
                             if r["pool_pct"] is not None), default=None),
        "think_seen": sorted({r["think"] for r in rows if r["think"]}),
    }
    return out


def render(s, label, unparsed):
    w = []
    w.append(f"served turns{' — ' + label if label else ''}: n={s['n']}")
    if unparsed:
        w.append(f"!! {len(unparsed)} finish-shaped lines did NOT parse — a "
                 f"renamed field reads as a quiet server, so check these:")
        for line in unparsed[:3]:
            w.append(f"   {line[:120]}")
    w.append("")
    w.append(f"{'prompt band':>20} {'n':>6} "
             f"{'prefill p50':>12} {'p90':>7} {'p99':>7} {'worst':>7}  "
             f"{'decode p50':>11} {'p99':>6} {'floor':>6}  {'par':>5}")
    def num(v, width, dec=1):
        return f"{v:>{width}.{dec}f}" if v is not None else f"{'—':>{width}}"
    for b in s["bands"]:
        to = b["to"] if b["to"] is not None else "inf"
        w.append(
            f"{b['from']:>9,}-{str(to):>10} {b['n']:>6} "
            f"{b['prefill_s']['p50']:>11.2f}s {b['prefill_s']['p90']:>6.2f}s "
            f"{b['prefill_s']['p99']:>6.2f}s {b['prefill_s_max']:>6.2f}s  "
            f"{num(b['decode_tps']['p50'], 10)} {num(b['decode_tps']['p99'], 5)} "
            f"{num(b['decode_tps_min'], 5)}  {b['parallel']:>5}")
    c = s["coverage"]
    w.append("")
    w.append("WHAT THIS LOG DOES NOT COVER is the point of these three lines:")
    w.append(f"  deepest prompt seen      {c['deepest_prompt']:,} tokens")
    w.append(f"  turns over 100k prompt   {c['over_100k']} of {s['n']}")
    w.append(f"  turns beside another     {c['parallel_total']} of {s['n']}")
    if c["pool_pct_max"] is not None:
        w.append(f"  highest pool occupancy   {c['pool_pct_max']}%")
    else:
        w.append("  highest pool occupancy   not reported by this release "
                 "(0.11.5 added it)")
    w.append("  A band of a dozen turns states a shape, not a rate. Deep and "
             "concurrent work has to be RUN to be measured;")
    w.append("  its absence from a log is not evidence that it is fast.")
    return "\n".join(w)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--log", required=True,
                    help="container log file, or - for stdin")
    ap.add_argument("--json", help="also write the summary as JSON here")
    ap.add_argument("--label", default="",
                    help="what this log is (e.g. the image tag)")
    a = ap.parse_args()

    text = sys.stdin.read() if a.log == "-" else open(
        a.log, encoding="utf-8", errors="replace").read()
    rows, unparsed = parse(text)
    if not rows:
        print("no served turns found in this log", file=sys.stderr)
        return 1
    s = summarise(rows)
    print(render(s, a.label, unparsed))
    if a.json:
        s["label"] = a.label
        s["unparsed"] = len(unparsed)
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2)
        print(f"\nsummary written to {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
