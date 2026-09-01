#!/usr/bin/env python3
"""imagebench — seconds per image, N reps, judged output. For ONE workload.

    python3 bench/sideserver.py --workload setup/workloads/sdxl.env \\
        --stop llama-user@qwen38 -- \\
        python3 bench/imagebench.py --workload setup/workloads/sdxl.env --reps 3

Runs INSIDE the sideserver fence — one production stop pays for all reps —
and REFUSES to start while a llama-server is serving (worklib.fence_refusal;
the 26.08. lesson is that a guard which can be walked around will be).

What it measures: wall seconds per image at the profile's own settings
(resolution, steps, sampler, seed — one profile, one experiment). What it
judges: every produced image through bench/imagecheck.py — a fast wrong
image is worth nothing. What it does NOT judge: prompt fidelity or
aesthetics; see imagecheck's docstring for why that line is drawn.

The shared mechanics — fence, profile, rep loop with timing and incremental
result.json, hashing, metadata, summary — live in bench/worklib.py since
01.09.2026 (rule of three, paid). What stays here is the payload: the
image judge and the dedupe of byte-identical outputs.

Report: bench/reports/<stamp>_image_<workload>/ with result.json, the
sd-cli log per rep, and the images — deduplicated: fixed seed makes this
pipeline DETERMINISTIC (measured 01.09.2026: four SDXL runs across two
process lifetimes, byte-identical), so a repeat image is recorded by its
hash and not stored twice. The hash is the sharper evidence anyway: an A/B
across builds can diff it, the strictest corruption check this repo has.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import imagecheck                                             # noqa: E402
import worklib                                                # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workload", required=True,
                    help="a setup/workloads/*.env profile")
    ap.add_argument("--reps", type=int, default=3,
                    help="3 by default, and that is the bench floor here: "
                         "single runs decided a model choice once and the "
                         "margin did not survive (bench/README.md)")
    ap.add_argument("--note", default="")
    ap.add_argument("--dest", default=None,
                    help="report directory override — for verification runs\n"
                         "(tests/live_media.sh) whose output is not evidence")
    a = ap.parse_args(argv)

    if worklib.fence_refusal(a.workload):
        return 1
    report = worklib.BenchReport("image", a.workload, a.reps, a.note, dest=a.dest)

    def do_rep(r):
        out = os.path.join(report.dest, "rep%d.png" % r)
        log = os.path.join(report.dest, "rep%d.log" % r)
        job = report.argv + ["-p", report.prompt, "-o", out]
        rc, seconds = worklib.timed_run(job, log)
        rep = {"exit": rc, "seconds": seconds}
        if rc == 0 and os.path.exists(out):
            digest = worklib.sha256_of(out)
            rep["sha256"] = digest
            try:
                ok, s, reasons = imagecheck.assess(out)
            except (ValueError, OSError) as e:
                # A file the checker cannot read is a BROKEN rep, not a
                # dead bench (bench/README.md). OSError is what PIL
                # actually raises for a truncated/unidentifiable file
                # (UnidentifiedImageError subclasses it) — this seam was
                # the one of the three benches with NO guard at all
                # (review, 01.09.2026).
                rep.update({"ok": False, "reasons": [str(e)]})
                return rep
            rep.update({"ok": ok, "reasons": reasons,
                        "spread": round(s["spread"], 2),
                        "neighbour_r": round(s["neighbour_r"], 3)})
            if any(p.get("sha256") == digest for p in report.reps):
                rep["image"] = ("identical to an earlier rep — not stored "
                                "twice")
                os.unlink(out)
        else:
            rep.update({"ok": False,
                        "reasons": ["sd-cli exit %d or no output — see %s"
                                    % (rc, os.path.basename(log))]})
        return rep

    def describe(rep):
        return "%6.1f s  %s" % (rep["seconds"],
                                "ok" if rep["ok"] else
                                "BROKEN/FAILED %s" % rep["reasons"])

    report.run_reps(do_rep, describe)
    return report.finalize(extra={
        "checker": {"spread_min": imagecheck.SPREAD_MIN,
                    "neighbour_r_min": imagecheck.NEIGHBOUR_R_MIN,
                    "thresholds": "heuristic, not derived — see "
                                  "bench/imagecheck.py"}})


if __name__ == "__main__":
    sys.exit(main())
