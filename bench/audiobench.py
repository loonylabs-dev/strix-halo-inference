#!/usr/bin/env python3
"""audiobench — seconds per synthesis, N reps, judged output. One workload.

    python3 bench/sideserver.py --workload setup/workloads/qwen3-tts.env \\
        --stop llama-user@qwen38 -- \\
        python3 bench/audiobench.py --workload setup/workloads/qwen3-tts.env --reps 3

imagebench's sibling for WORKLOAD_KIND=audio, sharing its mechanics through
bench/worklib.py (rule of three, paid 01.09.2026). What stays here is the
payload: the audio judge (bench/audiocheck.py) and the REALTIME FACTOR —
seconds of audio per second of wall clock, the number a TTS user feels.

Determinism is not assumed: hashes are recorded per rep and equality is
REPORTED as an observation (distinct_outputs). Diffusion earned its
hash-diff instrument by measuring byte-equality first; a sampling TTS has
to earn it the same way — and both TTS lanes did, on this stack, at their
pinned seeds (reports 2026-09-01_0527 and _0530).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audiocheck                                             # noqa: E402
import worklib                                                # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workload", required=True,
                    help="a setup/workloads/*.env profile (KIND=audio)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--note", default="")
    ap.add_argument("--dest", default=None,
                    help="report directory override — for verification runs\n"
                         "(tests/live_media.sh) whose output is not evidence")
    a = ap.parse_args(argv)

    if worklib.fence_refusal(a.workload):
        return 1
    report = worklib.BenchReport("audio", a.workload, a.reps, a.note, dest=a.dest)

    def do_rep(r):
        out = os.path.join(report.dest, "rep%d.wav" % r)
        log = os.path.join(report.dest, "rep%d.log" % r)
        job = report.argv + ["-p", report.prompt, "-o", out]
        rc, seconds = worklib.timed_run(job, log)
        rep = {"exit": rc, "seconds": seconds}
        if rc == 0 and os.path.exists(out):
            digest = worklib.sha256_of(out)
            rep["sha256"] = digest
            try:
                ok, s, reasons = audiocheck.assess(out)
            except ValueError as e:
                # A file the checker refuses to read is a BROKEN rep, not a
                # dead bench — a cell that fails is recorded rather than
                # fatal (bench/README.md; cost one fenced cycle 01.09.2026).
                rep.update({"ok": False, "reasons": [str(e)]})
                return rep
            rep.update({"ok": ok, "reasons": reasons,
                        "audio_seconds": round(s["seconds"], 2),
                        "rms": round(s["rms"], 4),
                        "clip_fraction": round(s["clip_fraction"], 4),
                        "flatness": round(s["flatness"], 3)})
            if any(p.get("sha256") == digest for p in report.reps):
                rep["file"] = ("identical to an earlier rep — not stored "
                               "twice")
                os.unlink(out)
        else:
            rep.update({"ok": False,
                        "reasons": ["job exit %d or no output — see %s"
                                    % (rc, os.path.basename(log))]})
        return rep

    def post(rep):
        # Derived AFTER the runner stamps the wall time — the realtime
        # factor is audio seconds per wall second.
        if rep.get("audio_seconds") and rep["seconds"]:
            rep["realtime_factor"] = round(rep["audio_seconds"]
                                           / rep["seconds"], 3)

    def describe(rep):
        if "audio_seconds" not in rep:
            return "%6.1f s wall, FAILED %s" % (rep["seconds"], rep["reasons"])
        return ("%6.1f s wall, %.1f s audio (rtf %.2f), %s"
                % (rep["seconds"], rep["audio_seconds"],
                   rep.get("realtime_factor") or -1,
                   "ok" if rep["ok"] else "BROKEN %s" % rep["reasons"]))

    def summary(result, ok_reps):
        rtfs = sorted(r["realtime_factor"] for r in ok_reps
                      if r.get("realtime_factor"))
        if rtfs:
            result["median_realtime_factor"] = rtfs[len(rtfs) // 2]
            print("  median realtime factor %.2f" % rtfs[len(rtfs) // 2])

    report.run_reps(do_rep, describe, post=post)
    return report.finalize(
        extra={"checker": {"rms_min": audiocheck.RMS_MIN,
                           "clip_fraction_max": audiocheck.CLIP_FRACTION_MAX,
                           "flatness_max": audiocheck.FLATNESS_MAX,
                           "thresholds": "heuristic, not derived — see "
                                         "bench/audiocheck.py"}},
        summary_hook=summary)


if __name__ == "__main__":
    sys.exit(main())
