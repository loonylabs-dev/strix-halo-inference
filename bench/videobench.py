#!/usr/bin/env python3
"""videobench — seconds per clip, N reps, judged frames. One workload.

    python3 bench/sideserver.py --workload setup/workloads/wan21-t2v.env \\
        --stop llama-user@qwen38 --deadline 75 --job-timeout 3600 -- \\
        python3 bench/videobench.py --workload setup/workloads/wan21-t2v.env --reps 3

Third sibling, sharing its mechanics through bench/worklib.py (rule of
three, paid 01.09.2026). What stays here is the payload:

The EVIDENCE is the frame sequence: sd-cli writes printf-style PNG frames
natively, bench/videocheck.py judges them spatially (imagecheck per frame)
and temporally (frozen clip, temporal noise), and the per-rep sequence
hash makes determinism an observation (distinct_outputs). Frames live in
frames-repN/ — GITIGNORED, ~25 MB per rep does not belong in a public
repo — and what the repo keeps is: result.json with the hashes, first/
middle/last frame of rep 1, and a .webm assembled by ffmpeg for humans
(a derived viewing artifact, and named as one).
"""
import argparse
import glob as globmod
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import videocheck                                             # noqa: E402
import worklib                                                # noqa: E402


def assemble_webm(paths, out, fps, ffmpeg_bin="ffmpeg"):
    """Failure is reported, not fatal — the webm is for humans, the frames
    are the evidence. That includes a MISSING ffmpeg: subprocess raises
    OSError for an absent binary, and the first version let that traceback
    kill the bench after all reps had run, leaving result.json 'partial'
    forever (review, 01.09.2026 — 'ffmpeg is present on this machine' was
    a one-machine assumption living in shared code)."""
    listfile = out + ".frames.txt"
    with open(listfile, "w") as fh:
        for p in paths:
            fh.write("file '%s'\nduration %.6f\n" % (os.path.abspath(p),
                                                     1.0 / fps))
    try:
        r = subprocess.run(
            [ffmpeg_bin, "-y", "-loglevel", "error", "-f", "concat",
             "-safe", "0", "-i", listfile, "-c:v", "libvpx-vp9",
             "-pix_fmt", "yuv420p", "-r", str(fps), out],
            check=False, capture_output=True, text=True)
    except OSError as e:
        print("  (webm assembly skipped: %s — the frames and their hashes "
              "are the evidence)" % e)
        return False
    finally:
        os.unlink(listfile)
    if r.returncode != 0:
        print("  (webm assembly failed: %s)" % (r.stderr or "")[:160])
        return False
    return True


def sequence_hash(paths):
    import hashlib
    h = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as fh:
            h.update(hashlib.sha256(fh.read()).digest())
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workload", required=True,
                    help="a setup/workloads/*.env profile (KIND=video)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--fps", type=float, default=16.0,
                    help="playback rate for the derived webm and the clip-"
                         "seconds arithmetic; Wan's native rate is 16")
    ap.add_argument("--note", default="")
    ap.add_argument("--dest", default=None,
                    help="report directory override — for verification runs\n"
                         "(tests/live_media.sh) whose output is not evidence")
    a = ap.parse_args(argv)

    if worklib.fence_refusal(a.workload):
        return 1
    report = worklib.BenchReport("video", a.workload, a.reps, a.note,
                                 hash_key="sequence_sha256",
                                 dest=a.dest)

    def do_rep(r):
        framedir = os.path.join(report.dest, "frames-rep%d" % r)
        os.makedirs(framedir, exist_ok=True)
        # Cleared BEFORE the run: a re-run into an existing dest globbed
        # the previous run's leftovers into sequence_sha256, frame count
        # and clip_seconds — evidence describing a sequence the tool
        # never produced (review, 01.09.2026).
        for stale in globmod.glob(os.path.join(framedir, "frame_*.png")):
            os.unlink(stale)
        pattern = os.path.join(framedir, "frame_%03d.png")
        log = os.path.join(report.dest, "rep%d.log" % r)
        job = report.argv + ["-p", report.prompt, "-o", pattern]
        rc, seconds = worklib.timed_run(job, log)
        frames = sorted(globmod.glob(os.path.join(framedir, "frame_*.png")))
        rep = {"exit": rc, "seconds": seconds, "frames": len(frames)}
        if rc == 0 and frames:
            rep["sequence_sha256"] = sequence_hash(frames)
            try:
                ok, s, reasons = videocheck.assess(frames)
            except (ValueError, OSError) as e:
                # OSError is the one an UNREADABLE frame actually raises:
                # PIL's UnidentifiedImageError subclasses OSError, not
                # ValueError — the original guard only covered the numpy
                # shape mismatch and the corrupt-file case it was written
                # for sailed past it (review, 01.09.2026).
                ok, s, reasons = False, {}, [str(e)]
            rep.update({"ok": ok, "reasons": reasons,
                        "clip_seconds": round(len(frames) / a.fps, 2),
                        "mean_frame_diff":
                            round(s.get("mean_frame_diff", -1), 3),
                        "temporal_r": round(s.get("temporal_r", -1), 3)})
        else:
            rep.update({"ok": False,
                        "reasons": ["job exit %d or no frames — see %s"
                                    % (rc, os.path.basename(log))]})
        return rep

    def post(rep):
        if rep.get("frames"):
            rep["seconds_per_frame"] = round(rep["seconds"] / rep["frames"], 2)

    def describe(rep):
        if not rep.get("frames"):
            return "%7.1f s wall, FAILED %s" % (rep["seconds"], rep["reasons"])
        return ("%7.1f s wall, %d frames (%.1f s clip, %.1f s/frame), %s"
                % (rep["seconds"], rep["frames"], rep.get("clip_seconds", 0),
                   rep.get("seconds_per_frame", 0),
                   "ok" if rep["ok"] else "BROKEN %s" % rep["reasons"][:1]))

    report.run_reps(do_rep, describe, post=post)

    # What the repo keeps of rep 1: three sample frames and the webm.
    first_frames = sorted(globmod.glob(
        os.path.join(report.dest, "frames-rep1", "frame_*.png")))
    if first_frames:
        for label, idx in (("first", 0), ("mid", len(first_frames) // 2),
                           ("last", len(first_frames) - 1)):
            shutil.copy2(first_frames[idx],
                         os.path.join(report.dest, "rep1-%s.png" % label))
        assemble_webm(first_frames, os.path.join(report.dest, "rep1.webm"),
                      a.fps)

    return report.finalize(extra={
        "fps": a.fps,
        "frames_kept": "frames-rep*/ is gitignored; evidence = sequence "
                       "hashes above, rep1-{first,mid,last}.png and the "
                       "derived rep1.webm",
        "checker": {"frozen_diff_max": videocheck.FROZEN_DIFF_MAX,
                    "temporal_r_min": videocheck.TEMPORAL_R_MIN,
                    "thresholds": "heuristic, not derived — see "
                                  "bench/videocheck.py"}})


if __name__ == "__main__":
    sys.exit(main())
