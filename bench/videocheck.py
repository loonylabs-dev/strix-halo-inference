#!/usr/bin/env python3
"""videocheck — is this frame sequence broken, spatially or temporally?

    python3 bench/videocheck.py rep1/frame_*.png     one verdict for the sequence
    python3 bench/videocheck.py --selftest           prove the checker can go red

Third sibling of imagecheck and audiocheck. A video job that exits 0 can
fail three ways the single-frame probe cannot see:

  * a BROKEN FRAME — any frame solid or noise; judged per frame by
    imagecheck, the same two statistics, nothing reinvented;
  * a FROZEN clip — every frame identical: the sampler produced one image
    and the pipeline repeated it. Real Wan output of a static scene still
    breathes (grain, light), so the threshold sits near digital zero;
  * TEMPORAL NOISE — consecutive frames unrelated: spatially each frame
    can look structured while the sequence flickers garbage.

The temporal statistic is the mean absolute inter-frame difference (0-255
scale) and the correlation between consecutive frames' pixels. Thresholds
HEURISTIC, not derived — set once against the selftest's synthetic cases
on 01.09.2026:

  * frozen:  mean |diff| < 0.05 of 255 across ALL consecutive pairs
  * noise:   mean consecutive-frame correlation < 0.5 (real motion keeps
             most of the image; the moving-gradient control measures ~1.0,
             temporal noise ~0.0)

Judged on PNG FRAMES, not on a container: sd-cli writes printf-style frame
sequences natively, frames need no codec to read, and the viewing .webm a
bench assembles afterwards is a derived artifact, not the evidence.
"""
import argparse
import glob as globmod
import json
import sys

import numpy as np
from PIL import Image

import imagecheck

FROZEN_DIFF_MAX = 0.05    # of 255 — heuristic, see module docstring
TEMPORAL_R_MIN = 0.5      # heuristic


def _grey(arr):
    return arr.astype(np.float64).mean(axis=2)


def temporal_stats(frames):
    """(mean abs diff, mean consecutive correlation) over the sequence."""
    diffs, corrs = [], []
    for a, b in zip(frames, frames[1:]):
        ga, gb = _grey(a), _grey(b)
        diffs.append(float(np.abs(ga - gb).mean()))
        if ga.std() < 1e-9 or gb.std() < 1e-9:
            corrs.append(0.0)
        else:
            corrs.append(float(np.corrcoef(ga.ravel(), gb.ravel())[0, 1]))
    return float(np.mean(diffs)), float(np.mean(corrs))


def assess(frames_or_paths):
    """(ok, stats, reasons) for a whole sequence. Every reason carries its
    number and, for a broken frame, the frame index."""
    frames, reasons = [], []
    for i, f in enumerate(frames_or_paths):
        arr = f if isinstance(f, np.ndarray) else np.asarray(
            Image.open(f).convert("RGB"))
        frames.append(arr)
        f_ok, f_stats, f_reasons = imagecheck.assess(arr)
        if not f_ok:
            reasons.append("frame %d: %s" % (i, "; ".join(f_reasons)))
    if len(frames) < 2:
        reasons.append("only %d frame(s) — not a video" % len(frames))
        return False, {"frames": len(frames)}, reasons
    diff, corr = temporal_stats(frames)
    s = {"frames": len(frames), "mean_frame_diff": diff,
         "temporal_r": corr}
    if diff < FROZEN_DIFF_MAX:
        reasons.append("mean inter-frame diff %.3f < %.2f — a FROZEN clip, "
                       "one image repeated" % (diff, FROZEN_DIFF_MAX))
    if corr < TEMPORAL_R_MIN:
        reasons.append("consecutive-frame correlation %.2f < %.1f — "
                       "temporal noise, frames unrelated"
                       % (corr, TEMPORAL_R_MIN))
    return (not reasons), s, reasons


# --- selftest ---------------------------------------------------------------

def _moving_gradient(n=9):
    """The sound control: imagecheck's structured image, drifting a few
    pixels per frame — real motion's shape."""
    base = imagecheck._natural()
    return [np.roll(base, 3 * i, axis=1) for i in range(n)]


def _frozen(n=9):
    return [imagecheck._natural()] * n


def _temporal_noise(n=9):
    return [imagecheck._noise(seed=i) for i in range(n)]


def _one_bad_frame(n=9):
    frames = _moving_gradient(n)
    frames[n // 2] = imagecheck._solid()
    return frames


def selftest():
    """Three broken shapes red, the sound one green — run before the first
    real verdict counts, cheap enough for the gate forever."""
    cases = (("frozen clip", _frozen(), False),
             ("temporal noise", _temporal_noise(), False),
             ("one broken frame", _one_bad_frame(), False),
             ("moving gradient", _moving_gradient(), True))
    failures = []
    for name, frames, want_ok in cases:
        ok, s, reasons = assess(frames)
        verdict = "ok" if ok else "BROKEN (%s)" % "; ".join(reasons[:2])
        print("  %-16s frames %2d  diff %6.2f  r %5.2f  -> %s"
              % (name, s.get("frames", 0), s.get("mean_frame_diff", -1),
                 s.get("temporal_r", -1), verdict))
        if ok != want_ok:
            failures.append(name)
    if failures:
        print("SELFTEST FAILED: %s judged wrongly" % ", ".join(failures))
        return 1
    print("selftest green: all three broken shapes go red, the sound one "
          "passes")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("frames", nargs="*",
                    help="frame files in order (globs welcome)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    paths = sorted(p for pat in a.frames for p in globmod.glob(pat)) \
        if a.frames else []
    if not paths:
        print("nothing to check — pass frame files or --selftest",
              file=sys.stderr)
        return 2
    ok, s, reasons = assess(paths)
    if a.json:
        print(json.dumps({"ok": ok, "reasons": reasons, **s}, indent=2))
    else:
        print("%d frames: %s  (diff %.2f, r %.2f)%s"
              % (s.get("frames", 0), "ok" if ok else "BROKEN",
                 s.get("mean_frame_diff", -1), s.get("temporal_r", -1),
                 "  " + "; ".join(reasons) if reasons else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
