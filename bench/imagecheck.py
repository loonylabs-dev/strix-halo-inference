#!/usr/bin/env python3
"""imagecheck — is this image broken, in the way gfx1151 breaks things?

    python3 bench/imagecheck.py out1.png out2.png     OK/BROKEN per file, exit 1 on any BROKEN
    python3 bench/imagecheck.py --selftest            prove the checker can go red

Why this exists: the defect register's whole theme is that gfx1151 degrades
SILENTLY — llama.cpp emitted '////' with no error anywhere, and a diffusion
backend fault presents the same way: the job exits 0 and the PNG is a solid
color, pure noise, or NaN-black. A bench that only reads exit codes would
call that a pass. This is the machine checker every task here is required to
carry ("no human reads an answer to score it").

WHAT IT CANNOT SEE. This is a corruption probe, not a quality judge: it
catches an image with no structure, not a bad rendering of the prompt —
deliberately, for the same reason the coding battery was removed (other
people benchmark model quality; nobody else checks whether THIS build on
THIS gpu corrupts output).

The two statistics, and the thresholds are HEURISTIC, not derived — chosen
once against the selftest's synthetic cases and a handful of real SDXL
outputs on 01.09.2026, never calibrated against a corpus:

  * spread: max per-channel stddev. A solid color (NaN-black, the classic
    fp16-VAE failure) sits near 0; real photos and paintings sit far above.
    Threshold 4/255 — a dark but real image keeps structure above it.
  * neighbour correlation: Pearson r between horizontally adjacent pixels.
    Natural images are locally smooth (r typically > 0.8); uniform noise
    sits near 0. Threshold 0.35, failing LOW only — high-frequency art is
    still far more correlated than backend garbage.

Both must pass. Each verdict prints its numbers, so a future threshold
argument is an edit with evidence rather than a guess.
"""
import argparse
import json
import sys

import numpy as np
from PIL import Image

SPREAD_MIN = 4.0        # of 255 — heuristic, see module docstring
NEIGHBOUR_R_MIN = 0.35  # heuristic, see module docstring


def stats(path_or_array):
    """The two statistics for one image (RGB, any size)."""
    if isinstance(path_or_array, np.ndarray):
        arr = path_or_array
    else:
        arr = np.asarray(Image.open(path_or_array).convert("RGB"))
    a = arr.astype(np.float64)
    spread = float(a.std(axis=(0, 1)).max())
    grey = a.mean(axis=2)
    left, right = grey[:, :-1].ravel(), grey[:, 1:].ravel()
    if left.std() < 1e-9 or right.std() < 1e-9:
        # A constant image has no defined correlation; report 0.0 — the
        # spread check is the one that catches it, and it will.
        r = 0.0
    else:
        r = float(np.corrcoef(left, right)[0, 1])
    return {"spread": spread, "neighbour_r": r}


def assess(path_or_array):
    """(ok, stats, reasons). Both checks must pass; reasons name the failed
    one WITH its number, so a red verdict is an argument and not a mood."""
    s = stats(path_or_array)
    reasons = []
    if s["spread"] < SPREAD_MIN:
        reasons.append("spread %.2f < %.1f — (near-)solid color, the "
                       "NaN/black failure shape" % (s["spread"], SPREAD_MIN))
    if s["neighbour_r"] < NEIGHBOUR_R_MIN:
        reasons.append("neighbour correlation %.2f < %.2f — no local "
                       "structure, the pure-noise failure shape"
                       % (s["neighbour_r"], NEIGHBOUR_R_MIN))
    return (not reasons), s, reasons


# --- selftest ---------------------------------------------------------------

def _solid(value=13):
    return np.full((256, 256, 3), value, dtype=np.uint8)


def _noise(seed=7):
    return np.random.default_rng(seed).integers(0, 256, (256, 256, 3),
                                                dtype=np.uint8)


def _natural():
    """A synthetic stand-in for a real image: smooth gradients plus shapes —
    high neighbour correlation, real spread. Not a photo, and does not need
    to be: it sits on the PASS side of both thresholds by a wide margin."""
    y, x = np.mgrid[0:256, 0:256].astype(np.float64)
    img = np.stack([x, y, (x + y) / 2], axis=2)
    img[64:192, 64:192] += 40.0
    img = (img / img.max() * 255).astype(np.uint8)
    return img


def selftest():
    """The checker goes RED on two broken shapes and GREEN on a sound one —
    run before its first real verdict counts ('a check that cannot fail is
    not a check'), and cheap enough to run in the gate forever."""
    cases = (("solid color", _solid(), False),
             ("uniform noise", _noise(), True is False),  # expected broken
             ("structured image", _natural(), True))
    failures = []
    for name, arr, want_ok in cases:
        ok, s, reasons = assess(arr)
        verdict = "ok" if ok else "BROKEN (%s)" % "; ".join(reasons)
        print("  %-16s spread %6.2f  r %5.2f  -> %s"
              % (name, s["spread"], s["neighbour_r"], verdict))
        if ok != want_ok:
            failures.append(name)
    if failures:
        print("SELFTEST FAILED: %s judged wrongly" % ", ".join(failures))
        return 1
    print("selftest green: both broken shapes go red, the sound one passes")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("images", nargs="*", help="PNG/JPG files to judge")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.images:
        print("nothing to check — pass image files or --selftest",
              file=sys.stderr)
        return 2
    any_broken = False
    out = []
    for path in a.images:
        ok, s, reasons = assess(path)
        any_broken |= not ok
        out.append({"image": path, "ok": ok, "reasons": reasons, **s})
        if not a.json:
            print("%-40s %s  (spread %.2f, r %.2f)%s"
                  % (path, "ok" if ok else "BROKEN", s["spread"],
                     s["neighbour_r"],
                     "  " + "; ".join(reasons) if reasons else ""))
    if a.json:
        print(json.dumps(out, indent=2))
    return 1 if any_broken else 0


if __name__ == "__main__":
    sys.exit(main())
