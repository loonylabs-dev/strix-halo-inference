#!/usr/bin/env python3
"""audiocheck — is this WAV broken, in the way a TTS pipeline breaks?

    python3 bench/audiocheck.py out1.wav out2.wav   OK/BROKEN per file, exit 1 on any BROKEN
    python3 bench/audiocheck.py --selftest          prove the checker can go red

imagecheck's sibling. A TTS job that exits 0 can still hand over silence
(a model that produced no codes), digital noise (a codec fed garbage), or a
clipped screech — all with a valid WAV header. This is the machine judge a
bench verdict rests on; it catches structural corruption, not bad prosody
or a wrong accent, for the same reason imagecheck does not judge aesthetics.

Reads 16/24/32-bit integer PCM via the stdlib wave module. Float WAVs are
refused by name — the wrappers here write PCM_S 16-bit on purpose, so a
float file means the writer changed and the checker must not silently
misread it.

Three statistics; thresholds HEURISTIC, not derived — set once against the
selftest's synthetic cases on 01.09.2026, never calibrated on a corpus:

  * rms: overall level. Below 0.003 full scale (~-50 dBFS) the file is
    (near-)silence — the produced-no-codes failure shape.
  * clipping: fraction of samples at >= 0.999 full scale. Above 1 % the
    output is saturated — real speech peaks touch full scale rarely.
  * spectral flatness: geometric over arithmetic mean of the power
    spectrum, frame-averaged. Hann-windowed uniform noise measures ~0.56
    here, the speech-like control 0.00; the threshold sits at 0.4 so both
    sides keep a margin — above it there is no spectral structure, the
    codec-garbage failure shape.
"""
import argparse
import json
import sys
import wave

import numpy as np

RMS_MIN = 0.003          # heuristic, see module docstring
CLIP_FRACTION_MAX = 0.01  # heuristic
FLATNESS_MAX = 0.4        # heuristic


def load(path):
    """(samples as float64 in [-1, 1], sample rate). Mono-folded."""
    try:
        fh_ctx = wave.open(path, "rb")
    except (wave.Error, EOFError) as e:
        # The stdlib rejects float/extensible WAVs before this module's own
        # width check can name them — measured 01.09.2026, when a wrapper
        # wrote format tag 3 and the promised by-name refusal turned out to
        # live only in the docstring. One error type for the caller.
        # EOFError since the review the same day: an EMPTY or header-
        # truncated file raises that, not wave.Error (measured on 3.14:
        # 0, 3 and 6 bytes all do), and it escaped both this guard and
        # the bench's ValueError handler — a 0-byte WAV from a job that
        # exited 0 killed the bench mid-fence.
        raise ValueError("not an integer-PCM WAV (%s) — the wrappers here "
                         "write PCM_S 16-bit; a float or truncated file "
                         "means the writer changed, not that this should "
                         "guess" % (e or "empty/truncated"))
    with fh_ctx as fh:
        width = fh.getsampwidth()
        rate = fh.getframerate()
        channels = fh.getnchannels()
        raw = fh.readframes(fh.getnframes())
    if width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        x = ((b[:, 0].astype(np.int32)) | (b[:, 1].astype(np.int32) << 8)
             | (b[:, 2].astype(np.int32) << 16))
        x = np.where(x >= 1 << 23, x - (1 << 24), x).astype(np.float64) / (1 << 23)
    else:
        raise ValueError("unsupported sample width %d — the wrappers write "
                         "16-bit PCM; a different width means the writer "
                         "changed, not that this should guess" % width)
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    return x, rate


def spectral_flatness(x, rate, frame=2048):
    """Frame-averaged flatness of the power spectrum, 0 (structured) .. 1
    (white). Frames quieter than -60 dBFS are skipped — the flatness of
    near-silence is numerical noise, and silence has its own check.

    A signal SHORTER than one frame gets a single whole-signal FFT instead
    of a free pass: 2000 samples of white noise measured ok before the
    review caught it (01.09.2026) — no frame fit, the loop body never ran,
    and 0.0 read as 'perfectly structured'. Below 64 samples there is no
    spectrum worth judging and the judge says structured-unknown (0.0);
    the rms check still owns near-empty files.
    """
    if len(x) < frame:
        if len(x) < 64 or np.sqrt((x ** 2).mean()) < 1e-3:
            return 0.0
        p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2 + 1e-12
        return float(np.exp(np.log(p).mean()) / p.mean())
    vals = []
    # `+ 1`: at exactly len(x) == frame the exclusive bound produced ZERO
    # frames and the whole-signal fallback above did not fire either —
    # 2047 and 2100 samples of noise were refused, 2048 passed (re-review,
    # measured, 01.09.2026). The boundary case IS the frame size.
    for start in range(0, len(x) - frame + 1, frame):
        seg = x[start:start + frame]
        if np.sqrt((seg ** 2).mean()) < 1e-3:
            continue
        p = np.abs(np.fft.rfft(seg * np.hanning(frame))) ** 2 + 1e-12
        vals.append(float(np.exp(np.log(p).mean()) / p.mean()))
    return float(np.mean(vals)) if vals else 0.0


def stats(path_or_pair):
    if isinstance(path_or_pair, tuple):
        x, rate = path_or_pair
    else:
        x, rate = load(path_or_pair)
    return {"seconds": len(x) / rate if rate else 0.0,
            "rms": float(np.sqrt((x ** 2).mean())) if len(x) else 0.0,
            "clip_fraction": float((np.abs(x) >= 0.999).mean()) if len(x) else 0.0,
            "flatness": spectral_flatness(x, rate)}


def assess(path_or_pair):
    """(ok, stats, reasons) — every reason carries its number."""
    s = stats(path_or_pair)
    reasons = []
    if s["rms"] < RMS_MIN:
        reasons.append("rms %.4f < %.3f — (near-)silence, the no-codes "
                       "failure shape" % (s["rms"], RMS_MIN))
    if s["clip_fraction"] > CLIP_FRACTION_MAX:
        reasons.append("clipping on %.1f %% of samples (> %.0f %%) — "
                       "saturated output"
                       % (s["clip_fraction"] * 100, CLIP_FRACTION_MAX * 100))
    if s["flatness"] > FLATNESS_MAX:
        reasons.append("spectral flatness %.2f > %.1f — no spectral "
                       "structure, the codec-garbage failure shape"
                       % (s["flatness"], FLATNESS_MAX))
    return (not reasons), s, reasons


# --- selftest ---------------------------------------------------------------

def _silence(rate=24000, seconds=2.0):
    return np.zeros(int(rate * seconds)), rate


def _white_noise(rate=24000, seconds=2.0, seed=7):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.5, 0.5, int(rate * seconds)), rate


def _speechlike(rate=24000, seconds=2.0):
    """Harmonics under a syllable-rate envelope with pauses — sits on the
    PASS side of all three thresholds by a wide margin, like speech does."""
    t = np.arange(int(rate * seconds)) / rate
    voiced = sum(np.sin(2 * np.pi * f * t) / (i + 1)
                 for i, f in enumerate((140, 280, 420, 560, 700)))
    envelope = np.clip(np.sin(2 * np.pi * 3.0 * t), 0, None)
    x = 0.4 * voiced / np.abs(voiced).max() * envelope
    return x, rate


def selftest():
    """Red on both broken shapes, green on the sound one — run once before
    the first real verdict counts, cheap enough for the gate forever."""
    cases = (("silence", _silence(), False),
             ("white noise", _white_noise(), False),
             ("speech-like", _speechlike(), True))
    failures = []
    for name, pair, want_ok in cases:
        ok, s, reasons = assess(pair)
        verdict = "ok" if ok else "BROKEN (%s)" % "; ".join(reasons)
        print("  %-12s rms %.4f  clip %.3f  flat %.2f  -> %s"
              % (name, s["rms"], s["clip_fraction"], s["flatness"], verdict))
        if ok != want_ok:
            failures.append(name)
    if failures:
        print("SELFTEST FAILED: %s judged wrongly" % ", ".join(failures))
        return 1
    print("selftest green: both broken shapes go red, the sound one passes")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="*", help="WAV files to judge")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.files:
        print("nothing to check — pass WAV files or --selftest",
              file=sys.stderr)
        return 2
    any_broken = False
    out = []
    for path in a.files:
        ok, s, reasons = assess(path)
        any_broken |= not ok
        out.append({"file": path, "ok": ok, "reasons": reasons, **s})
        if not a.json:
            print("%-40s %s  (%.1f s, rms %.4f, clip %.3f, flat %.2f)%s"
                  % (path, "ok" if ok else "BROKEN", s["seconds"], s["rms"],
                     s["clip_fraction"], s["flatness"],
                     "  " + "; ".join(reasons) if reasons else ""))
    if a.json:
        print(json.dumps(out, indent=2))
    return 1 if any_broken else 0


if __name__ == "__main__":
    sys.exit(main())
