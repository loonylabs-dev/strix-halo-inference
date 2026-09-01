"""bench/audiocheck.py — the corruption probe for audio workloads.

imagecheck's sibling, same gate rule: a TTS job that exits 0 can still hand
over silence, codec noise or a clipped screech, so the checker itself must
be provably able to go red before any bench verdict resting on it counts.
"""
import sys
import unittest

import common

sys.path.insert(0, str(common.REPO / "bench"))

try:
    import numpy  # noqa: F401
    HAVE_DEPS = True
except ImportError:                                   # pragma: no cover
    HAVE_DEPS = False


@unittest.skipUnless(HAVE_DEPS, "numpy not installed — audiocheck cannot "
                                "run here at all, and says so itself")
class TestTheCheckerCanGoRed(unittest.TestCase):
    def setUp(self):
        import audiocheck
        self.ac = audiocheck

    def test_an_empty_or_truncated_file_is_a_valueerror_not_a_crash(self):
        """Review finding (01.09.2026): wave.open raises EOFError — not
        wave.Error — on an empty or header-truncated file (measured on
        3.14: 0, 3 and 6 bytes all do), and that escaped both load()'s
        guard and the benches' except ValueError: a 0-byte WAV from a
        job that exited 0 killed the bench mid-fence."""
        import os
        import tempfile
        for size in (0, 3, 6):
            with tempfile.NamedTemporaryFile(suffix=".wav",
                                             delete=False) as fh:
                fh.write(b"RIF"[:size])
                path = fh.name
            self.addCleanup(os.unlink, path)
            with self.assertRaises(ValueError, msg="%d bytes" % size):
                self.ac.load(path)

    def test_silence_is_broken(self):
        """The produced-no-codes failure shape."""
        ok, s, reasons = self.ac.assess(self.ac._silence())
        self.assertFalse(ok)
        self.assertTrue(any("silence" in r for r in reasons), reasons)

    def test_white_noise_is_broken(self):
        """The codec-garbage failure shape."""
        ok, s, reasons = self.ac.assess(self.ac._white_noise())
        self.assertFalse(ok)
        self.assertTrue(any("codec-garbage" in r for r in reasons), reasons)

    def test_speechlike_audio_passes(self):
        ok, s, reasons = self.ac.assess(self.ac._speechlike())
        self.assertTrue(ok, reasons)

    def test_the_selftest_itself_reports_red_and_green(self):
        """Return 0 on shipped thresholds, 1 when a judgement flips —
        proven by flipping one. Prints swallowed: a gate narrating a
        deliberate failure reads like a failing gate."""
        import contextlib
        import io
        old = self.ac.RMS_MIN
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                green = self.ac.selftest()
                # An absurd silence threshold judges the speech-like case
                # broken, and the selftest must say so.
                self.ac.RMS_MIN = 10.0
                flipped = self.ac.selftest()
        finally:
            self.ac.RMS_MIN = old
        self.assertEqual(green, 0)
        self.assertEqual(flipped, 1)

    def test_a_float_wav_is_refused_by_name(self):
        """The wrappers write 16-bit PCM on purpose; a float WAV means the
        writer changed, and the checker must refuse rather than misread.

        A REAL format-tag-3 file, not just an odd width: the first version
        of this test covered only the width branch, the promised float
        refusal lived only in the docstring, and the first chatterbox run
        found the gap with a traceback (01.09.2026)."""
        import io
        import struct
        import wave
        data = struct.pack("<4f", 0.1, -0.1, 0.2, -0.2)
        hdr = (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
               + b"fmt " + struct.pack("<IHHIIHH", 16, 3, 1, 24000,
                                       24000 * 4, 4, 32)
               + b"data" + struct.pack("<I", len(data)))
        with self.assertRaises(ValueError) as ctx:
            self.ac.load(io.BytesIO(hdr + data))
        self.assertIn("PCM", str(ctx.exception))

        buf = io.BytesIO()
        with wave.open(buf, "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(1)          # 8-bit: unsupported on purpose
            fh.setframerate(24000)
            fh.writeframes(b"\x80" * 24000)
        buf.seek(0)
        with self.assertRaises(ValueError):
            self.ac.load(buf)

    def test_thresholds_are_tagged_as_heuristic(self):
        src = (common.REPO / "bench" / "audiocheck.py").read_text(
            encoding="utf-8")
        self.assertIn("HEURISTIC", src.upper())

    def test_a_short_noise_blip_is_still_judged(self):
        """Review finding 01.09.2026, measured: 2000 samples of white noise
        passed as ok because no 2048-sample frame fit and flatness fell
        back to 0.0 — an under-85-ms codec-garbage blip sailed through.
        The judge must not be blind below its own frame size."""
        import numpy as np
        rng = np.random.default_rng(7)
        short = (rng.uniform(-0.5, 0.5, 2000), 24000)
        ok, s, reasons = self.ac.assess(short)
        self.assertFalse(ok, "a 2000-sample noise blip must not pass")

    def test_exactly_one_frame_of_noise_is_judged_too(self):
        """The re-review's off-by-one (measured): 2047 red, 2100 red,
        2048 GREEN — `len < frame` did not catch equality and the frame
        loop's exclusive bound produced zero frames. The boundary case IS
        the frame size."""
        import numpy as np
        rng = np.random.default_rng(7)
        ok, s, reasons = self.ac.assess(
            (rng.uniform(-0.5, 0.5, 2048), 24000))
        self.assertFalse(ok, "2048 samples of noise must not pass while "
                             "2047 and 2100 are refused")

    def test_24_bit_pcm_loads_correctly(self):
        """The 24-bit path was verified by hand in the review and had no
        gate coverage — a regression would have been invisible. Two known
        samples: +1/2 full scale and -1/2 full scale."""
        import io
        import struct
        import wave
        pos, neg = 1 << 22, -(1 << 22)          # +/- half of 2^23
        data = b"".join(struct.pack("<i", v)[:3] for v in (pos, neg))
        buf = io.BytesIO()
        with wave.open(buf, "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(3)
            fh.setframerate(24000)
            fh.writeframes(data)
        buf.seek(0)
        x, rate = self.ac.load(buf)
        self.assertEqual(rate, 24000)
        self.assertAlmostEqual(x[0], 0.5, places=6)
        self.assertAlmostEqual(x[1], -0.5, places=6)

    def test_stereo_folds_to_the_channel_mean(self):
        import io
        import struct
        import wave
        frames = struct.pack("<4h", 16384, -16384, 8192, 8192)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as fh:
            fh.setnchannels(2)
            fh.setsampwidth(2)
            fh.setframerate(24000)
            fh.writeframes(frames)
        buf.seek(0)
        x, _ = self.ac.load(buf)
        self.assertAlmostEqual(x[0], 0.0, places=4)
        self.assertAlmostEqual(x[1], 0.25, places=3)


if __name__ == "__main__":
    unittest.main()
