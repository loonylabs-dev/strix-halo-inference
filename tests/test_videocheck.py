"""bench/videocheck.py — the corruption probe for video workloads.

Third sibling. A video job that exits 0 can hand over a broken frame, a
frozen clip, or temporal noise — the gate proves all three go red and the
moving control stays green, same contract as image and audio.
"""
import sys
import unittest

import common

sys.path.insert(0, str(common.REPO / "bench"))

try:
    import numpy  # noqa: F401
    from PIL import Image  # noqa: F401
    HAVE_DEPS = True
except ImportError:                                   # pragma: no cover
    HAVE_DEPS = False


@unittest.skipUnless(HAVE_DEPS, "numpy/PIL not installed — videocheck "
                                "cannot run here at all, and says so itself")
class TestTheCheckerCanGoRed(unittest.TestCase):
    def setUp(self):
        import videocheck
        self.vc = videocheck

    def test_a_frozen_clip_is_broken(self):
        """One image repeated is not a video — the sampler-produced-one-
        frame failure shape."""
        ok, s, reasons = self.vc.assess(self.vc._frozen())
        self.assertFalse(ok)
        self.assertTrue(any("FROZEN" in r for r in reasons), reasons)

    def test_temporal_noise_is_broken(self):
        ok, s, reasons = self.vc.assess(self.vc._temporal_noise())
        self.assertFalse(ok)

    def test_one_broken_frame_breaks_the_sequence(self):
        """Spatial judgement is per frame and names the frame — a single
        NaN-black frame in an otherwise sound clip must not pass."""
        ok, s, reasons = self.vc.assess(self.vc._one_bad_frame())
        self.assertFalse(ok)
        self.assertTrue(any(r.startswith("frame 4:") for r in reasons),
                        reasons)

    def test_a_moving_sequence_passes(self):
        ok, s, reasons = self.vc.assess(self.vc._moving_gradient())
        self.assertTrue(ok, reasons)

    def test_a_single_frame_is_not_a_video(self):
        ok, s, reasons = self.vc.assess(self.vc._moving_gradient(n=1))
        self.assertFalse(ok)

    def test_the_selftest_itself_reports_red_and_green(self):
        import contextlib
        import io
        old = self.vc.FROZEN_DIFF_MAX
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                green = self.vc.selftest()
                # An absurd frozen threshold judges the moving control
                # broken, and the selftest must say so.
                self.vc.FROZEN_DIFF_MAX = 1e9
                flipped = self.vc.selftest()
        finally:
            self.vc.FROZEN_DIFF_MAX = old
        self.assertEqual(green, 0)
        self.assertEqual(flipped, 1)

    def test_thresholds_are_tagged_as_heuristic(self):
        src = (common.REPO / "bench" / "videocheck.py").read_text(
            encoding="utf-8")
        self.assertIn("HEURISTIC", src.upper())


if __name__ == "__main__":
    unittest.main()
