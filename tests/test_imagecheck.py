"""bench/imagecheck.py — the corruption probe for image workloads.

gfx1151's defect theme is SILENT degradation: exit 0, output garbage. The
checker is the machine judge every bench verdict rests on, so the gate holds
it to its own rule: both broken shapes must go red, the sound one green —
and the selftest that proves it must actually fail when a judgement flips.
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


@unittest.skipUnless(HAVE_DEPS, "numpy/PIL not installed — imagecheck "
                                "cannot run here at all, and says so itself")
class TestTheCheckerCanGoRed(unittest.TestCase):
    def setUp(self):
        import imagecheck
        self.ic = imagecheck

    def test_a_solid_color_is_broken(self):
        """The NaN/black failure shape — the classic fp16 VAE fault."""
        ok, s, reasons = self.ic.assess(self.ic._solid())
        self.assertFalse(ok)
        self.assertTrue(any("solid" in r for r in reasons), reasons)

    def test_pure_noise_is_broken(self):
        ok, s, reasons = self.ic.assess(self.ic._noise())
        self.assertFalse(ok)
        self.assertTrue(any("noise" in r for r in reasons), reasons)

    def test_a_structured_image_passes(self):
        ok, s, reasons = self.ic.assess(self.ic._natural())
        self.assertTrue(ok, reasons)

    def test_the_selftest_itself_reports_red_and_green(self):
        """selftest() is what a human runs before trusting a verdict; it has
        to return 0 on the shipped thresholds and 1 if a judgement flips —
        checked by flipping one deliberately. Its prints are swallowed here:
        a gate that narrates a deliberate failure reads like a failing gate."""
        import contextlib
        import io
        old = self.ic.SPREAD_MIN
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                green = self.ic.selftest()
                # With an absurd threshold the sound image is judged broken,
                # and the selftest must SAY so rather than stay green.
                self.ic.SPREAD_MIN = 1e9
                flipped = self.ic.selftest()
        finally:
            self.ic.SPREAD_MIN = old
        self.assertEqual(green, 0)
        self.assertEqual(flipped, 1)

    def test_thresholds_are_tagged_as_heuristic(self):
        """The numbers carry a judgement nobody measured against a corpus;
        the module must say so where they are defined."""
        src = (common.REPO / "bench" / "imagecheck.py").read_text(
            encoding="utf-8")
        self.assertIn("HEURISTIC", src.upper())


if __name__ == "__main__":
    unittest.main()
