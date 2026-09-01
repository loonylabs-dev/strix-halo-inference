"""The bench payloads' failure paths — a broken rep is recorded, not fatal.

Review findings (01.09.2026): a produced-but-unreadable output killed the
bench instead of becoming a BROKEN rep. imagebench called assess with no
guard at all, and videobench's guard caught ValueError where PIL raises
UnidentifiedImageError/OSError (measured — UnidentifiedImageError
subclasses OSError, not ValueError). Both died inside the PAID fence
window, losing the remaining reps and leaving result.json 'partial'
forever — the exact failure audiobench's comment says cost one fenced
cycle. Plus: videobench folded STALE frames from a previous run in the
same --dest into the sequence hash — evidence describing a sequence the
tool never produced.

The jobs here are stub scripts writing deliberately corrupt output; the
fence is passed by patching budget.server_pid (production may well be
serving while the gate runs — the gate must not care).
"""
import json
import os
import stat
import sys
import tempfile
import unittest

import common

sys.path.insert(0, str(common.REPO / "bench"))
sys.path.insert(0, str(common.REPO / "setup" / "lib"))

try:
    import numpy  # noqa: F401
    import PIL    # noqa: F401
    HAVE_DEPS = True
except ImportError:                                   # pragma: no cover
    HAVE_DEPS = False


def write_script(tmp, name, body):
    path = os.path.join(tmp, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("#!/bin/sh\n" + body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    return path


def write_profile(tmp, cmd, kind):
    path = os.path.join(tmp, "wl.env")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("WORKLOAD_TITLE=a stub workload\n"
                 "WORKLOAD_KIND=%s\n"
                 "WORKLOAD_MODE=batch\n"
                 "WORKLOAD_CMD=%s\n"
                 "WORKLOAD_PROMPT=stub prompt\n" % (kind, cmd))
    return path


# Writes garbage to whatever -o names: exit 0, output exists, unreadable.
CORRUPT_OUT = """out=""
while [ $# -gt 0 ]; do
  if [ "$1" = "-o" ]; then out="$2"; fi
  shift
done
printf 'not an image at all' > "$out"
"""

# Writes two garbage frames into the -o pattern's directory.
TWO_FRAMES = """out=""
while [ $# -gt 0 ]; do
  if [ "$1" = "-o" ]; then out="$2"; fi
  shift
done
dir="$(dirname "$out")"
printf 'garbage frame' > "$dir/frame_000.png"
printf 'garbage frame' > "$dir/frame_001.png"
"""


@unittest.skipUnless(HAVE_DEPS, "numpy/PIL not installed — the checkers "
                                "these benches call cannot run here")
class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        import budget
        self._real_server_pid = budget.server_pid
        budget.server_pid = lambda: None
        self.addCleanup(setattr, budget, "server_pid",
                        self._real_server_pid)
        self.dest = os.path.join(self.tmp, "report")

    def result(self):
        with open(os.path.join(self.dest, "result.json")) as fh:
            return json.load(fh)


class TestAnUnreadableImageIsABrokenRep(Base):
    def test_imagebench_records_it_instead_of_dying(self):
        import imagebench
        job = write_script(self.tmp, "fakejob", CORRUPT_OUT)
        profile = write_profile(self.tmp, job, "image")
        rc = imagebench.main(["--workload", profile, "--reps", "1",
                              "--dest", self.dest])
        self.assertEqual(rc, 1, "a broken rep is a failed bench, not a "
                                "crashed one")
        reps = self.result()["reps"]
        self.assertEqual(len(reps), 1)
        self.assertFalse(reps[0]["ok"])


class TestAnUnreadableFrameIsABrokenRep(Base):
    def test_videobench_records_it_instead_of_dying(self):
        import videobench
        job = write_script(self.tmp, "fakejob", TWO_FRAMES)
        profile = write_profile(self.tmp, job, "video")
        rc = videobench.main(["--workload", profile, "--reps", "1",
                              "--dest", self.dest])
        self.assertEqual(rc, 1)
        reps = self.result()["reps"]
        self.assertEqual(len(reps), 1)
        self.assertFalse(reps[0]["ok"])


class TestStaleFramesAreNotEvidence(Base):
    def test_a_previous_runs_frames_are_cleared_first(self):
        """A re-run into an existing dest globbed the old run's leftovers
        into sequence_sha256, frame count and clip_seconds — a hash of a
        sequence the tool never produced."""
        import videobench
        framedir = os.path.join(self.dest, "frames-rep1")
        os.makedirs(framedir)
        with open(os.path.join(framedir, "frame_099.png"), "w") as fh:
            fh.write("stale frame from an earlier invocation")
        job = write_script(self.tmp, "fakejob", TWO_FRAMES)
        profile = write_profile(self.tmp, job, "video")
        videobench.main(["--workload", profile, "--reps", "1",
                         "--dest", self.dest])
        rep = self.result()["reps"][0]
        self.assertEqual(rep["frames"], 2,
                         "the stale frame_099.png must not be counted")
        self.assertFalse(os.path.exists(
            os.path.join(framedir, "frame_099.png")))


if __name__ == "__main__":
    unittest.main()
