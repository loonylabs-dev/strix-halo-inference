"""fetch-model.sh — the one rule that a file of the right NAME is not evidence.

This script had no tests at all until 04.09.2026, and the thing that got it
some is a defect nobody had hit yet but which was one command away.

The model directory here is ONE FLAT NAMESPACE — every profile's `-m` points
into it and the files keep the names their publisher gave them. Quantisers
name a vision encoder after its PRECISION rather than after its model, so
`mmproj-F16.gguf` is Qwen3.6-35B-A3B's file (899,283,680 B) and it is equally
Qwen3.8-27B's (927,607,488 B), which qwen38 has been serving from since
17.08.2026. The old code compared the size on disk with the published size,
found them different, and ran `curl -C -` — appending the new model's bytes
to the old model's working file.

What made it worth a test rather than a one-line fix: the sha256 check that
would have caught it fires AFTER the download, i.e. after the other model's
weights are already gone, and nothing in the script could have told the two
files apart at that point. A guard that reports damage is not a guard.

The network is replaced by FETCH_MODEL_LISTING (a seam documented in the
script) and never reached: every case here is decided before the first byte
would be fetched.
"""
import hashlib, os, re, shutil, subprocess, tempfile, unittest

import common

REPO = common.REPO
FETCH = str(REPO / "setup" / "scripts" / "fetch-model.sh")

# The two real files this defect was found on. Kept as the actual byte counts
# rather than round numbers, because the point is that they DIFFER while the
# name does not.
QWEN36_MMPROJ = 899283680
QWEN38_MMPROJ = 927607488


def free_gib(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1073741824


class FetchCase(unittest.TestCase):
    """One temp directory per test, standing in for the model directory."""

    def setUp(self):
        self.dest = tempfile.mkdtemp(prefix="fetchmodel-")
        self.addCleanup(shutil.rmtree, self.dest, True)
        # The script refuses before it starts when the partition is too tight
        # (total x 1.05 + 20 GiB), which is right for a 100 GiB model and
        # would make this a test of the temp filesystem. Named, not silent.
        if free_gib(self.dest) < 21:
            self.skipTest("less than 21 GiB free under %s — fetch-model.sh "
                          "refuses before it reaches anything this tests"
                          % tempfile.gettempdir())

    def listing(self, rows):
        """rows: [(size, sha256, path)] -> a file the seam can read."""
        path = os.path.join(self.dest, "listing.tsv")
        with open(path, "w", encoding="utf-8") as fh:
            for size, sha, p in rows:
                fh.write("%d\t%s\t%s\n" % (size, sha, p))
        return path

    def run_fetch(self, listing, pattern="mmproj"):
        env = dict(os.environ)
        env["FETCH_MODEL_LISTING"] = listing
        return subprocess.run(
            ["bash", FETCH, "some/repo", pattern, "--dest", self.dest],
            capture_output=True, text=True, timeout=120, env=env)

    def sparse(self, name, size):
        path = os.path.join(self.dest, name)
        with open(path, "wb") as fh:
            fh.truncate(size)
        return path


class TestAForeignFileOfTheSameNameIsRefused(FetchCase):
    """The defect itself, in the shape it was found in."""

    def test_it_does_not_resume_into_a_file_of_a_different_size(self):
        victim = self.sparse("mmproj-F16.gguf", QWEN38_MMPROJ)
        r = self.run_fetch(self.listing(
            [(QWEN36_MMPROJ, "0" * 64, "mmproj-F16.gguf")]))

        self.assertNotEqual(r.returncode, 0,
                            "a refusal has to be a non-zero exit, or a caller "
                            "reads it as success:\n%s%s" % (r.stdout, r.stderr))
        self.assertIn("NOT resuming", r.stdout + r.stderr)
        self.assertEqual(os.path.getsize(victim), QWEN38_MMPROJ,
                         "the other model's file was touched")

    def test_the_refusal_says_how_to_adopt_a_genuine_partial(self):
        """The same situation has a second, innocent cause — a download
        interrupted before this rule existed. The message has to name the way
        out, or the rule turns a resumable 100 GiB fetch into a fresh one."""
        self.sparse("mmproj-F16.gguf", QWEN38_MMPROJ)
        r = self.run_fetch(self.listing(
            [(QWEN36_MMPROJ, "0" * 64, "mmproj-F16.gguf")]))
        self.assertIn(".part", r.stdout + r.stderr)
        self.assertIn("mv ", r.stdout + r.stderr)

    def test_a_smaller_stranger_is_refused_the_same_way(self):
        """Direction must not matter. A foreign file that happens to be
        SHORTER than the wanted one looks exactly like a partial download,
        and that resemblance is the whole trap."""
        victim = self.sparse("mmproj-F16.gguf", QWEN36_MMPROJ)
        r = self.run_fetch(self.listing(
            [(QWEN38_MMPROJ, "0" * 64, "mmproj-F16.gguf")]))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("NOT resuming", r.stdout + r.stderr)
        self.assertEqual(os.path.getsize(victim), QWEN36_MMPROJ)


class TestAFileThatIsAlreadyRightIsKept(FetchCase):
    """The other half: this must not have become a script that re-downloads
    everything. A complete file is verified, not fetched again."""

    def test_a_complete_and_correct_file_is_accepted_without_a_download(self):
        body = b"gguf pretend weights, exactly these bytes"
        path = os.path.join(self.dest, "model-UD-Q4_K_XL.gguf")
        with open(path, "wb") as fh:
            fh.write(body)
        sha = hashlib.sha256(body).hexdigest()

        r = self.run_fetch(self.listing(
            [(len(body), sha, "model-UD-Q4_K_XL.gguf")]), pattern="Q4_K_XL")

        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("complete and verified", r.stdout)
        self.assertNotIn("fetching", r.stdout,
                         "a file that is already right was downloaded again")

    def test_the_right_size_with_the_wrong_content_is_reported_not_resumed(self):
        body = b"gguf pretend weights, exactly these bytes"
        path = os.path.join(self.dest, "model-UD-Q4_K_XL.gguf")
        with open(path, "wb") as fh:
            fh.write(body)

        r = self.run_fetch(self.listing(
            [(len(body), "1" * 64, "model-UD-Q4_K_XL.gguf")]),
            pattern="Q4_K_XL")

        self.assertNotEqual(r.returncode, 0)
        self.assertIn("WRONG CONTENT", r.stdout + r.stderr)


class TestNothingLandsUnverified(FetchCase):
    """The invariant the .part file exists for: a name in the model directory
    means those bytes hashed to what the publisher said. Checked as a property
    of the SCRIPT's text, because the alternative is a test that downloads."""

    def test_the_download_target_is_never_the_final_path(self):
        with open(FETCH, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        targets = re.findall(r'-o\s+("[^"]*"|\S+)', "\n".join(lines))
        self.assertTrue(targets,
                        "no curl output target found — did the script move?")
        self.assertEqual(
            [t for t in targets if t != '"$part"'], [],
            "curl writes somewhere other than the .part file: %s" % targets)


if __name__ == "__main__":
    unittest.main()
