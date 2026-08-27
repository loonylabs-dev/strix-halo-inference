"""build-llama.sh — what a rebase actually replays.

`git rebase <target>` replays every commit in `target..HEAD`. That is "the
patch" only while the patch branch is what its name says, and on 27.08.2026
it was not: `gfx1151-patched` had been rebased onto PR #27742 the day before
to build Flash-Next, so it carried 26 commits over master, and the base of
the PR the session was told to test sat 65 commits further back.

The documented command would therefore have replayed 91 commits. **It does
not fail.** Tried in a throwaway worktree before anything was built: the
rebase succeeds, exit 0, no conflict — and the build comes out stamped
`upstream_ref=pr-27311` while containing an entire unmerged 180B-model PR and
65 extra master commits. Every check downstream would have agreed with the
stamp, and the measurement would have been attributed to the wrong change.

That is the defect this repository keeps meeting, in the one script whose
whole job is to make a build trustworthy: it runs, it exits 0, and it does
something other than what it says. The count is cheap, so it is looked at
before the rebase rather than after the report.

No llama.cpp and no network: the fixture is a git repository of four empty
commits.
"""
import os, shutil, subprocess, tempfile, unittest

import common

REPO = common.REPO
SCRIPT = str(REPO / "setup" / "scripts" / "build-llama.sh")
MARKER = "gfx1151/ROCm: trusting prop.integrated"
CUDA = "ggml/src/ggml-cuda/ggml-cuda.cu"


def git(cwd, *args):
    env = dict(os.environ,
               GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.invalid")
    return subprocess.run(["git", "-C", cwd, *args], capture_output=True,
                          text=True, timeout=60, env=env, check=True)


@unittest.skipIf(shutil.which("cmake") is None,
                 "build-llama.sh refuses without cmake, so its preflight "
                 "never reaches the check under test — skipped LOUDLY rather "
                 "than passing on a machine that cannot run it")
class TestItLooksAtWhatWouldBeReplayed(unittest.TestCase):
    """Four commits, three branches, and the two answers that must differ."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        git(self.tmp, "init", "-q", "-b", "master")
        os.makedirs(os.path.join(self.tmp, os.path.dirname(CUDA)))
        self.commit("base", CUDA, "int x;\n")
        # the patch, as it really is: ONE commit carrying the marker
        git(self.tmp, "checkout", "-q", "-b", "patch")
        self.commit("the patch", CUDA, "// %s\nint x;\n" % MARKER)
        # a branch that has drifted, the way gfx1151-patched had
        git(self.tmp, "checkout", "-q", "-b", "fat")
        for i in range(5):
            self.commit("somebody else's work %d" % i, "other%d.txt" % i, "x")
        git(self.tmp, "checkout", "-q", "master")

    def commit(self, msg, path, body):
        full = os.path.join(self.tmp, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(body)
        git(self.tmp, "add", path)
        git(self.tmp, "commit", "-q", "-m", msg)

    def build(self, patch_branch, ref, **env):
        return subprocess.run(
            ["bash", SCRIPT, "--ref", ref, "--dry-run"],
            capture_output=True, text=True, timeout=180,
            env=dict(os.environ, LLAMA_SRC=self.tmp,
                     PATCH_BRANCH=patch_branch, **env))

    def test_a_branch_that_is_only_the_patch_goes_through(self):
        r = self.build("patch", "master")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("1 commit(s) to replay", r.stdout)

    def test_a_branch_carrying_somebody_elses_work_is_refused(self):
        """The whole point. Without this the build succeeds and lies."""
        r = self.build("fat", "master")
        self.assertNotEqual(r.returncode, 0, r.stdout)
        self.assertIn("carries 6 commits", r.stdout + r.stderr)

    def test_the_refusal_names_what_would_come_along(self):
        """A refusal that does not say WHAT gets overridden, because the
        reader has no way to judge it."""
        r = self.build("fat", "master")
        out = r.stdout + r.stderr
        self.assertIn("somebody else's work 4", out)
        self.assertIn("the patch", out)
        self.assertIn("MAX_REPLAY", out, "a guard with no way past it gets "
                                         "commented out instead of used")

    def test_it_can_be_overridden_deliberately(self):
        r = self.build("fat", "master", MAX_REPLAY="6")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("6 commit(s) to replay", r.stdout)

    def test_the_bound_is_not_so_loose_that_it_cannot_fire(self):
        """A guard whose default admits everything is not a guard. Pinned
        against the case that produced it: 91 commits."""
        import re
        src = open(SCRIPT, encoding="utf-8").read()
        m = re.search(r'MAX_REPLAY="\$\{MAX_REPLAY:-(\d+)\}"', src)
        self.assertIsNotNone(m, "MAX_REPLAY default not found in the script")
        self.assertLessEqual(int(m.group(1)), 10, "the case this exists for "
                                                  "was 91 commits, but a "
                                                  "loose bound would have "
                                                  "waved through 26 as well")


if __name__ == "__main__":
    unittest.main()
