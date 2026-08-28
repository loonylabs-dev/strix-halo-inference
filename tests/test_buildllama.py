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
        with open(SCRIPT, encoding="utf-8") as fh:
            src = fh.read()
        m = re.search(r'MAX_REPLAY="\$\{MAX_REPLAY:-(\d+)\}"', src)
        self.assertIsNotNone(m, "MAX_REPLAY default not found in the script")
        self.assertLessEqual(int(m.group(1)), 10, "the case this exists for "
                                                  "was 91 commits, but a "
                                                  "loose bound would have "
                                                  "waved through 26 as well")


@unittest.skipIf(shutil.which("cmake") is None,
                 "build-llama.sh refuses without cmake — skipped LOUDLY")
class TestBuildingWithoutThePatchOnPurpose(unittest.TestCase):
    """An unpatched build is a SUBJECT. It must never be a binary to serve.

    "Is the corruption still in upstream master, and does PR #27311 fix it"
    cannot be answered on a binary that already suppresses the symptom — which
    is exactly what 27.08. did, measuring the PR on top of its own competitor,
    because building without the patch was a case the script did not have.

    The thing that must never exist is an unpatched build that CLAIMS to be
    patched. On gfx1151 that is not an error, it is wrong answers once a second
    slot is used. So: its own directory family, its own stamp field, the marker
    proven ABSENT rather than assumed, and no path to the symlink production
    execs.

    The build itself is faked — a cmake on PATH that writes a llama-server
    reporting the commit it was built from. Everything else runs for real.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.src = os.path.join(self.tmp, "src")
        os.makedirs(os.path.join(self.src, os.path.dirname(CUDA)))
        git(self.src, "init", "-q", "-b", "master")
        self.commit("base", CUDA, "int x;\n")
        # llama.cpp ignores its build directories, and the preflight's
        # "working tree clean" check would otherwise see the build this script
        # just made and refuse the next one. Fixture fidelity, not decoration.
        self.commit("ignore builds", ".gitignore", "build*/\n")
        git(self.src, "checkout", "-q", "-b", "patch")
        self.commit("the patch", CUDA, "// %s\nint x;\n" % MARKER)
        git(self.src, "checkout", "-q", "master")
        # step 1 fetches from `origin` even when the ref is local. Point it at
        # the fixture itself: no network, and the step is exercised rather
        # than skipped, which is the difference between this and a dry run.
        git(self.src, "remote", "add", "origin", self.src)

        # A cmake that configures and "builds" without a compiler. The fake
        # binary reports the commit the checkout is on, so step 5's check that
        # the binary matches the source it claims is exercised for real.
        self.bin = os.path.join(self.tmp, "bin")
        os.makedirs(self.bin)
        fake = os.path.join(self.bin, "cmake")
        with open(fake, "w") as f:
            f.write(
                '#!/bin/sh\n'
                'B=""\n'
                'while [ $# -gt 0 ]; do\n'
                '  case "$1" in -B) B="$2"; shift 2 ;;\n'
                '    --build) B="$2"; shift 2 ;; *) shift ;; esac\n'
                'done\n'
                '[ -n "$B" ] || exit 1\n'
                'mkdir -p "$B/bin"\n'
                'printf \'#!/bin/sh\\necho "version: 0.0.0-dev (build 1, '
                'commit $(git -C %s rev-parse --short=9 HEAD))"\\n\' '
                '> "$B/bin/llama-server"\n'
                'chmod +x "$B/bin/llama-server"\n' % self.src)
        os.chmod(fake, 0o755)

    def commit(self, msg, path, body):
        full = os.path.join(self.src, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(body)
        git(self.src, "add", path)
        git(self.src, "commit", "-q", "-m", msg)

    def build(self, *args, **env):
        return subprocess.run(
            ["bash", SCRIPT, *args], capture_output=True, text=True,
            timeout=300,
            env=dict(os.environ, LLAMA_SRC=self.src,
                     PATH=self.bin + os.pathsep + os.environ["PATH"], **env))

    def stamp(self, d):
        out = {}
        with open(os.path.join(self.src, d, ".build-stamp")) as f:
            for line in f:
                k, sep, v = line.partition("=")
                if sep:
                    out[k.strip()] = v.strip()
        return out

    def test_it_builds_into_a_family_of_its_own(self):
        r = self.build("--ref", "master", "--no-patch")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        made = [d for d in os.listdir(self.src) if d.startswith("build-")]
        self.assertEqual(len(made), 1, made)
        self.assertTrue(made[0].startswith("build-rocm-unpatched-"), made[0])
        self.assertNotIn("patched-", made[0].replace("unpatched-", ""),
                         "an unpatched build must not sit in the patched "
                         "family, whose name is what the symlink resolves to")

    def test_the_stamp_says_it_is_not_patched(self):
        """A reader — and bench/suites/restore-safety.py, which records
        provenance from this file — must not have to infer it from a
        directory name that anybody can rename."""
        self.build("--ref", "master", "--no-patch")
        d = [x for x in os.listdir(self.src) if x.startswith("build-")][0]
        st = self.stamp(d)
        self.assertEqual(st["patched"], "no")
        self.assertEqual(st["family"], "rocm-unpatched")
        self.assertEqual(st["patch_commit"], "none")
        self.assertEqual(st["patch_branch"], "none")

    def test_a_patched_build_still_says_it_is_patched(self):
        """The positive control. If `patched` were hard-wired to `no` this
        whole class would pass and mean nothing."""
        r = self.build("--ref", "master", PATCH_BRANCH="patch")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        d = [x for x in os.listdir(self.src) if x.startswith("build-")][0]
        self.assertTrue(d.startswith("build-rocm-patched-"), d)
        st = self.stamp(d)
        self.assertEqual(st["patched"], "yes")
        self.assertEqual(st["patch_branch"], "patch")
        self.assertNotEqual(st["patch_commit"], "none")

    def test_the_marker_is_proven_absent_rather_than_assumed(self):
        """--no-patch is an intention. "The marker is not in the source I am
        about to compile" is a fact, and if upstream ever adopts the change
        itself this fires — which is good news, and still not an unpatched
        build."""
        r = self.build("--ref", "patch", "--no-patch")
        self.assertNotEqual(r.returncode, 0, r.stdout)
        self.assertIn("marker is PRESENT", r.stdout + r.stderr)

    def test_it_cannot_be_activated(self):
        r = self.build("--ref", "master", "--no-patch", "--activate")
        self.assertNotEqual(r.returncode, 0, r.stdout)
        self.assertIn("cannot be combined", r.stdout + r.stderr)
        self.assertFalse(
            [d for d in os.listdir(self.src) if d.startswith("build-")],
            "it must refuse in preflight, before anything is built")

    def test_use_refuses_a_directory_whose_stamp_says_unpatched(self):
        """Belt and braces beside the name: activate() only looks in the
        patched family, so an unpatched id cannot be found — but a directory
        can be renamed, and the stamp is what says what is in it."""
        d = os.path.join(self.src, "build-rocm-patched-renamed")
        os.makedirs(os.path.join(d, "bin"))
        b = os.path.join(d, "bin", "llama-server")
        open(b, "w").close()
        os.chmod(b, 0o755)
        with open(os.path.join(d, ".build-stamp"), "w") as f:
            f.write("build_id=renamed\npatched=no\n")
        r = self.build("--use", "renamed")
        self.assertNotEqual(r.returncode, 0, r.stdout)
        self.assertIn("patched=no", r.stdout + r.stderr)

    def test_it_puts_the_source_tree_back(self):
        """An unpatched checkout LEFT BEHIND is a trap.

        setup/check.sh reads the SOURCE, not the running binary, so a tree
        parked on an unpatched commit makes it say "THE PATCH IS GONE" — true
        of the tree, false of the server. That happened on 28.08. after a
        night of --no-patch builds, and check.sh was right to shout.
        """
        before = git(self.src, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        self.assertEqual(before, "master")
        r = self.build("--ref", "master", "--no-patch")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        after = git(self.src, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        self.assertEqual(after, before,
                         "the source tree was left on the unpatched commit")
        self.assertIn("source tree back on", r.stdout)

    def test_a_patched_build_still_leaves_the_patch_branch_checked_out(self):
        """The positive control: putting the tree back must not become
        unconditional. After a PATCHED build the tree should carry the patch,
        which is what the next build rebases from."""
        r = self.build("--ref", "master", PATCH_BRANCH="patch")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        after = git(self.src, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        self.assertEqual(after, "patch")

    def test_list_shows_both_families(self):
        """A build that is not listed is a build nobody knows they have —
        950 MB at a time, which is why --prune exists at all."""
        a = self.build("--ref", "master", "--no-patch")
        self.assertEqual(a.returncode, 0, a.stdout + a.stderr)
        b = self.build("--ref", "master", PATCH_BRANCH="patch")
        self.assertEqual(b.returncode, 0, b.stdout + b.stderr)
        r = self.build("--list")
        self.assertIn("[rocm-unpatched]", r.stdout)
        self.assertIn("[rocm-patched]", r.stdout)


if __name__ == "__main__":
    unittest.main()
