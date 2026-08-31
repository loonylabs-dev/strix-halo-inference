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
import os, re, shutil, subprocess, tempfile, unittest

import common

REPO = common.REPO
SCRIPT = str(REPO / "setup" / "scripts" / "build-llama.sh")
MARKER = "gfx1151/ROCm: trusting prop.integrated"
CUDA = "ggml/src/ggml-cuda/ggml-cuda.cu"
# THE STACK CARRIES TWO PATCHES since 30.08.2026, and build-llama.sh checks a
# marker for each. The fixture has to carry both or the checks cannot pass —
# and a fixture that carries only one would let a second patch go missing
# unnoticed, which is the exact failure these markers exist to prevent.
MARKER2 = "accepted token(s) past EOG"
SERVER = "tools/server/server-context.cpp"


def git(cwd, *args):
    env = dict(os.environ,
               GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.invalid")
    return subprocess.run(["git", "-C", cwd, *args], capture_output=True,
                          text=True, timeout=60, env=env, check=True)


def write_fake_cmake(tmp, src):
    """A cmake that configures and "builds" without a compiler.

    The fake binary reports the commit the checkout is on, so step 5's check
    that the binary matches the source it claims is exercised for real.

    It honours `--target`, and that is not decoration: a build that asks for
    llama-bench and silently gets llama-server would let `--with-bench` pass
    while building nothing, which is the one thing its test has to catch.
    Returns the directory to put on PATH.
    """
    bindir = os.path.join(tmp, "bin")
    os.makedirs(bindir, exist_ok=True)
    fake = os.path.join(bindir, "cmake")
    with open(fake, "w") as f:
        f.write(
            '#!/bin/sh\n'
            'B=""; T=""\n'
            'while [ $# -gt 0 ]; do\n'
            '  case "$1" in -B) B="$2"; shift 2 ;;\n'
            '    --build) B="$2"; shift 2 ;;\n'
            '    --target) T="$2"; shift 2 ;; *) shift ;; esac\n'
            'done\n'
            '[ -n "$B" ] || exit 1\n'
            'mkdir -p "$B/bin"\n'
            '[ -n "$T" ] || exit 0\n'
            'printf \'#!/bin/sh\\necho "version: 0.0.0-dev (build 1, '
            'commit $(git -C %s rev-parse --short=9 HEAD))"\\n\' '
            '> "$B/bin/$T"\n'
            'chmod +x "$B/bin/$T"\n' % src)
    os.chmod(fake, 0o755)
    return bindir


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
        self.commit("base", CUDA, "int x;\n", SERVER, "int y;\n")
        # the patches, as they really are: ONE commit carrying both markers
        git(self.tmp, "checkout", "-q", "-b", "patch")
        self.commit("the patch", CUDA, "// %s\nint x;\n" % MARKER,
                    SERVER, "// %s\nint y;\n" % MARKER2)
        # a branch that has drifted, the way gfx1151-patched had
        git(self.tmp, "checkout", "-q", "-b", "fat")
        for i in range(5):
            self.commit("somebody else's work %d" % i, "other%d.txt" % i, "x")
        git(self.tmp, "checkout", "-q", "master")

    def commit(self, msg, *pairs):
        """(path, body) pairs, all in ONE commit — the patch really is one."""
        for path, body in zip(pairs[::2], pairs[1::2]):
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


class _FakeBuildFixture(unittest.TestCase):
    """A source tree, a patch branch, and a cmake that builds nothing.

    Shared by every class below that runs build-llama.sh end to end. It carries
    no tests of its own, so unittest collects nothing from it directly.

    The build is faked — a cmake on PATH that writes a binary reporting the
    commit it was built from. Everything else runs for real.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.src = os.path.join(self.tmp, "src")
        os.makedirs(os.path.join(self.src, os.path.dirname(CUDA)))
        git(self.src, "init", "-q", "-b", "master")
        self.commit("base", CUDA, "int x;\n", SERVER, "int y;\n")
        # llama.cpp ignores its build directories, and the preflight's
        # "working tree clean" check would otherwise see the build this script
        # just made and refuse the next one. Fixture fidelity, not decoration.
        self.commit("ignore builds", ".gitignore", "build*/\n")
        git(self.src, "checkout", "-q", "-b", "patch")
        self.commit("the patch", CUDA, "// %s\nint x;\n" % MARKER,
                    SERVER, "// %s\nint y;\n" % MARKER2)
        git(self.src, "checkout", "-q", "master")
        # step 1 fetches from `origin` even when the ref is local. Point it at
        # the fixture itself: no network, and the step is exercised rather
        # than skipped, which is the difference between this and a dry run.
        git(self.src, "remote", "add", "origin", self.src)

        self.bin = write_fake_cmake(self.tmp, self.src)

    def commit(self, msg, *pairs):
        """(path, body) pairs, all in ONE commit — the patch really is one."""
        for path, body in zip(pairs[::2], pairs[1::2]):
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


@unittest.skipIf(shutil.which("cmake") is None,
                 "build-llama.sh refuses without cmake — skipped LOUDLY")
class TestBuildingWithoutThePatchOnPurpose(_FakeBuildFixture):
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
    """

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

    def test_use_succeeds_and_says_so_with_an_exit_code(self):
        """The POSITIVE half of --use, which was missing.

        --use is the rollback path: the thing you reach for when a build is
        wrong and the machine has to go back, under time pressure. On
        30.08.2026 it moved the symlink correctly and then exited 2, because
        the closing hint interpolated $BUILD_ID — a variable only the BUILD
        path sets. Everything worked and the exit code said it had not.

        The negative case below was tested and this one was not, which is why
        it survived: a check that can only ever reach one of its two answers
        is the shape this repository keeps finding.
        """
        d = os.path.join(self.src, "build-rocm-patched-goodone")
        os.makedirs(os.path.join(d, "bin"))
        b = os.path.join(d, "bin", "llama-server")
        with open(b, "w") as f:
            f.write("#!/bin/sh\necho 'version: 0.0.0 (build 1, commit deadbeef)'\n")
        os.chmod(b, 0o755)
        with open(os.path.join(d, ".build-stamp"), "w") as f:
            f.write("build_id=goodone\npatched=yes\n")
        r = self.build("--use", "goodone")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(
            os.path.basename(os.readlink(os.path.join(self.src, "build-rocm-patched"))),
            "build-rocm-patched-goodone")

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


UNROLL_FLAG = "--amdgpu-unroll-threshold-local=600"


class TestTheHelpKeepsUpWithTheOptions(unittest.TestCase):
    """`--help` printed a fixed line RANGE, so adding two options above the
    last one pushed `--dry-run` off the end. Nothing failed; the help was
    simply short, which is the kind of wrong this repository keeps meeting.

    Needs no cmake: -h exits before the preflight."""

    def help_text(self):
        r = subprocess.run(["bash", SCRIPT, "-h"], capture_output=True,
                           text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def test_every_option_the_parser_accepts_is_documented(self):
        with open(SCRIPT) as f:
            body = f.read()
        # The parser's own case labels, which is the list that cannot drift.
        # Bounded by the `done` that closes the while, not by the first `esac`:
        # the case arm is INDENTED, so an unanchored split ran on into the
        # script body and collected `--porcelain` out of `git status
        # --porcelain)` as though it were an option.
        arm = body.split("while [ $# -gt 0 ]; do", 1)[1].split("\ndone", 1)[0]
        opts = set(re.findall(r"(--[a-z-]+)\)", arm))
        opts |= set(re.findall(r"\|(--[a-z-]+)\)", arm))
        self.assertIn("--unroll", opts, "the fixture found no options at all")
        text = self.help_text()
        missing = sorted(o for o in opts if o not in text and o != "--help")
        self.assertEqual(missing, [], "options the help does not mention")

    def test_it_stops_before_the_prose(self):
        """The bound is a heading. If it ever swallows the whole file the
        help becomes the essay, and this says so."""
        text = self.help_text()
        self.assertNotIn("Why this exists", text)
        self.assertLess(len(text.splitlines()), 40, text)


@unittest.skipIf(shutil.which("cmake") is None,
                 "build-llama.sh refuses without cmake — skipped LOUDLY")
class TestBuildingWithTheUnrollFlag(_FakeBuildFixture):
    """A build carrying an extra compiler flag is a SUBJECT, like an
    unpatched one — and for the same reason it needs a family of its own.

    llama.cpp#19984 measures, on this exact hardware, a prefill collapse
    attributed to a loop-unrolling regression in ROCm 7+, and works around it
    with `-mllvm --amdgpu-unroll-threshold-local=600`. Whether that flag does
    anything HERE is an open question — which is the point: it cannot be
    answered by a build that quietly overwrites the reference it would be
    compared against.

    The family name is `rocm-unroll` and not `rocm-patched-unroll`, and that
    is load-bearing rather than taste. builds_of_backend() globs
    `build-$fam-*`, so a name starting with `rocm-patched-` would be swept up
    by every list and every prune of the patched family — an unroll build
    would appear there as a build whose id begins `unroll-`, and --prune could
    offer to delete it as a stale patched build. Hence the collision test
    below, which is the one that would catch a later rename.
    """

    def test_it_builds_into_a_family_of_its_own(self):
        r = self.build("--ref", "master", "--unroll", PATCH_BRANCH="patch")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        made = [d for d in os.listdir(self.src) if d.startswith("build-")]
        self.assertEqual(len(made), 1, made)
        self.assertTrue(made[0].startswith("build-rocm-unroll-"), made[0])

    def test_the_family_does_not_collide_with_the_patched_glob(self):
        """The rename guard. builds_of_backend() is a prefix glob, so this is
        what keeps --prune of the patched family away from an unroll build."""
        self.build("--ref", "master", "--unroll", PATCH_BRANCH="patch")
        unroll = [d for d in os.listdir(self.src)
                  if d.startswith("build-rocm-unroll-")][0]
        self.assertFalse(
            unroll.startswith("build-rocm-patched-"),
            "an unroll build inside the patched family's glob would be "
            "listed, ranked and eventually deleted as a patched build")
        # and the glob itself, asserted rather than reasoned about
        import glob as _glob
        swept = _glob.glob(os.path.join(self.src, "build-rocm-patched-*"))
        self.assertEqual(swept, [], swept)

    def test_the_stamp_carries_the_flag(self):
        """`cmake=` is where a reader finds out what a binary was built with.
        A flag that is not in it is a flag nobody can attribute a number to."""
        self.build("--ref", "master", "--unroll", PATCH_BRANCH="patch")
        d = [x for x in os.listdir(self.src) if x.startswith("build-")][0]
        st = self.stamp(d)
        self.assertIn(UNROLL_FLAG, st["cmake"])
        self.assertEqual(st["family"], "rocm-unroll")

    def test_an_ordinary_build_does_not_carry_the_flag(self):
        """The negative control. Without it the test above passes even if the
        flag were hard-wired into every build, and the A/B would compare a
        binary with itself."""
        self.build("--ref", "master", PATCH_BRANCH="patch")
        d = [x for x in os.listdir(self.src) if x.startswith("build-")][0]
        self.assertNotIn(UNROLL_FLAG, self.stamp(d)["cmake"])
        self.assertNotIn("mllvm", self.stamp(d)["cmake"])

    def test_it_is_still_a_patched_build(self):
        """An unroll build is patched — the flag is the only difference. If it
        stamped `patched=no` the comparison would carry two variables."""
        self.build("--ref", "master", "--unroll", PATCH_BRANCH="patch")
        d = [x for x in os.listdir(self.src) if x.startswith("build-")][0]
        st = self.stamp(d)
        self.assertEqual(st["patched"], "yes")
        self.assertEqual(st["patch_branch"], "patch")
        self.assertNotEqual(st["patch_commit"], "none")

    def test_it_cannot_be_activated(self):
        """Same rule as --no-patch: a subject is not a binary to serve. It is
        reachable for a measurement through sideserver's --bin, which restores
        production afterwards; the symlink is not that."""
        r = self.build("--ref", "master", "--unroll", "--activate",
                       PATCH_BRANCH="patch")
        self.assertNotEqual(r.returncode, 0, r.stdout)
        self.assertIn("cannot be combined", r.stdout + r.stderr)
        self.assertFalse(
            [d for d in os.listdir(self.src) if d.startswith("build-")],
            "it must refuse in preflight, before anything is built")

    def test_list_shows_the_unroll_family(self):
        """A directory nothing lists is a directory nobody prunes, and these
        are ~950 MB each. The two-family list was hard-wired."""
        self.build("--ref", "master", "--unroll", PATCH_BRANCH="patch")
        r = self.build("--list")
        self.assertIn("[rocm-unroll]", r.stdout)

    def test_prune_names_the_unroll_family_as_one_it_did_not_touch(self):
        self.build("--ref", "master", "--unroll", PATCH_BRANCH="patch")
        self.build("--ref", "master", PATCH_BRANCH="patch")
        r = self.build("--prune")
        self.assertIn("rocm-unroll", r.stdout,
                      "prune of the patched family must say that unroll "
                      "builds exist and how to prune them")


@unittest.skipIf(shutil.which("cmake") is None,
                 "build-llama.sh refuses without cmake — skipped LOUDLY")
class TestBuildingAgainstAnotherRocm(_FakeBuildFixture):
    """A build against a ROCm that is not the system's.

    THE FAILURE THIS GUARDS AGAINST IS SILENT AND WAS MEASURED. ROCm 10.1's
    libamdhip64 carries the SAME soname as Fedora's 7.1 — `.so.7`, pointing at
    7.16.26344 and 7.1.52802 respectively. So a binary built against the new
    SDK loads the OLD runtime through the system search path unless something
    sets LD_LIBRARY_PATH, and reports numbers either way. The stamp records
    the SDK precisely so bench/suites/speed-ab.py can put it back on the path
    and then CHECK, with ldd, which one actually gets loaded.

    The compiler is proven present rather than assumed: without that check a
    wrong --rocm-path silently falls back to the system toolchain and stamps
    the result as the other SDK's, which is a lie a measurement would inherit.
    """

    def fake_sdk(self, version="10.1.0", with_clang=True):
        sdk = os.path.join(self.tmp, "sdk")
        os.makedirs(os.path.join(sdk, "llvm", "bin"), exist_ok=True)
        os.makedirs(os.path.join(sdk, ".info"), exist_ok=True)
        with open(os.path.join(sdk, ".info", "version"), "w") as f:
            f.write(version + "\n")
        if with_clang:
            p = os.path.join(sdk, "llvm", "bin", "clang")
            with open(p, "w") as f:
                f.write("#!/bin/sh\nexit 0\n")
            os.chmod(p, 0o755)
        return sdk

    def test_it_builds_into_a_family_of_its_own(self):
        r = self.build("--ref", "master", "--rocm-path", self.fake_sdk(),
                       PATCH_BRANCH="patch")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        made = [d for d in os.listdir(self.src) if d.startswith("build-")]
        self.assertEqual(len(made), 1, made)
        self.assertTrue(made[0].startswith("build-rocm-altsdk-"), made[0])

    def test_the_stamp_records_which_sdk(self):
        """Without this speed-ab.py cannot put the SDK back on the library
        path, and the run silently measures the system ROCm."""
        sdk = self.fake_sdk(version="10.1.0")
        self.build("--ref", "master", "--rocm-path", sdk, PATCH_BRANCH="patch")
        d = [x for x in os.listdir(self.src) if x.startswith("build-")][0]
        st = self.stamp(d)
        self.assertEqual(st["rocm_path"], sdk)
        self.assertEqual(st["rocm_version"], "10.1.0")
        self.assertIn(os.path.join(sdk, "llvm", "bin", "clang"), st["cmake"])

    def test_the_compiler_flag_appears_exactly_once(self):
        """It appeared twice — the system path and then the override. cmake
        takes the last, so it WORKED, and the stamp showed both and settled
        nothing about which compiler had produced the binary."""
        sdk = self.fake_sdk()
        self.build("--ref", "master", "--rocm-path", sdk, PATCH_BRANCH="patch")
        d = [x for x in os.listdir(self.src) if x.startswith("build-")][0]
        cmake = self.stamp(d)["cmake"]
        self.assertEqual(cmake.count("-DCMAKE_HIP_COMPILER="), 1, cmake)
        self.assertNotIn("/usr/lib64/rocm", cmake)

    def test_an_ordinary_build_records_no_sdk(self):
        """The negative control: speed-ab.py keys on this field, so an
        ordinary build must not claim one."""
        self.build("--ref", "master", PATCH_BRANCH="patch")
        d = [x for x in os.listdir(self.src) if x.startswith("build-")][0]
        st = self.stamp(d)
        self.assertIn(st.get("rocm_path", "none"), ("none", ""))

    def test_a_path_without_a_compiler_is_refused(self):
        """The lie this prevents: falling back to the system toolchain and
        stamping the build as the other SDK's."""
        r = self.build("--ref", "master", "--rocm-path",
                       self.fake_sdk(with_clang=False), PATCH_BRANCH="patch")
        self.assertNotEqual(r.returncode, 0, r.stdout)
        self.assertIn("no clang", (r.stdout + r.stderr).lower())
        self.assertFalse(
            [d for d in os.listdir(self.src) if d.startswith("build-")],
            "it must refuse in preflight, before anything is built")

    def test_a_missing_directory_is_refused(self):
        r = self.build("--ref", "master", "--rocm-path",
                       os.path.join(self.tmp, "nope"), PATCH_BRANCH="patch")
        self.assertNotEqual(r.returncode, 0, r.stdout)

    def test_it_cannot_be_activated(self):
        r = self.build("--ref", "master", "--rocm-path", self.fake_sdk(),
                       "--activate", PATCH_BRANCH="patch")
        self.assertNotEqual(r.returncode, 0, r.stdout)
        self.assertIn("cannot be combined", r.stdout + r.stderr)

    def test_list_shows_the_altsdk_family(self):
        self.build("--ref", "master", "--rocm-path", self.fake_sdk(),
                   PATCH_BRANCH="patch")
        self.assertIn("[rocm-altsdk]", self.build("--list").stdout)


@unittest.skipIf(shutil.which("cmake") is None,
                 "build-llama.sh refuses without cmake — skipped LOUDLY")
class TestBuildingLlamaBenchAsWell(_FakeBuildFixture):
    """llama-bench is what llama.cpp#19984 measured with, and this stack has
    never built it — `--target llama-server` only. Reproducing somebody
    else's number needs their instrument, not an equivalent one."""

    def test_it_also_builds_llama_bench(self):
        r = self.build("--ref", "master", "--with-bench", PATCH_BRANCH="patch")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        d = [x for x in os.listdir(self.src) if x.startswith("build-")][0]
        self.assertTrue(
            os.path.exists(os.path.join(self.src, d, "bin", "llama-bench")),
            os.listdir(os.path.join(self.src, d, "bin")))

    def test_the_server_is_still_built(self):
        """The positive control: --with-bench must ADD a target, not swap one."""
        self.build("--ref", "master", "--with-bench", PATCH_BRANCH="patch")
        d = [x for x in os.listdir(self.src) if x.startswith("build-")][0]
        self.assertTrue(
            os.path.exists(os.path.join(self.src, d, "bin", "llama-server")))

    def test_without_it_there_is_no_bench(self):
        self.build("--ref", "master", PATCH_BRANCH="patch")
        d = [x for x in os.listdir(self.src) if x.startswith("build-")][0]
        self.assertFalse(
            os.path.exists(os.path.join(self.src, d, "bin", "llama-bench")))


@unittest.skipIf(shutil.which("cmake") is None,
                 "build-llama.sh refuses without cmake — skipped LOUDLY")
class TestAForeignTreeGetsItsOwnFamily(_FakeBuildFixture):
    """--family: the phase-3 case. Measuring somebody's fork means building a
    tree that is neither the serving one nor merely unpatched — and the
    handover entry that planned it names the trap: a fork build sitting in
    the patched family is one --use away from being served. So the family is
    the caller's to name, within the rules the naming section of the script
    already carries: no hyphen (build-<family>-* is a glob), no built-in name.
    """

    def test_it_builds_into_the_named_family_with_the_patches(self):
        r = self.build("--ref", "master", "--family", "gdnfork",
                       PATCH_BRANCH="patch")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        made = [d for d in os.listdir(self.src) if d.startswith("build-")]
        self.assertEqual(len(made), 1, made)
        self.assertTrue(made[0].startswith("build-rocm-gdnfork-"), made[0])
        st = self.stamp(made[0])
        self.assertEqual(st["family"], "rocm-gdnfork")
        self.assertEqual(st["patched"], "yes",
                         "a --family build still carries the patches — it is "
                         "a foreign TREE, not an unpatched one")

    def test_it_cannot_be_activated(self):
        r = self.build("--ref", "master", "--family", "gdnfork", "--activate",
                       PATCH_BRANCH="patch")
        self.assertNotEqual(r.returncode, 0, r.stdout)
        self.assertIn("cannot be combined", r.stdout + r.stderr)
        self.assertFalse(
            [d for d in os.listdir(self.src) if d.startswith("build-")],
            "it must refuse in preflight, before anything is built")

    def test_a_built_in_name_is_refused(self):
        """--family patched would put a foreign tree in the one family the
        symlink resolves against, wearing the family's own name."""
        r = self.build("--ref", "master", "--family", "patched",
                       PATCH_BRANCH="patch")
        self.assertNotEqual(r.returncode, 0, r.stdout)
        self.assertIn("built-in family", r.stdout + r.stderr)

    def test_a_hyphen_is_refused(self):
        """builds_of_backend() globs build-<family>-*, so a family name with
        a hyphen re-opens the collision the unroll naming note describes."""
        r = self.build("--ref", "master", "--family", "gdn-fork",
                       PATCH_BRANCH="patch")
        self.assertNotEqual(r.returncode, 0, r.stdout)
        self.assertIn("lowercase letters and digits only",
                      r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
