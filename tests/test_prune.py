"""build-llama.sh --prune — the only thing in this repo that DELETES.

Builds are ~950 MB each and nothing removed them, so three had accumulated,
two of them the same build under two names. --prune was written for that on
26.08. and two bugs were found in it BY HAND before it ever ran for real:

  * the keep-list was matched with a `case` over a newline-separated string,
    so an id preceded by a newline never matched " $id " — it offered to
    delete a build it had just decided to keep;
  * `sort -r` on lines whose first field can be empty put the STAMPLESS build
    FIRST under a normal locale, because collation ignores the leading tab.
    Prune kept it as "newest" and offered the real second-newest instead.
    Wrong by 944 MB, in a command that deletes.

Both were caught by reading output, which is luck. A delete needs a test, so
the whole thing runs against a scratch directory of fake builds here — no GPU,
no llama.cpp, nothing real to lose.
"""
import os, subprocess, tempfile, unittest

import common

REPO = common.REPO


class TestPrune(unittest.TestCase):
    def build(self, tmp, ident, built_at=None, active=False):
        d = os.path.join(tmp, "build-rocm-patched-" + ident)
        os.makedirs(os.path.join(d, "bin"), exist_ok=True)
        with open(os.path.join(d, ".build-stamp"), "w") as f:
            f.write("build_id=%s\nbackend=rocm\n" % ident)
            if built_at:
                f.write("built_at=%s\n" % built_at)
        if active:
            link = os.path.join(tmp, "build-rocm-patched")
            if os.path.lexists(link):
                os.unlink(link)
            os.symlink("build-rocm-patched-" + ident, link)
        return d

    def prune(self, tmp, *args, locale=None, models_repo=None):
        env = dict(os.environ, LLAMA_SRC=tmp)
        if models_repo:
            env["MODELS_REPO"] = models_repo
        if locale:
            env["LC_ALL"] = env["LANG"] = locale
        return subprocess.run(
            ["bash", str(REPO / "setup" / "scripts" / "build-llama.sh"),
             "--prune", *args],
            capture_output=True, text=True, timeout=120, env=env)

    def test_the_active_build_is_never_offered(self):
        with tempfile.TemporaryDirectory() as t:
            self.build(t, "new", "2026-08-26T09:00:00+02:00", active=True)
            self.build(t, "old", "2026-08-20T09:00:00+02:00")
            r = self.prune(t)
            self.assertNotIn("would remove new", r.stdout, r.stdout)
            self.assertIn("keep  new", r.stdout)

    def test_the_stampless_build_goes_first_not_last(self):
        """The sort bug, pinned by BEHAVIOUR.

        A directory with no built_at predates stamps and is the OLDEST
        candidate. In the first version its line began with an empty field, so
        `sort -r` saw a leading tab, dictionary collation ignored it, and the
        line sorted by what followed — putting the oldest build FIRST. Prune
        kept it as the newest and offered the real second-newest for deletion.
        Wrong by 944 MB, in a command that DELETES.
        """
        with tempfile.TemporaryDirectory() as t:
            self.build(t, "active", "2026-08-26T09:00:00+02:00", active=True)
            self.build(t, "dated", "2026-08-25T09:00:00+02:00")
            self.build(t, "stampless")
            os.unlink(os.path.join(t, "build-rocm-patched-stampless", ".build-stamp"))
            r = self.prune(t, "--keep", "1")
            self.assertIn("would remove stampless", r.stdout, r.stdout)
            self.assertIn("keep  dated", r.stdout, r.stdout)

    def test_the_order_does_not_depend_on_the_locale(self):
        """The same list under C and under a collating locale, byte for byte.

        Added 27.08. after a check that did not do what it looked like. The
        test above passed no matter which locale the suite ran under, and the
        reason is worth writing down: the ordering is protected TWICE.
        `build-llama.sh` writes an explicit `0000-00-00` sentinel instead of an
        empty field — so no line begins with a tab and collation has nothing to
        ignore — AND it pins `LC_ALL=C sort -r`. Remove either one and the
        behaviour is unchanged; remove both and the oldest build is kept as the
        newest again.

        That means no test can fail on the removal of one belt, and none
        should: that is what defence in depth is. What CAN be pinned is the
        property both belts exist for, and this is it. Measured the same day,
        with the shape the first version produced:

            sort -r on "1111" against "\t9999"
                C                        1111    first
                en_US / de_DE / fr_FR    \t9999  first
        """
        loc = common.collating_locale(self)
        out = {}
        # ONE directory for both runs: two temp dirs differ in their names and
        # the name is printed, so a per-run directory would make every
        # comparison fail on the path and none on the ordering. Without --yes
        # prune only says what it WOULD remove, so running it twice is safe.
        with tempfile.TemporaryDirectory() as t:
            self.build(t, "active", "2026-08-26T09:00:00+02:00", active=True)
            self.build(t, "dated", "2026-08-25T09:00:00+02:00")
            self.build(t, "stampless")
            os.unlink(os.path.join(t, "build-rocm-patched-stampless",
                                   ".build-stamp"))
            for name in ("C", loc):
                r = self.prune(t, "--keep", "1", locale=name)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                out[name] = r.stdout
            self.assertIn("build-rocm-patched-stampless", os.listdir(t),
                          "a dry run must not have deleted anything")
        self.assertEqual(out["C"], out[loc],
                         "prune proposes a different set under %s than under C "
                         "— the sentinel or the LC_ALL pin is gone, and this "
                         "command deletes" % loc)

    def test_keep_counts_fallbacks_and_excludes_the_active_one(self):
        """--keep 1 means one build to roll back to, not "one in total"."""
        with tempfile.TemporaryDirectory() as t:
            self.build(t, "a", "2026-08-26T09:00:00+02:00", active=True)
            self.build(t, "b", "2026-08-25T09:00:00+02:00")
            self.build(t, "c", "2026-08-24T09:00:00+02:00")
            r = self.prune(t, "--keep", "1")
            self.assertIn("keep  b", r.stdout, r.stdout)
            self.assertIn("would remove c", r.stdout, r.stdout)

    def test_it_deletes_NOTHING_without_yes(self):
        """The deliberate departure from --dry-run everywhere else here: a
        model switch is reversible in seconds, a deleted build is a
        fifteen-minute rebuild."""
        with tempfile.TemporaryDirectory() as t:
            self.build(t, "a", "2026-08-26T09:00:00+02:00", active=True)
            doomed = self.build(t, "b", "2026-08-20T09:00:00+02:00")
            self.prune(t, "--keep", "0")
            self.assertTrue(os.path.isdir(doomed),
                            "prune deleted without --yes")

    def test_with_yes_it_removes_exactly_the_ones_it_named(self):
        with tempfile.TemporaryDirectory() as t:
            keep = self.build(t, "a", "2026-08-26T09:00:00+02:00", active=True)
            doomed = self.build(t, "b", "2026-08-20T09:00:00+02:00")
            r = self.prune(t, "--keep", "0", "--yes")
            self.assertFalse(os.path.isdir(doomed), r.stdout)
            self.assertTrue(os.path.isdir(keep), "it removed the ACTIVE build")


class TestAPinnedBuildIsNotDeletable(unittest.TestCase):
    """in_use_by() only sees RUNNING processes, and that is not the same
    question as "is anything relying on this".

    setup/env/flashnext.env pins build-rocm-patched-b10636-20-g035e22731 by
    name and explains why: that PR moved 20 commits on its first day, and a
    profile following the symlink would have changed backend under a rebuild.
    Nothing is running out of it, so `--prune --yes` offered to delete it —
    953 MB, in the one command in this repository that deletes, leaving a
    profile pointing at nothing.

    Found 27.08. while making --prune family-aware, and it is the same shape
    as everything else here: the check ran, it was right about what it
    checked, and what it needed to know was somewhere else.
    """

    def fixture(self, tmp, bin_line):
        """A minimal registry: setup/lib/models.sh reads MODELS_REPO."""
        repo = os.path.join(tmp, "repo")
        os.makedirs(os.path.join(repo, "setup", "env"))
        with open(os.path.join(repo, "setup", "env", "pinner.env"), "w") as f:
            f.write("MODEL_TITLE=a profile that pins a build\n"
                    "LLAMA_ARGS=-m /nowhere.gguf\n"
                    "LLAMA_BIN=%s\n" % bin_line)
        return repo

    def build(self, tmp, ident, built_at=None, active=False):
        return TestPrune.build(self, tmp, ident, built_at, active)

    def test_a_build_a_profile_names_is_kept(self):
        with tempfile.TemporaryDirectory() as t:
            repo = self.fixture(
                t, "llama.cpp/build-rocm-patched-pinned/bin/llama-server")
            self.build(t, "new", "2026-08-26T09:00:00+02:00", active=True)
            self.build(t, "pinned", "2026-08-20T09:00:00+02:00")
            self.build(t, "junk", "2026-08-19T09:00:00+02:00")
            r = TestPrune.prune(self, t, "--keep", "0", models_repo=repo)
            self.assertIn("keep  pinned", r.stdout, r.stdout)
            self.assertIn("PINNED", r.stdout)
            self.assertIn("would remove junk", r.stdout,
                          "the positive control: an unpinned build in the "
                          "same position must still be offered")

    def test_a_profile_that_names_the_symlink_pins_nothing_extra(self):
        """qwen38.env names build-rocm-patched, the SYMLINK. That is the
        active build, which is kept for its own reason — it must not silently
        pin whatever the symlink happens to point at under a different name,
        or --prune would keep everything forever and say nothing."""
        with tempfile.TemporaryDirectory() as t:
            repo = self.fixture(
                t, "llama.cpp/build-rocm-patched/bin/llama-server")
            self.build(t, "new", "2026-08-26T09:00:00+02:00", active=True)
            self.build(t, "old", "2026-08-20T09:00:00+02:00")
            r = TestPrune.prune(self, t, "--keep", "0", models_repo=repo)
            self.assertIn("would remove old", r.stdout, r.stdout)


if __name__ == "__main__":
    unittest.main()
