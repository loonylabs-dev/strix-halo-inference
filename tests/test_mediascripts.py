"""setup/scripts/build-sd.sh and build-qwentts.sh — the bash edges.

Two review findings (01.09.2026), both measured before they were written
down: `--list` died with exit 2 and NO output on the first build directory
without a .build-stamp (set -euo makes the sed in the command substitution
fatal — and a stamp is written LAST, so an aborted build creates exactly
that state and then breaks the tool for inspecting it); and a rerun of
build-qwentts.sh wrote the -p adapter into the build root while the build
branch writes it beside the binary — a bin/-layout build got a decoy
adapter whose relative qwen-tts does not exist.
"""
import os
import stat
import subprocess
import tempfile
import unittest

import common

SD = str(common.REPO / "setup" / "scripts" / "build-sd.sh")
QW = str(common.REPO / "setup" / "scripts" / "build-qwentts.sh")


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.src = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def run_script(self, script, env_name, *args):
        env = dict(os.environ)
        env[env_name] = self.src
        return subprocess.run(["bash", script, *args], env=env,
                              capture_output=True, text=True, timeout=60)


class TestListSurvivesAStamplessBuild(Base):
    def test_build_sd_list(self):
        os.makedirs(os.path.join(self.src, "build-vulkan-deadbeef1"))
        r = self.run_script(SD, "SD_SRC", "--list")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("no-stamp", r.stdout)

    def test_build_qwentts_list(self):
        os.makedirs(os.path.join(self.src, "build-vulkan-deadbeef1"))
        r = self.run_script(QW, "QWENTTS_SRC", "--list")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("no-stamp", r.stdout)


class TestRefDoesNotBuildAStaleLocalBranch(Base):
    """Re-review sweep (01.09.2026): the FETCH_HEAD fix closed the
    stale-local-branch hole only on the online path. With the network
    down (or no origin at all) the fetch fails, the fallback checks the
    BRANCH NAME out locally, and a weeks-old tip is built and stamped as
    if it were current — the exact failure the adjacent comment declares
    closed. The fallback is for SHA pins already on disk, and only for
    them."""

    def _repo_with_local_branch(self):
        subprocess.run(["git", "init", "-q", "-b", "main", self.src],
                       check=True)
        subprocess.run(["git", "-C", self.src, "-c",
                        "user.email=t@t", "-c", "user.name=t",
                        "commit", "-q", "--allow-empty", "-m", "x"],
                       check=True)
        commit = subprocess.run(["git", "-C", self.src, "rev-parse",
                                 "--short=9", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        # A build for HEAD already exists: the old fallback then said
        # "already built" with rc 0 — proof it had silently checked out
        # the stale local branch.
        bindir = os.path.join(self.src, "build-vulkan-%s" % commit, "bin")
        os.makedirs(bindir)
        for exe in ("sd-cli", "qwen-tts"):
            p = os.path.join(bindir, exe)
            with open(p, "w") as fh:
                fh.write("#!/bin/sh\nexit 0\n")
            os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)
        return commit

    def test_build_sd_refuses_a_branch_name_when_the_fetch_failed(self):
        self._repo_with_local_branch()
        r = self.run_script(SD, "SD_SRC", "--ref", "main")
        self.assertNotEqual(r.returncode, 0,
                            "a branch name whose fetch failed must refuse, "
                            "not build the stale local tip:\n%s" % r.stdout)
        self.assertIn("refusing", (r.stdout + r.stderr).lower())

    def test_build_qwentts_refuses_a_branch_name_when_the_fetch_failed(self):
        self._repo_with_local_branch()
        r = self.run_script(QW, "QWENTTS_SRC", "--ref", "main")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("refusing", (r.stdout + r.stderr).lower())

    def test_a_sha_already_on_disk_still_works_offline(self):
        commit = self._repo_with_local_branch()
        full = subprocess.run(["git", "-C", self.src, "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
        r = self.run_script(SD, "SD_SRC", "--ref", full)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("already built", r.stdout)
        self.assertIn(commit, r.stdout)


class TestSubmoduleFailureIsNotSwallowed(unittest.TestCase):
    """Review finding (01.09.2026): `git submodule update ... || true`
    swallowed a failed sync after a --ref move, and the stamp records no
    submodule SHAs — new sd.cpp against old ggml, stamped as the new
    commit: a hybrid binary whose measured figures are attributed to a
    tree upstream never shipped."""

    def test_neither_script_silences_the_submodule_update(self):
        for script in (SD, QW):
            src = open(script, encoding="utf-8").read()
            for line in src.splitlines():
                if "submodule" in line and "update" in line:
                    self.assertNotIn("|| true", line,
                                     "%s: a failed submodule update must "
                                     "fail the build" % script)
                    self.assertNotIn("2>/dev/null", line,
                                     "%s: the submodule failure must stay "
                                     "visible" % script)


class TestRelockSurvivesAFailedFreeze(Base):
    """Review finding (01.09.2026): the --relock group redirect truncated
    requirements.lock BEFORE freeze ran. A failed or interrupted freeze
    left a header-only lock; the next plain run's `uv pip sync` then
    stripped the venv to match it — the measured torch stack destroyed
    and the committed lock clobbered, silently."""

    def test_the_lock_survives_a_freeze_that_dies(self):
        auddir = os.path.join(self.src, "audio")
        os.makedirs(auddir)
        for f in ("setup-venv.sh", "requirements.txt", "requirements.lock"):
            with open(str(common.REPO / "media" / "audio" / f),
                      encoding="utf-8") as fh:
                content = fh.read()
            with open(os.path.join(auddir, f), "w", encoding="utf-8") as fh:
                fh.write(content)
        original_lock = open(os.path.join(auddir, "requirements.lock"),
                             encoding="utf-8").read()
        venv = os.path.join(self.src, "venv")
        os.makedirs(os.path.join(venv, "bin"))
        py = os.path.join(venv, "bin", "python")
        with open(py, "w") as fh:
            fh.write("#!/bin/sh\nexit 0\n")
        os.chmod(py, os.stat(py).st_mode | stat.S_IEXEC)
        uv = os.path.join(self.src, "uv")
        with open(uv, "w") as fh:
            fh.write('#!/bin/sh\n'
                     'if [ "$1 $2" = "pip freeze" ]; then\n'
                     '  echo "freeze exploded" >&2; exit 1\n'
                     'fi\nexit 0\n')
        os.chmod(uv, os.stat(uv).st_mode | stat.S_IEXEC)

        env = dict(os.environ)
        env["MEDIA_AUDIO_VENV"] = venv
        env["UV_BIN"] = uv
        r = subprocess.run(["bash", os.path.join(auddir, "setup-venv.sh"),
                            "--relock"], env=env, capture_output=True,
                           text=True, timeout=60)
        self.assertNotEqual(r.returncode, 0, "the failed freeze must fail "
                                             "the script")
        self.assertEqual(open(os.path.join(auddir, "requirements.lock"),
                              encoding="utf-8").read(), original_lock,
                         "a failed relock must leave the measured lock "
                         "untouched — a header-only lock makes the next "
                         "sync strip the venv")


class TestTheAdapterLandsBesideTheBinary(Base):
    def test_rerun_on_a_bin_layout_build(self):
        subprocess.run(["git", "init", "-q", self.src], check=True)
        subprocess.run(["git", "-C", self.src, "-c",
                        "user.email=t@t", "-c", "user.name=t",
                        "commit", "-q", "--allow-empty", "-m", "x"],
                       check=True)
        commit = subprocess.run(["git", "-C", self.src, "rev-parse",
                                 "--short=9", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        bindir = os.path.join(self.src, "build-vulkan-%s" % commit, "bin")
        os.makedirs(bindir)
        binpath = os.path.join(bindir, "qwen-tts")
        with open(binpath, "w") as fh:
            fh.write("#!/bin/sh\nexit 0\n")
        os.chmod(binpath, os.stat(binpath).st_mode | stat.S_IEXEC)

        r = self.run_script(QW, "QWENTTS_SRC")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(os.path.exists(os.path.join(bindir, "qwen-tts-p")),
                        "the adapter must sit BESIDE the binary it wraps")
        self.assertFalse(
            os.path.exists(os.path.join(self.src, "build-vulkan-%s" % commit,
                                        "qwen-tts-p")),
            "a root-level decoy adapter points at a qwen-tts that is not "
            "there")


if __name__ == "__main__":
    unittest.main()
