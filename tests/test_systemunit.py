"""The system unit is DERIVED, and these are the properties that makes true.

There used to be two unit files. `setup/systemd/llama@.service` was written by
hand beside `llama-user@.service`, it had never been started on this machine
(SELinux refuses it), and by 27.08.2026 it had rotted in three ways at once:

    ExecStart      pinned build-vulkan/bin/llama-server, ignoring LLAMA_BIN —
                   so it would have served the wrong backend AND the build
                   without setup/patches/hip-integrated-off.patch, which is
                   the patch that stops the gfx1151 corruption. A stranger
                   enabling it would have got '////' and blamed the model.
    MemoryHigh     48G against the user unit's 96G
    MemoryMax      64G against the user unit's 108G

Nothing caught any of it, because nothing ran it. So the file is gone and the
unit is generated from the one that does run. These tests are what "generated"
buys: not that the system unit works — it has still never been started — but
that it cannot disagree with the unit that is exercised every day.
"""
import os, sys, unittest

import common

REPO = common.REPO
sys.path.insert(0, str(REPO / "setup" / "lib"))
import systemdfile                                            # noqa: E402
import systemunit                                             # noqa: E402

USER_UNIT = str(REPO / "setup" / "systemd" / "llama-user@.service")


class Base(unittest.TestCase):
    def setUp(self):
        self.text = systemunit.render(user="nobody")

    def directives(self, text, name):
        return [l.split("=", 1)[1] for l in text.splitlines()
                if l.startswith(name + "=")]

    def user(self, name):
        return systemdfile.directive(USER_UNIT, name)


class TestThereIsOnlyOneUnitFile(unittest.TestCase):
    def test_the_hand_written_system_unit_is_gone(self):
        """If it comes back, it will rot again — it did once, invisibly, for
        a fortnight."""
        self.assertFalse((REPO / "setup" / "systemd" / "llama@.service").exists(),
                         "a second hand-maintained unit is back in the repo")

    def test_the_generator_names_its_source(self):
        self.assertTrue(os.path.exists(systemunit.SOURCE))
        self.assertIn("llama-user@.service", systemunit.SOURCE)


class TestTheDerivationIsComplete(Base):
    def test_no_home_specifier_survives(self):
        """%h in a SYSTEM unit resolves to /root, not to the home of User=.
        That is a silent wrong answer — a server that looks configured and
        reads nothing — so it is an error rather than a warning."""
        self.assertEqual(systemunit.leftovers(self.text), [])

    def test_no_user_unit_instance_name_survives(self):
        """A Conflicts= naming llama-user@X would let a system instance and a
        user instance run at once, and the second loses the race for port
        8080: the service says active and the gateway talks to the wrong
        model. That is the bug lib/models.sh was written against."""
        for line in self.text.splitlines():
            if line.lstrip().startswith("#"):
                continue
            self.assertNotIn("llama-user@", line)

    def test_every_rule_still_has_something_to_do(self):
        """A substitution that matches nothing is a rule describing a unit
        that no longer looks like that — it will not fail, it will silently
        stop covering the case it was written for."""
        with open(systemunit.SOURCE, encoding="utf-8") as fh:
            source = fh.read()
        dead = [old for old, _ in systemunit.RULES if old not in source]
        self.assertFalse(dead, "these rules match nothing in the user unit "
                               "any more: %s" % dead)

    def test_it_is_marked_as_generated_in_the_file_itself(self):
        """Someone will find this in /etc without the repo in front of them."""
        head = self.text.splitlines()[0]
        self.assertIn("GENERATED", head)
        self.assertIn("systemunit.py", self.text)
        self.assertIn("install.sh --system-unit", self.text)


class TestItCannotDisagreeWithTheUnitThatRuns(Base):
    """Each of these is one of the three ways the hand-written file drifted."""

    def test_the_memory_ceilings_are_the_user_unit_s(self):
        for name in ("MemoryHigh", "MemoryMax"):
            self.assertEqual(self.directives(self.text, name), self.user(name),
                             "%s drifted — 48G/64G against 96G/108G is exactly "
                             "how the predecessor was wrong" % name)

    def code(self, text):
        return "\n".join(l for l in text.splitlines()
                         if not l.lstrip().startswith("#"))

    def test_it_goes_through_the_wrapper_and_never_names_a_binary(self):
        """The predecessor hard-wired build-vulkan/bin/llama-server, so it
        ignored the profile's LLAMA_BIN and would have run the UNPATCHED
        build. Only llamaexec reads LLAMA_BIN.

        Checked against the DIRECTIVES: the header names the old binary in
        order to say what went wrong, and forbidding it there would be
        forbidding the explanation.
        """
        execs = self.directives(self.text, "ExecStart")
        self.assertEqual(len(execs), 1, execs)
        self.assertIn("llm-exec", execs[0])
        body = self.code(self.text)
        for banned in ("llama-server", "build-vulkan", "build-rocm"):
            self.assertNotIn(banned, body,
                             "the derived unit names a binary instead of "
                             "letting the profile choose one")

    def body(self):
        """Everything from [Unit] on. The header above it is written HERE and
        may say anything, including what the mistake was — the first version
        of this test forbade a phrase that its own header uses to explain it."""
        return self.text[self.text.index("[Unit]"):]

    def test_the_reasoning_is_not_copied_into_the_derived_file(self):
        """Carrying the comments across was tried on 27.08. and the
        instance-name rule turned `systemctl --user enable --now
        llama-user@X` into `llama@X` — advice that does not work. The
        substitutions are right for directives and wrong for prose."""
        comments = [l for l in self.body().splitlines()
                    if l.lstrip().startswith("#")]
        self.assertLessEqual(len(comments), 2,
                             "the user unit's prose is being copied again: %r"
                             % comments)
        self.assertNotIn("systemctl --user", self.body())

    def test_the_guard_and_the_wait_both_survive_the_mapping(self):
        pre = self.directives(self.text, "ExecStartPre")
        self.assertEqual(len(pre), len(self.user("ExecStartPre")))
        self.assertTrue(any("llm-wait-for-model" in p for p in pre), pre)
        self.assertTrue(any("llm-check-room" in p for p in pre), pre)
        for p in pre:
            self.assertTrue(p.startswith("/usr/local/bin/"),
                            "a system unit may not exec out of a home: %s" % p)

    def test_restart_policy_and_timeouts_come_across_unchanged(self):
        for name in ("Restart", "RestartSec", "TimeoutStartSec",
                     "StartLimitIntervalSec", "StartLimitBurst", "Type"):
            self.assertEqual(self.directives(self.text, name), self.user(name), name)


class TestWhatOnlyASystemUnitHas(Base):
    def test_it_runs_as_a_named_user_and_not_as_root(self):
        self.assertIn("User=nobody", self.text)
        self.assertEqual(self.user("User"), [],
                         "the USER unit must not carry User= — it already is")

    def test_it_keeps_the_gpu_groups(self):
        """render and video, or the server cannot open the device at all."""
        self.assertIn("SupplementaryGroups=render video", self.text)

    def test_it_is_wanted_by_the_system_and_not_by_a_session(self):
        self.assertIn("WantedBy=multi-user.target", self.text)
        self.assertNotIn("WantedBy=default.target", self.text)

    def test_the_user_name_is_not_baked_into_the_repo(self):
        """It is substituted at generation time. The predecessor carried a real
        `User=` in a tracked file, which is what decision B began as before the
        drift turned out to be the bigger half.

        Asked with getuser() rather than a literal: writing the name down to
        check that it is not written down is a rule breaking itself, and
        tests/test_localenv.py caught exactly that.
        """
        import getpass
        default = systemunit.render()
        self.assertNotIn("User=nobody", default)
        me = getpass.getuser()
        for rel in ("setup/lib/systemunit.py", "setup/install.sh"):
            self.assertNotIn(me, (REPO / rel).read_text(encoding="utf-8"),
                             "%s names this machine's user" % rel)


class TestTheEnvironmentFilesMoveButKeepTheirMeaning(Base):
    def test_the_local_config_stays_optional_and_the_profile_stays_mandatory(self):
        """The '-' is the difference between 'this machine has not been set up
        yet' and 'this server would start with an empty $LLAMA_ARGS'."""
        files = self.directives(self.text, "EnvironmentFile")
        self.assertEqual(files, ["-/etc/llm-stack.env", "/etc/llm-profile/%i.env"])

    def test_they_are_the_same_two_files_the_user_unit_reads(self):
        user = self.user("EnvironmentFile")
        self.assertEqual(len(user), len(self.directives(self.text, "EnvironmentFile")))
        self.assertTrue(user[0].startswith("-"))
        self.assertFalse(user[-1].startswith("-"))


class TestTheCheckMode(Base):
    def test_check_reports_a_missing_installation_rather_than_claiming_a_match(self):
        rc = systemunit.main(["--check"])
        # 0 only if /etc/systemd/system/llama@.service happens to match here;
        # 1 if it is absent or stale. Never 2, which would mean the derivation
        # itself is broken.
        self.assertIn(rc, (0, 1))

    def test_a_broken_derivation_is_an_error_and_not_a_diff(self):
        """If a %h ever survives, --check must not go on to compare files:
        the answer is 'this generator is wrong', not 'your copy is stale'."""
        self.assertEqual(systemunit.leftovers("ExecStart=%h/x"), [
            "1: %h survives and would resolve to /root — ExecStart=%h/x"])


if __name__ == "__main__":
    unittest.main()
