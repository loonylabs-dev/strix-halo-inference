"""Editing the kernel command line, and the GTT arithmetic on top of it.

This is the one place in the repo where getting a string wrong does not
produce a wrong measurement but an unbootable machine: `root=UUID=…` is on
the same line as `ttm.pages_limit=`. The way back from a mangled command line
is a rescue stick, so the parser is tested against the machine's REAL command
line and against the ways it could go wrong.

The GTT ladder itself is measured, and published in
docs/setup/03-gpu-and-memory.md. It is worth
one number here so the formula and the table cannot drift apart:

    GiB    pages        rest for the host
     96    25165824     32 GiB   the conservative start, unchanged since setup
    108    28311552     20 GiB
    116    30408704     12 GiB   the runbook's "Arbeitseinstellung"
    120    31457280      8 GiB
"""
import os, subprocess, sys, tempfile, unittest

import common

REPO = common.REPO
sys.path.insert(0, str(REPO / "setup" / "lib"))
import kernelcmdline as kc                                   # noqa: E402

# A real command line in SHAPE — BOOT_IMAGE with a GPT partition, root by
# UUID, the ttm pair — because a toy string does not catch what this parser
# gets wrong. The UUID itself is fabricated: the property under test is that
# `root=UUID=` survives every edit, and which UUID it is decides nothing.
# It used to be this machine's actual root filesystem, which is a unique
# identifier of one installation published in a test that never needed it.
REAL = ("BOOT_IMAGE=(hd0,gpt8)/vmlinuz-6.19.10-300.fc44.x86_64 "
        "root=UUID=1f0a54c2-9e3b-4d17-8a60-c5b2e7d41839 ro rhgb quiet "
        "ttm.pages_limit=25165824 ttm.page_pool_size=25165824")

LADDER = {96: 25165824, 108: 28311552, 116: 30408704, 120: 31457280, 124: 32505856}


class TestLadder(unittest.TestCase):
    def test_the_formula_reproduces_the_runbook_table(self):
        for gib, pages in LADDER.items():
            with self.subTest(gib=gib):
                self.assertEqual(kc.pages_for_gib(gib), pages)
                self.assertEqual(kc.gib_for_pages(pages), gib)

    def test_the_value_this_machine_boots_with_is_96_gib(self):
        self.assertEqual(
            kc.gib_for_pages(int(kc.get_param(REAL, "ttm.pages_limit"))), 96)

    def test_fractional_gib_is_refused(self):
        with self.assertRaises(ValueError):
            kc.pages_for_gib(96.5)


class TestReading(unittest.TestCase):
    def test_a_value_containing_equals_survives(self):
        """root=UUID=… is the token that a naive split(\"=\") destroys."""
        self.assertEqual(kc.get_param(REAL, "root"),
                         "UUID=1f0a54c2-9e3b-4d17-8a60-c5b2e7d41839")

    def test_a_flag_without_a_value(self):
        self.assertEqual(kc.get_param(REAL, "quiet"), "")

    def test_an_absent_parameter_is_none_not_empty(self):
        """'' means present-without-value, None means absent. A caller that
        conflates them writes the parameter twice."""
        self.assertIsNone(kc.get_param(REAL, "amd_iommu"))

    def test_the_last_occurrence_wins_as_in_the_kernel(self):
        self.assertEqual(kc.get_param("a=1 b=2 a=3", "a"), "3")

    def test_quotes_are_refused_rather_than_guessed_at(self):
        with self.assertRaises(ValueError):
            kc.parse('root=UUID=x rd.luks.options="discard,foo"')


class TestSetting(unittest.TestCase):
    def test_raising_gtt_to_116_touches_only_the_two_ttm_values(self):
        pages = kc.pages_for_gib(116)
        got = kc.set_params(REAL, {"ttm.pages_limit": pages,
                                   "ttm.page_pool_size": pages})
        self.assertEqual(got,
            "BOOT_IMAGE=(hd0,gpt8)/vmlinuz-6.19.10-300.fc44.x86_64 "
            "root=UUID=1f0a54c2-9e3b-4d17-8a60-c5b2e7d41839 ro rhgb quiet "
            "ttm.pages_limit=30408704 ttm.page_pool_size=30408704")

    def test_root_and_boot_image_are_never_lost(self):
        """The property that matters more than any other in this file."""
        for params in ({"ttm.pages_limit": 1}, {"amd_iommu": "pt"},
                       {"ttm.pages_limit": 2, "iommu": "pt", "x": "y"}):
            with self.subTest(params=params):
                got = kc.set_params(REAL, params)
                self.assertIn("root=UUID=1f0a54c2-9e3b-4d17-8a60-c5b2e7d41839", got)
                self.assertIn("BOOT_IMAGE=(hd0,gpt8)/vmlinuz-6.19.10-300.fc44.x86_64", got)
                self.assertIn(" ro ", " %s " % got)

    def test_a_new_parameter_is_appended(self):
        got = kc.set_params(REAL, {"iommu": "pt"})
        self.assertTrue(got.endswith(" iommu=pt"))
        self.assertEqual(len(got.split()), len(REAL.split()) + 1)

    def test_an_existing_parameter_is_replaced_in_place(self):
        """Not removed-and-appended: a stable order keeps the diff between two
        edits readable, and the file is one people read under pressure."""
        got = kc.set_params(REAL, {"ttm.pages_limit": 7})
        self.assertEqual(got.split().index("ttm.pages_limit=7"),
                         REAL.split().index("ttm.pages_limit=25165824"))

    def test_duplicates_collapse_into_one(self):
        got = kc.set_params("a=1 b=2 a=3 c=4", {"a": 9})
        self.assertEqual(got, "a=9 b=2 c=4")

    def test_setting_is_idempotent(self):
        once = kc.set_params(REAL, {"ttm.pages_limit": 30408704})
        self.assertEqual(kc.set_params(once, {"ttm.pages_limit": 30408704}), once)

    def test_the_safety_net_fires_when_a_token_would_be_lost(self):
        """_assert_nothing_lost is the whole point of the module. Prove it is
        not decorative by feeding set_params a line it must not silently
        rewrite."""
        with self.assertRaises(ValueError):
            kc.set_params('root=UUID=x quiet foo="a b"', {"ttm.pages_limit": 1})


class TestRemoving(unittest.TestCase):
    def test_the_way_back(self):
        """The rollback the runbook names: drop both TTM parameters and the
        machine boots with the driver default again."""
        got = kc.remove_params(REAL, ["ttm.pages_limit", "ttm.page_pool_size"])
        self.assertEqual(got,
            "BOOT_IMAGE=(hd0,gpt8)/vmlinuz-6.19.10-300.fc44.x86_64 "
            "root=UUID=1f0a54c2-9e3b-4d17-8a60-c5b2e7d41839 ro rhgb quiet")

    def test_removing_something_absent_changes_nothing(self):
        self.assertEqual(kc.remove_params(REAL, ["not-there"]), REAL)


class TestGrubDefault(unittest.TestCase):
    """/etc/default/grub wraps the same line in a shell assignment. Fedora
    boots from BLS entries and not from this file, but leaving it stale means
    the next person reads a value the machine does not use."""

    FILE = ('GRUB_TIMEOUT=5\n'
            'GRUB_DISTRIBUTOR="$(sed \'s, release .*$,,g\' /etc/system-release)"\n'
            'GRUB_DEFAULT=saved\n'
            'GRUB_CMDLINE_LINUX="rhgb quiet ttm.pages_limit=25165824 ttm.page_pool_size=25165824"\n'
            'GRUB_ENABLE_BLSCFG=true\n')

    def test_only_the_cmdline_line_changes(self):
        got = kc.grub_default_set(self.FILE, {"ttm.pages_limit": 30408704,
                                              "ttm.page_pool_size": 30408704})
        before, after = self.FILE.splitlines(), got.splitlines()
        self.assertEqual(len(before), len(after))
        for b, a in zip(before, after):
            if b.startswith("GRUB_CMDLINE_LINUX="):
                self.assertIn("30408704", a)
            else:
                self.assertEqual(b, a, "an unrelated line was rewritten")

    def test_the_distributor_line_with_its_quotes_and_dollar_survives(self):
        got = kc.grub_default_set(self.FILE, {"ttm.pages_limit": 1})
        self.assertIn('GRUB_DISTRIBUTOR="$(sed \'s, release .*$,,g\' '
                      '/etc/system-release)"', got)

    def test_a_file_without_the_line_is_refused(self):
        with self.assertRaises(ValueError):
            kc.grub_default_set("GRUB_TIMEOUT=5\n", {"a": 1})


class TestRocmPool(unittest.TestCase):
    """Reading the GPU agent's pool out of rocminfo.

    The one-liner this replaced took the FIRST pool flagged COARSE GRAINED —
    which belongs to the CPU agent and is sized at the whole of system RAM.
    It therefore reported 124.9 GiB on a machine whose GPU limit was 96, in a
    line that called itself "the number that decides what loads". A check that
    prints a wrong number confidently is worse than no check, so this one is
    driven against a recorded rocminfo from this machine.
    """

    AWK = str(REPO / "setup" / "lib" / "rocm-gpu-pool.awk")
    FIXTURE = REPO / "tests" / "fixtures" / "rocminfo-gfx1151-gtt116.txt"

    def parse(self, text=None):
        if text is None:
            return subprocess.run(["awk", "-f", self.AWK, str(self.FIXTURE)],
                                  capture_output=True, text=True, timeout=30).stdout.strip()
        return subprocess.run(["awk", "-f", self.AWK], input=text,
                              capture_output=True, text=True, timeout=30).stdout.strip()

    def test_it_reads_the_gpu_agent_and_not_the_cpu_one(self):
        """The fixture was taken at GTT 116 GiB. The CPU agent's pool in the
        same file is 131002972 KB — the whole of RAM — and picking that one is
        the bug."""
        self.assertEqual(self.parse(), "121634816")
        # rocminfo reports KB, ttm counts 4 KiB pages — dividing by 4 is what
        # ties the two units together, and getting that backwards is how a
        # memory figure ends up off by a factor of 1024.
        self.assertEqual(kc.gib_for_pages(int(self.parse()) // 4), 116.0)

    def test_the_cpu_pool_really_is_in_the_fixture(self):
        """Otherwise the test above proves nothing: it has to be possible to
        get it wrong."""
        self.assertIn("131002972", self.FIXTURE.read_text(),
                      "the fixture no longer contains the pool that used to be "
                      "misread — it is not a regression test any more")

    def test_no_gpu_agent_yields_nothing_rather_than_a_wrong_number(self):
        cpu_only = "\n".join(l for l in self.FIXTURE.read_text().splitlines())
        cpu_only = cpu_only.replace("Device Type:             GPU",
                                    "Device Type:             CPU")
        self.assertEqual(self.parse(cpu_only), "")

    def test_the_live_machine_agrees_with_amdgpu(self):
        """The cross-check the script itself makes, asserted here too — this
        is the pair of numbers whose disagreement IS the amdgpu.gttsize trap
        (a ROCm issue documents 62.2 GB instead of 120)."""
        import glob
        live = subprocess.run("rocminfo 2>/dev/null | awk -f %s" % self.AWK,
                              shell=True, capture_output=True, text=True, timeout=60)
        if not live.stdout.strip():
            self.skipTest("no GPU agent here")
        sysfs = glob.glob("/sys/class/drm/card*/device/mem_info_gtt_total")
        if not sysfs:
            self.skipTest("no amdgpu sysfs here")
        rocm_gib = int(live.stdout.strip()) / 1048576
        with open(sysfs[0]) as fh:
            amdgpu_gib = int(fh.read()) / 1024**3
        self.assertAlmostEqual(rocm_gib, amdgpu_gib, delta=1.0,
                               msg="ROCm and amdgpu disagree about the GTT cap")


class TestScript(unittest.TestCase):
    """setup/scripts/gtt.sh must be safe to run with no arguments and must
    refuse a value that would starve the host."""

    GTT = str(REPO / "setup" / "scripts" / "gtt.sh")

    def run_(self, *args):
        return subprocess.run(["bash", self.GTT, *args],
                              capture_output=True, text=True, timeout=60)

    def test_showing_the_state_needs_no_arguments_and_no_root(self):
        r = self.run_()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("GiB", r.stdout)
        self.assertIn("116 GiB", r.stdout, "the ladder is not printed")

    def test_the_output_does_not_depend_on_the_locale(self):
        """The property, pinned differentially: the same script under C and
        under a comma-decimal locale must print the SAME thing.

        Three corrections went into this test on 27.08., and the third is the
        one that matters.

        1. It named de_DE, as if the bug were German. It is every
           comma-decimal locale — French, Spanish, Italian, Portuguese,
           Polish, Dutch. "German" was the locale of the person who found it.
        2. It ASKED for that locale instead of requiring one. Where
           de_DE.UTF-8 is not installed — most CI, most contributors — bash
           warns, falls back to C, and the test went green having compared
           nothing. It skips loudly now.
        3. It claimed to prove that `export LC_ALL=C` in gtt.sh works. IT
           CANNOT, and neither can any other test, because that line is
           REDUNDANT. Measured 27.08.: the real fix is that every float is
           formatted by awk and printed with %s, so bash's printf never parses
           one — and gawk 5.3 emits "16.9" under de_DE and fr_FR alike, since
           it honours LC_NUMERIC only under --posix. Remove `export LC_ALL=C`
           and nothing changes. That is what defence in depth means, and a
           test that failed when one of two independent belts came off would
           be pinning the implementation, not the behaviour.

        So this asserts the behaviour instead: locale in, identical bytes out.
        It fails when the script genuinely becomes locale-dependent again —
        which is the only condition worth failing on.
        """
        loc = common.comma_decimal_locale(self)
        runs = {}
        for name, env in (("C", dict(os.environ, LC_ALL="C", LANG="C")),
                          (loc, dict(os.environ, LC_ALL=loc, LANG=loc))):
            r = subprocess.run(["bash", self.GTT], capture_output=True,
                               text=True, timeout=60, env=env)
            self.assertEqual(r.returncode, 0, "%s: %s" % (name, r.stdout + r.stderr))
            self.assertNotIn("printf:", r.stderr, name)
            runs[name] = r.stdout
        self.assertEqual(runs["C"], runs[loc],
                         "gtt.sh prints differently under %s than under C — a "
                         "number is being formatted by something that reads "
                         "LC_NUMERIC" % loc)
        self.assertRegex(runs[loc], r"116 GiB\s+30408704\s+host keeps \d+\.\d GiB",
                         "the host figure lost its decimal point under %s" % loc)

    def test_a_value_larger_than_the_machine_is_refused(self):
        r = self.run_("--set", "999")
        self.assertEqual(r.returncode, 2)
        self.assertIn("RAM", r.stderr)

    @staticmethod
    def _ram_gib():
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1048576.0
        return None

    def test_a_value_that_leaves_the_host_nothing_is_refused(self):
        """The property is RELATIVE to the machine, and until 27.08. the test
        was not: it asked for 124 GiB, which is 'benchmarks only' on the
        124.9 GiB machine this was written on and is simply more RAM than a
        7.8 GiB CI runner has. Both are refused — but by DIFFERENT guards, and
        the one under test here is the second: enough RAM to take, not enough
        left to live on.

        Found by the first CI run this repository ever had. The simulated-clone
        run that was supposed to catch things like this shared the one thing
        that mattered: the machine."""
        total = self._ram_gib()
        self.assertIsNotNone(total, "no MemTotal — cannot pick a value")
        want = int(total) - 4          # takes most of it, leaves under 6
        if want < 1:
            self.skipTest("machine has %.1f GiB: no value both fits and "
                          "starves the host" % total)
        r = self.run_("--set", str(want))
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("host", r.stderr,
                      "refused, but not by the guard this test is about")

    def test_dry_run_shows_the_diff_and_touches_nothing(self):
        total = self._ram_gib()
        self.assertIsNotNone(total)
        want = min(116, int(total) - 6)      # accepted: leaves the host 6+
        if want < 1:
            self.skipTest("machine has %.1f GiB: no acceptable value" % total)
        r = self.run_("--set", str(want), "--dry-run")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn(str(want * 262144), r.stdout,
                      "the page count for %d GiB is not in the diff" % want)
        for line in r.stdout.splitlines():
            if "sudo " in line:
                self.assertIn("would:", line,
                              "a dry run named a sudo command without marking "
                              "it as a plan: %r" % line.strip())


if __name__ == "__main__":
    unittest.main()
