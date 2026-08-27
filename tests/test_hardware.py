"""What machine is this, and what follows from the answer.

This repo is measured on ONE configuration: Strix Halo (gfx1151) with 128 GB
of shared memory. The same silicon ships with 32 and 64 GB, and until 27.08.
nothing here said so — a newcomer with 64 GB could spend an afternoon before
finding out that one of seven profiles fits.

The answer is not to scale anything. Every number in this repo was measured on
one machine, and this day proved three times over that derived numbers are
wrong here. The answer is to say where you stand at the door, and to show the
profiles that fit AS WRITTEN — a filter over measured values, not a guess
about unmeasured ones.
"""
import os, re, subprocess, sys, unittest

import common

REPO = common.REPO
sys.path.insert(0, str(REPO / "setup" / "lib"))
import budget                                                  # noqa: E402
import defects                                                 # noqa: E402
import hardware                                                # noqa: E402
import systemdfile                                             # noqa: E402


class TestItIdentifiesTheGpuTwice(unittest.TestCase):
    """rocminfo is authoritative and is part of ROCm — which the person asking
    "is this repo for my machine" has not installed yet. So the PCI id is read
    too, out of sysfs, which needs nothing but a kernel."""

    def test_the_known_ids_map_to_the_target(self):
        self.assertTrue(hardware.KNOWN_GPUS)
        for pci, (gfx, name) in hardware.KNOWN_GPUS.items():
            self.assertRegex(pci, r"^[0-9a-f]{4}:[0-9a-f]{4}$")
            self.assertEqual(gfx, hardware.TARGET_GFX)
            self.assertIn("Strix Halo", name)

    def test_the_source_is_part_of_the_answer(self):
        """'the card is right but ROCm is missing' and 'ROCm says gfx1151' are
        different states, and a preflight that reported them the same would
        tell a newcomer they were ready when they were not."""
        g = hardware.gpu()
        self.assertIn(g["source"], ("rocm", "pci", "none"))
        self.assertIn("recognised", g)

    def test_every_id_in_the_table_says_where_it_was_seen(self):
        """An id that is not in the table is UNKNOWN, not wrong. Claiming to
        recognise hardware nobody has tested is how a preflight starts lying —
        and the first version of this table did exactly that, carrying a
        "second id" nobody had verified, one hour after the module's own
        docstring forbade it.

        So every entry has to be traceable: the comment above it must name the
        machine it was read from. That is checkable, and an unverifiable id
        cannot pass it by accident.
        """
        src = (REPO / "setup" / "lib" / "hardware.py").read_text(encoding="utf-8")
        table = src[src.index("KNOWN_GPUS = {"):src.index("TARGET_GFX")]
        for pci in hardware.KNOWN_GPUS:
            line = next(l for l in table.splitlines() if pci in l and not l.lstrip().startswith("#"))
            before = table[:table.index(line)].splitlines()
            comments = []
            for l in reversed(before):
                if l.lstrip().startswith("#"):
                    comments.insert(0, l)
                elif l.strip():
                    break
            self.assertTrue(comments, "%s has no provenance comment" % pci)
            self.assertRegex("\n".join(comments), r"verified|seen on",
                             "%s does not say where it was read" % pci)

    def test_the_ram_check_counts_what_the_bios_took(self):
        """MemTotal is what is LEFT. A BIOS holding a Windows-style UMA split
        makes a 128 GB machine report far less, and a preflight reading
        MemTotal alone would turn away the one owner this repo is for. Found
        on 27.08. by asking whether the preflight was actually right."""
        self.assertAlmostEqual(hardware.machine_ram_gib(60.0, 64.0), 124.0, delta=0.01)
        self.assertAlmostEqual(hardware.machine_ram_gib(124.9, 0.5), 125.4, delta=0.01)
        self.assertIsNone(hardware.machine_ram_gib(None, 0.5),
                          "an explicit None must mean 'not known' and not "
                          "'go and measure this machine'")
        self.assertIsNotNone(hardware.machine_ram_gib(),
                             "no arguments must still read this machine")

    def test_a_machine_hidden_behind_a_uma_split_is_still_the_target(self):
        r_small = hardware.machine_ram_gib(58.0, 64.0)
        self.assertGreaterEqual(r_small, hardware.TARGET_RAM_GIB,
                                "a 128 GB machine with a 64 GiB UMA split is "
                                "being turned away")

    def test_a_genuinely_small_machine_is_not_rescued_by_the_arithmetic(self):
        self.assertLess(hardware.machine_ram_gib(60.0, 0.5), hardware.TARGET_RAM_GIB)

    def test_a_large_split_is_reported_as_something_to_undo(self):
        src = (REPO / "setup" / "preflight.sh").read_text(encoding="utf-8")
        self.assertIn("uma_is_large", src)
        self.assertIn("does NOT use UMA", src,
                      "the preflight reports the split without saying that this "
                      "stack cannot use it")

    def test_memory_is_not_re_implemented_here(self):
        """budget.read_machine() owns it — one reader per fact, which is the
        rule the whole day was spent enforcing.

        Checked by looking for the FILE rather than for the word: the first
        version of this searched for "MemTotal" outside the module docstring
        and went red the moment a paragraph explained what MemTotal is.
        """
        src = (REPO / "setup" / "lib" / "hardware.py").read_text(encoding="utf-8")
        self.assertIn("budget.read_machine()", src)
        code = "\n".join(l for l in src.splitlines()
                          if not l.lstrip().startswith("#"))
        self.assertNotIn("/proc/meminfo", code,
                         "hardware.py parses meminfo itself — that is budget.py's")

    def test_the_gtt_cap_comes_from_the_running_kernel(self):
        """The parameter only takes effect at boot, so what is written in a
        config file and what is in force are different questions."""
        src = (REPO / "setup" / "lib" / "hardware.py").read_text(encoding="utf-8")
        self.assertIn("/proc/cmdline", src)

    def test_the_report_answers_both_target_questions(self):
        r = hardware.report()
        self.assertIn("is_target_gpu", r)
        self.assertIn("is_target_ram", r)
        self.assertIsInstance(r["is_target_gpu"], bool)


class TestTheDefectRegistryFinallyReadsItsOwnField(unittest.TestCase):
    """`applies_to: {"gpu": "gfx1151"}` sat on nine of twelve entries and
    nothing read it, so on any other card the registry reported all nine.

    Its own docstring calls that a failure — "a registry that cries about
    qwen4exp while qwen38 serves trains the reader to skip it" — and the
    hardware axis had exactly the bug the model axis was protected from.
    """

    def setUp(self):
        self.defects = defects.load()

    def gpu_scoped(self):
        return [d for d in self.defects
                if (d.get("applies_to") or {}).get("gpu")]

    def test_the_registry_still_scopes_defects_by_gpu(self):
        self.assertTrue(self.gpu_scoped(), "nothing is gpu-scoped any more — "
                                           "then this whole mechanism is dead code")

    def test_a_foreign_gpu_drops_them(self):
        for d in self.gpu_scoped():
            self.assertFalse(defects.applies(d, None, "gfx1100"), d["id"])

    def test_the_target_gpu_keeps_them(self):
        for d in self.gpu_scoped():
            self.assertIsNot(defects.applies(d, None, "gfx1151"), False, d["id"])

    def test_an_unknown_gpu_keeps_them_too(self):
        """Not knowing what you are on is a reason for MORE caution, not less.
        None must not be read as 'does not match'."""
        for d in self.gpu_scoped():
            self.assertIsNot(defects.applies(d, None, None), False, d["id"])

    def test_the_reason_says_which_gpu_it_was_about(self):
        d = self.gpu_scoped()[0]
        verdict, detail = defects.evaluate(d, None, None, "gfx1100")
        self.assertEqual(verdict, defects.NA)
        self.assertIn("gfx1151", detail)
        self.assertIn("gfx1100", detail)

    def test_the_decision_stays_pure(self):
        """The gpu is handed in, not read inside — so a test can present any
        machine, and the same function serves check.sh and the preflight."""
        src = (REPO / "setup" / "lib" / "defects.py").read_text(encoding="utf-8")
        body = src[src.index("def applies("):src.index("def evaluate(")]
        self.assertNotIn("hardware.", body)
        self.assertNotIn("subprocess", body)


class TestThePreflightIsHonestRatherThanHelpful(unittest.TestCase):
    def src(self):
        return (REPO / "setup" / "preflight.sh").read_text(encoding="utf-8")

    def test_it_changes_nothing(self):
        """It runs before install.sh, so it must be safe to run on a machine
        the reader has not decided about yet."""
        for forbidden in ("sudo ", "systemctl ", " > ", "install -m", "sed -i"):
            self.assertNotIn(forbidden, self.src(),
                             "the preflight modifies the system: %r" % forbidden)

    def test_it_names_the_configuration_it_was_measured_on(self):
        src = self.src()
        self.assertIn("128 GB", src)
        self.assertIn("gfx1151", src)

    def test_it_refuses_to_scale(self):
        """The one thing it must not do. A scaled value would be a number
        nobody measured, from a repo whose entire argument is that its numbers
        are measured."""
        src = self.src()
        self.assertIn("does not scale", src.replace("\n", " ").replace("  ", " "))

    def test_it_shows_which_profiles_fit(self):
        self.assertIn("fits_the_machine", self.src())

    def test_a_non_target_machine_gets_a_non_zero_exit(self):
        """So a script can ask, and so `&&` does the right thing."""
        self.assertIn("exit $STATUS", self.src())


class TestScriptsThatPrintDecimalsPinTheLocale(unittest.TestCase):
    """bash's printf parses its ARGUMENTS by LC_NUMERIC, so `printf '%.1f' 8.9`
    fails with "invalid number" in de_DE — while awk, which produced the 8.9,
    always writes a dot.

    setup/scripts/gtt.sh has carried `export LC_ALL=C` for this since 26.08.
    and tests/test_gtt.py pins it there. Generalising the rule on 27.08. found
    two scripts that had never had it — `llmprofile`, which prints telemetry,
    and `bench` sweep.sh — plus the preflight, which was written that morning
    and broke on the first run in a German locale.
    """

    DECIMAL_PRINTF = re.compile(r"printf[^|;\n]*%[-0-9]*\.[0-9]f")

    def scripts(self):
        out = []
        for f in subprocess.run(["git", "ls-files"], cwd=str(REPO),
                                capture_output=True, text=True).stdout.split():
            p = REPO / f
            if not p.is_file():
                continue
            if f.endswith(".sh") or (p.stat().st_mode & 0o111 and "." not in os.path.basename(f)):
                out.append(f)
        return out

    def test_every_script_that_prints_a_decimal_pins_LC_ALL(self):
        missing = []
        for f in self.scripts():
            text = (REPO / f).read_text(encoding="utf-8", errors="replace")
            if not self.DECIMAL_PRINTF.search(text):
                continue
            if not re.search(r"^export LC_ALL=C|LC_ALL=C ", text, re.M):
                missing.append(f)
        self.assertFalse(missing,
                         "these print a decimal with bash printf and do not pin "
                         "LC_ALL=C — they abort in any comma-decimal locale: %s"
                         % ", ".join(missing))

    def test_the_preflight_survives_a_comma_decimal_locale(self):
        """The regression itself, run rather than reasoned about. If the locale
        is not installed bash falls back to C and this passes vacuously — which
        is why the static check above exists as well."""
        env = dict(os.environ, LC_ALL="de_DE.UTF-8", LANG="de_DE.UTF-8")
        r = subprocess.run(["bash", str(REPO / "setup" / "preflight.sh")],
                           capture_output=True, text=True, env=env,
                           cwd=str(REPO), timeout=120)
        self.assertNotIn("invalid number", r.stdout + r.stderr)
        self.assertNotIn("Ungültige Zahl", r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
