"""The guard that stops a measurement from taking the machine down.

On 26.08.2026 the coding battery — since removed — moved from a 13.5 GiB model to a 68 GiB one
while production held 17 GiB. Two sets of weights went into GTT at once, the
kernel squeezed the page cache to 40 MiB, and the OOM killer took llama-server
and the desktop with it.

Two things made it possible and both are worth pinning:

  * GTT comes out of system RAM, is NOT swappable, and does not look like
    process memory. Starting a server that does not fit therefore does not
    page — it hangs the machine.
  * ttm.pages_limit had been raised from 96 to 116 GiB that morning. A higher
    cap is more room AND less protection: at 96 the allocation would have
    failed with a clean error. That second half was not thought through, so it
    is written down here.
"""
import os, sys, tempfile, unittest

import common

sys.path.insert(0, str(common.REPO / "bench"))
sys.path.insert(0, str(common.REPO / "tools"))
import run as runlib                                          # noqa: E402
import sideserver as SIDE                                     # noqa: E402


class TestModelSize(unittest.TestCase):
    """The guard is only as good as its estimate, and underestimating is the
    dangerous direction."""

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, self.d, ignore_errors=True)

    def write(self, name, mib):
        path = os.path.join(self.d, name)
        with open(path, "wb") as f:
            f.truncate(mib * 1024 * 1024)
        return path

    def test_a_single_file(self):
        p = self.write("m.gguf", 100)
        self.assertAlmostEqual(runlib._model_size_gib(["-m", p]), 100 / 1024.0, 3)

    def test_shards_are_all_counted(self):
        """A sharded GGUF names part one and finds the rest. Counting only the
        named part would have said Laguna is 3.6 MB."""
        base = os.path.join(self.d, "L")
        for i in (1, 2, 3):
            self.write("L-0000%d-of-00003.gguf" % i, 100)
        got = runlib._model_size_gib(["-m", base + "-00001-of-00003.gguf"])
        self.assertAlmostEqual(got, 300 / 1024.0, 3)

    def test_the_mmproj_counts_too(self):
        m = self.write("m.gguf", 100)
        v = self.write("mm.gguf", 50)
        self.assertAlmostEqual(runlib._model_size_gib(["-m", m, "--mmproj", v]),
                               150 / 1024.0, 3)

    def test_no_model_flag_means_no_opinion(self):
        self.assertIsNone(runlib._model_size_gib(["--host", "127.0.0.1"]))

    def test_an_unreadable_path_means_no_opinion(self):
        """Refusing on a path we cannot stat would break every caller that
        points at a model this process may not read."""
        self.assertIsNone(runlib._model_size_gib(["-m", "/does/not/exist.gguf"]))


class TestGuard(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, self.d, ignore_errors=True)
        self.saved = {k: getattr(runlib, k) for k in ("_gtt", "_mem_available_gib")}
        self.addCleanup(lambda: [setattr(runlib, k, v) for k, v in self.saved.items()])

    def model(self, gib):
        path = os.path.join(self.d, "m.gguf")
        with open(path, "wb") as f:
            f.truncate(int(gib * 1024 ** 3))
        return ["-m", path]

    def fake(self, total, used, avail):
        runlib._gtt = lambda which: {"total": total, "used": used}[which]
        runlib._mem_available_gib = lambda: avail

    def test_it_fits(self):
        self.fake(total=116.0, used=35.0, avail=81.0)
        runlib.check_room_for(self.model(13.5))        # must not raise

    def test_the_case_that_took_the_machine_down(self):
        """Production holding 35 GiB of GTT, 81 GiB available, and a 68 GiB
        model asked to start alongside it."""
        self.fake(total=116.0, used=35.0, avail=81.0)
        with self.assertRaises(SystemExit) as cm:
            runlib.check_room_for(self.model(68.0), "laguna")
        msg = str(cm.exception)
        self.assertIn("REFUSING TO START laguna", msg)
        self.assertIn("does not fit", msg)

    def test_the_gtt_cap_alone_can_refuse(self):
        """Plenty of RAM available, but the cap is nearly used up — which is
        what the cap is for."""
        self.fake(total=116.0, used=110.0, avail=100.0)
        with self.assertRaises(SystemExit) as cm:
            runlib.check_room_for(self.model(20.0))
        self.assertIn("GTT has", str(cm.exception))

    def test_the_host_reserve_alone_can_refuse(self):
        """The cap has room, but taking it would leave the desktop nothing.
        GTT cannot be swapped, so 'the host will page' is not an option."""
        self.fake(total=116.0, used=5.0, avail=20.0)
        with self.assertRaises(SystemExit) as cm:
            runlib.check_room_for(self.model(40.0))
        self.assertIn("must stay free", str(cm.exception))

    def test_the_override_exists_and_is_explicit(self):
        self.fake(total=116.0, used=110.0, avail=100.0)
        os.environ["BENCH_NO_MEMORY_GUARD"] = "1"
        self.addCleanup(os.environ.pop, "BENCH_NO_MEMORY_GUARD", None)
        runlib.check_room_for(self.model(40.0))       # must not raise

    def test_unreadable_sysfs_does_not_block_everything(self):
        """On a machine without amdgpu the guard has no opinion rather than
        refusing every measurement."""
        runlib._gtt = lambda which: None
        runlib._mem_available_gib = lambda: None
        runlib.check_room_for(self.model(400.0))      # must not raise


class TestStartServerIsGuarded(unittest.TestCase):
    def test_start_server_asks_before_it_starts(self):
        """The guard is worth nothing if a caller can go around it, and all
        six measurement tools go through start_server."""
        src = (common.REPO / "bench" / "run.py").read_text(encoding="utf-8")
        body = src[src.index("def start_server("):]
        self.assertLess(body.index("check_room_for"), body.index("subprocess.Popen"),
                        "start_server must check BEFORE it spawns")


if __name__ == "__main__":
    unittest.main()


class TestTheGuardIsReachable(unittest.TestCase):
    """The guard was never missing. It was SKIPPABLE, and that is worse.

    bench/run.py has had check_room_for() and wait_for_gtt_release() since the
    morning of 26.08., written after a measurement took the machine down. The
    machine went down again the same evening — because the throwaway scripts
    that ran the Flash-Next cells started llama-server directly, called
    neither, and did `kill; sleep 5` instead of waiting for GTT to fall. A
    second 87 GiB model landed on an allocation still being torn down.

    So the test is not "does the guard work". It is "is there one obvious way
    to start a side server that cannot skip it", and that is
    bench/sideserver.py.
    """

    def test_sideserver_uses_both_halves_of_the_guard(self):
        src = (common.REPO / "bench" / "sideserver.py").read_text(encoding="utf-8")
        self.assertIn("check_room_for", src, "a start that does not check the fit")
        self.assertIn("wait_for_gtt_release", src,
                      "a stop that waits for the port, not for the memory")

    def test_it_waits_for_gtt_before_starting_AND_after_stopping(self):
        """Twice, and both matter. Before: the unit it stopped may still hold
        GTT. After: the next thing to run must not land on ours."""
        src = (common.REPO / "bench" / "sideserver.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(src.count("wait_for_gtt_release"), 2)

    def test_the_teardown_is_in_a_finally(self):
        """Production has to come back even when the measurement crashes —
        which is exactly what happened on 26.08."""
        src = (common.REPO / "bench" / "sideserver.py").read_text(encoding="utf-8")
        self.assertIn("finally:", src)
        i = src.index("finally:")
        self.assertIn("systemctl", src[i:], "the unit is not restarted in the finally")


class TestTheMeasuredOverride(unittest.TestCase):
    """There has to be a way past a WRONG estimate that is not a way past the
    guard.

    The estimate reads the file size, which is right for every model here but
    Flash-Next: 103.7 GiB on disk, 87.4 in GTT, because llama.cpp keeps the
    26 GiB n-gram table out of it. So the guard refuses something that fits.

    Without a narrow escape the only escape is BENCH_NO_MEMORY_GUARD=1 — and a
    safety that is inconvenient at the wrong moment gets switched off entirely,
    which is how this machine hung twice in one day.
    """

    def setUp(self):
        self.run = common.load("bench/run.py", "bench_run_ovr")
        for k in ("BENCH_MODEL_GIB", "BENCH_NO_MEMORY_GUARD"):
            os.environ.pop(k, None)

    def tearDown(self):
        for k in ("BENCH_MODEL_GIB", "BENCH_NO_MEMORY_GUARD"):
            os.environ.pop(k, None)

    def test_the_override_is_still_checked_against_the_limits(self):
        """It corrects the INPUT. It does not skip the comparison."""
        os.environ["BENCH_MODEL_GIB"] = "200"
        self.run._model_size_gib = lambda argv: 200.0
        self.run._gtt = lambda w: {"total": 108.0, "used": 5.0}[w]
        self.run._mem_available_gib = lambda: 110.0
        with self.assertRaises(SystemExit):
            self.run.check_room_for(["-m", "/does/not/matter.gguf"], "test")

    def test_an_unreadable_model_is_not_our_call_override_or_not(self):
        """The size on disk is what the HOST check needs, and without the file
        there is no way to know it. Honouring an override here would let a
        stated GTT number stand in for a host number nobody has — which is the
        exact substitution that cost a hang."""
        os.environ["BENCH_MODEL_GIB"] = "10"
        self.run._model_size_gib = lambda argv: None
        self.run._gtt = lambda w: {"total": 108.0, "used": 107.0}[w]
        self.run._mem_available_gib = lambda: 1.0
        self.run.check_room_for(["-m", "/does/not/exist.gguf"], "test")

    def test_the_override_corrects_GTT_and_cannot_lower_the_HOST_check(self):
        """The mistake that cost a third hang, pinned.

        BENCH_MODEL_GIB=88 was passed for Flash-Next because only 87.4 GiB of
        its 103.7 lands in GTT. It silently lowered the host check too — and
        the 26 GiB that is not in GTT does not evaporate, it is page cache for
        the mmap and it is resident. The server started, the machine went to
        100 %, and the kernel OOM-killed it.

        A measurement can tell you what the GPU pins. Nothing makes the bytes
        on disk smaller.
        """
        os.environ["BENCH_MODEL_GIB"] = "88"
        self.run._model_size_gib = lambda argv: 103.7
        # GTT is roomy; the HOST is not. Under the old code 88 * 1.1 = 96.8
        # passed both. The file's 103.7 * 1.1 = 114.1 must not.
        self.run._gtt = lambda w: {"total": 108.0, "used": 1.0}[w]
        self.run._mem_available_gib = lambda: 118.0
        with self.assertRaises(SystemExit) as cm:
            self.run.check_room_for(["-m", "/does/not/matter.gguf"], "test")
        self.assertIn("resident somewhere", str(cm.exception))

    def test_it_still_lets_a_genuinely_fitting_model_through(self):
        os.environ["BENCH_MODEL_GIB"] = "88"
        self.run._model_size_gib = lambda argv: 103.7
        self.run._gtt = lambda w: {"total": 108.0, "used": 1.0}[w]
        self.run._mem_available_gib = lambda: 130.0
        self.run.check_room_for(["-m", "/does/not/matter.gguf"], "test")

    def test_without_the_override_the_file_size_governs_both(self):
        self.run._model_size_gib = lambda argv: 103.7
        self.run._gtt = lambda w: {"total": 108.0, "used": 1.0}[w]
        self.run._mem_available_gib = lambda: 130.0
        with self.assertRaises(SystemExit):
            self.run.check_room_for(["-m", "/does/not/matter.gguf"], "test")

    def test_nonsense_is_refused_rather_than_ignored(self):
        """Silently falling back to the file size would hide a typo in the one
        place where a number is being trusted instead of measured."""
        os.environ["BENCH_MODEL_GIB"] = "eighty-eight"
        self.run._model_size_gib = lambda argv: 50.0
        with self.assertRaises(SystemExit):
            self.run.check_room_for(["-m", "/does/not/matter.gguf"], "test")


class TestTheCeilingIsDerived(unittest.TestCase):
    """The transient unit's MemoryMax, and why a flat number was never one.

    It defaulted to `100G` until 27.08. That day Flash-Next was measured while
    it ran: 80 GiB pinned in GTT and 29.6 GiB of ANONYMOUS host memory beside
    it (RssAnon 27.1, Private_Dirty 28.1, no file mapping at all), on a machine
    with 124.9. Put the profile's own `-cram 32768` back — 32 GiB of RAM prompt
    cache — and the host side may reach ~57 GiB and the total ~137, which the
    machine does not have. **MemoryMax=100G would not have stopped that**,
    because 57 is less than 100. The ceiling sat above the cliff.

    So it is derived: RAM, minus what the model will PIN in GTT, minus a share
    for the host. GTT is not charged to the cgroup (verified the same day:
    qwen38 shows MemoryCurrent 32.9 GiB while holding 35.7 GiB of GTT), so the
    limit governs exactly the half that can take the machine down.
    """

    FLASHNEXT = ["-m", "/models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"]

    def setUp(self):
        os.environ.pop("BENCH_MODEL_GIB", None)

    tearDown = setUp

    # The machine is STATED rather than read. What ceiling() derives is a
    # fraction of the RAM present, so every assertion below was really an
    # assertion about the machine the suite ran on — invisible here, where it
    # is 124.9 GiB, and fatal on the 7.8 GiB runner the first CI run used.
    MACHINE = 124.9

    def test_a_big_model_gets_a_small_ceiling(self):
        os.environ["BENCH_MODEL_GIB"] = "81"
        mx, hi, why = SIDE.ceiling(argv=self.FLASHNEXT, total_gib=self.MACHINE)
        self.assertLess(int(mx.rstrip("G")), 40,
                        "80 GiB in GTT leaves nowhere near 100 GiB for the host")
        self.assertLess(int(hi.rstrip("G")), int(mx.rstrip("G")) + 1)
        self.assertIn("GTT", why)

    def test_a_small_model_is_not_strangled(self):
        os.environ["BENCH_MODEL_GIB"] = "17"
        mx, _, _ = SIDE.ceiling(argv=["-m", "/models/qwen38.gguf"],
                                total_gib=self.MACHINE)
        self.assertGreater(int(mx.rstrip("G")), 80)

    def test_an_explicit_value_always_wins(self):
        """A caller who has measured knows more than this arithmetic does."""
        os.environ["BENCH_MODEL_GIB"] = "81"
        self.assertEqual(SIDE.ceiling("40G", argv=self.FLASHNEXT)[0], "40G")

    def test_the_expected_gtt_is_used_and_not_the_live_one(self):
        """sideserver computes this AFTER waiting for GTT to fall, so a live
        reading would say ~0 and derive the same useless ceiling the flat
        default was."""
        os.environ["BENCH_MODEL_GIB"] = "81"
        self.assertEqual(SIDE.expected_gtt_gib(self.FLASHNEXT), 81.0)

    def test_memtotal_is_read_by_name_and_not_by_position(self):
        """/proc/meminfo's first token is the LABEL. Reading it positionally
        returned None and the ceiling silently fell back to its low default —
        which looks like caution and is actually a broken reader."""
        self.assertIsNotNone(SIDE._memtotal_gib())
        self.assertGreater(SIDE._memtotal_gib(), 1.0)

    def test_it_never_returns_a_ceiling_below_the_floor(self):
        """An absurd estimate must not produce a 0G ceiling that refuses
        everything — that is a different failure from the one being prevented."""
        os.environ["BENCH_MODEL_GIB"] = "999"
        mx, _, _ = SIDE.ceiling(argv=self.FLASHNEXT)
        self.assertGreaterEqual(int(mx.rstrip("G")), 8)


class TestTheWatchdogGoesDownWithProduction(unittest.TestCase):
    """A detector that cries wolf on every measurement gets ignored.

    llama-probe.timer asks the production server one question every ten
    minutes and reports a failure when nothing answers. sideserver stops
    production by design — so on 27.08., the first measurement run after the
    timer was armed left "UNREACHABLE ConnectionRefused" in the journal and a
    red line in check.sh. Nothing was wrong; the watchdog was doing its job
    against a server that had been stopped on purpose.

    It matters because this detector exists for the SILENT faults. A false
    alarm on every measurement is exactly how it stops being read.
    """

    def src(self):
        return (common.REPO / "bench" / "sideserver.py").read_text(encoding="utf-8")

    def test_the_probe_timer_is_stopped_with_production(self):
        self.assertIn('systemctl("stop", PROBE_TIMER)', self.src())

    def test_and_started_again_in_the_teardown(self):
        self.assertIn('systemctl("start", PROBE_TIMER)', self.src())

    def test_the_dead_man_switch_restores_it_too(self):
        """The half that matters. A `finally` does not run when the OOM killer
        takes this process — that is the 26.08. incident — so the watchdog
        must come back from the timer systemd owns, not from here."""
        src = self.src()
        block = src[src.index("--on-active="):src.index("--on-active=") + 400]
        self.assertIn("PROBE_TIMER", block,
                      "a killed sideserver would leave the watchdog off, which "
                      "is worse than the false alarm it prevents")


class TestSideserverAfterTheThirdIncident(unittest.TestCase):
    """23:11 on 26.08. The guard had been made unskippable that evening and the
    machine still went to 100 %. Two things the first two incidents had hidden:

    THE LIMIT DID NOT APPLY. A server started from a script inherits the
    CALLER's cgroup, and the kernel log named it:
    task_memcg=/user.slice/.../app-com.anthropic.Claude-13190.scope. So
    MemoryMax=108G on llama-user@.service guarded the service and nothing
    else. What actually stopped it was a global OOM.

    THE TEARDOWN DID NOT RUN. The OOM killer took the orchestrator with
    SIGKILL, `finally` never executed, and production stayed down until a
    human noticed. A `finally` is not a guarantee.
    """

    def src(self):
        return (common.REPO / "bench" / "sideserver.py").read_text(encoding="utf-8")

    def test_the_server_gets_its_own_cgroup_and_ceiling(self):
        s = self.src()
        self.assertIn("systemd-run", s, "a bare child inherits the caller's cgroup")
        self.assertIn("MemoryMax=", s)
        self.assertIn("MemoryHigh=", s)

    def test_the_dead_mans_switch_is_armed_BEFORE_production_is_stopped(self):
        """Order is the whole point. Armed after the stop, a crash in between
        leaves the machine with no model and no timer."""
        s = self.src()
        armed = s.index("on-active")
        stopped = s.index('systemctl("stop", a.stop)')
        self.assertLess(armed, stopped,
                        "the switch is armed after production is already down")

    def test_the_teardown_stops_the_unit_by_name(self):
        """A named unit can be stopped by anybody, including somebody cleaning
        up after a process that is no longer there. os.killpg on a handle this
        process no longer holds cannot."""
        s = self.src()
        self.assertIn('systemctl("stop", unit)', s)
        self.assertNotIn("killpg", s, "a pid-based teardown dies with its owner")


class TestTheHostOverrideHasAFloor(unittest.TestCase):
    """A measured host footprint may correct the crude `file * 1.10` estimate,
    and may never claim less than the file.

    The default is deliberately crude — 10 % for KV and buffers, about right
    for most models here and 9 GiB too high for Flash-Next, whose KV at the
    served window is 2.3 GiB. That 9 GiB is the difference between measuring
    this hardware and not being able to.

    The floor is the whole reason it is safe to have. BENCH_MODEL_GIB was used
    once to claim 88 GiB for a model with 103.7 GiB on disk, and the machine
    went to 100 %."""

    def setUp(self):
        self.run = common.load("bench/run.py", "bench_run_host")
        for k in ("BENCH_MODEL_GIB", "BENCH_HOST_GIB", "BENCH_NO_MEMORY_GUARD"):
            os.environ.pop(k, None)
        self.run._model_size_gib = lambda argv: 103.7
        self.run._gtt = lambda w: {"total": 108.0, "used": 0.5}[w]
        self.run._mem_available_gib = lambda: 118.1

    def tearDown(self):
        for k in ("BENCH_MODEL_GIB", "BENCH_HOST_GIB", "BENCH_NO_MEMORY_GUARD"):
            os.environ.pop(k, None)

    def test_below_the_file_size_is_refused_by_name(self):
        os.environ["BENCH_HOST_GIB"] = "88"
        with self.assertRaises(SystemExit) as cm:
            self.run.check_room_for(["-m", "/x.gguf"], "test")
        self.assertIn("below the", str(cm.exception))
        self.assertIn("on disk", str(cm.exception))

    def test_a_measured_footprint_above_the_floor_is_accepted(self):
        os.environ["BENCH_MODEL_GIB"] = "81"     # what lands in GTT
        os.environ["BENCH_HOST_GIB"] = "106"     # GTT + the anonymous rest
        self.run.check_room_for(["-m", "/x.gguf"], "test")

    def test_it_is_still_checked_against_what_the_host_has(self):
        os.environ["BENCH_MODEL_GIB"] = "81"
        os.environ["BENCH_HOST_GIB"] = "115"
        with self.assertRaises(SystemExit):
            self.run.check_room_for(["-m", "/x.gguf"], "test")

    def test_the_crude_default_still_governs_when_nobody_measured(self):
        with self.assertRaises(SystemExit):
            self.run.check_room_for(["-m", "/x.gguf"], "test")
