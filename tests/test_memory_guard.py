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
from unittest import mock

import common

sys.path.insert(0, str(common.REPO / "bench"))
sys.path.insert(0, str(common.REPO / "tools"))
sys.path.insert(0, str(common.REPO / "setup" / "lib"))
import run as runlib                                          # noqa: E402
import sideserver as SIDE                                     # noqa: E402
import budget                                                 # noqa: E402


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
        GTT. After: the next thing to run must not land on ours.

        The two waits ask DIFFERENT questions and are two functions since
        28.08.2026 — before, there is no baseline to fall back to, so the
        question is whether the teardown has finished; after, this process
        measured the baseline itself. Counted together, because what has to
        hold is that both moments are waited for, not which name does it.
        """
        src = (common.REPO / "bench" / "sideserver.py").read_text(encoding="utf-8")
        waits = (src.count("runlib.wait_for_gtt_release")
                 + src.count("runlib.wait_for_gtt_to_settle"))
        self.assertGreaterEqual(waits, 2)

    def test_the_teardown_is_in_a_finally(self):
        """Production has to come back even when the measurement crashes —
        which is exactly what happened on 26.08."""
        src = (common.REPO / "bench" / "sideserver.py").read_text(encoding="utf-8")
        self.assertIn("finally:", src)
        i = src.index("finally:")
        self.assertIn("systemctl", src[i:], "the unit is not restarted in the finally")


class TestWaitingForGttAsksAnAnswerableQuestion(unittest.TestCase):
    """A guard that cannot be satisfied refuses everything, silently.

    sideserver waited for GTT to fall below 1 GiB before starting a side
    server — wait_for_gtt_release(0.0), whose tolerance is +1.0. The desktop
    on this machine holds 1.5 GiB and never gives it back, so the condition
    was unmeetable while anyone was logged in. Measured 28.08.2026: three
    starts, production already stopped, GTT already at 1.5, all three refused
    after the full 180 s. Nothing was wrong with the memory and nothing was
    wrong with the refusal path — the question had no answer on this machine
    and the default answer was no.
    """

    def setUp(self):
        self.saved = runlib._gtt
        self.addCleanup(lambda: setattr(runlib, "_gtt", self.saved))
        self.slept = []
        p = mock.patch("time.sleep", self.slept.append)
        p.start(); self.addCleanup(p.stop)

    def readings(self, values):
        """Hand out `values` one per call, then repeat the last forever."""
        seq = list(values)
        def _gtt(which):
            if which != "used":
                return 108.0
            return seq.pop(0) if len(seq) > 1 else seq[0]
        runlib._gtt = _gtt

    def test_a_desktop_that_keeps_its_gtt_no_longer_blocks_a_start(self):
        """The regression itself: falls to the desktop floor and stays."""
        self.readings([35.0, 12.0, 1.5])
        self.assertEqual(runlib.wait_for_gtt_to_settle(timeout=60, quiet_s=0.0),
                         1.5)

    def test_it_waits_while_the_teardown_is_still_running(self):
        """Still falling is exactly the case the wait exists for — the one
        that took the machine down on 26.08."""
        falling = [80.0 - i for i in range(400)]
        self.readings(falling)
        with mock.patch("time.time", side_effect=[0.0] + [float(i) for i in range(1, 400)]):
            self.assertIsNone(runlib.wait_for_gtt_to_settle(timeout=60))

    def test_an_unreadable_gtt_does_not_block_a_measurement(self):
        """A machine without amdgpu cannot answer this, and refusing there
        would make the guard a portability bug. Same answer
        wait_for_gtt_release gives."""
        runlib._gtt = lambda which: None
        self.assertEqual(runlib.wait_for_gtt_to_settle(timeout=60), 0.0)

    def test_the_settled_value_is_not_judged_here(self):
        """It returns a large steady reading rather than refusing: whether
        there is ROOM is check_room_for's question, and it has the profile and
        the machine to answer it with. Two questions, two functions."""
        self.readings([40.0, 40.0, 40.0])
        self.assertEqual(runlib.wait_for_gtt_to_settle(timeout=60, quiet_s=0.0),
                         40.0)


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
        leaves the machine with no model and no timer.

        Since the workload path landed (01.09.2026) the arm and the stop live
        in shared helpers, so the order is proven at every CALL SITE — a
        linear-source index would compare function definitions, which say
        nothing about who calls whom first."""
        import re
        s = self.src()
        self.assertIn("on-active", s, "nothing arms a systemd timer at all")
        arms = [m.start() for m in
                re.finditer(r"(?<!def )arm_deadman\(deadman", s)]
        stops = [m.start() for m in
                 re.finditer(r"stop_production_and_settle\(a\.stop\)", s)]
        self.assertTrue(arms, "no call site arms the switch")
        self.assertEqual(len(arms), len(stops),
                         "a caller stops production without arming the switch")
        for armed, stopped in zip(arms, stops):
            self.assertLess(armed, stopped,
                            "the switch is armed after production is already "
                            "down")

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


class TestDemandPagedTensors(unittest.TestCase):
    """Every number in budget.py rested on one sentence: all of a model's bytes
    have to be resident somewhere. verdict() says it out loud, the LLM_HOST_GIB
    floor enforces it, and it was simply TRUE for every model this repo has
    served.

    llama.cpp #27794 (master, 27.08.2026) ended that. `--tensor-read-lazy`
    reads the rows of a tagged tensor from the mapping on demand, so for
    Qwen3.8-Flash-Next 26.82 of 103.7 GiB need never be resident at all.

    What is tested here is not the relief. It is the three locks on it — the
    dangerous direction is granting relief a build cannot deliver, which is the
    26.08. claim wearing a new name.
    """

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, self.d, ignore_errors=True)

    def profile(self, **fields):
        path = os.path.join(self.d, "p.env")
        with open(path, "w") as f:
            for k, v in fields.items():
                f.write("%s=%s\n" % (k, v))
        return path

    def binary(self, name="llama-server"):
        path = os.path.join(self.d, name)
        with open(path, "w") as f:
            f.write("#!/bin/sh\n")
        os.chmod(path, 0o755)
        return path

    def test_a_profile_that_says_nothing_gets_nothing(self):
        b = self.binary()
        env = self.profile(MODEL_GTT_BASE_GIB=78.1, LLAMA_BIN=b)
        self.assertEqual(budget.lazy_relief(env, [], b), 0.0)

    def test_a_different_build_gets_nothing(self):
        """The dangerous one, and the reason it is not "does the binary know
        the flag": on 28.08. three builds all knew it and only one delivered.
        b10665-1 read the table into 27.13 GiB of anonymous memory while
        answering --help exactly like the build that did not."""
        env = self.profile(MODEL_LAZY_GIB=26.82, LLAMA_BIN=self.binary("measured-on"))
        self.assertEqual(budget.lazy_relief(env, [], self.binary("something-else")), 0.0)

    def test_a_profile_that_pins_no_build_gets_nothing(self):
        """A figure with no build attached is an assertion about nothing."""
        env = self.profile(MODEL_LAZY_GIB=26.82)
        self.assertEqual(budget.lazy_relief(env, [], self.binary()), 0.0)

    def test_turning_it_off_in_the_ENVIRONMENT_gets_nothing_either(self):
        """llama.cpp registers this option with
        .set_env("LLAMA_ARG_TENSOR_READ_LAZY") — common/arg.cpp:2743 — so a
        systemd Environment= line switches it off exactly as an argument does.
        Checking argv alone made the guard asymmetric where llama.cpp is not."""
        env = self.profile(MODEL_LAZY_GIB=26.82, LLAMA_BIN=self.binary())
        os.environ["LLAMA_ARG_TENSOR_READ_LAZY"] = "off"
        self.addCleanup(os.environ.pop, "LLAMA_ARG_TENSOR_READ_LAZY", None)
        self.assertEqual(budget.lazy_relief(env, [], self.binary()), 0.0)

    def test_turning_it_off_on_the_command_line_gets_nothing(self):
        """--tensor-read-lazy off restores the old behaviour, so it has to
        restore the old arithmetic with it."""
        b = self.binary()
        env = self.profile(MODEL_LAZY_GIB=26.82, LLAMA_BIN=b)
        argv = ["-m", "x.gguf", "--tensor-read-lazy", "off"]
        self.assertEqual(budget.lazy_relief(env, argv, b), 0.0)

    def test_no_binary_named_gets_nothing(self):
        """A caller that cannot say which build will run cannot be told yes.
        Unknown must not read as granted."""
        env = self.profile(MODEL_LAZY_GIB=26.82, LLAMA_BIN=self.binary())
        self.assertEqual(budget.lazy_relief(env, [], None), 0.0)

    def test_the_observed_build_grants_it(self):
        b = self.binary()
        env = self.profile(MODEL_LAZY_GIB=26.82, LLAMA_BIN=b)
        self.assertAlmostEqual(budget.lazy_relief(env, ["-m", "x.gguf"], b), 26.82, 2)

    def test_a_symlink_does_not_pass_as_the_build_behind_it(self):
        """Paths are compared resolved. A symlink that has been repointed
        since the measurement is a different build wearing the same name —
        which is what the profile's pin was written against in the first
        place."""
        real = self.binary("real-build")
        link = os.path.join(self.d, "stable")
        os.symlink(real, link)
        env = self.profile(MODEL_LAZY_GIB=26.82, LLAMA_BIN=link)
        self.assertAlmostEqual(budget.lazy_relief(env, [], real), 26.82, 2)
        os.remove(link)
        os.symlink(self.binary("other-build"), link)
        self.assertEqual(budget.lazy_relief(env, [], real), 0.0)

    def test_auto_is_not_off(self):
        """Only the literal 'off' withdraws it; 'auto' and 'on' both read
        lazily. Matching the flag NAME rather than its value would have made
        the profile's own default look like a refusal."""
        b = self.binary()
        env = self.profile(MODEL_LAZY_GIB=26.82, LLAMA_BIN=b)
        for mode in ("auto", "on"):
            self.assertAlmostEqual(
                budget.lazy_relief(env, ["--tensor-read-lazy", mode], b), 26.82, 2, mode)

    def test_the_plan_subtracts_it_and_shows_it(self):
        """Shown as its own line, because the two figures have different
        provenance: host_anon was MEASURED on a build that loaded the table,
        and what moved it is a property of the new build, not of the model."""
        p = budget.plan([], 103.7, gtt_base=78.1, host_anon=27.1, declared=37.3,
                        lazy=26.82)
        lines = [i for i in p.items if "demand" in i[0]]
        self.assertEqual(len(lines), 1, [i[0] for i in p.items])
        self.assertAlmostEqual(lines[0][1], -26.82, 2)
        self.assertAlmostEqual(p.host_gib, 78.1 + 27.1 - 26.82, 1)

    def test_it_can_never_subtract_more_than_is_there(self):
        """A profile declaring more lazy bytes than the model holds outside GTT
        must not push the host figure below what is left of it."""
        p = budget.plan([], 103.7, gtt_base=78.1, host_anon=5.0, lazy=99.0)
        self.assertGreaterEqual(p.host_gib, 78.1)

    def test_the_host_floor_moves_by_the_relief_and_no_further(self):
        """LLM_HOST_GIB may not claim a model is smaller than its bytes. With
        demand paging the floor is the RESIDENT bytes instead — which is lower,
        but by exactly what lazy_relief() vouched for."""
        os.environ["LLM_HOST_GIB"] = "80"
        self.addCleanup(os.environ.pop, "LLM_HOST_GIB", None)
        budget.plan([], 103.7, gtt_base=78.1, lazy=26.82)     # 76.9 floor: fine
        with self.assertRaises(SystemExit) as cm:
            budget.plan([], 103.7, gtt_base=78.1, lazy=0.0)   # 103.7 floor: not
        self.assertIn("26.08", str(cm.exception))


class TestTheProfileReachesTheGuard(unittest.TestCase):
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

    """The measurements existed, were documented, and never arrived.

    budget.plan() takes `declared`, `gtt_base` and `host_anon`, and its
    docstring names Qwen3.8-Flash-Next as the reason all three exist.
    check_room_for() passed none of them. So on 28.08. the one model they were
    written for was refused on the estimate they were meant to replace: 122.9
    GiB of GTT and 126.9 resident, against 80.7 and 111.8 from its own profile.

    A signature can be widened again by accident, so this is pinned at the call
    sites rather than only in the arithmetic.
    """

    def test_the_guard_receives_the_profile_s_measurements(self):
        """A SPY, not a grep. The version before this asserted that the string
        "declared_kv" appeared in check_room_for's source — which would pass if
        the name occurred only in a comment, and would fail on a rename that
        changed nothing. What matters is the arguments budget.plan() actually
        gets."""
        env = os.path.join(self.d, "p.env")
        with open(env, "w") as f:
            f.write("MODEL_KV_KIB_PER_TOKEN=37.3\nMODEL_GTT_BASE_GIB=78.1\n"
                    "MODEL_HOST_ANON_GIB=27.1\n")
        seen = {}
        real = budget.plan

        def spy(argv, weights, **kw):
            seen.update(kw)
            return real(argv, weights, **kw)

        self.fake(total=116.0, used=0.0, avail=100.0)
        with mock.patch.object(budget, "plan", spy):
            try:
                runlib.check_room_for(self.model(1.0), "x", env=env)
            except SystemExit:
                pass
        self.assertAlmostEqual(seen.get("declared"), 37.3)
        self.assertAlmostEqual(seen.get("gtt_base"), 78.1)
        self.assertAlmostEqual(seen.get("host_anon"), 27.1)

    def test_without_a_profile_it_falls_back_to_the_file_size(self):
        """The old behaviour has to survive: most callers have no profile."""
        seen = {}
        real = budget.plan

        def spy(argv, weights, **kw):
            seen.update(kw)
            return real(argv, weights, **kw)

        self.fake(total=116.0, used=0.0, avail=100.0)
        with mock.patch.object(budget, "plan", spy):
            runlib.check_room_for(self.model(1.0), "x")
        self.assertIsNone(seen.get("gtt_base"))

    def test_both_call_sites_hand_over_the_profile(self):
        """start_server and sideserver are the only two ways in, and a guard
        that is merely ABLE to read the profile is not one that does."""
        for f in ("run.py", "sideserver.py"):
            src = (common.REPO / "bench" / f).read_text(encoding="utf-8")
            i = src.index("check_room_for(argv")
            call = src[i:i + 200]
            self.assertIn("env=", call, "%s calls the guard without the profile" % f)
            self.assertIn("binary=", call, "%s cannot say which build runs" % f)

    def test_the_ceiling_reads_it_too(self):
        """The same blindness, failing the other way: from the file size the
        derived MemoryMax for Flash-Next is 8G — under the clamp — for a server
        whose host side is 31 GiB. The cgroup would have killed it on start."""
        src = (common.REPO / "bench" / "sideserver.py").read_text(encoding="utf-8")
        body = src[src.index("def expected_gtt_gib("):src.index("def ceiling(")]
        self.assertIn("declared_gtt", body)
        self.assertIn("declared_kv", body)


if __name__ == "__main__":
    unittest.main()
