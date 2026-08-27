"""setup/lib/budget.py — the one memory budget, and the failures it exists for.

Three times on 26.08.2026 this machine stopped responding, and not once did
anything crash. GTT comes out of system RAM, is pinned, and is not swappable:
a model that does not fit does not page and does not get OOM-killed. The
desktop simply stops.

Until 27.08. the arithmetic that would have caught it lived in three places
with three different formulas — bench/run.py charged weights x 1.10 with a
host reserve of 10, bench/sideserver.py used 12 for the same machine, and
tests/test_models.py added -cram but no KV at all — and none of the three sat
where a model actually gets started. These tests pin the one that replaced
them, and the cases below are the real ones, with the real numbers.
"""
import os, sys, unittest

import common

sys.path.insert(0, str(common.REPO / "setup" / "lib"))
import budget                                             # noqa: E402

KNOBS = ("LLM_MODEL_GIB", "LLM_HOST_GIB", "LLM_KV_KIB_PER_TOKEN",
         "LLM_HOST_RESERVE_GIB", "LLM_NO_MEMORY_GUARD",
         "BENCH_MODEL_GIB", "BENCH_HOST_GIB", "BENCH_KV_KIB_PER_TOKEN",
         "BENCH_HOST_RESERVE_GIB", "BENCH_NO_MEMORY_GUARD")


class Base(unittest.TestCase):
    def setUp(self):
        self.saved = {k: os.environ.pop(k, None) for k in KNOBS}

    def tearDown(self):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def machine(self, total=124.9, avail=110.0, gtt_total=108.0, gtt_used=0.5):
        return budget.Machine(total, avail, gtt_total, gtt_used)

    def argv(self, ctx=None, cram=None, ctk=None):
        a = ["-m", "/models/x.gguf"]
        if ctx:
            a += ["-c", str(ctx)]
        if cram:
            a += ["-cram", str(cram)]
        if ctk:
            a += ["-ctk", ctk, "-ctv", ctk]
        return a


class TestTheArithmetic(Base):
    def test_gtt_is_weights_plus_kv_plus_measured_buffers(self):
        """Not `x 1.10`. Nine recorded measurements put the buffer term at
        3.1-4.6 GiB and show it barely moving when the KV triples — it is
        roughly constant, not proportional, so a percentage under-predicted
        every one of those points."""
        p = budget.plan(self.argv(ctx=204800), 17.6, declared=74.3)
        self.assertAlmostEqual(p.gtt_gib, 17.6 + 14.51 + budget.BUFFER_FLOOR_GIB,
                               delta=0.1)

    def test_the_floor_is_above_every_measurement_it_was_derived_from(self):
        """3.1 ROCm, 3.5 production, 4.6 Vulkan. A floor at or below the worst
        of them would be a guard calibrated to fail on the next Vulkan run."""
        for measured in (3.1, 3.5, 3.6, 4.6):
            self.assertGreater(budget.BUFFER_FLOOR_GIB, measured)

    def test_the_percentage_takes_over_only_far_above_what_was_measured(self):
        """A constant is the optimistic answer for a model much larger than
        anything here, so the fraction is kept as an upper branch."""
        self.assertEqual(budget.buffers_gib(20.0), budget.BUFFER_FLOOR_GIB)
        self.assertEqual(budget.buffers_gib(120.0), 12.0)
        self.assertGreater(budget.buffers_gib(1000.0), budget.BUFFER_FLOOR_GIB)

    def test_host_adds_the_ram_prompt_cache_on_top_of_gtt(self):
        """GTT is system RAM too, so it is the FIRST term of the host side and
        not a separate column. Conflating the two cost a hang on 26.08."""
        p = budget.plan(self.argv(ctx=204800, cram=32768), 17.6, declared=74.3)
        self.assertAlmostEqual(p.host_gib, p.gtt_gib + 32.0, delta=0.01)

    def test_cram_is_megabytes(self):
        """-cram 32768 is 32 GiB, not 32768. A factor of 1024 in the flag that
        was copied unread into five profiles."""
        self.assertEqual(budget.cram_gib(self.argv(cram=32768)), 32.0)
        self.assertEqual(budget.cram_gib(self.argv()), 0.0)

    def test_the_window_and_not_the_slot_count_decides_the_kv(self):
        """-c is the TOTAL number of cells; -np divides it among slots rather
        than multiplying the cost."""
        one = budget.kv_gib(["-c", "131072", "-np", "1"], 96.0)[0]
        four = budget.kv_gib(["-c", "131072", "-np", "4"], 96.0)[0]
        self.assertEqual(one, four)

    def test_no_window_means_no_kv_term(self):
        self.assertEqual(budget.kv_gib(["-m", "/x.gguf"], 96.0), (0.0, "no -c"))


class TestTheProfileMayCorrectTheEstimate(Base):
    """The two measurements a profile can carry, and the one that must not be
    charged slack."""

    def test_a_measured_gtt_base_is_not_charged_the_slack_again(self):
        """MODEL_GTT_BASE_GIB is an OBSERVATION of GTT and already contains
        the compute buffers. Slacking it double-counts them — 8 GiB on
        Flash-Next, which is the difference between 'tight' and 'refused'."""
        p = budget.plan(self.argv(ctx=65536), 103.7, declared=37.3, gtt_base=78.1)
        kv = 65536 * 37.3 / 1048576.0
        self.assertAlmostEqual(p.gtt_gib, 78.1 + kv * 1.10, delta=0.01)
        self.assertLess(p.gtt_gib, (78.1 + kv) * 1.10)

    def test_a_measured_anon_figure_beats_the_derivation(self):
        """file - gtt_base is 103.7 - 78.1 = 25.6, and the measurement is
        27.1: the derivation is LOW, because gtt_base carries the buffers.
        Low is the dangerous direction, so a measured figure wins."""
        derived = budget.plan(self.argv(ctx=65536), 103.7, 37.3, gtt_base=78.1)
        measured = budget.plan(self.argv(ctx=65536), 103.7, 37.3,
                               gtt_base=78.1, host_anon=27.1)
        self.assertGreater(measured.host_gib, derived.host_gib)
        self.assertAlmostEqual(measured.host_gib - derived.host_gib,
                               27.1 - (103.7 - 78.1), delta=0.01)

    def test_the_real_flashnext_profile_is_tight_and_fits(self):
        """Its own MODEL_TITLE says 'tight on a 124.9 GiB machine'. The
        arithmetic has to agree with the sentence, or one of them is wrong."""
        p = budget.plan(self.argv(ctx=65536, cram=4096), 103.7, 37.3,
                        gtt_base=78.1, host_anon=27.1)
        self.assertTrue(budget.fits_the_machine(p, self.machine()))
        self.assertGreater(p.host_gib + 12.0, 120.0, "not tight at all, then")


class TestWhereTheKvNumberComesFrom(Base):
    def test_declared_beats_estimated(self):
        self.assertEqual(budget.kv_kib_per_token([], 74.3), (74.3, "declared"))

    def test_an_override_beats_the_declaration(self):
        os.environ["LLM_KV_KIB_PER_TOKEN"] = "50"
        self.assertEqual(budget.kv_kib_per_token([], 74.3), (50.0, "stated"))

    def test_without_one_it_estimates_and_says_so(self):
        v, src = budget.kv_kib_per_token(["-c", "1024"])
        self.assertEqual(src, "estimated")
        self.assertEqual(v, budget.ESTIMATE_KV_KIB_PER_TOKEN)

    def test_the_estimate_follows_the_cache_type(self):
        """A q8_0 cache really is half of f16, and that is physics rather than
        optimism. An UNRECOGNISED type keeps the f16 figure — not knowing what
        something is, is not a licence to charge less for it."""
        f16 = budget.kv_kib_per_token(["-ctk", "f16", "-ctv", "f16"])[0]
        q8 = budget.kv_kib_per_token(["-ctk", "q8_0", "-ctv", "q8_0"])[0]
        odd = budget.kv_kib_per_token(["-ctk", "q3_banana", "-ctv", "q3_banana"])[0]
        self.assertAlmostEqual(q8, f16 / 2.0)
        self.assertEqual(odd, f16)

    def test_an_estimated_plan_is_flagged_as_estimated(self):
        self.assertTrue(budget.plan(self.argv(ctx=1024), 10.0).estimated)
        self.assertFalse(budget.plan(self.argv(ctx=1024), 10.0, 74.3).estimated)

    def test_the_verdict_says_out_loud_that_it_is_estimating(self):
        p = budget.plan(self.argv(ctx=1024), 10.0)
        v = budget.verdict(p, self.machine())
        self.assertTrue(any("ESTIMATE" in n for n in v.notes),
                        "an estimate that does not announce itself is how "
                        "-cram 32768 got into five profiles")




class TestTheObservationIsTheServersOwnGtt(Base):
    """The observation used to be SYSTEM-WIDE, and that was the open item.

    check.sh holds the prediction against what the machine actually pinned,
    and only the under-predicting direction is a defect. Reading GTT from
    mem_info_gtt_used counts the desktop in it, so an under-prediction of two
    or three GiB — exactly the size that decides whether a start fits — was
    absorbed by whatever the compositor happened to be holding, and the check
    reported "not under-predicting" either way.

    Measured 27.08. while qwen38 served: 35.64 GiB system-wide against 34.78
    the process itself. 0.86 GiB of slack on an IDLE desktop, and a busy one
    hides more.

    amdgpu accounts per DRM client in /proc/PID/fdinfo, so the server's own
    figure is available on this kernel. It is not available on every kernel,
    and the fallback is the point of half these tests: coarse-and-labelled
    beats sharp-and-silently-absent.
    """

    def proc(self, fdinfo=None, cmd=b"/usr/bin/llama-server\0-c\0128\0"):
        """A /proc with one llama-server in it. fdinfo maps fd name -> text."""
        import tempfile
        d = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, d, True)
        pid = os.path.join(d, "4242")
        os.makedirs(os.path.join(pid, "fdinfo"))
        with open(os.path.join(pid, "cmdline"), "wb") as fh:
            fh.write(cmd)
        with open(os.path.join(pid, "status"), "w") as fh:
            fh.write("Name:\tllama-server\nRssAnon:\t1048576 kB\n")
        for name, text in (fdinfo or {}).items():
            with open(os.path.join(pid, "fdinfo", name), "w") as fh:
                fh.write(text)
        return d

    @staticmethod
    def client(cid, gtt_kib):
        return ("drm-driver:\tamdgpu\n"
                "drm-client-id:\t%s\n"
                "drm-total-gtt:\t%d KiB\n"
                "drm-resident-gtt:\t%d KiB\n" % (cid, gtt_kib + 99, gtt_kib))

    def test_it_reads_the_servers_own_gtt(self):
        d = self.proc({"7": self.client(2160, 36474596)})
        self.assertAlmostEqual(budget.server_gtt_gib(d), 34.78, delta=0.01)

    def test_two_fds_of_one_client_are_not_counted_twice(self):
        """A process may hold the card node and the render node open. Both
        fdinfo files report the SAME allocation under the same client id, and
        summing the files instead of the clients doubles the model."""
        one = self.client(2160, 36474596)
        d = self.proc({"7": one, "8": one})
        self.assertAlmostEqual(budget.server_gtt_gib(d), 34.78, delta=0.01)

    def test_two_distinct_clients_do_add_up(self):
        d = self.proc({"7": self.client(1, 1048576),
                       "8": self.client(2, 2097152)})
        self.assertAlmostEqual(budget.server_gtt_gib(d), 3.0, delta=0.01)

    def test_resident_is_the_quantity_not_total(self):
        """drm-total-gtt counts what was allocated; drm-resident-gtt counts
        what is occupying memory now, which is what the guard predicts."""
        d = self.proc({"7": self.client(1, 1048576)})
        self.assertAlmostEqual(budget.server_gtt_gib(d), 1.0, delta=0.01)

    def test_a_kernel_without_the_keys_says_None_rather_than_zero(self):
        """Zero would read as 'the server holds no GTT', which is a
        measurement. None is the absence of one, and the caller falls back."""
        d = self.proc({"7": "pos:\t0\nflags:\t02\n"})
        self.assertIsNone(budget.server_gtt_gib(d))

    def test_no_server_no_figure(self):
        d = self.proc(cmd=b"/usr/bin/something-else\0")
        self.assertIsNone(budget.server_gtt_gib(d))
        self.assertIsNone(budget.server_pid(d))

    def test_observe_prefers_the_process_and_says_which(self):
        got = budget.observe(argv=self.argv(ctx=204800),
                             machine=self.machine(gtt_used=35.64),
                             weights=17.56, gtt_process=34.78)
        self.assertEqual(got["gtt_observed_gib"], 34.78)
        self.assertEqual(got["gtt_used_gib"], 35.64)
        self.assertIn("own", got["gtt_source"])
        self.assertAlmostEqual(got["kv_gib_observed"], 34.78 - 17.56, delta=0.01)

    def test_observe_falls_back_and_labels_it(self):
        got = budget.observe(argv=self.argv(ctx=204800),
                             machine=self.machine(gtt_used=35.64),
                             weights=17.56, gtt_process=None)
        self.assertEqual(got["gtt_observed_gib"], 35.64)
        self.assertIsNone(got["gtt_process_gib"])
        self.assertIn("system-wide", got["gtt_source"])

    def test_the_desktop_can_no_longer_hide_an_under_prediction(self):
        """The failure the open item described, as a case. A prediction of
        32.0 against a server truly holding 34.78 is an UNDER-prediction of
        8.7 %. System-wide it reads as 35.64 and looks worse still — but on a
        busy desktop holding 3 GiB the system-wide figure is what makes a bad
        prediction look fine, and that is the direction that matters."""
        p = budget.plan(self.argv(ctx=204800), 17.56, declared=74.3)
        sharp = budget.compare(p, budget.observe(
            argv=self.argv(ctx=204800), machine=self.machine(gtt_used=99.0),
            weights=17.56, gtt_process=34.78))
        self.assertAlmostEqual(sharp.observed, 34.78, delta=0.01)
        self.assertIn("own", sharp.note)
        coarse = budget.compare(p, budget.observe(
            argv=self.argv(ctx=204800), machine=self.machine(gtt_used=99.0),
            weights=17.56, gtt_process=None))
        self.assertAlmostEqual(coarse.observed, 99.0, delta=0.01)
        self.assertIn("system-wide", coarse.note)

    def test_one_scan_finds_the_process_for_all_three_readers(self):
        """server_pid() exists because running_argv(), _rss_anon_gib() and the
        GTT reader each walked /proc with the same test. Three readers of one
        thing is how most of this repo's bugs were found."""
        d = self.proc({"7": self.client(1, 1048576)})
        self.assertEqual(budget.server_pid(d), "4242")
        self.assertEqual(budget.running_argv(d)[0], "/usr/bin/llama-server")
        self.assertAlmostEqual(budget._rss_anon_gib(d), 1.0, delta=0.01)


class TestTheEstimateIsAboveEveryMeasurement(Base):
    """ESTIMATE_KV_KIB_PER_TOKEN is what EVERY model a stranger adds is
    charged, and until 27.08.2026 nothing checked it against anything.

    It was described as "deliberately high", which is a claim and not a
    measurement. Held against the seven profiles that DO declare a measured
    figure it turned out to be high for six of them and 50 % LOW for the
    seventh: laguna measures 96.0 KiB/token at q8_0 and the estimate charged
    64.0. Low is the direction that freezes the machine.

    The cause was a missing term rather than a wrong number. laguna is the
    only profile carrying --swa-full, which removes the sliding-window cap so
    that every layer keeps a full KV cache; the estimate scaled by cache type
    alone. gemma31 declares the same MODEL_SWA=yes WITHOUT the switch and
    measures 44.5.

    These two tests are the open item turned into a property. A profile whose
    measured figure ever exceeds what the estimate would have charged it makes
    this red, which is the only way a constant like that stays honest.
    """

    # How far above the tightest measurement the estimate may sit before it
    # has stopped being connected to evidence. Raising the constant until
    # nothing can fail it would pass the first test and defeat its purpose.
    TOO_LOOSE = 4.0

    def measured(self):
        """(name, what the estimate WOULD charge, what was measured)."""
        import glob as _g, os as _os
        systemdfile = __import__("systemdfile")
        out = []
        for env in sorted(_g.glob(str(common.REPO / "setup" / "env" / "*.env"))):
            declared = budget.declared_kv(env)
            if declared is None:
                continue
            argv = systemdfile.llama_args(env)
            estimate, source = budget.kv_kib_per_token(argv, declared=None)
            self.assertEqual(source, "estimated")
            out.append((_os.path.basename(env)[:-4], estimate, declared))
        return out

    def test_it_covers_every_profile_that_declares_a_measurement(self):
        rows = self.measured()
        self.assertTrue(rows, "no profile declares a KV figure — nothing to "
                              "hold the estimate against, which is itself the "
                              "state this test exists to prevent")
        short = ["    %-11s measured %6.1f, estimate charges %6.1f  (%.0f %% LOW)"
                 % (n, m, e, (m - e) / e * 100)
                 for n, e, m in rows if e < m]
        self.assertFalse(short,
                         "the estimate charges LESS than these profiles were "
                         "measured to cost. A model a stranger adds without "
                         "declaring MODEL_KV_KIB_PER_TOKEN would be passed a "
                         "start that does not fit, and against pinned GTT that "
                         "is a frozen machine rather than an error:\n%s"
                         % "\n".join(short))

    def test_it_is_still_connected_to_what_was_measured(self):
        """The other half. Passing the test above by raising the constant
        until nothing can reach it would make the guard refuse profiles that
        fit, and would make the first test unfalsifiable."""
        rows = self.measured()
        tightest = min((e / m, n) for n, e, m in rows)
        self.assertLess(tightest[0], self.TOO_LOOSE,
                        "the closest any measured profile comes to the "
                        "estimate is %s at %.1fx. Nothing measured is near it "
                        "any more, so it has become a number rather than a "
                        "bound — re-derive it from the profiles."
                        % (tightest[1], tightest[0]))

    def test_swa_full_is_what_the_missing_term_was(self):
        """The mechanism, not just the constant: the same profile with and
        without the switch has to be charged differently, because the switch
        is what stops the window from capping the cache."""
        windowed = budget.kv_kib_per_token(["-ctk", "q8_0", "-ctv", "q8_0"])[0]
        full = budget.kv_kib_per_token(
            ["-ctk", "q8_0", "-ctv", "q8_0", "--swa-full"])[0]
        self.assertGreater(full, windowed)
        self.assertAlmostEqual(full / windowed, budget.SWA_FULL_FACTOR, places=6)
        self.assertGreaterEqual(full, 96.0,
                                "laguna measures 96.0 at exactly these flags")

    def test_a_declared_figure_is_never_touched_by_the_term(self):
        """The term belongs to the ESTIMATE. A profile that measured its own
        cost with --swa-full already has the effect in the number."""
        self.assertEqual(
            budget.kv_kib_per_token(["--swa-full"], declared=96.0),
            (96.0, "declared"))


class TestWhatTheRamPromptCacheBuys(Base):
    """`-cram` was computed ONCE, for qwen38, and copied into every profile
    that came after.

    The audit of 27.08. found it 30 GiB over on Flash-Next and fixed that one.
    It never asked the other question — not "does it fit" but "what does it
    BUY" — and the answer ranged from 1.3 full windows to 197.
    """

    # A window is the worst case one prefix can cost. qwen38, the profile
    # where the number was actually computed and defended, buys 2.2. Anything
    # up to a couple of dozen is arguable headroom on a machine with room.
    # Fifty is not a judgement call any more: it is a number nobody read.
    ABSURD = 50

    def profiles(self):
        import glob as _g, os as _os
        out = []
        for env in sorted(_g.glob(str(common.REPO / "setup" / "env" / "*.env"))):
            argv = __import__("systemdfile").llama_args(env)
            out.append((_os.path.basename(env)[:-4], argv, budget.declared_kv(env)))
        return out

    def test_no_profile_reserves_a_cache_nobody_could_defend(self):
        import sys as _s
        _s.path.insert(0, str(common.REPO / "setup" / "lib"))
        bad, rows = [], []
        for name, argv, kv in self.profiles():
            cram, per, n = budget.cache_windows(argv, kv)
            if not cram or n is None:
                continue
            rows.append("    %-11s %5.0f GiB / %5.2f per window = %6.1f windows"
                        % (name, cram, per, n))
            if n > self.ABSURD:
                bad.append(name)
        self.assertFalse(bad, "-cram in these profiles buys more than %d full "
                              "windows, which is a copied number rather than a "
                              "choice — %s\n%s"
                              % (self.ABSURD, ", ".join(bad), "\n".join(rows)))

    def test_the_reference_profile_is_still_the_tightest_reasoned_one(self):
        """qwen38's 2.2 windows is the figure the repo argued for. If some
        other profile ever drops below it, that one is the new reference and
        this test should be read again rather than edited."""
        vals = {}
        for name, argv, kv in self.profiles():
            _, _, n = budget.cache_windows(argv, kv)
            if n is not None:
                vals[name] = n
        self.assertIn("qwen38", vals)
        self.assertLess(vals["qwen38"], 5.0, "the reference has drifted upward")

    def test_a_window_is_derived_from_the_declared_kv_and_not_guessed(self):
        cram, per, n = budget.cache_windows(
            self.argv(ctx=204800, cram=32768), 74.3)
        self.assertAlmostEqual(per, 204800 * 74.3 / 1048576.0, delta=0.01)
        self.assertAlmostEqual(n, 32.0 / per, delta=0.01)

    def test_no_cram_means_no_opinion(self):
        self.assertEqual(budget.cache_windows(self.argv(ctx=1024), 74.3)[2], None)


class TestTheThreeDisasters(Base):
    """Every one of these actually happened, or would have."""

    def test_a_side_server_beside_production_is_refused(self):
        """26.08.2026: the coding battery moved from a 13.5 GiB model to a 68
        GiB one while production held 17. Two sets of weights went into GTT,
        the page cache was squeezed to 40 MiB, and the OOM killer took
        llama-server and the desktop with it."""
        p = budget.plan(["-m", "/models/laguna.gguf"], 68.0, what="laguna")
        v = budget.verdict(p, self.machine(avail=81.0, gtt_total=116.0, gtt_used=35.0))
        self.assertFalse(v.fits)
        self.assertIn("REFUSING TO START laguna", budget.refusal(p, self.machine(), v))

    def test_the_production_profile_on_a_64_gb_machine_is_refused(self):
        """The case this guard was actually built for. Strix Halo ships with
        32, 64 and 128 GB; qwen38's profile is written for 128, and the flags
        that make it good there — -c 204800, -cram 32768 — are what make it
        fatal on 64. Nothing in this repo said so before 27.08."""
        p = budget.plan(self.argv(ctx=204800, cram=32768), 17.6, 74.3, "qwen38")
        small = budget.Machine(mem_total=62.0, mem_available=58.0,
                               gtt_total=48.0, gtt_used=0.5)
        self.assertFalse(budget.fits_the_machine(p, small))
        self.assertFalse(budget.verdict(p, small).fits)
        self.assertTrue(budget.fits_the_machine(p, self.machine()),
                        "and it must still fit the machine it was written for")

    def test_lagunas_original_cram_did_not_fit_and_the_old_test_missed_it(self):
        """68.4 of weights + 32 of RAM cache + 12 for the host is 112.4 of
        124.9 and looks fine — which is what tests/test_models.py used to
        compute. Add the 12 GiB of KV that -c 131072 costs at a measured 96.0
        KiB/token and it is 132.4. The KV term was the whole difference."""
        old = budget.plan(["-c", "131072", "-cram", "32768"], 68.4, 96.0, "laguna")
        self.assertFalse(budget.fits_the_machine(old, self.machine()))
        now = budget.plan(["-c", "131072", "-cram", "16384"], 68.4, 96.0, "laguna")
        self.assertTrue(budget.fits_the_machine(now, self.machine()))


class TestItFailsOpenOnWhatItCannotSee(Base):
    """A safety that blocks everything it cannot see gets switched off, and
    then it is gone. Refusing needs a REASON, not an absence of one."""

    def test_unreadable_weights_are_not_our_call(self):
        p = budget.plan(["-m", "/does/not/exist.gguf"], None)
        v = budget.verdict(p, self.machine(avail=1.0, gtt_used=107.0))
        self.assertTrue(v.fits)
        self.assertTrue(any("not our call" in n for n in v.notes))

    def test_a_machine_without_amdgpu_still_checks_the_host(self):
        """No GTT facts is not no opinion: MemAvailable still bounds
        everything, and that check must survive on its own."""
        p = budget.plan(self.argv(ctx=204800, cram=32768), 17.6, 74.3)
        no_gpu = budget.Machine(mem_total=32.0, mem_available=30.0,
                                gtt_total=None, gtt_used=None)
        v = budget.verdict(p, no_gpu)
        self.assertFalse(v.fits)
        self.assertTrue(any("GTT is not readable" in n for n in v.notes))

    def test_a_machine_with_nothing_readable_refuses_nothing(self):
        p = budget.plan(self.argv(ctx=204800, cram=32768), 400.0, 74.3)
        blind = budget.Machine(None, None, None, None)
        self.assertTrue(budget.verdict(p, blind).fits)

    def test_read_machine_returns_four_independently_optional_facts(self):
        m = budget.read_machine()
        self.assertIsNotNone(m.mem_total, "/proc/meminfo is readable everywhere")
        self.assertIsInstance(m, budget.Machine)


class TestTheEscapeHatches(Base):
    """There has to be a way past a WRONG estimate that is not a way past the
    guard. Without one the only escape is the off switch, and a safety that is
    inconvenient at the wrong moment gets switched off entirely — which is how
    this machine hung twice in one day."""

    def test_an_override_corrects_the_input_and_every_check_still_runs(self):
        os.environ["LLM_MODEL_GIB"] = "200"
        p = budget.plan(["-m", "/x.gguf"], 200.0)
        self.assertFalse(budget.verdict(p, self.machine()).fits)

    def test_a_stated_host_figure_may_not_fall_below_the_file(self):
        """LLM_HOST_GIB=88 was once used for a 103.7 GiB model. The server
        started, the machine went to 100 %, and the kernel OOM-killed it.
        A measurement can say where the bytes land; nothing makes a file
        smaller."""
        os.environ["LLM_HOST_GIB"] = "88"
        with self.assertRaises(SystemExit) as cm:
            budget.plan(["-m", "/x.gguf"], 103.7)
        self.assertIn("below the 103.7 GiB", str(cm.exception))

    def test_nonsense_in_a_knob_is_refused_rather_than_ignored(self):
        """Silently falling back would hide a typo in the one place where a
        number is being trusted instead of measured."""
        os.environ["LLM_MODEL_GIB"] = "eighty-eight"
        with self.assertRaises(SystemExit):
            budget.plan(["-m", "/x.gguf"], 50.0)

    def test_the_bench_names_still_mean_the_same_thing(self):
        """bench/run.py shipped with BENCH_*. Renaming them without aliases
        would change the meaning of a measurement already in flight."""
        os.environ["BENCH_HOST_RESERVE_GIB"] = "20"
        self.assertEqual(budget.host_reserve_gib(), 20.0)
        os.environ["BENCH_NO_MEMORY_GUARD"] = "1"
        self.assertTrue(budget.guard_disabled())

    def test_the_new_name_wins_over_the_old_one(self):
        os.environ["BENCH_HOST_RESERVE_GIB"] = "20"
        os.environ["LLM_HOST_RESERVE_GIB"] = "8"
        self.assertEqual(budget.host_reserve_gib(), 8.0)

    def test_the_off_switch_is_a_different_and_worse_thing(self):
        """It is named in the refusal so nobody has to go looking for it — and
        named LAST, after the three corrections that keep the guard running."""
        p = budget.plan(["-m", "/x.gguf"], 68.0, what="x")
        m = self.machine(avail=20.0)
        text = budget.refusal(p, m, budget.verdict(p, m))
        self.assertIn("LLM_MODEL_GIB", text)
        self.assertGreater(text.index("NO_MEMORY_GUARD"), text.index("LLM_MODEL_GIB"))


class TestStaticIsADifferentQuestionFromNow(Base):
    def test_a_sound_profile_can_still_be_unstartable_right_now(self):
        p = budget.plan(self.argv(ctx=204800, cram=32768), 17.6, 74.3)
        busy = self.machine(avail=20.0, gtt_used=90.0)
        self.assertTrue(budget.fits_the_machine(p, busy),
                        "the profile is fine; the machine is busy")
        self.assertFalse(budget.verdict(p, busy).fits,
                         "and it still must not start on top of that")


class TestTheObservationClosesTheLoop(Base):
    """A declared number with nothing checking it drifts back into an
    assertion. This is the half that re-checks it after every start."""

    def plan_(self):
        return budget.plan(self.argv(ctx=204800, cram=32768), 17.6, 74.3, "qwen38")

    def test_observed_below_predicted_is_the_guard_doing_its_job(self):
        c = budget.compare(self.plan_(), {"gtt_used_gib": 33.0})
        self.assertTrue(c.ok)
        self.assertLess(c.margin, 0)

    def test_observed_above_predicted_is_the_dangerous_direction(self):
        c = budget.compare(self.plan_(), {"gtt_used_gib": 60.0})
        self.assertFalse(c.ok)

    def test_the_real_reading_from_this_machine_passes(self):
        """27.08.2026: `mem_info_gtt_used` showed 35.6 GiB for the production
        profile, and the prediction must sit at or above it.

        Two earlier versions of this got it wrong in opposite directions. The
        first derived KiB/token from a single reading, called it +24 % and
        flagged a correct profile red — one reading cannot separate the KV
        from the compute buffers, so the whole buffer term landed on the KV.
        The second asserted the prediction was within 5 % of the observation,
        which quietly demanded that the guard be TIGHT. It must be
        conservative: below the observation is the only failing direction.
        """
        c = budget.compare(self.plan_(), {"gtt_used_gib": 35.6})
        self.assertTrue(c.ok, "a correct profile must not be flagged")
        self.assertLess(c.margin, 0, "the guard is predicting less than was "
                                     "actually pinned — the dangerous direction")
        self.assertGreater(c.margin, -0.25, "conservative is right, but a "
                                            "prediction this far above the "
                                            "reading stops meaning anything")

    def test_it_has_no_opinion_without_an_observation(self):
        self.assertIsNone(budget.compare(self.plan_(), None))
        self.assertIsNone(budget.compare(self.plan_(), {"gtt_used_gib": None}))
        self.assertIsNone(budget.compare(budget.plan([], None), {"gtt_used_gib": 3.0}))

    def test_the_caveat_travels_with_the_number(self):
        """Both sides overstate — system-wide GTT includes the desktop, and
        the prediction carries its slack. A reader who does not know that will
        read a harmless +1 % as a finding."""
        self.assertIn("system-wide", budget.compare(self.plan_(), {"gtt_used_gib": 35.6}).note)


class TestTheGuardSaysSoWhenItPasses(Base):
    """A guard that is silent on success cannot be told from a guard that is
    not installed — and checkroom deliberately exits 0 when it cannot find
    budget.py, so those two really do produce the same journal. Verified on
    27.08. from a real restart: `systemctl show` said the ExecStartPre had run
    and exited 0, and the journal said nothing at all."""

    def test_the_brief_line_carries_both_numbers(self):
        p = budget.plan(self.argv(ctx=204800, cram=32768), 17.6, 74.3, "qwen38")
        line = budget.brief(p, self.machine(), budget.verdict(p, self.machine()))
        self.assertIn("qwen38 fits", line)
        self.assertIn("%.1f" % p.gtt_gib, line)
        self.assertIn("%.1f" % p.host_gib, line)
        self.assertEqual(line.count("\n"), 0, "one line means one line")

    def test_it_says_when_the_kv_figure_was_guessed(self):
        p = budget.plan(self.argv(ctx=204800, cram=32768), 17.6)
        line = budget.brief(p, self.machine(), budget.verdict(p, self.machine()))
        self.assertIn("ESTIMATED", line)

    def test_a_refusal_reads_as_one(self):
        p = budget.plan(self.argv(ctx=204800, cram=32768), 17.6, 74.3, "qwen38")
        small = budget.Machine(62.0, 58.0, 48.0, 0.5)
        self.assertIn("DOES NOT FIT", budget.brief(p, small, budget.verdict(p, small)))

    def test_unreadable_weights_do_not_claim_a_verdict(self):
        p = budget.plan(["-m", "/nope.gguf"], None, what="x")
        self.assertIn("not weighed", budget.brief(p, self.machine(),
                                                  budget.verdict(p, self.machine())))

    def test_checkroom_prints_it_on_the_success_path(self):
        """Inside the branch that PASSES, not merely somewhere in the file.

        The first version of this test compared against the file's first
        `exit 0`, which is the one for "budget.py not found" — so it failed on
        correct code and would have passed on a printf that never ran.
        """
        src = (common.REPO / "setup" / "checkroom").read_text(encoding="utf-8")
        self.assertIn("--brief", src, "checkroom asks for the table, not the line")
        start = src.index('if [ "$rc" -eq 0 ]; then')
        branch = src[start:src.index("\n  fi", start)]
        self.assertIn("llm-check-room: $out", branch,
                      "the success branch exits without saying what it decided")
        self.assertLess(branch.index("llm-check-room: $out"), branch.index("exit 0"),
                        "the verdict has to be printed before the exit that ends it")


class TestTheGuardSaysWhichCopyAnsweredIt(Base):
    """checkroom's search path has a tail, and the tail is dangerous.

        "$HERE"  "$HERE/lib"  "$HOME/.claude/bin"  /usr/local/lib/llm-profile

    The last entry is populated only by `install.sh --system-unit`, and once
    the system unit is removed, check.sh stops watching it. A stale budget.py
    left there would not fail — it would guard with OLD arithmetic and say
    nothing. That is this guard's own failure mode occurring inside its own
    lookup, found on 27.08. while deciding whether the leftovers of the
    removed system unit could simply stay.

    So the fallback is named. It cannot be removed: a system-installed
    checkroom at /usr/local/bin genuinely needs it.
    """

    def src(self):
        return (common.REPO / "setup" / "checkroom").read_text(encoding="utf-8")

    def test_the_search_path_still_has_the_system_directory(self):
        """Removing it would break the system unit, which is the one caller
        that has no other place to look."""
        self.assertIn("/usr/local/lib/llm-profile", self.src())

    def test_a_copy_that_is_not_beside_the_script_is_named(self):
        src = self.src()
        self.assertIn('FROM=', src)
        self.assertIn('"$(dirname "$BUDGET")" != "$HERE"', src,
                      "the guard no longer notices where its arithmetic came from")

    def test_the_name_travels_on_the_verdict_line(self):
        """Not on a separate line: the verdict is what a reader greps for in a
        journal, and a provenance note they have to correlate is one they will
        not see."""
        src = self.src()
        self.assertIn('"llm-check-room: $out$FROM"', src)

    def test_check_sh_says_the_leftovers_can_still_be_found(self):
        """The orphan report is the place a reader decides whether to delete
        them, so that is where the reason belongs."""
        src = (common.REPO / "setup" / "check.sh").read_text(encoding="utf-8")
        self.assertIn("stale", src)
        self.assertIn("/usr/local/LIB/llm-profile", src,
                      "three different paths are spelled llm-profile and the "
                      "rm line names only one of them")


class TestTheFormulaLivesInExactlyOnePlace(Base):
    """The reason this file exists. Three copies had already drifted — 10 vs
    12 GiB of host reserve, with KV vs without — and a fourth would drift too."""

    def test_bench_does_not_carry_its_own_arithmetic(self):
        src = (common.REPO / "bench" / "run.py").read_text(encoding="utf-8")
        self.assertIn("import budget", src)
        self.assertNotIn("* 1.10", src, "bench/run.py has its own slack again")

    def test_sideserver_does_not_carry_its_own_host_reserve(self):
        src = (common.REPO / "bench" / "sideserver.py").read_text(encoding="utf-8")
        self.assertIn("budget.host_reserve_gib()", src)
        self.assertNotIn("HOST_RESERVE_GIB = 12.0", src)

    def test_one_host_reserve_for_the_whole_repo(self):
        import importlib
        run = common.load("bench/run.py", "bench_run_reserve")
        self.assertEqual(run.HOST_RESERVE_GIB, budget.HOST_RESERVE_GIB)
        del importlib


if __name__ == "__main__":
    unittest.main()
