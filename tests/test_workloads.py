"""setup/workloads/ — foreign workloads under the same memory authority.

A diffusion job that pins 8 GiB of GTT beside qwen38's 35 is exactly the
problem budget.py exists for: GTT is pinned, an over-large start does not
page, it takes the machine down (three times on 26.08.2026 — see
bench/sideserver.py). Until now only llama-server starts were weighed. These
tests pin the contract that lets ANY workload be weighed:

  * a workload profile is one file in setup/workloads/, systemd
    EnvironmentFile syntax, readable by the same parser the services use;
  * its footprint fields are born EMPTY (unmeasured) and the guard then
    charges an estimate from the model files and SAYS so — the same honesty
    rule the llama profiles follow for a missing KV figure;
  * a measured footprint is an observation with a date and a machine, and
    the guard refuses a workload that does not fit what is left RIGHT NOW —
    the refusal case is tested, not assumed ("a check that cannot fail is
    not a check").

No GPU, no network: machines are handed in as values, file sizes as fakes.
"""
import os
import sys
import tempfile
import unittest

import common

sys.path.insert(0, str(common.REPO / "setup" / "lib"))
import budget                                             # noqa: E402
import systemdfile                                        # noqa: E402

WORKLOADS_DIR = common.REPO / "setup" / "workloads"


def profile(tmp, name="wl.env", **fields):
    """Write a minimal workload profile and return its path."""
    base = {
        "WORKLOAD_TITLE": "a test workload",
        "WORKLOAD_KIND": "image",
        "WORKLOAD_MODE": "batch",
        "WORKLOAD_CMD": "@HOME@/sd/bin/sd-cli -m @MODELS@/image/x.safetensors "
                        "-p prompt -o @HOME@/out.png",
        "WORKLOAD_FILES": "@MODELS@/image/x.safetensors",
    }
    base.update(fields)
    path = os.path.join(tmp, name)
    with open(path, "w", encoding="utf-8") as fh:
        for k, v in base.items():
            if v is not None:
                fh.write("%s=%s\n" % (k, v))
    return path


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def machine(self, total=124.9, avail=110.0, gtt_total=108.0, gtt_used=0.5):
        return budget.Machine(total, avail, gtt_total, gtt_used)


class TestReadingAWorkloadProfile(Base):
    def test_the_command_is_read_by_the_one_parser(self):
        """WORKLOAD_CMD goes through systemdfile like LLAMA_ARGS does — same
        continuation lines, same token expansion, quotes passed through as
        data. A second parser is where this repo's bugs have lived."""
        p = profile(self.tmp)
        argv = systemdfile.args_of(p, "WORKLOAD_CMD",
                                   home="/h", models="/m")
        self.assertEqual(argv[0], "/h/sd/bin/sd-cli")
        self.assertIn("/m/image/x.safetensors", argv)

    def test_a_missing_assignment_is_none_not_an_error(self):
        p = profile(self.tmp)
        self.assertIsNone(systemdfile.args_of(p, "NO_SUCH_VAR"))

    def test_files_are_summed_and_unreadable_means_none(self):
        """None means "not our call" — an unreadable path is not evidence
        that a workload is small. Same rule as budget.weights_gib()."""
        p = profile(self.tmp, WORKLOAD_FILES="@MODELS@/a.st @MODELS@/b.st")
        sizes = {"/m/a.st": 2 * 1073741824, "/m/b.st": 1073741824}
        got = budget.workload_files_gib(p, size_of=sizes.__getitem__,
                                        models="/m")
        self.assertAlmostEqual(got, 3.0)

        def missing(_):
            raise OSError("gone")
        self.assertIsNone(budget.workload_files_gib(p, size_of=missing,
                                                    models="/m"))

    def test_globs_are_expanded_like_weights_gib_does(self):
        """Review finding 01.09.2026: an unexpanded glob (or a typo)
        raised OSError and quietly degraded the guard of an UNMEASURED
        profile to a note. weights_gib globs shards; this globs too, and
        an EMPTY glob is None — a pattern matching nothing is a typo, not
        a small workload."""
        import pathlib
        d = pathlib.Path(self.tmp)
        (d / "part-1.st").write_bytes(b"x" * 1024)
        (d / "part-2.st").write_bytes(b"x" * 1024)
        p = profile(self.tmp, WORKLOAD_FILES="@MODELS@/part-*.st")
        got = budget.workload_files_gib(p, models=self.tmp)
        self.assertAlmostEqual(got, 2048 / 1073741824.0)
        p2 = profile(self.tmp, name="wl2.env",
                     WORKLOAD_FILES="@MODELS@/no-such-*.st")
        self.assertIsNone(budget.workload_files_gib(p2, models=self.tmp))

    def test_hf_cache_is_a_recognized_keyword(self):
        """WORKLOAD_FILES=hf-cache is a CONTRACT, not a never-resolving
        path exploited for its side effect: budget names the state out
        loud instead of mumbling 'could not be read'."""
        p = profile(self.tmp, WORKLOAD_FILES="hf-cache")
        plan = budget.workload_plan(p, models="/m")
        self.assertIsNone(plan.gtt_gib)
        v = budget.workload_verdict(plan, self.machine())
        self.assertTrue(any("framework-cached" in n for n in v.notes),
                        v.notes)


class TestTheUnmeasuredPlan(Base):
    """A profile is born with empty footprint fields, and the guard still has
    an opinion: the files it will load, plus the buffer term the llama
    measurements calibrated — and it says ESTIMATE out loud."""

    def test_estimate_from_files_plus_buffers(self):
        p = profile(self.tmp)
        sizes = {"/m/image/x.safetensors": 7 * 1073741824}
        plan = budget.workload_plan(p, size_of=sizes.__getitem__, models="/m")
        self.assertAlmostEqual(plan.gtt_gib,
                               7.0 + budget.buffers_gib(7.0), delta=0.01)
        self.assertAlmostEqual(plan.host_gib, plan.gtt_gib, delta=0.01)
        self.assertTrue(plan.estimated)

    def test_the_estimate_announces_itself_in_the_verdict(self):
        """An estimate that does not announce itself is how -cram 32768 got
        copied into five profiles."""
        p = profile(self.tmp)
        sizes = {"/m/image/x.safetensors": 7 * 1073741824}
        plan = budget.workload_plan(p, size_of=sizes.__getitem__, models="/m")
        v = budget.workload_verdict(plan, self.machine())
        self.assertTrue(any("WORKLOAD_GTT_GIB" in n for n in v.notes))
        self.assertFalse(any("MODEL_KV_KIB_PER_TOKEN" in n for n in v.notes),
                         "the llama wording would send a reader to the wrong "
                         "field")

    def test_unreadable_files_are_not_our_call(self):
        """A guard that blocks everything it cannot see gets switched off,
        and then it is gone. Same philosophy as read_machine()."""
        def missing(_):
            raise OSError("gone")
        p = profile(self.tmp)
        plan = budget.workload_plan(p, size_of=missing, models="/m")
        v = budget.workload_verdict(plan, self.machine())
        self.assertIsNone(plan.gtt_gib)
        self.assertTrue(v.fits)
        self.assertTrue(v.notes)


class TestTheMeasuredPlan(Base):
    # The machine the fake profiles claim to be measured on — injected so
    # these tests are hermetic: since the unknown-machine degrade
    # (01.09.2026) a plan trusts figures only when the running identity
    # POSITIVELY matches, and CI has no kfd to answer with.
    HERE = {"gfx": "gfx1151", "mem_total_gib": 124.9}

    def measured(self, gtt=8.2, rss=1.4):
        return profile(
            self.tmp,
            WORKLOAD_GTT_GIB=str(gtt),
            WORKLOAD_GTT_SOURCE="01.09.2026, peak of mem_info_gtt_used minus "
                                "settled baseline during one run",
            WORKLOAD_HOST_RSS_GIB=str(rss),
            WORKLOAD_HOST_RSS_SOURCE="01.09.2026, peak RssAnon of the job",
            WORKLOAD_MEASURED_ON="gfx1151, 124.9 GiB RAM")

    def test_a_measured_peak_keeps_a_margin_and_says_so(self):
        """An observed peak of a BATCH job is not a static allocation — the
        next prompt or resolution can sit above it. 10 % on top, visible as
        its own line, because overestimating is the direction a guard may
        err in (heuristic, not derived — see budget.py)."""
        plan = budget.workload_plan(self.measured(), models="/m",
                                    identity=self.HERE)
        self.assertAlmostEqual(plan.gtt_gib, 8.2 * 1.10, delta=0.01)
        self.assertFalse(plan.estimated)
        self.assertTrue(any(it.source == "measured" for it in plan.items))

    def test_host_is_gtt_plus_resident_outside(self):
        plan = budget.workload_plan(self.measured(gtt=8.2, rss=1.4),
                                    models="/m", identity=self.HERE)
        self.assertAlmostEqual(plan.host_gib, plan.gtt_gib + 1.4, delta=0.01)

    def test_it_refuses_what_does_not_fit_beside_production(self):
        """THE refusal case. qwen38 holds ~36 GiB of GTT; a workload whose
        measured need exceeds what is left must come back with problems —
        this is the line between "guard" and "comment"."""
        plan = budget.workload_plan(self.measured(gtt=80.0), models="/m",
                                    identity=self.HERE)
        v = budget.workload_verdict(
            plan, self.machine(gtt_total=108.0, gtt_used=36.0, avail=80.0))
        self.assertFalse(v.fits)
        self.assertTrue(v.problems)

    def test_it_passes_what_fits(self):
        plan = budget.workload_plan(self.measured(gtt=8.2, rss=1.4),
                                    models="/m", identity=self.HERE)
        v = budget.workload_verdict(
            plan, self.machine(gtt_total=108.0, gtt_used=36.0, avail=80.0))
        self.assertTrue(v.fits, v.problems)

    def test_figures_from_another_machine_degrade_to_estimates(self):
        """The Medusa insurance (architecture review, 01.09.2026): figures
        measured on gfx1151 guarding on a different gpu are claims again —
        in the small-to-big direction merely conservative, in the
        big-to-small direction the freeze direction. The plan keeps the
        numbers (they beat file sizes) but says ESTIMATE again."""
        plan = budget.workload_plan(self.measured(gtt=8.2, rss=1.4),
                                    models="/m",
                                    identity={"gfx": "gfx1250",
                                              "mem_total_gib": 250.0})
        self.assertTrue(plan.estimated)
        v = budget.workload_verdict(plan, self.machine())
        self.assertTrue(any("different machine" in n for n in v.notes),
                        v.notes)

    def test_figures_on_the_same_machine_stay_measured(self):
        plan = budget.workload_plan(self.measured(gtt=8.2, rss=1.4),
                                    models="/m",
                                    identity={"gfx": "gfx1151",
                                              "mem_total_gib": 124.9})
        self.assertFalse(plan.estimated)

    def test_an_unknown_running_machine_degrades_too(self):
        """Re-review sweep (01.09.2026): `if here and there and ...`
        treated here=None (kfd unreadable, non-AMD box) as 'same
        machine' — gfx1151 figures then guarded as measured on a machine
        nobody identified, the exact freeze direction the insurance was
        bought against. Unknown is not a match."""
        plan = budget.workload_plan(self.measured(gtt=8.2, rss=1.4),
                                    models="/m",
                                    identity={"gfx": None,
                                              "mem_total_gib": 7.8})
        self.assertTrue(plan.estimated)
        v = budget.workload_verdict(plan, self.machine())
        self.assertTrue(any("tied to" in n for n in v.notes), v.notes)

    def test_a_measured_on_without_a_gfx_token_degrades(self):
        p = profile(
            self.tmp,
            WORKLOAD_GTT_GIB="8.2",
            WORKLOAD_GTT_SOURCE="01.09.2026, metered",
            WORKLOAD_HOST_RSS_GIB="1.4",
            WORKLOAD_HOST_RSS_SOURCE="01.09.2026, metered",
            WORKLOAD_MEASURED_ON="a machine someone forgot to name")
        plan = budget.workload_plan(p, models="/m",
                                    identity={"gfx": "gfx1151",
                                              "mem_total_gib": 124.9})
        self.assertTrue(plan.estimated)

    def test_a_cmd_repointed_to_another_build_degrades(self):
        """The runtime half of test_measured_on_names_the_pinned_build:
        the tree IS the installation, so a hand-edited WORKLOAD_CMD (no
        gate run) must not keep stale figures guarding as measured —
        the llama path has _is_the_observed_binary for exactly this
        (review, 01.09.2026)."""
        p = profile(
            self.tmp,
            WORKLOAD_CMD="@HOME@/sd/build-vulkan-fffffff01/bin/sd-cli "
                         "-m @MODELS@/image/x.safetensors",
            WORKLOAD_GTT_GIB="8.2",
            WORKLOAD_GTT_SOURCE="01.09.2026, metered",
            WORKLOAD_HOST_RSS_GIB="1.4",
            WORKLOAD_HOST_RSS_SOURCE="01.09.2026, metered",
            WORKLOAD_MEASURED_ON="gfx1151, 124.9 GiB RAM, "
                                 "build vulkan-aaaaaaa01")
        plan = budget.workload_plan(p, models="/m",
                                    identity={"gfx": "gfx1151",
                                              "mem_total_gib": 124.9})
        self.assertTrue(plan.estimated)
        v = budget.workload_verdict(plan, self.machine())
        self.assertTrue(any("fffffff01" in n for n in v.notes), v.notes)

    def test_a_cmd_matching_the_measured_build_stays_measured(self):
        p = profile(
            self.tmp,
            WORKLOAD_CMD="@HOME@/sd/build-vulkan-aaaaaaa01/bin/sd-cli "
                         "-m @MODELS@/image/x.safetensors",
            WORKLOAD_GTT_GIB="8.2",
            WORKLOAD_GTT_SOURCE="01.09.2026, metered",
            WORKLOAD_HOST_RSS_GIB="1.4",
            WORKLOAD_HOST_RSS_SOURCE="01.09.2026, metered",
            WORKLOAD_MEASURED_ON="gfx1151, 124.9 GiB RAM, "
                                 "build vulkan-aaaaaaa01")
        plan = budget.workload_plan(p, models="/m",
                                    identity={"gfx": "gfx1151",
                                              "mem_total_gib": 124.9})
        self.assertFalse(plan.estimated)

    def test_the_workload_refusal_names_workload_fields(self):
        """budget.refusal()'s default advice names LLM_MODEL_GIB and
        friends — none of which workload_plan reads. Following that advice
        ends at LLM_NO_MEMORY_GUARD=1, the off-switch escalation the
        design warns about (review, 01.09.2026)."""
        plan = budget.workload_plan(self.measured(gtt=80.0), models="/m",
                                    identity=self.HERE)
        machine = self.machine(gtt_total=108.0, gtt_used=36.0, avail=80.0)
        v = budget.workload_verdict(plan, machine)
        text = budget.refusal(plan, machine, v,
                              advice=budget.WORKLOAD_ADVICE)
        self.assertIn("WORKLOAD_GTT_GIB", text)
        self.assertNotIn("LLM_MODEL_GIB", text)

    def test_brief_carries_the_workload_hint(self):
        """--brief on the workload path sent readers to
        MODEL_KV_KIB_PER_TOKEN — a field a workload profile cannot fill."""
        p = profile(self.tmp)
        sizes = {"/m/image/x.safetensors": 7 * 1073741824}
        plan = budget.workload_plan(p, size_of=sizes.__getitem__, models="/m")
        line = budget.brief(plan, self.machine(),
                            budget.workload_verdict(plan, self.machine()),
                            estimate_hint=budget.WORKLOAD_BRIEF_HINT)
        self.assertIn("WORKLOAD_GTT_GIB", line)
        self.assertNotIn("MODEL_KV_KIB_PER_TOKEN", line)


class TestTheLockIdentity(Base):
    """Ultrareview finding (01.09.2026): the torch lane's measured figures
    were tied to nothing — `setup-venv.sh --relock` rewrites the venv the
    figures were measured on and workload_plan kept guarding them as
    measured. The lock file is the venv's identity (the torch lane's
    LLAMA_BIN); a profile that declares WORKLOAD_LOCK must carry the
    lock's sha256 in WORKLOAD_MEASURED_ON, checked at plan time — the
    exact symmetric of the pinned-build check beside it."""

    HERE = {"gfx": "gfx1151", "mem_total_gib": 124.9}

    def lock_file(self, content="torch==2.6.0\n"):
        import hashlib
        path = os.path.join(self.tmp, "requirements.lock")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return path, hashlib.sha256(content.encode()).hexdigest()

    def measured(self, lock_path, measured_on):
        return profile(
            self.tmp,
            WORKLOAD_LOCK=lock_path,
            WORKLOAD_GTT_GIB="0.0",
            WORKLOAD_GTT_SOURCE="01.09.2026, metered",
            WORKLOAD_HOST_RSS_GIB="4.4",
            WORKLOAD_HOST_RSS_SOURCE="01.09.2026, metered",
            WORKLOAD_MEASURED_ON=measured_on)

    def test_a_matching_lock_stays_measured(self):
        lock, digest = self.lock_file()
        p = self.measured(lock, "gfx1151, 124.9 GiB RAM, lock sha256:%s"
                          % digest[:12])
        plan = budget.workload_plan(p, models="/m", identity=self.HERE)
        self.assertFalse(plan.estimated)

    def test_a_relocked_venv_degrades_the_figures(self):
        lock, _ = self.lock_file("torch==2.7.0\nsomething-new==1.0\n")
        p = self.measured(lock, "gfx1151, 124.9 GiB RAM, lock "
                          "sha256:aaaaaaaaaaaa")
        plan = budget.workload_plan(p, models="/m", identity=self.HERE)
        self.assertTrue(plan.estimated)
        v = budget.workload_verdict(plan, self.machine())
        self.assertTrue(any("lock" in n for n in v.notes), v.notes)

    def test_a_measured_on_without_a_lock_stamp_degrades(self):
        lock, _ = self.lock_file()
        p = self.measured(lock, "gfx1151, 124.9 GiB RAM")
        plan = budget.workload_plan(p, models="/m", identity=self.HERE)
        self.assertTrue(plan.estimated)

    def test_an_unreadable_lock_degrades(self):
        p = self.measured(os.path.join(self.tmp, "no-such.lock"),
                          "gfx1151, 124.9 GiB RAM, lock sha256:aaaaaaaaaaaa")
        plan = budget.workload_plan(p, models="/m", identity=self.HERE)
        self.assertTrue(plan.estimated)


class TestFinishParity(Base):
    """The _finish refactor's own justification is that a flag added to
    one path cannot silently not exist on the other — pinned here: both
    paths emit the same --json shape."""

    def _json_keys(self, argv):
        import contextlib
        import io
        import json
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = budget.main(argv)
        self.assertEqual(rc, 0, buf.getvalue())
        return set(json.loads(buf.getvalue()).keys())

    def test_workload_and_profile_json_share_one_shape(self):
        self.assertEqual(self._json_keys(["--workload", "sdxl", "--json"]),
                         self._json_keys(["--profile", "qwen38", "--json"]))

    def test_static_and_check_run_on_both_paths(self):
        """The parity claim covered --json only; --static and --check are
        the other _finish forks (re-review, 01.09.2026). Both paths must
        take them without error."""
        import contextlib
        import io
        for argv in (["--workload", "sdxl", "--static"],
                     ["--profile", "qwen38", "--static"],
                     ["--workload", "sdxl", "--check", "--brief"],
                     ["--profile", "qwen38", "--check", "--brief"]):
            with contextlib.redirect_stdout(io.StringIO()):
                rc = budget.main(argv)
            self.assertIn(rc, (0, 1), argv)

    def test_brief_names_the_hf_cache_state(self):
        """--brief for an unmeasured hf-cache profile said 'the weights
        could not be read' while plan and verdict already named the state
        correctly — the one line a journal keeps told the wrong story
        (re-review, 01.09.2026)."""
        p = profile(self.tmp, WORKLOAD_FILES="hf-cache")
        plan = budget.workload_plan(p, models="/m")
        line = budget.brief(plan, self.machine(),
                            budget.workload_verdict(plan, self.machine()),
                            estimate_hint=budget.WORKLOAD_BRIEF_HINT)
        self.assertIn("hf-cache", line)
        self.assertNotIn("could not be read", line)


class TestMachineIdentity(Base):
    """Reports and measured profiles say WHICH machine, explicitly — the
    Medusa preparation that costs nothing. Deliberately WITHOUT the hostname
    the work order suggested: reports are committed and the repo is public,
    so the field carries the architecture and the RAM, which reproduce, and
    not the name, which only identifies."""

    def kfd(self, versions):
        d = os.path.join(self.tmp, "nodes")
        for i, v in enumerate(versions):
            nd = os.path.join(d, str(i))
            os.makedirs(nd)
            with open(os.path.join(nd, "properties"), "w") as fh:
                fh.write("cpu_cores_count 16\ngfx_target_version %d\n" % v)
        return os.path.join(d, "*", "properties")

    def test_gfx_from_kfd_target_version(self):
        ident = budget.machine_identity(kfd_glob=self.kfd([0, 110501]))
        self.assertEqual(ident["gfx"], "gfx1151")

    def test_no_kfd_is_unknown_not_a_crash(self):
        ident = budget.machine_identity(
            kfd_glob=os.path.join(self.tmp, "nowhere", "*", "properties"))
        self.assertIsNone(ident["gfx"])
        self.assertIn("mem_total_gib", ident)

    def test_the_report_writer_carries_it(self):
        """bench/run.py records the machine beside build and flags. Pinned at
        source level: the field must be written into the report dict."""
        src = (common.REPO / "bench" / "run.py").read_text(encoding="utf-8")
        self.assertIn("machine_identity", src)


class TestTheRegistry(Base):
    """Every profile actually checked into setup/workloads/ keeps the
    contract. Same shape as the model-profile checks in test_models.py.

    Each test asserts the registry is NON-EMPTY before looping — the positive
    control test_vacuity.py demands, and it is real: sdxl.env is the first
    consumer, and a test that goes green because the directory vanished would
    be the quiet kind of wrong."""

    KINDS = ("image", "audio", "video")
    MODES = ("batch", "server")

    def profiles(self):
        return sorted(WORKLOADS_DIR.glob("*.env"))

    def test_required_fields(self):
        envs = self.profiles()
        self.assertTrue(envs, "setup/workloads/ holds no profile — sdxl.env "
                              "was the first consumer")
        for env in envs:
            for field in ("WORKLOAD_TITLE", "WORKLOAD_KIND", "WORKLOAD_MODE",
                          "WORKLOAD_CMD", "WORKLOAD_FILES"):
                self.assertTrue(
                    systemdfile.variable(str(env), field),
                    "%s lacks %s" % (env.name, field))
            self.assertIn(systemdfile.variable(str(env), "WORKLOAD_KIND"),
                          self.KINDS, env.name)
            self.assertIn(systemdfile.variable(str(env), "WORKLOAD_MODE"),
                          self.MODES, env.name)

    def test_server_mode_names_its_port_and_probe(self):
        envs = self.profiles()
        self.assertTrue(envs)
        for env in envs:
            if systemdfile.variable(str(env), "WORKLOAD_MODE") != "server":
                continue
            self.assertTrue(systemdfile.variable(str(env), "WORKLOAD_PORT"),
                            env.name)
            self.assertTrue(
                systemdfile.variable(str(env), "WORKLOAD_READY_PATH"),
                env.name)

    def test_a_generation_batch_names_its_prompt(self):
        """WORKLOAD_CMD is split on whitespace with no shell quoting — a
        multi-word prompt cannot live in it. It lives in WORKLOAD_PROMPT,
        which the runner appends as ONE argument; a text-to-image or
        text-to-speech profile without one cannot run its own smoke test."""
        envs = self.profiles()
        self.assertTrue(envs)
        for env in envs:
            if systemdfile.variable(str(env), "WORKLOAD_KIND") not in (
                    "image", "audio", "video"):
                continue
            self.assertTrue(systemdfile.variable(str(env), "WORKLOAD_PROMPT"),
                            "%s: generation workload without WORKLOAD_PROMPT"
                            % env.name)

    def test_measured_figures_travel_with_source_and_machine(self):
        """Both figures or neither; a figure without a dated source and a
        machine is a copied number waiting to happen."""
        import re
        date = re.compile(r"\d{2}\.\d{2}\.\d{4}|\d{4}-\d{2}-\d{2}")
        envs = self.profiles()
        self.assertTrue(envs)
        for env in envs:
            gtt = systemdfile.variable(str(env), "WORKLOAD_GTT_GIB")
            rss = systemdfile.variable(str(env), "WORKLOAD_HOST_RSS_GIB")
            self.assertEqual(bool(gtt), bool(rss),
                             "%s: measure both or neither" % env.name)
            if not gtt:
                continue
            for src_field in ("WORKLOAD_GTT_SOURCE", "WORKLOAD_HOST_RSS_SOURCE"):
                src = systemdfile.variable(str(env), src_field) or ""
                self.assertTrue(date.search(src),
                                "%s: %s carries no date" % (env.name, src_field))
            self.assertTrue(
                systemdfile.variable(str(env), "WORKLOAD_MEASURED_ON"),
                "%s: measured figures without WORKLOAD_MEASURED_ON" % env.name)

    def test_every_profile_names_its_sources(self):
        """WORKLOAD_SOURCE: where the weights come from, machine-readable —
        the model registry has MODEL_SOURCE and get-model.sh; the workload
        registry's provenance lived in prose comments (including the
        valuable gated-VAE detour of flux) until the architecture review
        called the gap (01.09.2026). URLs, space-separated."""
        envs = self.profiles()
        self.assertTrue(envs)
        for env in envs:
            src = systemdfile.variable(str(env), "WORKLOAD_SOURCE")
            self.assertTrue(src, "%s lacks WORKLOAD_SOURCE" % env.name)
            for token in src.split():
                self.assertTrue(token.startswith("https://"),
                                "%s: source %r is not a URL" % (env.name,
                                                                token))

    def test_no_machine_paths_in_commands(self):
        """@HOME@ and @MODELS@, not /home/<someone> — the discipline
        test_localenv.py holds the rest of the repo to."""
        envs = self.profiles()
        self.assertTrue(envs)
        for env in envs:
            raw = (systemdfile.variable(str(env), "WORKLOAD_CMD") or "")
            self.assertNotIn("/home/", raw, env.name)

    def test_the_shell_registry_command_actually_runs(self):
        """`bash setup/lib/models.sh workloads` — EXECUTED, not grepped.
        The first version of this test grepped the source for the word
        'workloads', matched the function definition, and let a missing
        CLI verb ship: the documented command exited 2 with usage. A test
        of a command is a run of the command (review, 01.09.2026)."""
        import subprocess
        r = subprocess.run(
            ["bash", str(common.REPO / "setup" / "lib" / "models.sh"),
             "workloads"],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("sdxl", r.stdout)

    def test_measured_profiles_carry_the_determinism_pin(self):
        """WORKLOAD_SMOKE_SHA256: a measured profile without its output
        hash pin cannot ride the determinism lane (tests/live_media.sh) —
        and a pin without a dated source is a copied number waiting to
        happen. 64 hex chars, source with a date."""
        import re
        envs = self.profiles()
        self.assertTrue(envs)
        date = re.compile(r"\d{2}\.\d{2}\.\d{4}|\d{4}-\d{2}-\d{2}")
        pinned = 0
        for env in envs:
            measured = systemdfile.variable(str(env), "WORKLOAD_GTT_GIB")
            pin = systemdfile.variable(str(env), "WORKLOAD_SMOKE_SHA256")
            if measured:
                self.assertTrue(pin, "%s is measured but not pinned — the "
                                     "determinism lane cannot see it"
                                % env.name)
            if not pin:
                continue
            pinned += 1
            self.assertRegex(pin, r"^[0-9a-f]{64}$", env.name)
            src = systemdfile.variable(
                str(env), "WORKLOAD_SMOKE_SHA256_SOURCE") or ""
            self.assertTrue(date.search(src),
                            "%s: pin without a dated source" % env.name)
        self.assertTrue(pinned, "no profile carries a pin — the lane "
                                "checks nothing")

    def test_measured_on_names_the_pinned_build(self):
        """A profile whose CMD pins build-vulkan-<id> must carry the same
        id in WORKLOAD_MEASURED_ON: editing the CMD onto a new build while
        the old figures keep guarding is the silent-staleness the
        LLAMA_BIN rule exists for."""
        import re
        envs = self.profiles()
        self.assertTrue(envs)
        checked = 0
        for env in envs:
            cmd = systemdfile.variable(str(env), "WORKLOAD_CMD") or ""
            measured_on = systemdfile.variable(
                str(env), "WORKLOAD_MEASURED_ON") or ""
            m = re.search(r"build-vulkan-([0-9a-f]{7,})", cmd)
            if not m or not measured_on:
                continue
            checked += 1
            self.assertIn(m.group(1), measured_on,
                          "%s: figures measured on a build the CMD no "
                          "longer names" % env.name)
        self.assertTrue(checked, "no profile pinned a build — the contract "
                                 "checked nothing")

    def test_a_venv_run_profile_declares_its_lock(self):
        """The torch lane's symmetric of the pinned-build contract
        (ultrareview, 01.09.2026): a CMD that runs from a venv has no
        build id — the venv's identity is its requirements.lock. A
        measured venv profile must name the lock (WORKLOAD_LOCK,
        repo-relative) and carry its sha256 in WORKLOAD_MEASURED_ON, or
        `setup-venv.sh --relock` rewrites the measured stack while the
        old figures keep guarding."""
        import hashlib
        import re
        envs = self.profiles()
        self.assertTrue(envs)
        checked = 0
        for env in envs:
            cmd = systemdfile.variable(str(env), "WORKLOAD_CMD") or ""
            if "/.venvs/" not in cmd:
                continue
            if not systemdfile.variable(str(env), "WORKLOAD_GTT_GIB"):
                continue
            checked += 1
            lock_rel = systemdfile.variable(str(env), "WORKLOAD_LOCK")
            self.assertTrue(lock_rel,
                            "%s: a measured venv profile without "
                            "WORKLOAD_LOCK — the figures are tied to "
                            "nothing" % env.name)
            lock_path = common.REPO / lock_rel
            self.assertTrue(lock_path.exists(),
                            "%s: WORKLOAD_LOCK names %s, which does not "
                            "exist" % (env.name, lock_rel))
            digest = hashlib.sha256(lock_path.read_bytes()).hexdigest()
            measured_on = systemdfile.variable(
                str(env), "WORKLOAD_MEASURED_ON") or ""
            m = re.search(r"lock sha256:([0-9a-f]{8,})", measured_on)
            self.assertTrue(m and digest.startswith(m.group(1)),
                            "%s: WORKLOAD_MEASURED_ON does not carry the "
                            "current lock's sha256 (%s…) — relocked "
                            "without re-measuring?"
                            % (env.name, digest[:12]))
        self.assertTrue(checked, "no measured venv profile — the contract "
                                 "checked nothing")


if __name__ == "__main__":
    unittest.main()
