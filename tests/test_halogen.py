"""test_halogen — the container backend, and the places it reaches into.

Halogen Flash Server is the first backend here that is NOT llama-server: a
rootless podman container speaking OpenAI only, reached through the Anthropic
bridge. It therefore touches four things that every llama profile gets for
free — the mutual exclusion in 128 GB UMA, the reasoning vocabulary, who is
serving, and the prefix store — and each of those is a guard this file keeps.

tests/test_anthropic_bridge.py owns the protocol translation. This file owns
everything around it.
"""
import ast
import os
import re
import subprocess
import unittest
from unittest import mock

import common

REPO = common.REPO
GW = common.load("setup/gateway/gateway.py", "gateway_halo")
MODES = common.load("setup/gateway/modes.py", "modes_halo")


def profiles():
    """Every llama profile the repo knows, by name."""
    return sorted(p.stem for p in (REPO / "setup" / "env").glob("*.env"))


def unit_text(name):
    return (REPO / "setup" / "systemd" / name).read_text(encoding="utf-8")


# --------------------------------------------------------------- exclusion ---
class TestTheUnitExcludesEveryModel(unittest.TestCase):
    """Halogen and a llama-server in 128 GB UMA at once takes the machine
    down — 26.08.2026, three times. systemd's Conflicts= is the only thing
    that cannot be talked out of it, and it is a HAND-WRITTEN list.

    Exact set equality against the profiles, for the same reason
    test_models.TestConflicts demands it of llama-user@.service: a profile
    added tomorrow is a model Halogen would happily run beside.
    """

    def test_it_names_every_profile(self):
        for u in ("halogen.service", "halogen-qwen38flash.service", "halogen-qwen38.service"):
            with self.subTest(unit=u):
                text = unit_text(u)
                line = " ".join(
                    l.strip().rstrip("\\")
                    for l in text.splitlines()
                    if l.strip().startswith("Conflicts=") or _continues(text, l))
                got = sorted(w for w in line.replace("Conflicts=", " ").split()
                             if w.endswith(".service") and w.startswith("llama-user@"))
                self.assertEqual(
                    got, sorted("llama-user@%s.service" % m for m in profiles()),
                    "%s's Conflicts= no longer matches setup/env/ — a "
                    "profile it does not name can be started beside it, and both in "
                    "UMA is how this machine went down on 26.08.2026" % u)
                if u == "halogen.service":
                    self.assertIn("halogen-qwen38.service", line)
                    self.assertIn("halogen-qwen38flash.service", line)
                elif u == "halogen-qwen38flash.service":
                    self.assertIn("halogen-qwen38.service", line)
                    self.assertIn("halogen.service", line)
                elif u == "halogen-qwen38.service":
                    self.assertIn("halogen-qwen38flash.service", line)
                    self.assertIn("halogen.service", line)


# ------------------------------------------------- fit-to-room / unit scope ---
class TestFitToRoomIsPerUnit(unittest.TestCase):
    """HALOGEN_FIT_TO_ROOM clamps an over-budget max_tokens to the window the
    prompt left instead of a 400 the interactive client cannot act on. It is
    enabled PER UNIT on EVERY halogen unit, because which unit serves an
    interactive agent client is switch-model's call at runtime and the defect is
    not model-specific. It is deliberately NOT in the shared llm-stack.env
    template and NOT on a bench side server: a budget quietly shrunk under a
    measurement is the failure serve_api already documents, and a bench side
    server never runs through a unit, so the unit-only placement is also the
    isolation."""

    UNITS = ("halogen.service", "halogen-qwen38flash.service", "halogen-qwen38.service")

    def test_every_halogen_unit_sets_it(self):
        for u in self.UNITS:
            with self.subTest(unit=u):
                self.assertIn("Environment=HALOGEN_FIT_TO_ROOM=1", unit_text(u),
                              "%s lost the fit-to-room knob" % u)

    def test_not_in_the_shared_template(self):
        tpl = (REPO / "setup" / "local.env.template").read_text(encoding="utf-8")
        self.assertNotIn("HALOGEN_FIT_TO_ROOM", tpl,
                         "fit-to-room must stay per unit, never in the shared "
                         "template, or a bench side server --env llm-stack "
                         "would inherit it and shrink budgets under a run")

    def test_not_in_any_profile_env(self):
        profiles_dir = REPO / "setup" / "env"
        self.assertTrue(
            any(profiles_dir.glob("*.env")),
            "no setup/env/*.env profiles to scan — the guard is vacuous")
        for p in profiles_dir.glob("*.env"):
            self.assertNotIn("HALOGEN_FIT_TO_ROOM", p.read_text(encoding="utf-8"),
                             "%s must not carry fit-to-room" % p.name)

    def test_not_in_any_bench_side_server(self):
        bench_dir = REPO / "bench"
        self.assertTrue(
            (bench_dir / "sideserver.py").exists(),
            "bench/sideserver.py is the bench side server the guard must find")
        for p in bench_dir.glob("*.py"):
            self.assertNotIn("HALOGEN_FIT_TO_ROOM", p.read_text(encoding="utf-8"),
                             "%s must not set fit-to-room" % p.name)


# ------------------------------------------------ quality sidecar / off ---
class TestTheQualitySidecarIsServed(unittest.TestCase):
    """The Flash-Next units serve the checkpoint WITH its quality sidecar —
    upstream's default — by not setting HALOGEN_CK_OVERLAY at all.

    From 23.09. to 24.09.2026 they served it bare (HALOGEN_CK_OVERLAY=none):
    five real stalls ("Jetzt die Änderungen am Command selbst:" then
    <|im_end|>) ended again 17/50 with the sidecar and 0/50 bare. Those five
    were picked where the sidecar stalled. At 10 positions it did NOT stall
    on, bare ended 10/100 (0.12.3: 15/100) and the sidecar 3/100, 24.09.2026
    (bench/reports/2026-09-24_halogen-0.13.8/, upstream #89): each checkpoint
    has its own positions, roughly 6 % vs 10 % of announcements — an
    estimate. Bare's case fell; the sidecar's ~3.8 % perplexity on prose
    (upstream's figure) decided it, operator's call 24.09.

    halogenexec still forwards the variable: bench side servers set it.
    """

    UNITS = ("halogen.service", "halogen-qwen38flash.service")

    def test_no_flash_unit_turns_it_off(self):
        for u in self.UNITS:
            with self.subTest(unit=u):
                set_lines = [l for l in unit_text(u).splitlines()
                             if l.startswith("Environment") and "HALOGEN_CK_OVERLAY" in l]
                self.assertEqual(set_lines, [], "%s overrides the sidecar default" % u)

    def test_halogenexec_forwards_it(self):
        text = (REPO / "setup" / "halogenexec").read_text(encoding="utf-8")
        m = re.search(r"^for var in (.*?); do$", text, re.M | re.S)
        self.assertIsNotNone(m, "halogenexec's env allowlist loop is gone")
        names = m.group(1).replace("\\\n", " ").split()
        self.assertIn("HALOGEN_TEMPERATURE", names,
                      "parsed the wrong loop — the guard would be vacuous")
        self.assertIn("HALOGEN_CK_OVERLAY", names,
                      "a side server's HALOGEN_CK_OVERLAY would not reach "
                      "the container")


# ------------------------------------------------------- prompt cache on disk ---
class TestTheDiskCacheReachesTheContainer(unittest.TestCase):
    """HALOGEN_CACHE_DIR (upstream 0.10.0) names a directory INSIDE the
    container, and halogenexec's allowlist drops every variable it does not
    name — so setting it in llm-stack.env did nothing at all, silently. The
    host directory has to be mounted and the container told where it went.
    Measured need 24.09.2026: a main session evicted while its subagents ran
    came back cold, 53,199 tokens, 51.7 s of prefill.
    """

    def text(self):
        return (REPO / "setup" / "halogenexec").read_text(encoding="utf-8")

    def test_the_host_directory_is_mounted_and_named_inside(self):
        t = self.text()
        self.assertRegex(t, r'-v "\$CACHE_DIR_WANT:/cache', "no mount of the cache dir")
        self.assertIn("HALOGEN_CACHE_DIR=/cache", t,
                      "the container is not told where the mount is")

    def test_its_bounds_are_forwarded(self):
        m = re.search(r"^for var in (.*?); do$", self.text(), re.M | re.S)
        names = m.group(1).replace("\\\n", " ").split()
        for v in ("HALOGEN_CACHE_DISK_GIB", "HALOGEN_CACHE_PRUNE_OLD"):
            self.assertIn(v, names, "%s would not reach the container" % v)

    def test_it_is_opt_in(self):
        t = self.text()
        i = t.index('-v "$CACHE_DIR_WANT:/cache')
        guard = t.rfind('if [ -n "$CACHE_DIR_WANT" ]', 0, i)
        self.assertGreater(guard, 0, "the mount is not behind an opt-in")


def _continues(text, line):
    """A continuation line of a backslash-wrapped directive."""
    lines = text.splitlines()
    try:
        i = lines.index(line)
    except ValueError:
        return False
    while i > 0:
        prev = lines[i - 1].rstrip()
        if not prev.endswith("\\"):
            return False
        if prev.lstrip().startswith("Conflicts="):
            return True
        i -= 1
    return False


# ------------------------------------------------------------ the modes ------
class TestTheReasoningVocabularyIsMeasured(unittest.TestCase):
    """What this model's template does with a level was MEASURED, and the
    declaration has to match the measurement.

    Rendered inside the running container against
    /models/tokenizer/chat_template.jinja, 12.09.2026:

        xhigh   a system block, "think carefully through the task…"
        medium  NO system block at all — the neutral middle
        low     a system block, "keep your thinking brief and focused…"
        absent  renders exactly like xhigh: xhigh IS the default
        high · max · none · minimal   the template RAISES

    So `high` is not a level this template has, and declaring it as one is
    an HTTP 500 waiting for whoever selects it — rescued today only by
    serve_api.py's EFFORT_MAP, which is a second translation nobody named.
    modes.check_modes() exists to refuse exactly that, and it can only do so
    if the levels are declared beside the modes.
    """

    RENDERS = {"low", "medium", "xhigh"}

    def test_the_declared_levels_are_the_measured_ones(self):
        self.assertEqual(set(GW.HALOGEN_TEMPLATE_LEVELS), self.RENDERS)

    def test_every_mode_survives_check_modes(self):
        """The same guard every profile passes at gateway start."""
        MODES.check_modes(GW.HALOGEN_MODES, set(GW.HALOGEN_TEMPLATE_LEVELS))

    def test_no_mode_sends_a_level_the_template_raises_on(self):
        # Positive control: an empty table would make every assertion below
        # vacuous, and "no modes at all" is a state this backend can reach by
        # a typo in one dict.
        self.assertGreaterEqual(len(GW.HALOGEN_MODES), 3)
        for name, value in GW.HALOGEN_MODES.items():
            for part in str(value).split("+"):
                if part in (MODES.ON, MODES.OFF):
                    continue
                with self.subTest(mode=name):
                    self.assertIn(
                        part, self.RENDERS,
                        "%s:%s sends %r, which this template answers with a "
                        "TemplateError" % (name, value, part))


class TestNoTwoNamesMeanTheSameThing(unittest.TestCase):
    """A picker that offers one behaviour twice is the defect modes.names()
    was written against ("naming all three offers one choice three times").

    The bare alias is special here and deliberately so: llama-server has a
    command line for the gateway to leave alone, a container has none, so
    the gateway spells the default out — thinking OFF, because Claude Code's
    tool loops are what this backend is for. That makes a `none` mode an
    exact synonym of the bare alias, and measuring it is how it was found:
    the two produced a byte-identical upstream body.
    """

    ALIAS = "halogen-qwen3.8-flash-next"

    def bodies(self, *models):
        out = []
        for m in models:
            p = {"model": m, "messages": [{"role": "user", "content": "hi"}],
                 "max_tokens": 8}
            p, _ = GW.inject_model_kwargs(p, served=self.ALIAS,
                                          modes=dict(GW.HALOGEN_MODES))
            out.append({k: v for k, v in p.items()
                        if k in ("enable_thinking", "reasoning_effort")})
        return out

    def setUp(self):
        self.flavor = GW.BACKEND_FLAVOR
        GW.BACKEND_FLAVOR = "openai"
        self.addCleanup(lambda: setattr(GW, "BACKEND_FLAVOR", self.flavor))
        self.log = mock.patch.object(GW, "log", lambda *a: None)
        self.log.start()
        self.addCleanup(self.log.stop)

    def test_every_offered_name_behaves_differently(self):
        """Within ONE naming scheme. The stable `local` family deliberately
        repeats every behaviour under a second name — that is its whole
        purpose and it is a different thing from offering one choice twice
        inside one scheme, which is what this guards."""
        names = [n for n in MODES.names(self.ALIAS, dict(GW.HALOGEN_MODES))
                 if n == self.ALIAS or n.startswith(self.ALIAS + "-")]
        # Positive control: the bare alias plus one name per distinct
        # behaviour. Fewer than that and the comparison below has nothing to
        # compare.
        self.assertGreaterEqual(len(names), 4, names)
        bodies = self.bodies(*names)
        self.assertEqual(len(bodies), len(names))
        seen = {}
        for name, body in zip(names, bodies):
            key = tuple(sorted(body.items()))
            self.assertNotIn(
                key, seen,
                "%s and %s send the identical body %r — one of them is a name "
                "for nothing" % (seen.get(key), name, body))
            seen[key] = name

    def test_the_stable_family_mirrors_them_exactly(self):
        """`local-low` must be the same request as the model's own `-low`,
        byte for byte — otherwise the two names are two cache keys for one
        prompt."""
        for word in ("low", "medium", "high"):
            with self.subTest(word=word):
                own = self.bodies("%s-%s" % (self.ALIAS, word))[0]
                stable = self.bodies("%s-%s" % (MODES.STABLE, word))[0]
                self.assertEqual(own, stable)
        self.assertEqual(self.bodies(MODES.STABLE)[0],
                         self.bodies(self.ALIAS)[0])

    def test_the_bare_alias_does_not_think(self):
        self.assertEqual(self.bodies(self.ALIAS)[0],
                         {"enable_thinking": False})

    def test_an_unknown_model_name_falls_back_to_not_thinking(self):
        """A stale ANTHROPIC_MODEL, or another provider's name. Falling
        through to the template default would be xhigh — the most expensive
        mode this model has, chosen by an accident."""
        self.assertEqual(self.bodies("claude-sonnet-4-5")[0],
                         {"enable_thinking": False})


class TestAnotherModelsNameDoesNotResolveHere(unittest.TestCase):
    """The defect modes.py was written to end, arriving from the side.

    Its docstring: `qwen38-think` was still answered after a switch —
    "injecting qwen38's thinking mode into requests bound for another model,
    over a command line that had set it otherwise, with no error anywhere".
    The fix was to DERIVE the names from what is served, so a name belonging
    to another model cannot exist.

    The Halogen path then added a fallback that tried `flashnext` as an alias
    as well, so `flashnext-low` — which is what setup/claude/local.json still
    says — resolved against this backend's modes and turned thinking ON for
    a consumer that had been configured for a different model. Convenient,
    silent, and the same defect.

    The two spellings of THIS backend stay: the served alias is the long
    name, and `halogen-low` is the short one for the same thing.
    """

    SERVED = "halogen-qwen3.8-flash-next"

    def setUp(self):
        self.flavor = GW.BACKEND_FLAVOR
        GW.BACKEND_FLAVOR = "openai"
        self.addCleanup(lambda: setattr(GW, "BACKEND_FLAVOR", self.flavor))
        self.log = mock.patch.object(GW, "log", lambda *a: None)
        self.log.start()
        self.addCleanup(self.log.stop)

    def ask(self, model):
        p = {"model": model, "messages": [], "max_tokens": 4}
        p, _ = GW.inject_model_kwargs(p, served=self.SERVED,
                                      modes=dict(GW.HALOGEN_MODES))
        return {k: v for k, v in p.items()
                if k in ("enable_thinking", "reasoning_effort")}

    def test_the_short_spelling_of_this_backend_resolves(self):
        self.assertEqual(self.ask("halogen-low"),
                         {"enable_thinking": True, "reasoning_effort": "low"})

    def test_the_long_spelling_resolves(self):
        self.assertEqual(self.ask(self.SERVED + "-medium"),
                         {"enable_thinking": True,
                          "reasoning_effort": "medium"})

    def test_another_models_slug_does_not(self):
        """It falls through to the bare alias — thinking off — and the
        gateway says so once per name. A stale ANTHROPIC_MODEL then produces
        a visible note instead of a silent mode change."""
        self.assertEqual(self.ask("flashnext-low"), {"enable_thinking": False})
        self.assertEqual(self.ask("qwen38-high"), {"enable_thinking": False})


class TestWhatTheGatewayInjectsForAContainerBackend(unittest.TestCase):
    def setUp(self):
        self.flavor = GW.BACKEND_FLAVOR
        GW.BACKEND_FLAVOR = "openai"
        self.addCleanup(lambda: setattr(GW, "BACKEND_FLAVOR", self.flavor))

    def inject(self, p, modes=None):
        p, _ = GW.inject_model_kwargs(p, served="halogen-qwen3.8-flash-next",
                                      modes=modes if modes is not None
                                      else dict(GW.HALOGEN_MODES))
        return p

    def test_a_stream_asks_for_the_usage_block(self):
        """Without it Halogen emits no usage at all and every column in the
        trace stays empty."""
        p = self.inject({"model": "halogen-qwen3.8-flash-next", "stream": True,
                         "messages": []})
        self.assertEqual(p["stream_options"], {"include_usage": True})

    def test_a_client_that_asked_for_its_own_stream_options_keeps_them(self):
        p = self.inject({"model": "halogen-qwen3.8-flash-next", "stream": True,
                         "messages": [], "stream_options": {"include_usage": False}})
        self.assertEqual(p["stream_options"], {"include_usage": False})

    def test_the_two_end_tokens_are_added_as_stop_strings(self):
        """Belt and braces with serve_api.py's own EOS registration: the
        patched server turns client stop STRINGS into token ids, so these two
        arrive as EOS ids even if the mount is ever lost. Qwen-specific, and
        that is why they are keyed on the served model rather than sent to
        every OpenAI backend."""
        p = self.inject({"model": "halogen-qwen3.8-flash-next", "messages": []})
        self.assertEqual(p["stop"], ["<|im_end|>", "<|endoftext|>"])

    def test_a_clients_own_stop_list_is_kept(self):
        p = self.inject({"model": "halogen-qwen3.8-flash-next", "messages": [],
                         "stop": "###"})
        self.assertEqual(p["stop"][0], "###")

    def test_nothing_is_injected_for_a_llama_backend(self):
        GW.BACKEND_FLAVOR = "anthropic"
        p = self.inject({"model": "qwen36", "stream": True, "messages": []},
                        modes={})
        self.assertNotIn("stop", p)
        self.assertNotIn("stream_options", p)
        self.assertNotIn("enable_thinking", p)


class TestTheInjectionExistsOnce(unittest.TestCase):
    """Two copies of the same eight lines, one in each arm of
    inject_model_kwargs. The lesson is a day old at this point: a guard that
    exists twice loses a clause in one of them — that is what happened to
    switch-model.sh's store block, and it is why this one is a function."""

    def test_the_stop_tokens_are_written_down_once(self):
        src = (REPO / "setup" / "gateway" / "gateway.py").read_text(
            encoding="utf-8")
        self.assertEqual(
            src.count('"<|im_end|>", "<|endoftext|>"'), 1,
            "the Qwen end tokens are spelled out more than once")

    def test_both_arms_go_through_it(self):
        """With modes and without: the second arm is what an unknown served
        model falls into, and it must still get the stop tokens."""
        flavor = GW.BACKEND_FLAVOR
        GW.BACKEND_FLAVOR = "openai"
        self.addCleanup(lambda: setattr(GW, "BACKEND_FLAVOR", flavor))
        for modes in ({}, dict(GW.HALOGEN_MODES)):
            with self.subTest(modes=bool(modes)):
                p = {"model": "whatever", "messages": [], "stream": True}
                with mock.patch.object(GW, "log", lambda *a: None):
                    p, _ = GW.inject_model_kwargs(
                        p, served="halogen-qwen3.8-flash-next", modes=modes)
                self.assertEqual(p["stop"], ["<|im_end|>", "<|endoftext|>"])
                self.assertIs(p["enable_thinking"], False)
                self.assertEqual(p["stream_options"], {"include_usage": True})


class TestTheSlotCountIsNotGuessedForALlamaBackend(unittest.TestCase):
    """query_slots() decides MAX_INFLIGHT. A container has no /slots, so a
    404 means "one" — but the same function also learned to accept /health as
    an answer on ANY exception, and for llama-server that turns one flaky
    moment during startup into a permanent MAX_INFLIGHT of 1 instead of the
    retry the wait parameter exists for."""

    def stub(self, slots_raises, health_ok):
        import urllib.error
        import io

        class Resp(io.BytesIO):
            status = 200
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def fake(url, timeout=None):
            if url.endswith("/slots"):
                raise slots_raises
            if url.endswith("/health"):
                if health_ok:
                    return Resp(b"{}")
                raise OSError("refused")
            raise AssertionError(url)
        return fake

    def test_a_404_on_slots_is_one_slot(self):
        import urllib.error
        import urllib.request
        flavor = GW.BACKEND_FLAVOR
        GW.BACKEND_FLAVOR = "openai"
        self.addCleanup(lambda: setattr(GW, "BACKEND_FLAVOR", flavor))
        err = urllib.error.HTTPError("u", 404, "nope", {}, None)
        with mock.patch.object(urllib.request, "urlopen",
                               self.stub(err, True)):
            self.assertEqual(GW.query_slots(wait=0), 1)

    def test_a_llama_backend_does_not_take_health_for_an_answer(self):
        """A refused connection is not "one slot". It is "ask again", and
        with wait=0 it is None — which main() already knows how to report."""
        import urllib.request
        flavor = GW.BACKEND_FLAVOR
        GW.BACKEND_FLAVOR = "anthropic"
        self.addCleanup(lambda: setattr(GW, "BACKEND_FLAVOR", flavor))
        with mock.patch.object(urllib.request, "urlopen",
                               self.stub(OSError("refused"), True)):
            self.assertIsNone(GW.query_slots(wait=0))

    def test_a_container_backend_does(self):
        import urllib.request
        flavor = GW.BACKEND_FLAVOR
        GW.BACKEND_FLAVOR = "openai"
        self.addCleanup(lambda: setattr(GW, "BACKEND_FLAVOR", flavor))
        with mock.patch.object(urllib.request, "urlopen",
                               self.stub(OSError("refused"), True)):
            self.assertEqual(GW.query_slots(wait=0), 1)


class TestTheBackendFlavourIsPinnedAtStartup(unittest.TestCase):
    def test_main_declares_the_global_it_writes(self):
        """`BACKEND_FLAVOR = "openai"` inside main() without a global
        statement is a local variable and a no-op. It went unnoticed because
        backend_is_openai_only() sniffs SERVED as a second route — a fallback
        hiding a dead line."""
        src = (REPO / "setup" / "gateway" / "gateway.py").read_text(
            encoding="utf-8")
        tree = ast.parse(src)
        main = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "main")
        writes = any(isinstance(n, ast.Assign)
                     and any(getattr(t, "id", None) == "BACKEND_FLAVOR"
                             for t in n.targets)
                     for n in ast.walk(main))
        if not writes:
            return
        declared = {n for g in ast.walk(main) if isinstance(g, ast.Global)
                    for n in g.names}
        self.assertIn("BACKEND_FLAVOR", declared,
                      "main() assigns BACKEND_FLAVOR without declaring it "
                      "global — the assignment never leaves the function")


class TestTheImageAndTheModelsAreResolvedOnce(unittest.TestCase):
    """setup/lib/models.sh exists because the set of models used to live in
    six places. The container backend promptly grew three: halogenexec,
    switch-model.sh and bench/run_halogen.py each resolved the model
    directory with their own copy of the same fallback chain, and two of them
    wrote the image tag out by hand.

    Bump one tag and the others serve or MEASURE a different version while
    reporting the new one — the silent direction, and the reason the vendored
    front-end's BASE_SHA256 is tied to whichever tag is actually started.
    """

    USERS = ("setup/halogenexec", "setup/switch-model.sh",
             "bench/run_halogen.py")

    def test_the_image_is_written_down_in_exactly_one_place(self):
        import subprocess
        # --untracked: a file that is written but not yet committed is
        # exactly the state this integration was in when the tag got spelled
        # out for the third time.
        hits = subprocess.run(
            ["git", "-C", str(REPO), "grep", "-l", "--untracked",
             "--exclude-standard", "halogen-flash-server:[0-9]"],
            capture_output=True, text=True)
        # bench/reports/** is exempt and must be: a report states the
        # conditions a measurement ran under, and the image it ran against is
        # one of them. A recorded fact about a past run is not a
        # configuration anybody will bump.
        files = sorted(f for f in hits.stdout.split()
                       if f and not f.startswith("bench/reports/"))
        self.assertEqual(
            files, ["setup/halogen/serve_api.py", "setup/lib/models.sh",
                    "tests/fixtures/halogen/tool_parse.py"],
            "the image tag is spelled out in %s — one of them will be "
            "forgotten on the next bump. models.sh answers `halogen-image`; "
            "serve_api.py names the tag it was CUT from, which is a different "
            "statement and is compared against the running one by halogenexec; "
            "the tool_parse fixture names the release the OFFLINE tests parse "
            "against, which nothing refuses at runtime and only "
            "test_the_fixture_is_the_release_that_is_started notices." % files)

    def test_models_sh_answers_for_the_image(self):
        import subprocess
        r = subprocess.run(["bash", str(REPO / "setup" / "lib" / "models.sh"),
                            "halogen-image"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertRegex(r.stdout.strip(),
                         r"^\S+/halogen-flash-server:\d+\.\d+\.\d+$")

    def test_models_sh_answers_for_the_model_directory(self):
        import subprocess
        r = subprocess.run(["bash", str(REPO / "setup" / "lib" / "models.sh"),
                            "halogen-models"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        out = r.stdout.strip()
        self.assertTrue(out.startswith("/"), out)

    def test_the_fetcher_downloads_where_the_runner_looks(self):
        """124 GiB is an expensive way to find out that two scripts disagree.

        fetch-halogen.sh defaulted to `$(models_dir)/halogen-models` — INSIDE
        the .gguf directory — while halogenexec looked for
        `$(dirname $(models_dir))/halogen-models` BESIDE it. On a machine
        without the bundle already in place, the fetch would land somewhere
        nothing serves from and the service would refuse to start with "no
        halogen-models directory found".
        """
        src = (REPO / "setup" / "scripts" / "fetch-halogen.sh").read_text(
            encoding="utf-8")
        code = "\n".join(l for l in src.splitlines()
                         if not l.lstrip().startswith("#"))
        self.assertIn("halogen_models_dir", code,
                      "the fetcher works out its own destination instead of "
                      "asking the one function that knows")
        self.assertNotIn("$(models_dir", code)

    def test_no_user_resolves_the_directory_on_its_own(self):
        import os as _os
        offenders = []
        for rel in self.USERS:
            text = (REPO / rel).read_text(encoding="utf-8")
            code = "\n".join(l for l in text.splitlines()
                             if not l.lstrip().startswith("#"))
            if "halogen-models" in code and "halogen_models_dir" not in code \
                    and "halogen-models\"" not in code:
                offenders.append(rel)
        self.assertFalse(
            offenders,
            "these carry their own copy of the model-directory fallback "
            "chain: %s" % offenders)


# ------------------------------------------------------------- who serves ---
class TestServingIsReadOffTheProcess(unittest.TestCase):
    """CLAUDE.md: "Ask `serving`, never `is-active`: only the first can say
    which of two started instances won the race for the port."

    The halogen branch arrived asking `systemctl is-active`, which is true
    from the moment podman starts — roughly twelve seconds before the engine
    listens. A caller that stops production on that answer stops it for a
    backend that is not up yet.
    """

    def test_the_shell_function_does_not_ask_is_active(self):
        src = (REPO / "setup" / "lib" / "models.sh").read_text(encoding="utf-8")
        body = src[src.index("models_serving()"):]
        body = body[:body.index("\n}")]
        # Comments stripped: the line explaining why is-active is NOT used is
        # the point of this change, and a test that cannot tell an
        # explanation from a call would delete it.
        code = "\n".join(l for l in body.splitlines()
                         if not l.lstrip().startswith("#"))
        self.assertNotIn("is-active", code,
                         "models_serving() asks systemd instead of the "
                         "process that holds the port")

    def test_it_answers_with_whatever_is_actually_up(self):
        """Run for real. Whatever this machine serves, the answer has to be
        a name this repo can map back to a unit — or nothing at all."""
        r = subprocess.run(["bash", str(REPO / "setup" / "lib" / "models.sh"),
                            "serving"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        for name in r.stdout.split():
            with self.subTest(name=name):
                self.assertTrue(name.startswith("halogen") or name in profiles(),
                                "%r is neither a profile nor halogen" % name)

    def test_there_is_a_verb_that_names_the_unit(self):
        """Callers used to build `llama-user@%s` from the alias. With a
        container in the picture that produces `llama-user@halogen`, a unit
        that does not exist — so the mapping belongs where the detection is."""
        r = subprocess.run(["bash", str(REPO / "setup" / "lib" / "models.sh"),
                            "serving-unit"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        for unit in r.stdout.split():
            with self.subTest(unit=unit):
                self.assertTrue(unit.endswith(".service"), unit)
                self.assertNotEqual(unit, "llama-user@halogen.service")


class TestTheHealthCheckLooksAtTheRealStore(unittest.TestCase):
    """setup/check.sh reports on the prefix store, and it was reading the same
    hard-wired path switch-model.sh used to.

    Measured on this machine 12.09.2026, with the container backend serving:

        = prefixes belong to halogen (4,0K)

    — green, on a 4 KiB directory holding one marker file, while the store the
    gateway actually uses held 89 GiB marked `.owner=qwen36`. The line exists
    to catch exactly that mismatch ("a restore would feed one model's KV state
    to another") and it was looking at the wrong directory to see it.

    And the same run printed, in red:

        ! MAX_INFLIGHT is 1, the server has /home/…/.cache/llama-slots slots
          — restart the gateway

    because `SLOTS` holds the store PATH until it is overwritten with a slot
    COUNT — and that overwrite sits inside `if pgrep -x llama-server`, which a
    container backend never satisfies. One name, two meanings, and the
    operator is told to restart something over a string comparison.
    """

    SRC = REPO / "setup" / "check.sh"

    def setUp(self):
        self.src = self.SRC.read_text(encoding="utf-8")

    def code(self):
        return "\n".join(l for l in self.src.splitlines()
                          if not l.lstrip().startswith("#"))

    def test_the_store_path_is_resolved_not_written_down(self):
        self.assertNotIn('$HOME/.cache/llama-slots', self.code(),
                         "check.sh writes the store path out by hand — the "
                         "gateway resolves it and the two will drift")
        self.assertIn("slots_dir", self.code())

    def test_the_path_and_the_count_are_different_names(self):
        """`SLOTS=` for a directory and `SLOTS=` for a number is how the
        second one silently keeps the first one's value."""
        assigns = [l.split("=")[0].strip() for l in self.src.splitlines()
                   if not l.lstrip().startswith("#")
                   and l.lstrip().startswith("SLOTS=")]
        self.assertLessEqual(
            len(assigns), 1,
            "SLOTS is assigned %d times and means two different things"
            % len(assigns))

    def test_a_parked_store_is_not_reported_as_a_hazard(self):
        """"a restore would feed one model's KV state to another" is true of
        llama-server and false of a backend that never touches the store. The
        line fired red on this machine with the container serving, where the
        gateway's AUTO_SAVE and session store are both off — an alarm that
        cannot be acted on is an alarm the operator learns to skip."""
        self.assertIn("backend_uses_slot_store", self.code())

    def test_the_slot_count_check_needs_a_count(self):
        """No llama-server means no /slots and therefore no count. Comparing
        MAX_INFLIGHT against whatever the variable happens to hold is the
        defect; saying nothing is the correct answer."""
        code = self.code()
        self.assertIn("N_SLOTS", code,
                      "the slot COUNT has no name of its own")
        self.assertNotIn('[ "$SLOTS" != "0" ]', code)


# ------------------------------------------------------- the prefix store ---
class TestTheSwitchKeepsItsGuards(unittest.TestCase):
    """switch-model.sh grew a parallel `IS_HALOGEN` preflight, and two guards
    did not travel with the copy.
    """

    def setUp(self):
        self.src = (REPO / "setup" / "switch-model.sh").read_text(
            encoding="utf-8")

    def test_an_unattributable_store_stops_the_switch_either_way(self):
        """The die exists because restoring one model's KV state into
        another is silent and ruinous. The halogen preflight had its own copy
        of the store logic WITHOUT it: a store nobody could attribute was not
        refused, it was relabelled `.owner=halogen` and its real owner lost.
        There is one block now, and it runs for every backend."""
        block = self.src[self.src.index('PARK_AS=""; RESTORE=0'):]
        block = block[:block.index('[ "$DRY" = 1 ]')]
        self.assertIn("nothing says which model wrote them", block)
        self.assertNotIn("IS_HALOGEN", block,
                         "the shared store block branches on the backend "
                         "again")

    def test_only_one_server_is_verified_for_halogen_too(self):
        """"The check no unit file can argue away" — dropped for halogen,
        where a hand-written Conflicts= line is the only thing standing
        between two engines and 128 GB of UMA. One implementation now, so it
        cannot be dropped from one branch again."""
        wait = self.src[self.src.index('step "5/7'):]
        wait = wait[:wait.index("# ONLY NOW the gateway")]
        self.assertEqual(wait.count("models_serving"), 1,
                         "step 5 asks who is serving %d times — one branch "
                         "with the check and one without is how it was lost"
                         % wait.count("models_serving"))
        self.assertEqual(wait.count("MORE THAN ONE"), 1)
        self.assertNotIn('if [ "$IS_HALOGEN" = 1 ]; then', wait,
                         "step 5 has forked again")

    def test_the_store_block_exists_once(self):
        """Both preflights carried a copy, and the halogen one had lost the
        refusal. A guard that exists twice loses a clause in one of them."""
        self.assertEqual(self.src.count('PARK_AS=""; RESTORE=0'), 1)
        self.assertEqual(
            self.src.count("nothing says which model wrote them"), 1)

    def test_the_store_is_resolved_not_spelled_out(self):
        """switch-model.sh parked and relabelled ~/.cache/llama-slots while
        the gateway saved into whatever LLAMA_SLOTS names — 89 GiB in one
        directory, an 8-byte marker in the other. tests/test_models.py
        TestTheStoreHasOneAddress owns the contract; this is the Halogen-side
        reminder of how it was found."""
        line = [l for l in self.src.splitlines() if l.startswith("SLOTS=")]
        self.assertEqual(len(line), 1)
        self.assertIn("slots_dir", line[0])


# ------------------------------------------------- the vendored serve_api ---
class TestTheVendoredServerDeclaresWhatItPatches(unittest.TestCase):
    """setup/halogen/serve_api.py is mounted OVER the file in the image.

    FOUR changes in five places — both EOS ids, per-request stop strings as
    EOS ids, the app.state bindings these tests use, and the fit-to-room gate
    — shipped as a 260 KB whole-file copy. Bump the image and the old copy
    silently reverts every upstream change in that file while `podman ps`
    reports the new tag — the exact shape of a defect this repo keeps
    finding. So the file has to say which image it was cut from, and
    halogenexec has to refuse the mount when that is no longer true.

    The count and the size were wrong here until 21.09.2026 ("three hunks",
    "150 KB"), the third place carrying that error after the header and
    setup/README.md were corrected on 17.09. — which is the argument for
    asserting on the file rather than describing it in prose.

    tool_parse.py was vendored beside it until 21.09.2026 and is not any
    more: its only change was the header, so the mount froze a file it did
    not patch and would have reverted 0.10.2's merge of several leading
    system messages.
    """

    PATH = REPO / "setup" / "halogen" / "serve_api.py"

    def test_it_names_the_image_it_was_cut_from(self):
        head = self.PATH.read_text(encoding="utf-8")[:4000]
        self.assertRegex(
            head, r"halogen-flash-server:\d+\.\d+\.\d+",
            "the vendored copy does not say which image tag it patches")
        self.assertRegex(
            head, r"BASE_SHA256\s*[:=]\s*[0-9a-f]{64}",
            "without the base file's hash nothing can notice that the image "
            "moved underneath this copy")

    def test_tool_parse_is_not_vendored_any_more(self):
        """Retired 21.09.2026, and it must not come back by reflex.

        The copy carried no change but its own header, so the mount froze a
        file it did not patch: 0.10.2 added the merge of several leading
        system messages to it (public issue #60) and a stale copy would have
        reverted that. Anything that needs patching there needs a hunk and a
        line in the header, not a whole-file copy.
        """
        self.assertFalse(
            (REPO / "setup" / "halogen" / "tool_parse.py").exists(),
            "tool_parse.py is vendored again — if that is deliberate it needs "
            "a hunk it actually changes, a BASE_SHA256 in halogenexec and a "
            "retirement condition; if it is not, delete it")

    def test_the_four_changes_are_actually_in_the_file(self):
        """The header CLAIMS four changes; this asserts they are there.

        Written 21.09.2026 because the prose count drifted three times
        (header, README, this class's docstring) while nothing checked the
        file. A re-cut that loses one of these silently serves a
        half-reverted front-end, which is the failure the whole vendoring
        apparatus exists to prevent.
        """
        src = self.PATH.read_text(encoding="utf-8")
        for marker, what in (
                ("def get_eos_ids", "change 1/2: the end-token function"),
                ("<|endoftext|>", "change 1: the second end token"),
                ("req_eos = get_eos_ids(stops)",
                 "change 2: per-request stop strings as EOS ids"),
                ("app.state.run", "change 3: the test bindings"),
                ("app.state.serve", "change 3: the test bindings"),
                ("FIT_TO_ROOM", "change 4: the fit-to-room gate"),
                ("fit-to-room clamped max_tokens",
                 "change 4: the clamp's own stderr line")):
            self.assertIn(marker, src,
                          "%s is missing from the vendored copy" % what)

    def test_halogenexec_asks_for_the_image_rather_than_naming_it(self):
        src = (REPO / "setup" / "halogenexec").read_text(encoding="utf-8")
        code = "\n".join(l for l in src.splitlines()
                         if not l.lstrip().startswith("#"))
        self.assertIn("halogen_image", code)
        self.assertNotIn("halogen-flash-server:", code)

    def test_halogenexec_refuses_a_mount_it_cannot_verify(self):
        src = (REPO / "setup" / "halogenexec").read_text(encoding="utf-8")
        self.assertIn("BASE_SHA256", src,
                      "halogenexec mounts the copy without checking that the "
                      "image still carries the file it was cut from")
        code = "\n".join(l for l in src.splitlines()
                         if not l.lstrip().startswith("#"))
        self.assertNotIn("tool_parse", code,
                         "halogenexec still mounts or verifies tool_parse.py, "
                         "which was retired 21.09.2026 — see "
                         "test_tool_parse_is_not_vendored_any_more")

    def test_the_image_tag_agrees_with_the_one_that_is_started(self):
        """The header names the image the copy was CUT from; models.sh names
        the one that gets STARTED. When those differ the mount is a patch
        against a file that is no longer there — which halogenexec would
        refuse at runtime, loudly, but only after a failed start."""
        import subprocess
        head = self.PATH.read_text(encoding="utf-8")[:4000]
        want = re.search(r"halogen-flash-server:(\d+\.\d+\.\d+)", head).group(1)
        r = subprocess.run(["bash", str(REPO / "setup" / "lib" / "models.sh"),
                            "halogen-image"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        got = re.search(r"halogen-flash-server:(\d+\.\d+\.\d+)",
                        r.stdout).group(1)
        self.assertEqual(want, got,
                         "the vendored file was cut from %s and models.sh "
                         "starts %s — re-cut the copy or move the tag back"
                         % (want, got))


class TestTheMountCheckActuallyRefuses(unittest.TestCase):
    """A guard is worth what it does when it fires.

    Driven with a stub `podman` on PATH, so it runs the real halogenexec and
    never touches this machine's container runtime. Only the REFUSING
    directions are exercised: the passing one needs the image's real bytes,
    and a test that fabricated them would be testing the fabrication.
    """

    import os as _os

    def run_with_stub(self, cp_writes, create_rc=0):
        import os
        import stat
        import subprocess
        import tempfile
        d = tempfile.mkdtemp(prefix="halogenexec-")
        self.addCleanup(__import__("shutil").rmtree, d, True)
        models = os.path.join(d, "models")
        os.makedirs(models)
        # halogenexec refuses earlier if the checkpoint is missing, and that
        # refusal is not the one under test.
        open(os.path.join(models, "qwen38-flash-next-w4b.hgn"), "w").close()
        stub = os.path.join(d, "bin")
        os.makedirs(stub)
        podman = os.path.join(stub, "podman")
        with open(podman, "w", encoding="utf-8") as fh:
            fh.write("#!/usr/bin/env bash\n"
                     'case "$1" in\n'
                     '  create) exit %d ;;\n'
                     '  cp)     printf %%s %r > "$3"; exit 0 ;;\n'
                     '  rm)     exit 0 ;;\n'
                     '  run)    echo "STARTED" ; exit 0 ;;\n'
                     '  *)      exit 0 ;;\n'
                     "esac\n" % (create_rc, cp_writes))
        os.chmod(podman, os.stat(podman).st_mode | stat.S_IEXEC)
        env = dict(os.environ)
        env["PATH"] = stub + os.pathsep + env["PATH"]
        env["HALOGEN_MODELS"] = models
        return subprocess.run(["bash", str(REPO / "setup" / "halogenexec")],
                              capture_output=True, text=True, env=env,
                              timeout=60)

    def test_a_changed_base_file_stops_the_start(self):
        r = self.run_with_stub("this is not the file it was cut from")
        self.assertEqual(r.returncode, 1,
                         "halogenexec started anyway:\n%s%s" % (r.stdout, r.stderr))
        self.assertNotIn("STARTED", r.stdout,
                         "podman run was reached despite the mismatch")
        self.assertIn("no longer carries the file", r.stderr)
        self.assertIn("podman cp", r.stderr,
                      "the refusal has to say how to re-cut the copy")

    def test_an_unreadable_image_stops_the_start_too(self):
        """"Could not check" is not "checked and fine". Mounting an
        unverified override is the thing the check exists to prevent."""
        r = self.run_with_stub("irrelevant", create_rc=1)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertNotIn("STARTED", r.stdout)
        self.assertIn("could not read", r.stderr)


class TestTheDefectIsFiled(unittest.TestCase):
    """CLAUDE.md's filing table: a defect goes to setup/defects.json, not
    into a comment. The missing `<|endoftext|>` cost a day of runaway
    generations and it is an upstream bug in a pinned image — exactly what
    that file is for."""

    def test_defects_json_carries_the_end_token_entry(self):
        import json
        data = json.loads((REPO / "setup" / "defects.json").read_text(
            encoding="utf-8"))
        entries = data if isinstance(data, list) else data.get("defects", [])
        ids = [e.get("id", "") for e in entries]
        self.assertTrue(
            any("halogen" in i for i in ids),
            "no defects.json entry mentions halogen — the EOS defect lives "
            "only in a commit message")
        self.assertIn("halogen-greedy-reasoning-loop", ids,
                      "halogen-greedy-reasoning-loop is not filed in setup/defects.json")


def load_serve_api():
    """Loads setup/halogen/serve_api.py, stubbing container-only packages if needed."""
    import sys, types, importlib.util
    if "serve_api_module" in sys.modules:
        return sys.modules["serve_api_module"]
    # serve_api.py imports tool_parse at module level. That module is the
    # image's, not this repo's, since 21.09.2026 — the unmodified copy under
    # tests/fixtures/halogen/ is here so this load resolves against the REAL
    # exports rather than a stub that would hide a renamed one. See that
    # file's header for why it is not in setup/halogen/.
    fixtures = str(REPO / "tests" / "fixtures" / "halogen")
    if fixtures not in sys.path:
        sys.path.insert(0, fixtures)
    class DummyBaseModel:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)
    for name in ["fastapi", "fastapi.exceptions", "fastapi.responses", "pydantic",
                 "starlette", "starlette.exceptions", "uvicorn", "PIL", "PIL.Image"]:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    if not hasattr(sys.modules["pydantic"], "BaseModel"):
        sys.modules["pydantic"].BaseModel = DummyBaseModel
        sys.modules["pydantic"].model_validator = lambda **kwargs: lambda f: f
    if not hasattr(sys.modules["fastapi"], "FastAPI"):
        class DummyFastAPI:
            def __init__(self, **kw):
                self.state = types.SimpleNamespace()
            def post(self, *a, **k):
                return lambda f: f
            def get(self, *a, **k):
                return lambda f: f
            def middleware(self, *a, **k):
                return lambda f: f
            def exception_handler(self, *a, **k):
                return lambda f: f
        sys.modules["fastapi"].FastAPI = DummyFastAPI
        sys.modules["fastapi"].HTTPException = type("HTTPException", (Exception,), {})
        # Request and Response are new in the 0.12.3 base (0.8.1 imported
        # neither). A stub that lags the base fails at IMPORT with
        # "cannot import name X from fastapi", which reads like a broken
        # environment rather than a moved base file — so when a re-cut breaks
        # here, check this list against the base's import block first.
        sys.modules["fastapi"].Request = type("Request", (), {})
        sys.modules["fastapi.exceptions"].RequestValidationError = type("RequestValidationError", (Exception,), {})
        sys.modules["fastapi.responses"].JSONResponse = type("JSONResponse", (), {})
        sys.modules["fastapi.responses"].Response = type("Response", (), {})
        sys.modules["fastapi.responses"].StreamingResponse = type("StreamingResponse", (), {})
        sys.modules["starlette.exceptions"].HTTPException = type("StarletteHTTPException", (Exception,), {})

    spec = importlib.util.spec_from_file_location("serve_api_module", REPO / "setup" / "halogen" / "serve_api.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["serve_api_module"] = mod
    return mod


class TestThinkingBudgetResolution(unittest.TestCase):
    """0.8.1 native thinking budget: defaults, request fields, and server_default resolution."""

    @classmethod
    def setUpClass(cls):
        cls.mod = load_serve_api()

    def test_thinking_budget_defaults_and_request_override(self):
        mod = self.mod
        import types
        # Default fallback when request has None
        req_none = types.SimpleNamespace(max_thinking_tokens=None)
        self.assertEqual(mod.server_default(req_none, "max_thinking_tokens"), mod.DEFAULTS["max_thinking_tokens"])
        # Explicit request override
        req_custom = types.SimpleNamespace(max_thinking_tokens=1500)
        self.assertEqual(mod.server_default(req_custom, "max_thinking_tokens"), 1500)

    def test_chat_request_schema_declares_max_thinking_tokens(self):
        mod = self.mod
        req = mod.ChatReq(messages=[{"role": "user", "content": "hi"}], max_thinking_tokens=1234)
        self.assertEqual(req.max_thinking_tokens, 1234)


class TestThinkingBudgetWireProtocol(unittest.IsolatedAsyncioTestCase):
    """0.8.1 wire protocol: GEN ... THINK <budget> <end_id> <len> <close_ids...>"""

    @classmethod
    def setUpClass(cls):
        cls.mod = load_serve_api()

    def test_engine_generate_formats_think_line(self):
        mod = self.mod
        import asyncio

        class BufferWriter:
            def __init__(self):
                self.buf = bytearray()
            def write(self, data):
                self.buf.extend(data)
            async def drain(self):
                pass
            def is_closing(self):
                return False

        class DummyReader:
            async def readline(self):
                return b"D 1 stop 3 1 0.0 0.0\n"

        eng = mod.Engine("127.0.0.1", 8730)
        eng.r = DummyReader()
        eng.w = BufferWriter()
        eng.n_slots = 1
        eng.info = {"ctx": 32768}

        think_spec = (26000, 248069, [10, 20, 30])
        async def run_gen():
            async for _ in eng.generate([1, 2, 3], max_tokens=100, eos=[248046, 248044], think=think_spec):
                pass

        asyncio.run(run_gen())
        line = eng.w.buf.decode("utf-8")
        self.assertIn("THINK 26000 248069 3 10 20 30", line)

    async def test_serve_propagates_think_tuple_to_engine(self):
        mod = self.mod
        import types, asyncio

        class DummyTokenizer:
            clean_up_tokenization_spaces = False
            eos_token_id = 248046
            def convert_tokens_to_ids(self, s):
                return {"<|im_end|>": 248046, "<|endoftext|>": 248044, "</think>": 248069}.get(s, None)
            def __call__(self, text, **kw):
                return {"input_ids": [42]}
            def decode(self, ids, **kw):
                return ""

        captured_think = []
        captured_eos = []

        class DummyEngine:
            def __init__(self):
                # the shape production serves: slot_ctx 262144 on 0.12.3.
                # serve() reads the window from engine.info, not from
                # build_app's ctx, so this is the number that decides whether
                # a 32000-token budget fits.
                self.info = {"ctx": 262144, "version": "0.12.3"}
                self.slots = asyncio.Semaphore(10)
                self.waiting = 0
                self.inflight = {}
                self.reserved = {}
                self.busy_since = None
                self.metrics = types.SimpleNamespace(record=lambda *a, **k: None)
            async def abort(self):
                pass
            # seg and snap3 are new in the 0.12.3 base — snap3 is 0.12.1's
            # third cache save point (the start of the last message). Named
            # rather than swallowed by **kw ON PURPOSE: a re-cut that adds
            # another argument should fail here and say so, which is the only
            # signal this offline suite gets that the engine protocol moved.
            # It did so on the 0.13.4 re-cut: `guard` is the end-of-turn
            # guard's clause (#84), 22.09.2026.
            async def generate(self, ids, max_tokens, eos, drafter=None, sample=None, penalty="",
                                snap=0, snap2=0, images=None, schema=None, after=None, escape=(), think=None,
                                seg=None, snap3=0, guard=None):
                captured_think.append(think)
                captured_eos.append(list(eos))
                yield None, {"reason": "stop", "n_gen": 0, "n_prompt": len(ids),
                             "decode_ms": 1.0, "prefill_ms": 1.0}, None

        engine = DummyEngine()
        tok = DummyTokenizer()
        # max_cap raised from build_app's 4096 default: the shape asserted on
        # below is the one Claude Code actually sends (max_tokens=32000), and
        # both the cap and the context window are checked BEFORE the answer
        # room, so the defaults would refuse the request with a different 400
        # before any budget was computed.
        app = mod.build_app(tok, engine, ctx=262144, max_cap=32000)
        serve_fn = app.state.serve

        # 1. With thinking enabled and budget set, at a budget the ANSWER ROOM
        #    leaves alone. This assertion used max_tokens=100 until
        #    21.09.2026 and passed on 0.8.1; on 0.12.3 it read 1, which is
        #    0.11.0's answer room doing its job and not a regression:
        #        room = max(1024, max_tokens * 15 // 100)
        #        cap  = max(1, max_tokens - room)      # the think budget's ceiling
        #    At max_tokens=100 the room (1024) exceeds the whole budget, so
        #    one token of thinking is all that is left. Case 3 below pins that.
        #    32000 is what Claude Code sends, so this is the live shape: room
        #    4800, cap 27200, and 26000 passes untouched — with 1200 tokens to
        #    spare. The margin is thin on purpose to be visible: below a
        #    max_tokens of about 30,600 this repo's 26000 starts being cut.
        await serve_fn([1, 2, 3], max_tokens=32000, stops=[], stream=False, chat=True, prefix="chat",
                       thinking=True, think_budget=26000)
        self.assertEqual(len(captured_think), 1)
        self.assertIsNotNone(captured_think[0])
        budget, end_id, close_ids = captured_think[0]
        self.assertEqual(budget, 26000,
                         "the operator's think budget did not reach the engine "
                         "at a max_tokens the answer room leaves alone")
        self.assertEqual(end_id, 248069)
        # Check EOS includes both <|im_end|> and <|endoftext|>
        self.assertIn(248046, captured_eos[0])
        self.assertIn(248044, captured_eos[0])

        # 2. With thinking disabled
        captured_think.clear()
        captured_eos.clear()
        await serve_fn([1, 2, 3], max_tokens=100, stops=[], stream=False, chat=True, prefix="chat",
                       thinking=False, think_budget=26000)
        self.assertEqual(len(captured_think), 1)
        self.assertIsNone(captured_think[0])

        # 3. The answer room wins over a larger budget (0.11.0, new since the
        #    0.8.1 this repo served until 21.09.2026). A small max_tokens keeps
        #    max(1024, 15%) for the answer, so a 26000-token think budget is
        #    cut to what is left — one token here, since the room alone is more
        #    than the whole budget. Before 0.11.0 this request thought until it
        #    hit max_tokens and returned `finish_reason: length` with EMPTY
        #    content, which is the failure the room exists to remove.
        #
        #    Pinned because it changes what a client with a small budget gets
        #    from this stack, and because the repo's own 26000 is only
        #    untouched above a max_tokens of about 30,600 (cap = 85% of
        #    max_tokens there; Claude Code's 32000 clears it, 30000 does not).
        captured_think.clear()
        await serve_fn([1, 2, 3], max_tokens=100, stops=[], stream=False, chat=True, prefix="chat",
                       thinking=True, think_budget=26000)
        self.assertEqual(len(captured_think), 1)
        self.assertEqual(captured_think[0][0], 1,
                         "the answer room did not bound a think budget larger "
                         "than the whole of max_tokens")

        # 4. And it does not touch a budget that already fits beside the room.
        captured_think.clear()
        await serve_fn([1, 2, 3], max_tokens=32000, stops=[], stream=False, chat=True, prefix="chat",
                       thinking=True, think_budget=1000)
        self.assertEqual(captured_think[0][0], 1000,
                         "the answer room moved a budget that fits")


class TestFitToRoom(unittest.IsolatedAsyncioTestCase):
    """HALOGEN_FIT_TO_ROOM: a budget that does not fit the remaining window is
    clamped to the room when the gate is on, and refused when it is off or when
    less than the floor is left — so a near-empty budget is never a silent
    one-token answer, and an ungated side server (bench) keeps the hard 400."""

    @classmethod
    def setUpClass(cls):
        cls.mod = load_serve_api()

    def setUp(self):
        self._fit = self.mod.FIT_TO_ROOM
        self._floor = self.mod.FIT_TO_ROOM_FLOOR

    def tearDown(self):
        self.mod.FIT_TO_ROOM = self._fit
        self.mod.FIT_TO_ROOM_FLOOR = self._floor

    async def _generate_maxtok(self, prompt_len, want):
        """Runs serve() once on a 1000-token context and returns the budgets
        the engine was actually asked for."""
        import types, asyncio
        mod = self.mod
        seen = []

        class Tok:
            clean_up_tokenization_spaces = False
            eos_token_id = 248046
            def convert_tokens_to_ids(self, s):
                return {"</think>": 248069}.get(s)
            def __call__(self, text, **kw):
                return {"input_ids": [42]}
            def decode(self, ids, **kw):
                return ""

        class Eng:
            def __init__(self):
                self.info = {"ctx": 1000, "version": "0.8.1"}
                self.slots = asyncio.Semaphore(10)
                self.waiting = 0
                self.inflight = {}
                self.reserved = {}
                self.busy_since = None
                self.metrics = types.SimpleNamespace(record=lambda *a, **k: None)
            async def abort(self):
                pass
            # see the note on the other generate() stub: seg/snap3 are 0.12.x
            # arguments and guard is 0.13.4's, named so a further one fails
            # loudly.
            async def generate(self, ids, max_tokens, eos, drafter=None, sample=None,
                               penalty="", snap=0, snap2=0, images=None, schema=None,
                               after=None, escape=(), think=None, seg=None, snap3=0,
                               guard=None):
                seen.append(max_tokens)
                yield None, {"reason": "stop", "n_gen": 0, "n_prompt": len(ids),
                             "decode_ms": 1.0, "prefill_ms": 1.0}, None

        app = mod.build_app(Tok(), Eng(), ctx=1000)
        await app.state.serve([1] * prompt_len, max_tokens=want, stops=[],
                              stream=False, chat=True, prefix="chat",
                              thinking=False)
        return seen

    async def test_gate_off_refuses(self):
        """Default: the refusal that names the numbers stays."""
        self.mod.FIT_TO_ROOM = False
        with self.assertRaises(Exception) as cm:
            await self._generate_maxtok(950, 100)          # room = 50, want = 100
        self.assertIn("does not fit", str(cm.exception))

    async def test_gate_on_clamps_to_room(self):
        """Gated: the budget is what the prompt left, not the refusal."""
        self.mod.FIT_TO_ROOM = True
        self.mod.FIT_TO_ROOM_FLOOR = 16
        seen = await self._generate_maxtok(950, 100)        # room = 50 >= floor 16
        self.assertEqual(seen, [50])

    async def test_gate_on_below_floor_still_refuses(self):
        """Gated but the prompt has eaten the window: still an honest error."""
        self.mod.FIT_TO_ROOM = True
        self.mod.FIT_TO_ROOM_FLOOR = 1024
        with self.assertRaises(Exception) as cm:
            await self._generate_maxtok(950, 100)          # room = 50 < floor 1024
        self.assertIn("below the floor", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
