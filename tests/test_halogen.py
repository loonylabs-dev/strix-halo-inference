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
            files, ["setup/halogen/serve_api.py", "setup/halogen/tool_parse.py",
                    "setup/lib/models.sh"],
            "the image tag is spelled out in %s — one of them will be "
            "forgotten on the next bump. models.sh answers `halogen-image`; "
            "serve_api.py and tool_parse.py name the tag they were CUT from, "
            "which is a different statement and is compared against the running "
            "one by halogenexec." % files)

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

    Three hunks of real fix (both EOS ids, and per-request stop strings as
    EOS ids) shipped as a 150 KB whole-file copy. Bump the image and the old
    copy silently reverts every upstream change in that file while `podman
    ps` reports the new tag — the exact shape of a defect this repo keeps
    finding. So the file has to say which image it was cut from, and
    halogenexec has to refuse the mount when that is no longer true.
    """

    PATH = REPO / "setup" / "halogen" / "serve_api.py"
    TOOL_PARSE_PATH = REPO / "setup" / "halogen" / "tool_parse.py"

    def test_it_names_the_image_it_was_cut_from(self):
        head = self.PATH.read_text(encoding="utf-8")[:4000]
        self.assertRegex(
            head, r"halogen-flash-server:\d+\.\d+\.\d+",
            "the vendored copy does not say which image tag it patches")
        self.assertRegex(
            head, r"BASE_SHA256\s*[:=]\s*[0-9a-f]{64}",
            "without the base file's hash nothing can notice that the image "
            "moved underneath this copy")

    def test_tool_parse_names_the_image_it_was_cut_from(self):
        head = self.TOOL_PARSE_PATH.read_text(encoding="utf-8")[:4000]
        self.assertRegex(
            head, r"halogen-flash-server:\d+\.\d+\.\d+",
            "the vendored tool_parse.py does not say which image tag it patches")
        self.assertRegex(
            head, r"BASE_SHA256\s*[:=]\s*[0-9a-f]{64}",
            "without the base file's hash nothing can notice that the image "
            "moved underneath this copy")

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
        self.assertIn("TOOL_PARSE_BASE_SHA256", src,
                      "halogenexec mounts tool_parse.py without checking that the "
                      "image still carries the file it was cut from")

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

    def test_tool_parse_tag_agrees_with_the_one_that_is_started(self):
        import subprocess
        head = self.TOOL_PARSE_PATH.read_text(encoding="utf-8")[:4000]
        want = re.search(r"halogen-flash-server:(\d+\.\d+\.\d+)", head).group(1)
        r = subprocess.run(["bash", str(REPO / "setup" / "lib" / "models.sh"),
                            "halogen-image"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        got = re.search(r"halogen-flash-server:(\d+\.\d+\.\d+)",
                        r.stdout).group(1)
        self.assertEqual(want, got,
                         "the vendored tool_parse.py was cut from %s and models.sh "
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
        sys.modules["fastapi.exceptions"].RequestValidationError = type("RequestValidationError", (Exception,), {})
        sys.modules["fastapi.responses"].JSONResponse = type("JSONResponse", (), {})
        sys.modules["fastapi.responses"].StreamingResponse = type("StreamingResponse", (), {})
        sys.modules["starlette.exceptions"].HTTPException = type("StarletteHTTPException", (Exception,), {})

    spec = importlib.util.spec_from_file_location("serve_api_module", REPO / "setup" / "halogen" / "serve_api.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["serve_api_module"] = mod
    return mod


class TestThinkingBudgetResolution(unittest.TestCase):
    """resolve_thinking_budget sizes the thinking cap per effort tier and request field."""

    @classmethod
    def setUpClass(cls):
        cls.mod = load_serve_api()

    def test_disabled_thinking_returns_none(self):
        import types
        req = types.SimpleNamespace()
        self.assertIsNone(self.mod.resolve_thinking_budget(req, thinking=False, max_tokens=65536))

    def test_insufficient_tokens_returns_none(self):
        import types
        req = types.SimpleNamespace()
        self.assertIsNone(self.mod.resolve_thinking_budget(req, thinking=True, max_tokens=1))
        self.assertIsNone(self.mod.resolve_thinking_budget(req, thinking=True, max_tokens=0))

    def test_explicit_request_fields(self):
        import types
        req1 = types.SimpleNamespace(max_thinking_tokens=1500)
        self.assertEqual(self.mod.resolve_thinking_budget(req1, True, 65536), 1500)
        req2 = types.SimpleNamespace(budget_tokens=2500)
        self.assertEqual(self.mod.resolve_thinking_budget(req2, True, 65536), 2500)
        # Clamped to max_tokens - 1
        req3 = types.SimpleNamespace(max_thinking_tokens=1000)
        self.assertEqual(self.mod.resolve_thinking_budget(req3, True, 100), 99)

    def test_chat_template_kwargs_nested_budget(self):
        import types
        req = types.SimpleNamespace(chat_template_kwargs={"budget_tokens": 3000})
        self.assertEqual(self.mod.resolve_thinking_budget(req, True, 65536), 3000)

    def test_low_effort_budget(self):
        import types
        req = types.SimpleNamespace(reasoning_effort="low")
        # Low effort: min(26000, 50% of max_tokens)
        self.assertEqual(self.mod.resolve_thinking_budget(req, True, 65536), 26000)
        self.assertEqual(self.mod.resolve_thinking_budget(req, True, 20000), 10000)

    def test_high_effort_budget(self):
        import types
        req_med = types.SimpleNamespace(reasoning_effort="medium")
        self.assertEqual(self.mod.resolve_thinking_budget(req_med, True, 65536), 49152)
        req_high = types.SimpleNamespace(reasoning_effort="high")
        self.assertEqual(self.mod.resolve_thinking_budget(req_high, True, 65536), 49152)
        self.assertEqual(self.mod.resolve_thinking_budget(req_high, True, 20000), 15000)

    def test_unspecified_effort_defaults_to_low(self):
        import types
        req = types.SimpleNamespace()
        # Default tier is low -> 26000 at 64k
        self.assertEqual(self.mod.resolve_thinking_budget(req, True, 65536), 26000)


class TestThinkingBudgetContinuation(unittest.IsolatedAsyncioTestCase):
    """When thinking exceeds the budget, the engine is aborted, </think>\n\n is injected,
    and a warm KV continuation stream is launched for the answer."""

    @classmethod
    def setUpClass(cls):
        cls.mod = load_serve_api()

    async def test_budget_cutoff_forces_tag_and_continues(self):
        mod = self.mod
        class DummyTokenizer:
            clean_up_tokenization_spaces = False
            eos_token_id = 248046
            def convert_tokens_to_ids(self, s):
                return {"<|im_end|>": 248046, "<|endoftext|>": 248044}.get(s, None)
            def __call__(self, text, **kw):
                if text == "</think>\n\n":
                    return {"input_ids": [248069, 271]}
                return {"input_ids": [hash(text) % 100000]}
            def decode(self, ids, **kw):
                id_map = {100: "<think>\nThinking", 101: " deeply", 248069: "</think>",
                          271: "\n\n", 200: "The answer is 42."}
                return "".join(id_map.get(i, f"[tok_{i}]") for i in ids)

        class DummyEngine:
            def __init__(self):
                self.info = {"ctx": 32768}
                self.aborted = 0
                self.calls = []
            async def abort(self):
                self.aborted += 1
            async def generate(self, ids, max_tokens, stops, drafter, sample, penalty, snap=0, snap2=0, images=None):
                self.calls.append(list(ids))
                if len(self.calls) == 1:
                    yield 100, None, None
                    yield 101, None, None
                else:
                    yield 200, None, None
                    yield None, {"reason": "stop", "n_gen": 1, "n_prompt": len(ids),
                                 "decode_ms": 10.0, "prefill_ms": 1.0}, None

        engine = DummyEngine()
        tok = DummyTokenizer()
        app = mod.build_app(tok, engine, ctx=32768)
        run_fn = app.state.run

        deltas = []
        done_record = None
        prompt_ids = [1, 2, 3]
        async for delta, d in run_fn(prompt_ids, max_tokens=100, stops=[], thinking=True, max_thinking_tokens=2):
            if delta is not None:
                deltas.append(delta)
            if d is not None:
                done_record = d

        full_output = "".join(deltas)
        self.assertGreaterEqual(engine.aborted, 1, "engine.abort() must be called on budget cutoff")
        self.assertEqual(len(engine.calls), 2, "continuation generation must be called")
        self.assertIn("</think>\n\n", full_output, "</think> must be injected")
        self.assertIn("The answer is 42.", full_output, "answer from continuation must be streamed")
        self.assertEqual(done_record["reason"], "stop")
        # Initial 2 tokens + close_ids (2 tokens: 248069, 271) + 1 answer token = 5 total
        self.assertEqual(done_record["n_gen"], 5)


if __name__ == "__main__":
    unittest.main()
