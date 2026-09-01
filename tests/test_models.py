"""The model registry and the contracts around it.

Adding a model used to mean editing six places. Five of them are derived from
setup/env/*.env now (setup/lib/models.sh). The sixth CANNOT be derived —
systemd has no wildcard in `Conflicts=`, so a template unit cannot conflict
with every other instance of itself, and every model has to be named there by
hand.

That is precisely the kind of thing this test directory exists for. A model
missing from `Conflicts=` breaks nothing visibly: both llama-servers start,
one loses the race for port 8080, systemd still reports "active", and the
gateway answers from whichever won. No error, no log line, just the wrong
model. So the list is not trusted — it is compared against the registry here,
and a new profile turns that into a red test instead of a silent switch.
"""
import glob, json, os, re, shutil, subprocess, sys, tempfile, unittest

import common

sys.path.insert(0, str(common.REPO / "setup" / "lib"))
import systemdfile                                        # noqa: E402
import budget                                             # noqa: E402

REPO = common.REPO
LIB = str(REPO / "setup" / "lib" / "models.sh")
SWITCH = str(REPO / "setup" / "switch-model.sh")


def lib(*args, env_=None, cwd=None):
    u = dict(os.environ)
    u.update(env_ or {})
    return subprocess.run(["bash", LIB, *args], capture_output=True, text=True,
                          timeout=60, env=u, cwd=cwd)


def conflicts_of(unit_name):
    """The Conflicts= units of a unit file — read with the SAME parser
    switch-model.sh uses, so the test cannot pass against a reading the
    script does not share."""
    return " ".join(systemdfile.directive(
        str(REPO / "setup" / "systemd" / unit_name), "Conflicts")).split()


def profiles():
    return sorted(p.stem for p in (REPO / "setup" / "env").glob("*.env"))


class TestRegistry(unittest.TestCase):
    """setup/env/*.env IS the list of models. Nothing else may hold one."""

    def test_the_registry_lists_exactly_the_profiles(self):
        r = lib("list")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.split(), profiles())

    def test_known_accepts_a_profile_and_rejects_a_typo(self):
        self.assertEqual(lib("known", profiles()[0]).returncode, 0)
        self.assertEqual(lib("known", "qwen38-typo").returncode, 1)
        self.assertEqual(lib("known", "").returncode, 1)

    def test_every_profile_declares_its_metadata(self):
        for m in profiles():
            with self.subTest(model=m):
                title = lib("meta", m, "MODEL_TITLE").stdout.strip()
                swa = lib("meta", m, "MODEL_SWA").stdout.strip()
                self.assertTrue(title, "%s has no MODEL_TITLE" % m)
                self.assertIn(swa, ("yes", "no", "unknown"),
                              "%s: MODEL_SWA=%r — say yes, no, or unknown, but say it"
                              % (m, swa))

    def test_metadata_lookup_does_not_confuse_two_variables_with_a_prefix(self):
        # MODEL_SWA and MODEL_TITLE share 'MODEL_'. A sloppy grep returns both.
        for m in profiles():
            with self.subTest(model=m):
                self.assertNotIn("MODEL_", lib("meta", m, "MODEL_SWA").stdout)

    def test_every_profile_carries_llama_args(self):
        for m in profiles():
            with self.subTest(model=m):
                r = lib("args", m)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertIn("--alias", r.stdout,
                              "%s: LLAMA_ARGS without --alias — the gateway and "
                              "check.sh both identify the running model by it" % m)

    def test_the_alias_is_the_profile_name(self):
        """The store of saved prefixes, check.sh and the gateway all key on
        --alias. A profile called flashnext.env that aliases itself qwen38
        would park its prefixes under the wrong name."""
        for m in profiles():
            with self.subTest(model=m):
                args = lib("args", m).stdout.split()
                self.assertEqual(args[args.index("--alias") + 1], m)


class TestArgsReader(unittest.TestCase):
    """ONE reader for LLAMA_ARGS. Two bugs lived in the copy that used to sit
    in check.sh, and switch-model.sh would have made a third copy."""

    def _read(self, body):
        with tempfile.TemporaryDirectory() as d:
            env = os.path.join(d, "probe.env")
            with open(env, "w", encoding="utf-8") as f:
                f.write(body)
            return lib("args", env).stdout.strip()

    def test_continuations_are_joined(self):
        self.assertEqual(self._read("LLAMA_ARGS=--alias x \\\n  -c 8192\n"),
                         "--alias x -c 8192")

    def test_it_stops_at_the_end_of_the_assignment(self):
        """The old regex ran on to the next VAR= and swallowed the comment
        lines in between, then reported their words as missing arguments."""
        got = self._read(
            "LLAMA_ARGS=--alias x \\\n  -c 8192\n"
            "\n# a comment with the word --swa-full in it\n"
            "LLAMA_BIN=llama.cpp/build-vulkan/bin/llama-server\n")
        self.assertEqual(got, "--alias x -c 8192")
        self.assertNotIn("LLAMA_BIN", got)
        self.assertNotIn("--swa-full", got)

    def test_quotes_are_data_not_syntax(self):
        """systemd splits on whitespace and passes quotes through. shlex.split
        ate them out of --chat-template-kwargs {"a":false}, so the value never
        matched the real command line and check.sh reported a phantom
        difference."""
        self.assertEqual(
            self._read('LLAMA_ARGS=--chat-template-kwargs {"enable_thinking":false}\n'),
            '--chat-template-kwargs {"enable_thinking":false}')

    def test_metadata_before_llama_args_is_skipped(self):
        self.assertEqual(
            self._read("MODEL_TITLE=something with -c 4 in it\nMODEL_SWA=no\n"
                       "LLAMA_ARGS=--alias x\n"),
            "--alias x")

    # A file that both names LLAMA_ARGS and joins continuation lines is a
    # parser, whatever else it calls itself.
    JOINERS = ('rstrip("\\\\")', 're.sub(r"\\\\\\n"', 'replace("\\\\\\n"',
               'startswith("LLAMA_ARGS=")')

    def test_only_one_copy_of_this_reader_exists(self):
        """Duplicated code that is allowed to drift apart, drifts apart —
        the same rule tests/test_scripts.py applies to the token expression.

        Three copies existed on 26.08. and they disagreed. The one in
        bench/suites/slot-scaling.sh appended the words of the comment lines
        after LLAMA_ARGS to the server's command line; the one in bench/run.py
        was fine but separate, which means it was one edit away from not
        being. A bench harness that reads the profile differently from the
        service is not measuring the service.
        """
        carriers = []
        for root, dirs, files in os.walk(REPO):
            dirs[:] = [d for d in dirs
                       if d not in (".git", "__pycache__", "reports", "archive")]
            if os.path.relpath(root, REPO).split(os.sep)[0] == "tests":
                continue          # a test has to be able to name the idioms
            for f in files:
                if not f.endswith((".sh", ".py")):
                    continue
                p = os.path.join(root, f)
                try:
                    with open(p, encoding="utf-8") as fh:
                        text = fh.read()
                except (OSError, UnicodeDecodeError):
                    continue
                if "LLAMA_ARGS" in text and any(j in text for j in self.JOINERS):
                    carriers.append(os.path.relpath(p, REPO))
        self.assertEqual(sorted(carriers), ["setup/lib/systemdfile.py"],
                         "the LLAMA_ARGS reader has grown a second copy")

    def test_every_caller_gets_the_same_answer(self):
        """The contract that actually matters, checked on the real profiles
        and across the two languages: the shell path (switch-model.sh,
        check.sh), the module path (bench/run.py, bench/sweep.py) and the
        command path must agree word for word."""
        import importlib.util
        for p in (REPO / "tools", REPO / "bench", REPO / "setup" / "lib"):
            if str(p) not in os.sys.path:
                os.sys.path.insert(0, str(p))
        spec = importlib.util.spec_from_file_location("bench_run", REPO / "bench/run.py")
        run = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run)
        import systemdfile as envfile
        for m in profiles():
            path = str(REPO / "setup" / "env" / (m + ".env"))
            with self.subTest(model=m):
                direct = envfile.llama_args(path)
                self.assertEqual(lib("args", m).stdout.split(), direct)
                self.assertEqual(run.args_from_env(path), direct)

    def test_the_scalar_reader_agrees_with_the_shell_one(self):
        """model_meta is a sed one-liner and envfile.variable is Python —
        two implementations of a deliberately trivial job. They still have to
        say the same thing about every profile in the house."""
        for p in (REPO / "setup" / "lib",):
            if str(p) not in os.sys.path:
                os.sys.path.insert(0, str(p))
        import systemdfile as envfile
        for m in profiles():
            path = str(REPO / "setup" / "env" / (m + ".env"))
            for name in ("MODEL_TITLE", "MODEL_SWA", "LLAMA_BIN"):
                with self.subTest(model=m, variable=name):
                    self.assertEqual(lib("meta", m, name).stdout.strip(),
                                     (envfile.variable(path, name) or "").strip())



class TestFlagReadsWhatTheServerReads(unittest.TestCase):
    """systemdfile.flag() must resolve a repeated option the way llama-server
    does, which is LAST wins.

    It read the FIRST until 27.08.2026, and nothing said so. bench/sideserver.py
    appends its --extra to the profile's own arguments, so an override arrives
    as a SECOND `-c` — the server allocated for it and every reader in this
    repo computed for the profile's. Caught by a measurement that disagreed
    with itself: gemma31 pinned 27.2 GiB of GTT at `-c 131072` while
    budget.py reported `ctx 32768` and a KV figure four times too high.

    The direction is why this is a test and not a note. When an override
    RAISES the context the guard under-predicts, and under-prediction against
    pinned GTT does not produce an error — it produces a frozen machine.
    """

    def test_a_repeated_flag_keeps_the_last_value(self):
        self.assertEqual(
            systemdfile.flag(["-c", "32768", "-c", "131072"], "-c", "--ctx-size"),
            "131072")

    def test_short_and_long_name_are_one_option_decided_by_position(self):
        """They are aliases, so neither outranks the other — the later one
        wins, exactly as it would on the real command line."""
        self.assertEqual(
            systemdfile.flag(["--ctx-size", "8192", "-c", "4096"],
                             "-c", "--ctx-size"), "4096")
        self.assertEqual(
            systemdfile.flag(["-c", "4096", "--ctx-size", "8192"],
                             "-c", "--ctx-size"), "8192")

    def test_a_trailing_name_without_a_value_is_not_a_value(self):
        """`-c` as the final token has nothing after it. It must not erase an
        earlier occurrence that did."""
        self.assertEqual(
            systemdfile.flag(["-c", "32768", "-c"], "-c", "--ctx-size"), "32768")

    def test_absent_gives_the_default(self):
        self.assertIsNone(systemdfile.flag(["-ngl", "999"], "-c", "--ctx-size"))
        self.assertEqual(
            systemdfile.flag(["-ngl", "999"], "-c", default="0"), "0")

    def test_the_guard_sees_an_appended_override(self):
        """The end-to-end shape of the bug: a profile's arguments with an
        override appended, as bench/sideserver.py builds it. The KV term has
        to follow the LARGER window, because that is what gets allocated."""
        base = ["--alias", "probe", "-c", "32768", "-ctk", "q8_0", "-ctv", "q8_0"]
        small = budget.kv_gib(base, declared=44.5)[0]
        large = budget.kv_gib(base + ["-c", "131072"], declared=44.5)[0]
        self.assertGreater(large, small)
        self.assertAlmostEqual(large, 131072 * 44.5 / 1048576.0, places=3)


class TestConflicts(unittest.TestCase):
    """The one list systemd will not derive for us.

    Exact set equality, not 'contains': a profile that is deleted has to
    disappear from the unit too, or the next person reads a model list that
    is no longer true.
    """

    def test_the_user_unit_names_every_model(self):
        self.assertEqual(
            sorted(conflicts_of("llama-user@.service")),
            sorted("llama-user@%s.service" % m for m in profiles()))

    def test_the_system_unit_names_every_model(self):
        """Derived, so it cannot disagree — and checked anyway, because the
        derivation is a string substitution and a substitution can be wrong.
        See tests/test_systemunit.py for the rest of the mapping."""
        import systemunit, tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".service", delete=False,
                                         encoding="utf-8") as fh:
            fh.write(systemunit.render(user="nobody"))
            path = fh.name
        self.addCleanup(os.unlink, path)
        got = " ".join(systemdfile.directive(path, "Conflicts")).split()
        self.assertEqual(sorted(got),
                         sorted("llama@%s.service" % m for m in profiles()))


class TestSlidingWindow(unittest.TestCase):
    """A model WITH a sliding window and WITHOUT --swa-full runs every Claude
    Code turn cold: measured 100.2 s against 10.4 s on the same body. The
    switch is therefore a property the profile owes its own MODEL_SWA."""

    # Three profiles are knowingly missing it — the memory cost of --swa-full
    # is unmeasured for them and they are not in production use
    # (docs/DOCUMENTS.md, "Open points"). They stand here BY NAME so that the
    # gap stays visible and no NEW model can quietly join them.
    KNOWN_MISSING = {"gemma26", "gemma31", "gptoss", "batch"}

    def test_swa_models_carry_swa_full(self):
        missing = set()
        for m in profiles():
            if lib("meta", m, "MODEL_SWA").stdout.strip() != "yes":
                continue
            if "--swa-full" not in lib("args", m).stdout:
                missing.add(m)
        self.assertEqual(missing - self.KNOWN_MISSING, set(),
                         "these models declare a sliding window but do not "
                         "switch it off — every changed question re-processes "
                         "the whole prompt, silently")

    def test_the_known_gap_has_not_quietly_closed(self):
        """If someone adds --swa-full, this list has to shrink with it —
        otherwise it turns into folklore about a problem that is gone."""
        still = {m for m in self.KNOWN_MISSING
                 if m in profiles() and "--swa-full" not in lib("args", m).stdout}
        self.assertEqual(still, self.KNOWN_MISSING & set(profiles()),
                         "KNOWN_MISSING lists a model that now HAS --swa-full — "
                         "take it out of the list")


class TestEveryProfileFitsTheMachineItIsOn(unittest.TestCase):
    """A profile must not promise more memory than the machine has.

    This test used to carry its own arithmetic — `weights + cram + host` — and
    it was the third copy of a formula that also lived in bench/run.py and
    bench/sideserver.py, each of them slightly different. This one had no KV
    term at all, which is why it passed laguna: 68.4 GiB of weights and 32 of
    RAM prompt cache fit in 124.9, and the 12 GiB of KV at -c 131072 that made
    it 132.4 was simply not in the sum.

    So the formula is gone from here. setup/lib/budget.py owns it, the unit
    runs it as ExecStartPre, switch-model.sh runs it as preflight, and this
    test runs the same function — which is the only arrangement in which
    "the tests pass" and "the machine is safe" mean the same thing.

    What is checked here is the STATIC question: would this profile fit an
    IDLE machine of this size? Not "does it fit right now", which depends on
    what happens to be running and is the unit's question, not the repo's.
    """

    def machine(self):
        m = budget.read_machine()
        if m.mem_total is None:
            self.skipTest("no /proc/meminfo — cannot judge any of this")
        return m

    def test_no_profile_promises_more_than_the_machine_has(self):
        machine, rows, over, unknown, estimated = self.machine(), [], [], [], []
        for name in profiles():
            env = str(REPO / "setup" / "env" / (name + ".env"))
            argv = systemdfile.llama_args(env)
            weights = budget.weights_gib(argv)
            if weights is None:
                unknown.append(name)
                continue
            p = budget.plan(argv, weights, budget.declared_kv(env), name,
                            gtt_base=budget.declared_gtt(env),
                            host_anon=budget.declared_anon(env))
            fits = budget.fits_the_machine(p, machine)
            rows.append("    %-11s GTT %6.1f + cram %5.1f -> host %6.1f + %.0f "
                        "reserve of %.1f%s"
                        % (name, p.gtt_gib, budget.cram_gib(argv), p.host_gib,
                           budget.host_reserve_gib(), machine.mem_total,
                           "   (KV ESTIMATED)" if p.estimated else ""))
            if p.estimated:
                estimated.append(name)
            if fits is False:
                over.append(name)

        # A profile that is over budget on an ESTIMATED KV figure is a warning
        # and not a failure. The estimate is deliberately pessimistic, so it
        # can refuse a configuration that is in fact fine — and a red test that
        # can be wrong about the repo's own contents gets edited away rather
        # than acted on. At RUNTIME the same case still refuses, because there
        # the cost of being wrong is a machine that stops responding.
        hard = [n for n in over if n not in estimated]
        self.assertFalse(
            hard,
            "these profiles need more than this machine has, on MEASURED "
            "numbers — %s\n%s" % (", ".join(hard), "\n".join(rows)))
        soft = [n for n in over if n in estimated]
        if soft:
            print("\n    (over budget, but on an ESTIMATED KV figure — measure it: %s)"
                  % ", ".join(soft))
        if unknown:
            # Named, not silent. A profile whose model is not on this disk is
            # not evidence of anything, and a test that quietly skips it looks
            # like a test that passed.
            print("\n    (not judged, model not on this machine: %s)"
                  % ", ".join(unknown))

    def test_a_declared_kv_figure_carries_its_provenance(self):
        """A number without a source is an assertion, and assertions are what
        this whole registry exists to replace. MODEL_KV_KIB_PER_TOKEN decides
        whether a start is refused; where it came from has to travel with it,
        or the next reader cannot tell a measurement from a copy — which is
        precisely what happened to `-cram 32768`."""
        missing = []
        for name in profiles():
            env = str(REPO / "setup" / "env" / (name + ".env"))
            if budget.declared_kv(env) is None:
                continue
            if not systemdfile.variable(env, "MODEL_KV_SOURCE"):
                missing.append(name)
        self.assertFalse(missing,
                         "these profiles declare MODEL_KV_KIB_PER_TOKEN with no "
                         "MODEL_KV_SOURCE saying where it was measured: %s"
                         % ", ".join(missing))


class TestNothingStartsAModelWithoutAskingFirst(unittest.TestCase):
    """The guard is worth nothing if a caller can go around it.

    Same shape as tests/test_memory_guard.py::TestStartServerIsGuarded, which
    pins the measurement path. This one pins the PRODUCTION path — the one
    that was unguarded until 27.08. while three copies of the arithmetic sat
    in the bench harness.

    Every way a model can be started has to pass through it:

        systemctl --user start llama-user@X   ->  ExecStartPre=…/checkroom
        sudo systemctl start llama@X          ->  ExecStartPre=/usr/local/bin/llm-check-room
        bash setup/switch-model.sh X          ->  preflight, before anything stops
    """

    def unit(self, name):
        return (REPO / "setup" / "systemd" / name).read_text(encoding="utf-8")

    def test_the_user_unit_checks_before_it_starts(self):
        pre = systemdfile.directive(
            str(REPO / "setup" / "systemd" / "llama-user@.service"), "ExecStartPre")
        self.assertTrue(any("checkroom" in x for x in pre),
                        "llama-user@.service starts a model without weighing it: %r" % pre)

    def test_the_system_unit_checks_before_it_starts(self):
        """It is derived now, so this asks the derivation rather than a file.
        Its predecessor was hand-written and had drifted to the wrong binary —
        which is the reason there is no file left to ask."""
        import systemunit
        text = systemunit.render(user="nobody")
        pre = [l.split("=", 1)[1] for l in text.splitlines()
               if l.startswith("ExecStartPre=")]
        self.assertTrue(any("llm-check-room" in x for x in pre),
                        "the system unit starts a model without weighing it: %r" % pre)

    def test_neither_guard_is_prefixed_with_a_dash(self):
        """A leading '-' makes systemd ignore the exit code. On ExecStartPost
        that is the right call and this repo learned it the hard way; here it
        would turn the guard into a comment."""
        import systemunit
        texts = {"llama-user@.service": self.unit("llama-user@.service"),
                 "llama@.service (derived)": systemunit.render(user="nobody")}
        for unit, text in texts.items():
            for line in text.splitlines():
                if line.startswith("ExecStartPre=") and "check" in line:
                    self.assertNotIn("=-", line,
                                     "%s ignores the guard's exit code: %s" % (unit, line))

    def test_switch_model_weighs_it_in_the_preflight(self):
        src = (REPO / "setup" / "switch-model.sh").read_text(encoding="utf-8")
        self.assertIn("budget.py", src, "switch-model.sh does not weigh the profile")
        self.assertLess(src.index("budget.py"), src.index("systemctl --user daemon-reload"),
                        "the budget is checked AFTER the switch has begun — the "
                        "whole point of a preflight is that it decides first")

    def test_install_puts_the_guard_where_both_units_look_for_it(self):
        """budget.py imports systemdfile.py from its own directory, so a
        symlink of one without the other is a guard that cannot run — and
        checkroom then starts the model unguarded, by design."""
        src = (REPO / "setup" / "install.sh").read_text(encoding="utf-8")
        for needed in ("checkroom", "lib/budget.py", "lib/systemdfile.py",
                       "llm-check-room", "/usr/local/lib/llm-profile",
                       "--system-unit"):
            self.assertIn(needed, src, "install.sh does not install %s" % needed)



class TestSwitchPreflight(unittest.TestCase):
    """switch-model.sh must decide everything BEFORE it touches anything.

    setup/tunnel/switch.sh cost that lesson: it died on an unset $TOKEN, but
    only after it had swapped the tunnel configuration and restarted the
    container (tests/test_scripts.py::TestSwitch). The same property is
    checked here against a scratch repo, so the test never depends on which
    model this machine happens to be serving.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.repo = os.path.join(self.tmp, "repo")
        os.makedirs(self.repo)
        shutil.copytree(REPO / "setup", os.path.join(self.repo, "setup"),
                        ignore=shutil.ignore_patterns("__pycache__"))
        self.shrink_the_budget_constants()
        # a model file that really exists, so waitformodel is happy
        self.gguf = os.path.join(self.tmp, "ghost.gguf")
        open(self.gguf, "wb").close()
        self.write_profile()
        # a home with the symlink the SERVICE would read, and a binary
        self.home = os.path.join(self.tmp, "home")
        os.makedirs(os.path.join(self.home, ".config", "llm-profile"))
        os.symlink(os.path.join(self.repo, "setup", "env", "ghost.env"),
                   os.path.join(self.home, ".config", "llm-profile", "ghost.env"))
        binp = os.path.join(self.home, "llama.cpp", "build-vulkan", "bin")
        os.makedirs(binp)
        stub = os.path.join(binp, "llama-server")
        open(stub, "w").close()
        os.chmod(stub, 0o755)

    def shrink_the_budget_constants(self):
        """Make the ghost profile fit on ANY machine — in the COPY only.

        These tests are about preflight ORDER and about nothing being written
        before every decision is made. They are not about the memory verdict;
        tests/test_budget.py and tests/test_memory_guard.py are, with injected
        machine facts.

        But switch-model.sh runs the real guard, and the real guard needs a
        real machine: BUFFER_FLOOR_GIB 6.0 plus HOST_RESERVE_GIB 12.0 is an
        18 GiB floor before a single weight is loaded. A GitHub runner has
        7.8 GiB, so EVERY test in this class aborted at the memory step —
        which is the guard working correctly and the tests measuring the
        runner instead of the script.

        Found by the first CI run this repository ever had. It had never run
        anywhere, and the simulated-clone run that was meant to catch such
        things shared the one thing that decided the outcome: this machine.

        The copy is a scratch tree that exists for the length of one test. The
        real constants are never touched, and a test that ASSERTS them would
        fail here rather than pass quietly.
        """
        budget = os.path.join(self.repo, "setup", "lib", "budget.py")
        with open(budget, encoding="utf-8") as fh:
            text = fh.read()
        for name, value in (("BUFFER_FLOOR_GIB", "0.25"),
                            ("HOST_RESERVE_GIB", "0.25"),
                            ("ESTIMATE_KV_KIB_PER_TOKEN", "8.0")):
            before = text
            # A trailing comment is allowed: BUFFER_FLOOR_GIB carries its
            # own measurement on the same line, which the first version of
            # this rewrite did not match. The assertion below caught that.
            text = re.sub(r"(?m)^%s = [0-9.]+\b" % name,
                          "%s = %s" % (name, value), text, count=1)
            self.assertNotEqual(before, text,
                                "%s is no longer a module-level constant in "
                                "budget.py — this rewrite silently stopped "
                                "applying" % name)
        with open(budget, "w", encoding="utf-8") as fh:
            fh.write(text)

    def write_profile(self, name="ghost", extra=""):
        with open(os.path.join(self.repo, "setup", "env", name + ".env"),
                  "w", encoding="utf-8") as f:
            f.write("MODEL_TITLE=synthetic model for tests\n"
                    "MODEL_SWA=no\n"
                    "LLAMA_ARGS=--alias %s -m %s \\\n"
                    "  -ngl 999 -c 8192 -np 1 --host 127.0.0.1 --port 8080\n"
                    "LLAMA_BIN=llama.cpp/build-vulkan/bin/llama-server\n%s"
                    % (name, self.gguf, extra))

    def add_to_conflicts(self, name="ghost"):
        p = os.path.join(self.repo, "setup", "systemd", "llama-user@.service")
        with open(p, encoding="utf-8") as fh:
            text = fh.read()
        text = text.replace("Conflicts=", "Conflicts=llama-user@%s.service " % name, 1)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(text)

    def switch(self, *args):
        u = dict(os.environ)
        u["HOME"] = self.home
        u["SLOTS"] = os.path.join(self.home, ".cache", "llama-slots")
        return subprocess.run(
            ["bash", os.path.join(self.repo, "setup", "switch-model.sh"), *args],
            capture_output=True, text=True, timeout=120, env=u)

    def untouched(self):
        """Everything the script could have written, and did not.

        .config is excluded when the test put a gateway env there itself —
        what matters is that the SCRIPT wrote nothing.
        """
        cache = os.path.join(self.home, ".cache")
        if os.path.exists(cache):
            # A test may have planted a store on purpose; what matters is that
            # the SCRIPT neither parked nor re-keyed it.
            self.assertFalse(
                any(n.startswith("llama-slots.") for n in os.listdir(cache)),
                "the script parked a prefix store")
            owner = os.path.join(cache, "llama-slots", ".owner")
            if os.path.exists(owner):
                with open(owner) as fh:
                    owner_text = fh.read()
                self.assertNotIn("ghost", owner_text,
                                 "the script re-keyed the prefix store")
        cfg = os.path.join(self.home, ".config")
        if os.path.exists(cfg):
            # The fixture plants llm-profile/ always and llm-gateway.env on
            # demand; anything beyond those two was written by the script.
            extra = set(os.listdir(cfg)) - {"llm-profile", "llm-gateway.env"}
            self.assertEqual(extra, set(),
                             "the script wrote into %s" % cfg)

    # --- the aborts -------------------------------------------------------

    def test_preflight_weighs_the_profiles_measured_figures(self):
        """A profile whose FILE-SIZE estimate does not fit but whose measured
        MODEL_GTT_BASE_GIB / MODEL_HOST_ANON_GIB do must pass the memory
        preflight — flashnext is exactly this shape (103.7 GiB of file, 80.8
        in GTT, 0.31 resident). On 01.09.2026 the switch refused it with the
        file arithmetic because the preflight forwarded MODEL_WEIGHTS_GTT_GIB,
        a name budget.py --from-env does not read, and forwarded the two
        measured figures not at all."""
        self.write_profile("ghost",
                           extra="MODEL_GTT_BASE_GIB=0.4\n"
                                 "MODEL_HOST_ANON_GIB=0.05\n")
        self.add_to_conflicts()
        # LLM_MODEL_GIB overstates the weights far beyond the shrunk machine
        # floor; only the measured figures can make this profile fit.
        u = dict(os.environ)
        u["HOME"] = self.home
        u["SLOTS"] = os.path.join(self.home, ".cache", "llama-slots")
        u["LLM_MODEL_GIB"] = "500"
        r = subprocess.run(
            ["bash", os.path.join(self.repo, "setup", "switch-model.sh"),
             "ghost", "--dry-run"],
            capture_output=True, text=True, timeout=120, env=u)
        self.assertNotIn("does not fit", r.stdout + r.stderr)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.untouched()

    def test_an_unknown_model_aborts_and_lists_the_real_ones(self):
        r = self.switch("not-a-model", "--dry-run")
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("unknown model", r.stderr)
        self.assertIn("ghost", r.stderr, "the abort should say what DOES exist")
        self.untouched()

    def test_no_model_at_all_aborts(self):
        r = self.switch("--dry-run")
        self.assertEqual(r.returncode, 2)
        self.assertIn("no model given", r.stderr)
        self.untouched()

    def test_a_profile_the_service_cannot_read_aborts(self):
        """The unit names its EnvironmentFile without a leading '-', so a
        missing profile has to stop the switch rather than start llama-server
        with an empty $LLAMA_ARGS."""
        self.write_profile("orphan")
        self.add_to_conflicts("orphan")          # only the symlink is missing
        r = self.switch("orphan", "--dry-run")
        self.assertEqual(r.returncode, 2, r.stdout)
        self.assertIn("does not exist", r.stderr)
        self.untouched()

    def test_a_model_missing_from_conflicts_aborts(self):
        """The silent failure this whole file is about: without the
        Conflicts= entry, two llama-servers start and the unit still says
        'active'."""
        r = self.switch("ghost", "--dry-run")       # not added to Conflicts
        self.assertEqual(r.returncode, 2, r.stdout)
        self.assertIn("MISSING from the Conflicts", r.stderr)
        self.untouched()

    def test_a_missing_model_file_aborts_and_says_how_to_get_it(self):
        """The directory is there, the file is not — which for anyone who has
        just cloned the repo means "not fetched yet", not "the mount is gone".
        The old message named only the fstab cause and sent a newcomer into
        the machine's plumbing over a missing download."""
        self.add_to_conflicts()
        os.unlink(self.gguf)
        r = self.switch("ghost", "--dry-run")
        self.assertEqual(r.returncode, 2, r.stdout)
        self.assertIn(self.gguf, r.stderr, "the abort must name the path")
        self.assertIn("get-model.sh ghost", r.stderr)
        self.untouched()

    def test_a_missing_DIRECTORY_still_points_at_the_mount(self):
        """The other cause, and it wants the opposite action: a model volume in
        fstab with nofail passes silently when it is not mounted, so an absent
        mount looks exactly like this."""
        self.add_to_conflicts()
        os.unlink(self.gguf)
        # Point the profile INTO a directory that does not exist, rather than
        # trying to remove the scratch directory the fixtures live in.
        gone = os.path.join(os.path.dirname(self.gguf), "no-such-mount", "ghost.gguf")
        env = os.path.join(self.repo, "setup", "env", "ghost.env")
        with open(env, encoding="utf-8") as fh:
            text = fh.read().replace(self.gguf, gone)
        with open(env, "w", encoding="utf-8") as fh:
            fh.write(text)
        r = self.switch("ghost", "--dry-run")
        self.assertEqual(r.returncode, 2, r.stdout)
        self.assertIn("nofail", r.stderr)
        self.assertNotIn("get-model.sh", r.stderr,
                         "fetching cannot help when the directory is gone")
        self.untouched()

    def move_ghost_to(self, port):
        path = os.path.join(self.repo, "setup", "env", "ghost.env")
        with open(path, encoding="utf-8") as fh:
            text = fh.read().replace("--port 8080", "--port %d" % port)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)

    def plant_gateway(self, port=8080):
        """A gateway config, which is one of the signals that say a
        gateway lives on this machine."""
        gwenv = os.path.join(self.home, ".config", "llm-gateway.env")
        os.makedirs(os.path.dirname(gwenv), exist_ok=True)
        with open(gwenv, "w") as fh:
            fh.write("LLAMA_URL=http://127.0.0.1:%d\n" % port)

    def test_every_profile_says_where_its_model_comes_from(self):
        """MODEL_SOURCE is what makes `get-model.sh <name>` possible, and
        without it a fresh clone on another machine cannot satisfy a single
        profile — which was true of this repo until 26.08. The shape is
        checked, not the contents: whether the repo exists is a network
        question and belongs to fetch-model.sh."""
        import systemdfile as envfile
        for env in sorted((REPO / "setup" / "env").glob("*.env")):
            with self.subTest(profile=env.name):
                src = envfile.variable(str(env), "MODEL_SOURCE", "").strip()
                self.assertTrue(src, "%s has no MODEL_SOURCE" % env.name)
                parts = src.split()
                self.assertGreaterEqual(
                    len(parts), 2,
                    "%s: MODEL_SOURCE needs a repo AND at least one pattern" % env.name)
                self.assertIn("/", parts[0],
                              "%s: %r is not an owner/repo" % (env.name, parts[0]))

    def test_a_profile_on_another_port_than_the_gateway_aborts(self):
        """Three profiles in the repo serve on 8081-8083. Switching to one of
        them used to start the model and then wait fifteen minutes on 8080
        before failing — with the old model already stopped and disabled. The
        model was fine; the script was looking in the wrong place.

        The gateway config is planted explicitly. Until 26.08. this test
        passed without one, because the script fell back to assuming a
        gateway on 8080 — which is exactly the coupling that made a machine
        without a harness unable to switch models at all."""
        self.add_to_conflicts()
        self.plant_gateway(8080)
        self.move_ghost_to(8081)
        r = self.switch("ghost", "--dry-run")
        self.assertEqual(r.returncode, 2, r.stdout)
        self.assertIn("port 8081", r.stderr)
        self.assertIn("dead port", r.stderr)
        self.untouched()

    def test_without_a_gateway_the_port_does_not_block_the_switch(self):
        """The inference layer may notice the harness; it must not need it.

        The abort above protects consumers from being left talking to a dead
        port. Where no gateway is installed there is no such consumer, and
        the wait in step 5 targets the port the PROFILE names anyway. Aborting
        there would refuse a switch for a reason that has nothing to do with
        the model — the same shape as ExecStartPost without a leading '-'."""
        self.add_to_conflicts()
        self.move_ghost_to(8081)          # and deliberately NO gateway config
        r = self.switch("ghost", "--dry-run")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("no gateway here", r.stdout)

    def test_without_a_gateway_nothing_is_restarted_or_smoked_through_it(self):
        """The plan must not contain a step that needs a component that is
        not there. --dry-run prints what it WOULD do, so the absence is
        checkable without a machine."""
        self.add_to_conflicts()
        r = self.switch("ghost", "--dry-run")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("none installed, skipped", r.stdout)
        self.assertNotIn("restart llm-gateway", r.stdout)
        self.assertNotIn("restart cc-gateway", r.stdout)
        self.assertIn("smoke against the server", r.stdout)

    def test_the_wait_targets_the_port_the_profile_serves_on(self):
        """Not a hard-wired 8080. Proven through the gateway env, so the
        check follows a moved gateway too."""
        self.add_to_conflicts()
        self.plant_gateway(8081)
        self.move_ghost_to(8081)
        r = self.switch("ghost", "--dry-run")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("port 8081", r.stdout)

    def test_a_nonsense_owner_marker_aborts_before_any_delete(self):
        """$SLOTS/.owner decides the target of an `rm -rf` two steps later.
        A file that steers a recursive delete is checked, not trusted."""
        self.add_to_conflicts()
        slots = os.path.join(self.home, ".cache", "llama-slots")
        os.makedirs(slots)
        open(os.path.join(slots, "something.bin"), "w").close()
        with open(os.path.join(slots, ".owner"), "w") as f:
            f.write("../../../etc\n")
        r = self.switch("ghost", "--dry-run")
        self.assertEqual(r.returncode, 2, r.stdout)
        self.assertIn("not a model in this repo", r.stderr)
        self.assertTrue(os.path.exists(os.path.join(slots, "something.bin")),
                        "the prefix store was touched")

    def test_a_missing_binary_aborts(self):
        self.add_to_conflicts()
        os.unlink(os.path.join(self.home, "llama.cpp", "build-vulkan",
                               "bin", "llama-server"))
        r = self.switch("ghost", "--dry-run")
        self.assertEqual(r.returncode, 2, r.stdout)
        self.assertIn("not executable", r.stderr)
        self.untouched()

    # --- the happy path, still changing nothing ---------------------------

    def test_a_complete_profile_passes_preflight_and_dry_run_writes_nothing(self):
        self.add_to_conflicts()
        r = self.switch("ghost", "--dry-run")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("DRY RUN", r.stdout)
        self.assertIn("would:", r.stdout, "a dry run should say what it would do")
        self.untouched()

    def test_the_dry_run_never_calls_sudo_or_systemctl_for_real(self):
        self.add_to_conflicts()
        r = self.switch("ghost", "--dry-run")
        # POSITIVE CONTROL FIRST. Everything below is a loop over the output,
        # and a loop over nothing passes: if the dry run aborted early or wrote
        # to stderr instead, this test used to go green precisely BECAUSE the
        # thing it checks never ran. Found 27.08. by sweeping the suite for
        # assertions that only live inside a loop.
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        candidates = [l for l in r.stdout.splitlines()
                      if "sudo" in l or "systemctl" in l
                      or l.strip().startswith("mv ")]
        self.assertGreaterEqual(
            len(candidates), 2,
            "the plan named fewer than two privileged steps, so there was "
            "essentially nothing to check. A switch installs to /etc and calls "
            "systemctl at least twice; if that stopped being true, this test "
            "needs rewriting rather than passing.\n%s" % r.stdout)
        for line in candidates:
            self.assertIn("would:", line,
                          "this line was not announced as a plan: %r" % line)

    def test_the_gateway_is_restarted_only_after_the_model_answers(self):
        """Order, not presence. The gateway asks the server for its slot count
        at startup (query_slots) and falls back to a DEFAULT OF 2 when nobody
        answers. Restarting it before the model is up therefore left
        MAX_INFLIGHT at 2 against a one-slot server on every single switch —
        which is not visible: the gateway just admits a second request that
        llama.cpp then queues internally, where the priority ordering between
        local, LAN and remote no longer applies. Found on 26.08. by rehearsing
        a real switch; check.sh had been reporting it all along."""
        text = (REPO / "setup" / "switch-model.sh").read_text(encoding="utf-8")
        body = text[text.index("From here on the system is changed"):]
        wait = body.index("/slots")
        restart = body.index('restart "$GW_UNIT"')
        self.assertLess(wait, restart,
                        "the gateway is restarted before the server is known "
                        "to answer — MAX_INFLIGHT will not match the slot count")

    def test_list_names_the_synthetic_model_too(self):
        r = self.switch("--list")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("ghost", r.stdout)
        self.assertIn("synthetic model for tests", r.stdout)


class TestNoModelNamesInCode(unittest.TestCase):
    """A model name on an executable line is a list that will be forgotten.

    In prose it is documentation and welcome — the switch to qwen38 is worth
    naming in a comment. What must not come back is `case "$NEW" in qwen38)`.
    """

    FILES = ("setup/switch-model.sh", "setup/check.sh", "setup/lib/models.sh")

    def scripts(self):
        """Every shell script in setup/. The .gguf check below is safe to run
        this widely; the profile-NAME check above is not, because one of the
        profiles is called `batch` and that is also an ordinary English word.
        Two rules, two scopes, and the difference is the reason."""
        import glob as _g
        out = _g.glob(str(REPO / "setup" / "*.sh")) + _g.glob(str(REPO / "setup" / "scripts" / "*.sh"))
        out += [str(REPO / p) for p in ("setup/llamaexec", "setup/waitformodel",
                                        "setup/checkroom", "setup/llmprofile")]
        return sorted(f for f in out if os.path.isfile(f))

    def test_no_script_names_a_weights_FILE(self):
        """The other half of the same rule, and the half that was missing.

        The original checked profile names in three files. The four
        measurement scripts named the WEIGHTS instead —
        `gemma-4-26B_q4_0-it.gguf`, `gpt-oss-120b-MXFP4.gguf` — which the rule
        could not see and which is worse: a filename does not follow a switch,
        so the script quietly measures whatever used to be called that. Ask
        `model_gguf <profile>` instead.
        """
        files = set()
        for name in profiles():
            argv = systemdfile.llama_args(str(REPO / "setup" / "env" / (name + ".env")))
            for flag_ in ("-m", "--model", "--mmproj"):
                v = systemdfile.flag(argv, flag_)
                if v:
                    files.add(os.path.basename(v))
        bad = []
        for f in self.scripts():
            rel = os.path.relpath(f, str(REPO))
            with open(f, encoding="utf-8") as fh:
                lines = fh.read().splitlines()
            for n, line in enumerate(lines, 1):
                if line.lstrip().startswith("#"):
                    continue
                for g in files:
                    if g in line:
                        bad.append("%s:%d  %s" % (rel, n, line.strip()[:70]))
        self.assertFalse(bad, "these name a weights file instead of asking the "
                              "registry (`model_gguf <profile>`):\n    "
                              + "\n    ".join(bad))

    def test_the_registry_consumers_name_no_model(self):
        names = set(profiles())
        for rel in self.FILES:
            text = (REPO / rel).read_text(encoding="utf-8")
            for n, line in enumerate(text.splitlines(), 1):
                if line.lstrip().startswith("#"):
                    continue                      # prose is allowed to be concrete
                for m in names:
                    with self.subTest(file=rel, line=n, model=m):
                        self.assertNotIn(m, line,
                                         "%s:%d hardcodes the model %r: %s"
                                         % (rel, n, m, line.strip()))


class TestLocalJsonMatchesAModel(unittest.TestCase):
    """setup/claude/local.json points Claude Code at a model name, and nothing
    reconciles that file with the profile that declares the modes.

    Two files, one agreement, no derivation possible — the shape TestConflicts
    exists for. Checking only the STEM was not enough: `qwen38-nonsense`
    starts with a real profile and resolves to nothing, and the gateway then
    serves it as the bare alias. Since 28.08. that is at least a log line, but
    the user's only other signal is that thinking silently stopped.
    """

    MODES = common.load("setup/gateway/modes.py", "modes")

    def test_the_configured_model_names_resolve_to_a_profile(self):
        env = json.loads((REPO / "setup/claude/local.json").read_text())["env"]
        known = set(profiles())
        for key in ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL"):
            value = env.get(key, "")
            base = value.split("-")[0] if value else ""
            with self.subTest(key=key, value=value):
                self.assertIn(base, known,
                              "%s=%r does not begin with any model in "
                              "setup/env/" % (key, value))

    def test_the_suffix_is_a_mode_that_profile_declares(self):
        """The stem is not the agreement. What has to hold is that the gateway
        would RESOLVE this exact name against that profile's MODES."""
        import sys
        sys.path.insert(0, str(REPO / "setup" / "lib"))
        import systemdfile as SDF
        env = json.loads((REPO / "setup/claude/local.json").read_text())["env"]
        checked = 0
        for key in ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL"):
            value = env.get(key, "")
            if not value:
                continue
            alias = value.split("-")[0]
            path = REPO / "setup" / "env" / ("%s.env" % alias)
            if not path.exists():
                continue
            modes = self.MODES.parse_modes(SDF.variable(str(path), "MODES"))
            checked += 1
            with self.subTest(key=key, value=value):
                _, hit = self.MODES.resolve(value, alias, modes)
                self.assertTrue(hit,
                                "%s=%r is not a mode %s declares. It offers: %s"
                                % (key, value, alias,
                                   "  ".join(self.MODES.names(alias, modes))))
        self.assertGreater(checked, 0, "nothing was checked")


if __name__ == "__main__":
    unittest.main()
