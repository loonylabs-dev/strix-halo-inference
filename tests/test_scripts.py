"""Tests for the shell parts: waitformodel, token reading, switch.sh.

Shell can be tested if the scripts are written so that they have a return
value and take their input from the environment. That is the case here — and
these are the places where a mistake either keeps the model server from
booting or touches the public tunnel.
"""
import os, re, subprocess, sys, tempfile, unittest

import common

REPO = common.REPO
WAIT = str(REPO / "setup" / "waitformodel")


def run_one(path, args=(), env_=None, stdin_=None):
    u = dict(os.environ)
    u.update(env_ or {})
    return subprocess.run(["bash", path, *args], capture_output=True, text=True,
                          env=u, timeout=60, input=stdin_)


class TestWaitForModel(unittest.TestCase):
    """Without this script the service fails three times in fifteen seconds
    and then stays down for good — unnoticed at boot."""

    def test_existing_model_immediately(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf") as f:
            r = run_one(WAIT, env_={"LLAMA_ARGS": "--alias x -m %s -ngl 999" % f.name})
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_long_flag_is_recognised_too(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf") as f:
            r = run_one(WAIT, env_={"LLAMA_ARGS": "--model %s" % f.name})
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_missing_model_fails_with_a_reason(self):
        r = run_one(WAIT, env_={"LLAMA_ARGS": "-m /does/not/exist.gguf",
                                   "WAIT_MAX": "2"})
        self.assertEqual(r.returncode, 1)
        self.assertIn("not readable", r.stderr)
        self.assertIn("ntfs3", r.stderr, "the hint at the cause is missing")

    def test_no_model_flag_is_no_obstacle(self):
        # A profile without -m must not block the service.
        r = run_one(WAIT, env_={"LLAMA_ARGS": "--host 127.0.0.1"})
        self.assertEqual(r.returncode, 0)

    def test_no_llama_args_is_no_obstacle(self):
        r = run_one(WAIT, env_={"LLAMA_ARGS": ""})
        self.assertEqual(r.returncode, 0)


class TestTokenReading(unittest.TestCase):
    """The awk expression with which smoketest.sh and switch.sh take the
    secret out of the access file — it MUST read like cc-gateway
    (split(None, 1))."""
    EXPR = r"""awk '!/^#/ && NF>=2 {sub(/^[ \t]*[^ \t]+[ \t]+/, ""); print; exit}'"""

    def read_tokens(self, content):
        with tempfile.NamedTemporaryFile("w", suffix=".tokens", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            r = subprocess.run(["bash", "-c", '%s "%s"' % (self.EXPR, path)],
                               capture_output=True, text=True, timeout=30)
            return r.stdout.strip("\n")
        finally:
            os.unlink(path)

    def test_simple_access(self):
        self.assertEqual(self.read_tokens("martin-mobile secret\n"), "secret")

    def test_secret_with_spaces(self):
        # cc-gateway nimmt alles nach dem Namen. Mit '{print $2}' waere hier
        # only "two" would have come out and every check would report 401.
        self.assertEqual(self.read_tokens("a two three four\n"), "two three four")

    def test_comments_are_skipped(self):
        self.assertEqual(self.read_tokens("# header\n\nname value\n"), "value")

    def test_an_empty_file_yields_nothing(self):
        self.assertEqual(self.read_tokens("# only a comment\n"), "")

    def test_both_scripts_use_the_same_expression(self):
        """Duplicated code that is allowed to drift apart, drifts apart."""
        pattern = re.compile(r"sub\(/\^\[ \\t\]\*\[\^ \\t\]\+\[ \\t\]\+/, \"\"\)")
        for filename in ("setup/smoketest.sh", "setup/tunnel/switch.sh"):
            with self.subTest(filename=filename):
                text = (REPO / filename).read_text(encoding="utf-8")
                self.assertTrue(pattern.search(text),
                                "%s reads the secret differently from cc-gateway" % filename)


class TestSwitch(unittest.TestCase):
    def test_aborts_without_access_before_anything_happens(self):
        """The script used to die on an unset $TOKEN — but only AFTER it had
        already swapped the tunnel configuration and restarted the
        container."""
        with tempfile.TemporaryDirectory() as home_:
            # put down a configuration that must not be touched
            os.makedirs(os.path.join(home_, ".cloudflared"))
            important = os.path.join(home_, ".cloudflared", "config.new.yml")
            with open(important, "w") as f:
                f.write("tunnel: must-not-vanish\n")
            r = run_one(str(REPO / "setup" / "tunnel" / "switch.sh"),
                        ["example.invalid"],
                        env_={"HOME": home_, "TOKEN_FILE": "/does/not/exist"})
            # check inside the with — afterwards the directory is gone
            self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
            self.assertIn("stays as it is", r.stdout)
            self.assertTrue(os.path.exists(important),
                            "the prepared configuration was touched")



class TestAnImportStaysWithoutConsequences(unittest.TestCase):
    """tests/common.py states the precondition in so many words: the scripts
    here are loaded BY PATH, and "the precondition for that is an import
    without consequences: no network, no token file, no web.run_app".

    It was stated and not checked, and on 27.08.2026 two files broke exactly
    two of the three named items:

        bench/suites/cc-tap.py             web.run_app() at module level
        bench/suites/gateway-concurrency.py  read ~/.config/cc-gateway-tokens

    cc-tap is the sharper one. Importing it tried to BIND PORT 8090 — the port
    cc-gateway serves on — and the proof was the import failing with "address
    already in use" against the running production gateway.

    Nothing imports bench/suites/ today. That was the entire defence, and it
    is the kind nobody decided to rely on.

    CHECKED IN THE SOURCE, not by importing. Importing to find out would run
    the suite, which is the thing being guarded against — and nine other
    files in that directory legitimately execute when run. What is forbidden
    is not "doing something on import"; it is doing THESE things, and a file
    that wants to do them puts them behind `if __name__ == "__main__":`, which
    is what both now do.
    """

    # Matched as CALLS, not as text. The first version of this test looked
    # for the strings and flagged the gateway, which computes the token
    # path with expanduser at module level and never opens it — the rule is
    # about reading the file, not about naming it.
    CALLED = {"run_app": "starts a server",
              "urlopen": "goes to the network"}
    OPENS = "llm-gateway-tokens"

    def files(self):
        import glob as _g
        out = []
        for pattern in ("bench/suites/*.py", "bench/*.py", "setup/claude/*.py",
                        "setup/gateway/*.py", "tools/*.py", "setup/lib/*.py"):
            out += sorted(_g.glob(str(REPO / pattern)))
        return out

    @staticmethod
    def module_level(path):
        """Source lines that run on import: everything not inside a function,
        a class, or an `if __name__ == "__main__"` block."""
        import ast
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        tree = ast.parse(text, path)
        lines = text.splitlines()
        out = []

        def guarded(node):
            return (isinstance(node, ast.If)
                    and "__main__" in ast.dump(node.test))

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)) or guarded(node):
                continue
            end = getattr(node, "end_lineno", node.lineno)
            out += [(n, lines[n - 1]) for n in range(node.lineno, end + 1)
                    if n <= len(lines)]
        return out

    def test_the_scan_sees_module_level_code(self):
        """Positive control. If this returns nothing the test below passes by
        reading nothing — which is the defect it exists to prevent, one level
        up."""
        total = sum(len(self.module_level(f)) for f in self.files())
        self.assertGreater(total, 200,
                           "only %d module-level lines across %d files — the "
                           "scanner is not parsing them"
                           % (total, len(self.files())))

    def forbidden_calls(self, path):
        """(line, what) for every forbidden CALL that runs on import."""
        import ast
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), path)

        def guarded(node):
            return (isinstance(node, ast.If)
                    and "__main__" in ast.dump(node.test))

        found = []
        for top in tree.body:
            if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef,
                                ast.ClassDef)) or guarded(top):
                continue
            for node in ast.walk(top):
                if not isinstance(node, ast.Call):
                    continue
                name = (node.func.attr if isinstance(node.func, ast.Attribute)
                        else node.func.id if isinstance(node.func, ast.Name)
                        else "")
                if name in self.CALLED:
                    found.append((node.lineno, self.CALLED[name]))
                elif name == "open":
                    try:
                        rendered = ast.unparse(node)
                    except Exception:              # pragma: no cover
                        rendered = ""
                    if self.OPENS in rendered:
                        found.append((node.lineno, "reads the token file"))
        return found

    def test_nothing_forbidden_runs_at_import(self):
        bad = []
        for path in self.files():
            rel = os.path.relpath(path, str(REPO))
            for n, what in self.forbidden_calls(path):
                bad.append("    %s:%d  %s" % (rel, n, what))
        self.assertFalse(
            bad,
            "these run while the file is being IMPORTED, and tests/common.py "
            "names all three as the precondition for loading a script by "
            "path. Put them behind `if __name__ == \"__main__\":`:\n%s"
            % "\n".join(bad))


class TestNoStdlibShadowing(unittest.TestCase):
    """No file in the repo may be named like a standard library module.

    Python puts a script's own directory first on the search path. A file
    called bisect.py next to a script therefore hides the real `bisect` — and
    because `random` imports it and `tempfile` imports `random` and
    `urllib.request` imports `tempfile`, every script in that directory dies on
    `import urllib.request`. That happened here: renaming bisekt.py to
    bisect.py broke all ten measurement suites at once, and nothing noticed
    until one of them was run.
    """

    def test_no_file_hides_a_standard_module(self):
        stdlib = set(sys.stdlib_module_names)
        clashes = []
        for root, dirs, files in os.walk(REPO):
            dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
            for f in files:
                if f.endswith(".py") and f[:-3] in stdlib:
                    clashes.append(os.path.relpath(os.path.join(root, f), REPO))
        self.assertEqual(clashes, [],
                         "these files hide a standard library module for every "
                         "script beside them")


if __name__ == "__main__":
    unittest.main()


class TestUserUnit(unittest.TestCase):
    """The user service must not depend on root, and must not restore at
    boot. Both cost an incident on 25.08.: the root-owned profile under
    /etc made the stack unchangeable for a remote operator who cannot
    authenticate on the machine's own screen, and the eager boot restore
    poisoned a freshly started server within seconds — a restore landing
    during another slot's prompt processing corrupts the KV state
    (bench/suites/restore-safety.py, six cells)."""

    UNIT = str(REPO / "setup/systemd/llama-user@.service")

    def _directives(self, name):
        # Read with setup/lib/systemdfile.py, the same parser switch-model.sh
        # and tests/test_models.py use. The version that stood here split on
        # "=" line by line and would have missed any directive written across
        # continuation lines — which Conflicts= now is.
        import sys
        sys.path.insert(0, str(REPO / "setup" / "lib"))
        import systemdfile
        return systemdfile.directive(self.UNIT, name)

    def test_the_profile_comes_from_the_users_own_directory(self):
        """Two EnvironmentFiles since 27.08., and the difference between them
        is the whole point of the '-'.

        The machine's own answers come FIRST and are OPTIONAL: a machine that
        has never run install.sh must still be able to start a profile whose
        model path is absolute. The profile comes second and is MANDATORY —
        a missing profile would start llama-server with an empty $LLAMA_ARGS,
        which the repo learned means a server in router mode that listens,
        answers /slots, and serves no model at all.
        """
        files = self._directives("EnvironmentFile")
        self.assertEqual(files, ["-%h/.config/llm-stack.env",
                                 "%h/.config/llm-profile/%i.env"])
        self.assertTrue(files[0].startswith("-"),
                        "the local config must be optional: install.sh has to "
                        "be able to run before it exists")
        self.assertFalse(files[-1].startswith("-"),
                         "a missing profile must fail loudly, not start the "
                         "server with an empty $LLAMA_ARGS")

    def test_nothing_restores_slots_at_start(self):
        """The eager boot restore poisoned a freshly started server on 25.08.:
        ExecStartPost pulled the saved states in over the HTTP API while the
        first requests were already arriving, and a restore landing during
        another slot's prompt processing corrupts the KV state.

        THIS TEST WAS VACUOUS. It looped over ExecStartPost and asserted
        inside the loop — and the unit has no ExecStartPost at all any more,
        so the body never ran. It has been passing without reading anything
        since the directive was removed, which is the same failure it exists
        to prevent: the effect stopped happening and nothing said so.
        """
        # Positive control: the parser has to be finding this unit and
        # matching directives in it. Without this, "no ExecStartPost restores"
        # is equally true of a file that was never opened.
        self.assertTrue(self._directives("ExecStart"),
                        "the directive reader found no ExecStart in %s — it is "
                        "not reading the unit, so nothing below means anything"
                        % self.UNIT)
        for cmd in self._directives("ExecStartPost"):
            self.assertNotIn("prewarm.py restore", cmd,
                             "the boot restore poisons the server; the "
                             "gateway reloads lazily and idle-guarded")
        # And the form that does not depend on the directive being present:
        # no EFFECTIVE line of the unit may restore at all. Comments may —
        # line 161 explains why the directive was removed, and that history is
        # worth keeping.
        with open(self.UNIT, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        effective = [l for l in lines
                     if l.strip() and not l.lstrip().startswith("#")]
        self.assertTrue(effective, "the unit file is empty or unreadable")
        self.assertFalse([l for l in effective if "prewarm.py restore" in l],
                         "something in the unit restores slots at start")

    def test_the_binary_still_has_a_fallback(self):
        """The property is unchanged — a profile without LLAMA_BIN has to keep
        starting — but on 26.08. it moved out of the unit and into
        setup/llamaexec, because an inline `bash -c` in ExecStart cannot work:
        systemd substitutes $VAR there before any shell sees it. So the
        assertion follows the fallback rather than the line it used to be on."""
        start = self._directives("ExecStart")
        self.assertEqual(len(start), 1)
        self.assertIn("llamaexec", start[0])
        wrapper = (REPO / "setup" / "llamaexec").read_text(encoding="utf-8")
        self.assertIn("LLAMA_BIN:-", wrapper,
                      "a profile without LLAMA_BIN has to keep starting")


class TestPlaceholdersAgree(unittest.TestCase):
    """@HOME@ and @MODELS@ are expanded in three places, and all three have to
    know both. They exist because every profile in this repo used to name one
    person's directories, which made the whole thing unusable on anybody
    else's machine.

    The three cannot share an implementation: one is a Python module, one is
    an ExecStartPre in bash, and one is the ExecStart wrapper. What they CAN
    share is a test that none of them silently forgets a token — which would
    not fail loudly. It would start a server that looks for a file called
    `@MODELS@/…`, or, as happened on 26.08. while this was being built, one
    with no arguments at all: llama-server came up in router mode, listening
    and modelless, and /slots answered so it looked alive.
    """

    TOKENS = ("@HOME@", "@MODELS@")
    PLACES = ("setup/lib/systemdfile.py", "setup/waitformodel", "setup/llamaexec")

    def test_all_three_expanders_know_both_tokens(self):
        for rel in self.PLACES:
            text = (REPO / rel).read_text(encoding="utf-8")
            for tok in self.TOKENS:
                with self.subTest(place=rel, token=tok):
                    self.assertIn(tok, text, "%s does not mention %s" % (rel, tok))

    # The rule that used to live here — "no profile names a real home or model
    # directory" — is now tests/test_localenv.py. It had covered setup/env/,
    # the one directory whose files never contained such a path, while
    # eighteen files elsewhere did. The wider version scans every line that
    # RUNS, in the whole repo, and keeps the narrow profile check as well.

    def test_the_unit_execs_the_wrapper_and_not_an_inline_shell(self):
        """systemd substitutes $VAR inside ExecStart before any shell sees it,
        so bash's ${VAR//a/b} there expands to nothing. A file has no such
        layer."""
        import systemdfile
        unit = str(REPO / "setup" / "systemd" / "llama-user@.service")
        execs = systemdfile.directive(unit, "ExecStart")
        self.assertEqual(len(execs), 1, execs)
        self.assertIn("llamaexec", execs[0])
        self.assertNotIn("bash -c", execs[0])
