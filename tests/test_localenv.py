"""What belongs to the MACHINE, and the rule that it may not be in the repo.

Until 27.08.2026 this repo answered three questions with string constants:
where the models live (`/mnt/shared/LLM`, in eighteen files), where the repo
lives (`/mnt/shared/Development/inference-stack`, in seven), and what the
gateway's public hostname is. All three are true on one computer.

One of them had a consequence rather than an inconvenience. `setup/smoketest.sh`
used the private hostname as its DEFAULT, so `git clone && bash
setup/smoketest.sh` sent requests at somebody else's tunnel.

The answers now live in ~/.config/llm-stack.env, outside the repo and
gitignored, written once by setup/install.sh. These tests pin the two readers
that consult it — one in Python, one in shell, and they must agree — and the
rule that keeps the constants from coming back.
"""
import glob, os, re, subprocess, sys, tempfile, unittest
import os as _os

import common

REPO = common.REPO
sys.path.insert(0, str(REPO / "setup" / "lib"))
import systemdfile                                        # noqa: E402


def sh(script, env=None):
    """Run a snippet with setup/lib/models.sh sourced."""
    e = dict(os.environ, **(env or {}))
    return subprocess.run(
        ["bash", "-c", '. "%s/setup/lib/models.sh"\n%s' % (REPO, script)],
        capture_output=True, text=True, env=e, cwd=str(REPO))


class Base(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, self.d, ignore_errors=True)
        self.saved = {k: os.environ.pop(k, None)
                      for k in ("LLAMA_MODELS", "LLM_STACK_ENV", "MODELLPFAD")}

    def tearDown(self):
        for k, v in self.saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def config(self, text):
        p = os.path.join(self.d, "llm-stack.env")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.environ["LLM_STACK_ENV"] = p
        return p

    def models(self, *names):
        d = os.path.join(self.d, "models")
        os.makedirs(d, exist_ok=True)
        for n in names or ("m.gguf",):
            open(os.path.join(d, n), "wb").close()
        return d


class TestTheLocalConfig(Base):
    def test_a_missing_file_is_not_an_error(self):
        """install.sh has to be able to run on a machine that has never run
        it — which means everything it calls must survive the file's absence."""
        os.environ["LLM_STACK_ENV"] = os.path.join(self.d, "nope.env")
        self.assertIsNone(systemdfile.local_var("LLAMA_MODELS"))
        self.assertEqual(systemdfile.local_var("LLAMA_MODELS", "fallback"), "fallback")

    def test_it_is_read_as_systemd_syntax_and_not_sourced(self):
        """`. file.env` makes bash read `VAR=value command args`. The repo
        documents that trap in three places; this reader must not fall into
        it, which is why it is systemdfile and not a `source`."""
        self.config('LLAMA_MODELS=/srv/models\nGATEWAY_HOST=host.example\n')
        self.assertEqual(systemdfile.local_var("GATEWAY_HOST"), "host.example")

    def test_the_shell_reader_agrees_with_the_python_one(self):
        """Two readers of one file is how the three LLAMA_ARGS parsers began."""
        p = self.config('LLAMA_MODELS=/srv/models\nGATEWAY_HOST=host.example\n')
        for name in ("LLAMA_MODELS", "GATEWAY_HOST"):
            got = sh('local_var %s' % name, {"LLM_STACK_ENV": p}).stdout.strip()
            self.assertEqual(got, systemdfile.local_var(name), name)

    def test_an_empty_value_reads_as_absent(self):
        """GATEWAY_HOST is empty in the template on purpose. Empty has to mean
        'not configured' and not the empty string, or smoketest.sh builds
        https:///v1/messages out of it."""
        self.config("GATEWAY_HOST=\n")
        self.assertIsNone(systemdfile.local_var("GATEWAY_HOST"))
        self.assertEqual(sh('local_var GATEWAY_HOST',
                            {"LLM_STACK_ENV": os.environ["LLM_STACK_ENV"]}).stdout.strip(), "")


class TestWhereTheModelsAre(Base):
    def test_the_environment_wins(self):
        self.config("LLAMA_MODELS=/from/config\n")
        os.environ["LLAMA_MODELS"] = "/from/env"
        self.assertEqual(systemdfile.models_dir(), "/from/env")

    def test_then_the_local_config(self):
        self.config("LLAMA_MODELS=%s\n" % self.models())
        self.assertEqual(systemdfile.models_dir(), self.models())

    def test_then_a_convention_that_actually_holds_gguf_files(self):
        """A conventional directory counts only if the files are IN it. An
        empty ~/models is not an answer, it is a coincidence."""
        self.config("")
        d = self.models()
        old = systemdfile.MODELS_CONVENTIONS
        self.addCleanup(setattr, systemdfile, "MODELS_CONVENTIONS", old)
        systemdfile.MODELS_CONVENTIONS = (os.path.join(self.d, "empty"), d)
        os.makedirs(os.path.join(self.d, "empty"), exist_ok=True)
        self.assertEqual(systemdfile.models_dir(), d)

    def test_no_entry_in_the_conventions_names_one_machine(self):
        """The whole point. A fallback containing /mnt/shared/LLM would be the
        same hard-coding one level down — and that is exactly what the first
        draft of this mechanism contained.

        The rule is NOT "must be relative", which is what the first version of
        this test said. `/srv/models` and `/var/lib/llm` are FHS locations and
        `/mnt/*/LLM` matches by SHAPE — all three are conventions on any
        machine. What is forbidden is a concrete path under a mount point,
        because that names somebody's disk.
        """
        for d in systemdfile.MODELS_CONVENTIONS:
            with self.subTest(convention=d):
                if d.startswith("~") or d.startswith("."):
                    continue                       # under $HOME or relative
                if "*" in d:
                    self.assertTrue(d.count("*") >= 1 and "/mnt/" in d,
                                    "%s globs something other than a mount" % d)
                    continue
                self.assertTrue(d.startswith(("/srv/", "/var/lib/", "/opt/")),
                                "%s is a machine, not a convention" % d)
        for banned in ("/mnt/shared/LLM", "/mnt/shared/Development"):
            self.assertNotIn(banned, systemdfile.MODELS_CONVENTIONS)

    def test_the_convention_list_exists_exactly_once(self):
        """Written after reproducing, in three hours, the exact failure this
        repo keeps writing about: models.sh and install.sh each grew their own
        copy of this list, and install.sh's was already WIDER — so a directory
        it found when writing the config was one models_dir() would then not
        have found on its own.
        """
        import re
        for rel in ("setup/lib/models.sh", "setup/install.sh"):
            src = (REPO / rel).read_text(encoding="utf-8")
            code = "\n".join(l for l in src.splitlines()
                              if not l.lstrip().startswith("#"))
            self.assertIn("systemdfile.py", code,
                          "%s does not read the convention list" % rel)
            self.assertNotRegex(
                code, r'\.cache/llama\.cpp["\s]',
                "%s carries its own copy of the conventions again" % rel)

    def test_the_shell_side_sees_the_same_conventions(self):
        got = sh("llm_conventions").stdout.split()
        self.assertEqual(got, list(systemdfile.MODELS_CONVENTIONS))

    def test_a_glob_resolves_deterministically(self):
        """/mnt/*/LLM can match twice. Sorted, or the model directory depends
        on readdir order and two runs on one machine disagree."""
        src = (REPO / "setup" / "lib" / "systemdfile.py").read_text(encoding="utf-8")
        self.assertIn("sorted(glob.glob", src)

    def test_it_gives_up_rather_than_guessing(self):
        self.config("")
        old = systemdfile.MODELS_CONVENTIONS
        self.addCleanup(setattr, systemdfile, "MODELS_CONVENTIONS", old)
        systemdfile.MODELS_CONVENTIONS = (os.path.join(self.d, "empty"),)
        with self.assertRaises(SystemExit) as cm:
            systemdfile.models_dir()
        self.assertIn("LLAMA_MODELS", str(cm.exception))
        self.assertIn("install.sh", str(cm.exception))
        self.assertIsNone(systemdfile.models_dir(required=False))

    def test_expand_only_needs_an_answer_when_it_is_asked_for_one(self):
        """A unit file without @MODELS@ must stay readable on a machine that
        has no models at all — otherwise a fresh checkout cannot read its own
        configuration in order to be told where they are."""
        self.config("")
        old = systemdfile.MODELS_CONVENTIONS
        self.addCleanup(setattr, systemdfile, "MODELS_CONVENTIONS", old)
        systemdfile.MODELS_CONVENTIONS = ()
        self.assertEqual(systemdfile.expand("@HOME@/x"),
                         os.path.expanduser("~") + "/x")
        with self.assertRaises(SystemExit):
            systemdfile.expand("@MODELS@/x")


class TestRecordingIsTheInverseOfRunning(unittest.TestCase):
    """"Expanded to run, unexpanded to record" needed an implementation.

    The rule was learned on 27.08. and lived only in bench/sweep.py, which
    never had to fold anything: it still had the raw value from the variants
    file. A tool that COMPUTES a path at runtime has no raw value, and the
    first one that needed it — bench/suites/restore-safety.py, recording
    which binary produced a report — wrote a home directory into three
    reports before test_no_report_records_a_home_directory caught it.

    So the fold is the inverse of expand() and sits beside it, rather than
    being re-derived by every future recorder. That re-derivation is where
    this repository's bugs live; the convention list for model directories
    existed in three places for three hours and cost a whole afternoon.
    """

    def test_it_is_the_inverse_of_expand(self):
        home = os.path.expanduser("~")
        for raw in ("@HOME@/llama.cpp/build-rocm-patched/bin/llama-server",
                    "@HOME@/.cache/llama-slots", "no placeholder at all"):
            self.assertEqual(systemdfile.expand(systemdfile.unexpand(
                systemdfile.expand(raw, home=home, models="/m"),
                home=home, models="/m"), home=home, models="/m"),
                systemdfile.expand(raw, home=home, models="/m"), raw)

    def test_a_models_directory_under_the_home_still_folds(self):
        """The ordering property, and it is not cosmetic: fold the home
        first and "@HOME@/models" is left with nothing for @MODELS@ to
        match, so the more specific placeholder never appears."""
        got = systemdfile.unexpand("/h/models/qwen.gguf",
                                   home="/h", models="/h/models")
        self.assertEqual(got, "@MODELS@/qwen.gguf")

    def test_a_trailing_slash_does_not_defeat_it(self):
        self.assertEqual(
            systemdfile.unexpand("/h/x", home="/h/", models=None), "@HOME@/x")

    def test_the_report_writer_actually_uses_it(self):
        """The rule is only worth having where it is applied. Pinned in the
        source, because the alternative is running a measurement to find out.

        It lives in bench/run.py — provenance() moved there when a second
        suite needed it, which is the point: one implementation, so the rule
        is applied once rather than remembered twice."""
        src = (REPO / "bench/run.py").read_text(encoding="utf-8")
        self.assertIn("systemdfile.unexpand(binary)", src,
                      "bench/run.py:provenance() must fold the path it "
                      "records")


class TestNothingInTheRepoNamesOneMachine(unittest.TestCase):
    """The rule, applied to EFFECTIVE code rather than to prose.

    A hard-coded path in a comment is documentation — several of them explain
    why this mechanism exists and must stay. A hard-coded path in a line that
    RUNS is a default, and a default that is true on one computer is what this
    whole step removed.

    setup/env/*.env has had this rule since 26.08. It covered the files that
    never contained the path, and not the eighteen that did.
    """

    # Three corrections to one line, all on 27.08.2026, and each looked right
    # until it ran somewhere else:
    #
    #   * it was the maintainer's login name as a LITERAL — published in the
    #     test that exists to keep such things out, so it is not repeated here
    #   * getpass.getuser() looked strictly better and was not: on a GitHub
    #     runner that returns "runner", which appears legitimately in
    #     `runner = web.AppRunner(...)` and in three comments about CI. Red on
    #     all three Python versions
    #   * the home PATH as a plain substring then matched `/nonexistent-token`
    #     in tests/test_gateway.py, because the suite runs with
    #     HOME=/nonexistent in a simulated clone
    #
    # What a line actually leaks is the path, and only when it appears AS a
    # path. HOME_PATH is therefore matched with a boundary; the other two stay
    # plain substrings, which is what they have always been.
    FORBIDDEN = ("/mnt/shared", "loonylabs")
    HOME_PATH = os.path.expanduser("~")

    # What is still outstanding, each named with the decision it belongs to.
    # Named rather than remembered: an exception nobody can see is how a
    # temporary state becomes permanent.
    # Empty since 27.08.2026, and it has been through three entries:
    # setup/systemd/llama@.service (deleted, the system unit is derived now),
    # bench/variants/qwen38.json (@HOME@) and docs/CONSUMERS.md (generic, with
    # the operator's values read from the running stack instead). Each one was
    # taken off by test_the_outstanding_list_is_still_accurate going red — an
    # exception that no longer applies fails here rather than being forgotten.
    OUTSTANDING = {}

    SCAN = ("*.py", "*.sh", "*.json", "*.service", "*.timer", "*.env", "*.template",
            "*.awk", "*.conf")
    # Markdown is NOT scanned wholesale: docs/SECURITY.md and the machine
    # records under docs/px13/ name this machine because they are EVIDENCE
    # about it, and a rule that forbade that would be a rule against writing
    # measurements down. These two are different — somebody READS them in
    # order to DO something, so a name in them is an instruction that is wrong
    # for everyone else. docs/CONSUMERS.md sat on the OUTSTANDING list for a
    # day without ever being scanned, which is how a decorative exception
    # looks from the inside.
    ALSO_SCAN = ("docs/CONSUMERS.md", "setup/README.md", "bench/README.md",
                 "tests/README.md", "README.md")
    EXTRALESS = ("setup/llamaexec", "setup/waitformodel", "setup/checkroom",
                 "setup/llmprofile", "setup/get-model.sh", "setup/lib/models.sh")
    SKIP_DIRS = ("/.git/", "/__pycache__/", "/reports/", "/docs/px13/",
                 "/docs/archive/")

    def files(self):
        out = []
        for pat in self.SCAN:
            out += glob.glob(str(REPO / "**" / pat), recursive=True)
        out += [str(REPO / p) for p in self.EXTRALESS + self.ALSO_SCAN]
        return sorted({f for f in out
                       if not any(d in f.replace("\\", "/") for d in self.SKIP_DIRS)})

    def code_lines(self, path):
        """Lines a reader would RUN. Prose may name the history; a command
        may not.

        In a script that means skipping `#` comments. In Markdown there is no
        comment marker — the prose IS the file — so the equivalent is: only
        what sits in a code block counts. `setup/README.md` has to be able to
        say that `/mnt/shared/LLM` was once a default in eighteen files, which
        is the paragraph explaining why it no longer is; what it must not
        contain is a command with that path in it, because somebody will paste
        it.

        Fenced blocks and 4-space indented blocks both count: this repo uses
        the indented form for nearly every command it shows.
        """
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        markdown = path.endswith(".md")
        fenced = False
        for n, line in enumerate(text.splitlines(), 1):
            if not markdown:
                if line.lstrip().startswith("#"):
                    continue
                yield n, line
                continue
            if line.lstrip().startswith("```"):
                fenced = not fenced
                continue
            indented = line.startswith("    ") or line.startswith("\t")
            if fenced or indented:
                yield n, line

    def test_no_running_line_names_one_machine(self):
        bad = []
        for f in self.files():
            rel = os.path.relpath(f, str(REPO))
            if rel in self.OUTSTANDING or rel.startswith("tests/test_localenv"):
                continue
            for n, line in self.code_lines(f):
                hit = any(token in line for token in self.FORBIDDEN)
                # The home path only counts as a path: followed by a separator
                # or ending the token, never as the prefix of a longer word.
                if not hit and self.HOME_PATH:
                    hit = re.search(re.escape(self.HOME_PATH) + r'(?![\w-])', line)
                if hit:
                    bad.append("%s:%d  %s" % (rel, n, line.strip()[:90]))
        self.assertFalse(bad, "these lines name one machine:\n    "
                              + "\n    ".join(bad))

    def test_the_markdown_rule_catches_a_command_and_spares_the_prose(self):
        """The distinction the rule rests on, checked directly rather than
        trusted. setup/README.md must be able to SAY that /mnt/shared/LLM was
        a default in eighteen files — that paragraph is why it no longer is —
        and must not contain a command anybody could paste."""
        import tempfile, os as _os
        doc = ("Until 27.08. the default was /mnt/shared/LLM in eighteen files.\n"
               "\n"
               "    sudo loginctl enable-linger $USER\n"
               "\n"
               "```\n"
               "echo fine\n"
               "```\n")
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False,
                                         encoding="utf-8") as fh:
            fh.write(doc); path = fh.name
        self.addCleanup(_os.unlink, path)
        lines = [l for _, l in self.code_lines(path)]
        self.assertNotIn("Until 27.08. the default was /mnt/shared/LLM in eighteen files.",
                         lines, "prose is being scanned")
        self.assertTrue(any("enable-linger" in l for l in lines),
                        "an indented command is not being scanned")
        self.assertTrue(any("echo fine" in l for l in lines),
                        "a fenced command is not being scanned")

    def test_the_documents_a_reader_acts_on_are_in_the_scan(self):
        """docs/CONSUMERS.md sat on the OUTSTANDING list for a day while the
        scan did not cover Markdown at all — so the exception was decorative
        and the rule would never have caught the domain coming back."""
        scanned = {_os.path.relpath(f, str(REPO)) for f in self.files()}
        for rel in self.ALSO_SCAN:
            self.assertIn(rel, scanned)

    def test_the_outstanding_list_is_still_accurate(self):
        """An exception that no longer applies is worse than no exception: it
        says a problem exists where it has been fixed, and the next reader
        stops trusting the list."""
        stale = []
        for rel in self.OUTSTANDING:
            path = REPO / rel
            if not path.exists():
                stale.append("%s no longer exists" % rel)
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if not any(t in text for t in self.FORBIDDEN):
                stale.append("%s is clean — take it off the list" % rel)
        self.assertFalse(stale, "\n    ".join(stale))

    def test_no_variants_file_names_one_machine_s_builds(self):
        """The narrow rule for setup/env/*.env, applied to the other place a
        path gets written down by hand.

        bench/variants/qwen38.json carried eight absolute
        `/home/<user>/llama.cpp/build-*/bin/llama-server` until 27.08. — one
        per cell — so a sweep could not be repeated from another machine OR
        from a second checkout on this one. They go through the same expander
        as the profiles now, and bench/sweep.py applies it.
        """
        import json as _json
        for f in sorted((REPO / "bench" / "variants").glob("*.json")):
            spec = _json.loads(f.read_text(encoding="utf-8"))
            paths = [v.get("binary", "") for v in spec.get("variants", [])]
            paths += [a for a in spec.get("base_args", []) if a.startswith("/")]
            for p_ in paths:
                with self.subTest(variants=f.name, path=p_):
                    self.assertFalse(p_.startswith("/"),
                                     "%s names an absolute path; use @HOME@ / "
                                     "@MODELS@, which sweep.py expands" % f.name)

    def test_sweep_expands_the_variants_it_reads(self):
        """A placeholder nobody expands is worse than an absolute path: the
        server is handed a literal `@HOME@/...` and fails at load with a
        message about a missing file rather than about a missing expansion."""
        src = (REPO / "bench" / "sweep.py").read_text(encoding="utf-8")
        self.assertIn("systemdfile.expand", src)
        self.assertIn('v["binary"]', src,
                      "the binary is read straight from the file without "
                      "being expanded")

    def test_no_report_records_a_home_directory(self):
        """`bench/reports/` is skipped by the rule above, because a report is
        a dated record and rewriting one is what this repository argues
        against. That exemption hid a live leak: `bench/sweep.py` expanded
        `@HOME@` before writing `variant.json`, so every sweep recorded the
        operator's home directory into a file that gets published — fourteen
        of them by 27.08.

        The path is redundant (the build stamp identifies the binary and is
        recorded beside it), so folding it back changed no measurement. This
        checks the narrow thing the wider rule cannot: that the leak stays
        closed.
        """
        import json as _json, subprocess as _sp
        home = os.path.expanduser("~")
        bad = []
        files = _sp.run(["git", "ls-files", "bench/reports"], cwd=str(REPO),
                        capture_output=True, text=True).stdout.split()
        for f in files:
            p_ = REPO / f
            if not p_.is_file() or not f.endswith(".json"):
                continue
            text = p_.read_text(encoding="utf-8", errors="replace")
            if home in text or "/home/" in text:
                bad.append(f)
        self.assertFalse(bad, "these reports record a home directory — "
                              "bench/sweep.py must record the UNEXPANDED path: %s"
                              % ", ".join(bad))

    def test_sweep_records_unexpanded_and_runs_expanded(self):
        """Both halves, because getting only one is the bug: recording the
        expanded path leaks, and running the unexpanded one starts nothing."""
        src = (REPO / "bench" / "sweep.py").read_text(encoding="utf-8")
        self.assertIn('"binary": raw_binary', src,
                      "sweep.py records the expanded path again")
        self.assertIn('binary=systemdfile.expand(raw_binary)', src,
                      "sweep.py no longer expands the path it actually runs")

    def test_the_profiles_rule_still_holds(self):
        """The narrow rule this one grew out of, kept explicitly: a profile
        may name neither a home nor a model directory, comment or not."""
        for env in sorted((REPO / "setup" / "env").glob("*.env")):
            text = env.read_text(encoding="utf-8")
            with self.subTest(profile=env.name):
                self.assertNotIn("/home/", text, "%s hard-codes a home" % env.name)
                self.assertNotIn("/mnt/", text, "%s hard-codes a mount" % env.name)


if __name__ == "__main__":
    unittest.main()
