"""Does anything published point at something that is not published?

Five files are the maintainer's own and are excluded by `.gitignore`: a
2,000-line session log, a closed plan for a model that is not served, a
watcher for two upstream conditions, its wrapper, and the test that imports
it. They are the record of DEVELOPING this stack, not of running it.

Making them internal broke 61 references from files that stay. Most were code
comments — "see docs/HANDOVER.md, section 2" — pointing at the narrative
rather than at the evidence, and repointing them at the bench suite or the
profile that actually holds the measurement made them better comments. But 61
is more than anyone re-checks by hand, and the next file to become internal
will break more.

So the check is here rather than in a habit.

The rule is the same one tests/test_localenv.py uses for machine names, for
the same reason: **prose may name what is excluded — a document explaining
what is not in the repository has to be able to say what** — but a LINK or a
COMMAND may not, because somebody will click or paste it.
"""
import os, re, subprocess, unittest

import common

REPO = common.REPO

# Kept as a literal list rather than parsed out of .gitignore: the point is to
# state which files this rule is about, and a test that derives its own subject
# from the thing it is testing cannot fail when that thing is wrong.
INTERNAL = (
    "docs/HANDOVER.md",
    "docs/FLASHNEXT-PLAN.md",
    "setup/scripts/watchflashnext.py",
    "setup/scripts/watch-flashnext.sh",
    "tests/test_watchflashnext.py",
)

# Whole directories that are the maintainer's own. docs/px13/ holds the five
# German machine documents that docs/setup/ and the translated measurement
# records were made FROM; docs/archive/ holds withdrawn predecessors, some of
# them actively wrong.
INTERNAL_DIRS = ("docs/px13/", "docs/archive/")

# Directories whose contents are dated records and are never edited. A
# measurement from August may name a document that has since gone; correcting
# it would be editing evidence.
EVIDENCE = ("bench/reports/", "docs/px13/", "docs/archive/")

PATH = re.compile(r"(?<![\w/.-])((?:docs|setup|bench|tools|tests)/[\w./@-]*[\w])")
MD_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


# The files that DECLARE the exclusions have to name them: .gitignore does the
# excluding, DOCUMENTS.md tells the reader what is missing, and this file is
# the rule itself.
#
# This test was not on that list at first and passed anyway — because it was
# not committed yet, so `git ls-files` did not see it. Running the suite
# against a SIMULATED CLONE, where the file list comes from the tree, is what
# surfaced it; in the repository it would have gone red on the first commit.
DECLARES_THE_RULE = (".gitignore", "docs/DOCUMENTS.md", "tests/test_docs.py",
                     # names docs/px13/ in its own skip list, for the same
                     # reason: a rule has to be able to state its subject.
                     "tests/test_localenv.py")


def in_a_git_repo():
    """Two of the checks below ask git, and a released tarball has no .git.

    Failing there would be a red test for a reason that has nothing to do with
    the reader — found by running this suite against a simulated clone, which
    is the only way that case ever shows up.
    """
    r = subprocess.run(["git", "rev-parse", "--git-dir"], cwd=str(REPO),
                       capture_output=True, text=True)
    return r.returncode == 0


def tracked():
    """The published files. From git where there is git, from the tree
    otherwise — the tree is what a tarball reader actually has, and the link
    checks below are just as valid against it."""
    if in_a_git_repo():
        out = subprocess.run(["git", "ls-files"], cwd=str(REPO),
                             capture_output=True, text=True)
        names = out.stdout.split()
    else:
        names = []
        for root, dirs, files in os.walk(str(REPO)):
            dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
            for f in files:
                names.append(os.path.relpath(os.path.join(root, f), str(REPO)))
    return [f for f in names
            if not f.startswith(EVIDENCE) and f not in DECLARES_THE_RULE]


class TestNothingPublishedIsGitignored(unittest.TestCase):
    def setUp(self):
        if not in_a_git_repo():
            self.skipTest("no git repository here — this is what a released "
                          "tarball looks like, and the two checks in this "
                          "class are about the repository, not about the tree")

    def test_the_internal_files_are_not_tracked(self):
        """`git rm --cached` and a .gitignore line are two different acts, and
        only doing the second leaves the file in the repository forever."""
        t = set(subprocess.run(["git", "ls-files"], cwd=str(REPO),
                               capture_output=True, text=True).stdout.split())
        for f in INTERNAL:
            self.assertNotIn(f, t, "%s is still tracked — .gitignore does not "
                                   "remove a file that is already in the index" % f)

    def test_each_one_is_actually_ignored(self):
        r = subprocess.run(["git", "check-ignore", "--stdin"], cwd=str(REPO),
                           input="\n".join(INTERNAL), capture_output=True, text=True)
        ignored = set(r.stdout.split())
        for f in INTERNAL:
            self.assertIn(f, ignored, "%s is untracked but NOT ignored — the "
                                      "next `git add -A` puts it back" % f)


class TestNoPublishedLinkOrCommandLeavesTheRepository(unittest.TestCase):
    """A link into a file nobody has is worse than no link: it tells the reader
    the answer exists and that they cannot see it."""

    def code_lines(self, path):
        """Lines a reader would click or paste. Same criterion as
        tests/test_localenv.py: in Markdown that is a link or a code block; in
        a script it is anything that is not a comment."""
        with open(os.path.join(str(REPO), path), encoding="utf-8",
                  errors="replace") as fh:
            text = fh.read()
        markdown = path.endswith(".md")
        fenced = False
        for n, line in enumerate(text.splitlines(), 1):
            if not markdown:
                if line.lstrip().startswith("#"):
                    continue
                yield n, line, "code"
                continue
            if line.lstrip().startswith("```"):
                fenced = not fenced
                continue
            if fenced or line.startswith(("    ", "\t")):
                yield n, line, "block"
            for m in MD_LINK.finditer(line):
                yield n, m.group(1), "link"

    def test_no_link_or_command_names_an_internal_file(self):
        bad = []
        for f in tracked():
            if not os.path.isfile(os.path.join(str(REPO), f)):
                continue
            for n, line, kind in self.code_lines(f):
                for internal in INTERNAL:
                    if internal in line or os.path.basename(internal) in line:
                        bad.append("%s:%d (%s)  %s" % (f, n, kind, line.strip()[:70]))
                for d in INTERNAL_DIRS:
                    if d in line:
                        bad.append("%s:%d (%s)  %s" % (f, n, kind, line.strip()[:70]))
        self.assertFalse(bad, "these point at a file that is not published:\n    "
                              + "\n    ".join(bad))

    def test_no_markdown_link_points_at_a_path_that_does_not_exist(self):
        """The other half, and it found rot that predated this change:
        `bench/quality.py`, `docs/MODELLWAHL.md` and `bench/suites/np2-restore.py`
        were still cited after being deleted."""
        bad = []
        for f in tracked():
            full = os.path.join(str(REPO), f)
            if not os.path.isfile(full) or not f.endswith(".md"):
                continue
            base = os.path.dirname(f)
            for n, target, kind in self.code_lines(f):
                if kind != "link" or "://" in target or target.startswith("#"):
                    continue
                rel = os.path.normpath(os.path.join(base, target.split("#")[0]))
                if not os.path.exists(os.path.join(str(REPO), rel)):
                    bad.append("%s:%d  ->  %s" % (f, n, target))
        self.assertFalse(bad, "dead links:\n    " + "\n    ".join(bad))



# Programs a documented line may start with. Anything else in a code block is
# output, a config fragment, or prose — not something to check.
COMMAND_STARTS = ("bash ", "sh ", "python3 ", "python ")


class TestEveryDocumentedCommandCouldBeRun(unittest.TestCase):
    """A command in a document is an instruction, and instructions rot.

    Four of the ten defects found on 27.08.2026 were documented commands that
    nobody had executed: a simulated-clone `rsync` whose exclude list excluded
    nothing, a `switch-model.sh gemma31` that aborts in preflight because the
    profile serves on a port the gateway does not ask, a `--observe` reading
    that is four times too high taken on its own, and a `# shellcheck source=`
    directive voided by the prose after it.

    RUNNING them is not something a test can do — most need root, a GPU, or
    the network. Two things it can do, and both would have caught a rename or
    a dropped flag before a reader did:

      * the script a command invokes has to EXIST
      * every long option has to appear in that script

    The second is a literal search rather than a parse. argparse spells its
    flags as strings and a bash `case` spells them as patterns, so both are
    findable, and a flag that is only forwarded to another program would be a
    false alarm — there is none today, and an exception would belong here by
    name if one ever appeared.

    Placeholders are skipped: `<model>`, `$ENDPOINT`, `@MODELS@` are the
    document telling the reader to substitute, not naming a file.
    """

    def documented_commands(self):
        """(file, line number, command) for every command in a code block."""
        out = []
        for path in tracked():
            if not path.endswith(".md"):
                continue
            with open(os.path.join(str(REPO), path), encoding="utf-8",
                      errors="replace") as fh:
                text = fh.read()
            fenced = False
            for n, line in enumerate(text.splitlines(), 1):
                if line.lstrip().startswith("```"):
                    fenced = not fenced
                    continue
                if not (fenced or line.startswith(("    ", "\t"))):
                    continue
                # A documented line often carries its explanation in the same
                # row, separated by alignment. The command is the first field.
                cmd = re.split(r"\s{2,}", line.strip())[0].strip()
                cmd = cmd.split(" #")[0].strip()
                if cmd.startswith(COMMAND_STARTS):
                    out.append((path, n, cmd))
        return out

    @staticmethod
    def target(cmd):
        """(script path, arguments) or (None, []) if it is not a repo script."""
        tok = cmd.split()
        if len(tok) > 1 and not tok[1].startswith("-"):
            path, args = tok[1], tok[2:]
        else:
            return None, []
        if any(c in path for c in "<>$@*"):
            return None, []
        if not path.startswith(("setup/", "bench/", "tests/", "tools/", "docs/")):
            return None, []
        return path, args

    def test_the_scan_finds_commands_to_check(self):
        """Without this the two tests below pass by parsing nothing, which is
        the shape of defect this class exists for."""
        found = self.documented_commands()
        self.assertGreater(len(found), 40,
                           "only %d commands found in the documents — the "
                           "block scanner is not reading them" % len(found))

    def test_every_documented_command_names_a_script_that_exists(self):
        missing = []
        for path, n, cmd in self.documented_commands():
            script, _ = self.target(cmd)
            if script and not os.path.isfile(os.path.join(str(REPO), script)):
                missing.append("    %s:%d  %s" % (path, n, cmd))
        self.assertFalse(missing, "these documented commands invoke a script "
                                  "that is not in the repository:\n%s"
                                  % "\n".join(missing))

    def test_every_documented_option_exists_in_the_script(self):
        bad = []
        for path, n, cmd in self.documented_commands():
            script, args = self.target(cmd)
            if not script:
                continue
            full = os.path.join(str(REPO), script)
            if not os.path.isfile(full):
                continue                      # the test above owns that case
            with open(full, encoding="utf-8", errors="replace") as fh:
                body = fh.read()
            for arg in args:
                if not arg.startswith("--") or any(c in arg for c in "<>$@"):
                    continue
                flag = arg.split("=")[0]
                if flag not in body:
                    bad.append("    %s:%d  %s  does not accept %s"
                               % (path, n, cmd, flag))
        self.assertFalse(bad, "documented options that the script does not "
                              "mention anywhere:\n%s" % "\n".join(bad))


def real_test_count():
    """How many tests this suite actually has — the WHOLE suite.

    A FRESH loader, never `unittest.defaultTestLoader`. `-m unittest -k
    PATTERN` sets `testNamePatterns` on that shared loader, so counting
    through it returns whatever the caller happened to filter for; a loader
    made here carries no pattern and counts all of them. Measured 04.09.2026:
    1277 either way with a fresh one, 1277 then 11 with the shared one.
    """
    import unittest as ut
    loader = ut.TestLoader()
    return loader.discover(str(REPO / "tests"),
                           top_level_dir=str(REPO / "tests")).countTestCases()


class TestTheAdvertisedTestCountIsTrue(unittest.TestCase):
    """README.md tells the reader how many tests there are, and that number
    drifted three times in a single day — 298, then 429, while the suite was
    at 510. A number in prose that nobody checks is a number that is wrong.

    The tolerance is wide on purpose: this is a claim about the order of
    magnitude ("a few hundred, seconds to run"), not a count that should send
    somebody to edit a document every time a test is added.
    """

    TOLERANCE = 0.15

    def test_the_readme_number_is_close_to_the_real_one(self):
        import re, unittest as ut
        text = (REPO / "README.md").read_text(encoding="utf-8")
        m = re.search(r"\((\d+) tests", text)
        self.assertIsNotNone(m, "README.md no longer says how many tests there are")
        claimed = int(m.group(1))
        real = real_test_count()
        self.assertLess(abs(claimed - real) / real, self.TOLERANCE,
                        "README.md says %d tests, there are %d. Either update it "
                        "or widen this tolerance deliberately." % (claimed, real))

    def test_the_count_survives_an_active_k_filter(self):
        """`bash tests/run.sh -k readme` was RED, and the red was about the
        FILTER rather than about anything it selected.

        `-m unittest -k PATTERN` sets `testNamePatterns` on the SHARED
        `defaultTestLoader`, so a `discover()` through that loader counts only
        what the filter selected — 11 under `-k test_docs`, 1 under
        `-k readme` — and this claim then compared README's number against a
        fraction of the suite. Under a pattern matching nothing it was worse
        than a failure: `real` is 0 and the division raises.

        The unfiltered gate never saw it, which is why it stood. It matters
        because `tests/run.sh`'s own header advertises `-k`, and a gate that
        goes red at its own filtering teaches the reader to distrust the red —
        the reader most likely to type `-k test_docs` being whoever is editing
        this file.

        Fixed with a FRESH loader rather than by skipping the assertion under
        a filter: skipping would make the check stop checking in exactly the
        situation where somebody is working on it, which is this repository's
        own definition of a check that is not one.
        """
        import unittest as ut
        prev = ut.defaultTestLoader.testNamePatterns
        ut.defaultTestLoader.testNamePatterns = ["*matches_no_test_at_all*"]
        try:
            self.assertGreater(
                real_test_count(), 100,
                "the count is narrowed by whatever -k the caller passed")
        finally:
            ut.defaultTestLoader.testNamePatterns = prev


class TestProseMayStillSayWhatIsMissing(unittest.TestCase):
    def test_documents_md_names_the_internal_files(self):
        """The rule allows it, and this is the document that needs it — a map
        of the documentation that silently omitted three files would leave the
        reader wondering whether they had a broken checkout."""
        text = (REPO / "docs" / "DOCUMENTS.md").read_text(encoding="utf-8")
        self.assertIn("HANDOVER.md", text)
        self.assertIn("FLASHNEXT-PLAN.md", text)
        self.assertIn("gitignore", text.lower())


class TestNoTestFileHidesItsOwnTail(unittest.TestCase):
    """`if __name__ == "__main__": unittest.main()` must be the LAST thing in
    a test file.

    Found 03.09.2026 while adding tests to test_dialects.py: the block sat at
    line 265 of 484, so `python3 tests/test_dialects.py` ran 25 of 51 tests
    and printed OK. unittest.main() executes at that point — every class
    defined further down does not exist yet and is never collected.

    Six more files were in the same state, hiding 42 classes between them.
    Nothing was actually untested: tests/run.sh uses `unittest discover`,
    which IMPORTS the module, so __name__ is not "__main__" and the whole file
    is collected. That is exactly what made it survive — the gate stayed
    honest while the obvious way to run one file by hand did not, and a green
    OK that has silently skipped half the file is the worst shape a test
    result can take.
    """

    def files(self):
        import glob
        return sorted(glob.glob(str(common.REPO / "tests" / "test_*.py")))

    def test_there_are_test_files_to_check(self):
        self.assertGreater(len(self.files()), 10)

    def test_nothing_is_defined_after_the_entry_point(self):
        import re
        bad = []
        for path in self.files():
            lines = open(path, encoding="utf-8").read().split("\n")
            at = next((i for i, l in enumerate(lines)
                       if l.startswith("if __name__")), None)
            if at is None:
                continue
            after = [l for l in lines[at + 1:]
                     if re.match(r"^(class|def) ", l)]
            if after:
                bad.append("%s: %d definition(s) after line %d"
                           % (os.path.basename(path), len(after), at + 1))
        self.assertEqual(bad, [], "these run under `tests/run.sh` but are "
                                  "silently skipped when the file is executed "
                                  "directly:\n  " + "\n  ".join(bad))


if __name__ == "__main__":
    unittest.main()
