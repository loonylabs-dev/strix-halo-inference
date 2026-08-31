"""install.sh — a module found by directory has to BE in the directory.

setup/install.sh states the rule in one line and the tree obeys it twice:
`dialects.py` sits beside `gateway.py` and `prewarm.py` because they
`import dialects`, and `systemdfile.py` travels with `budget.py` for the same
reason. The directory is ~/.local/lib/llm-stack since 09/2026; the history
below happened in its predecessor ~/.claude/bin and is kept as written.

`modes.py` was added on 28.08.2026 as a third such module and was NOT linked.
The gateway kept starting anyway, which is why nothing noticed: since Python
3.11 the interpreter puts the RESOLVED script directory on `sys.path[0]`, and
for a symlinked `~/.claude/bin/cc-gateway.py` that is the repo — so `import
modes` found the file there. Verified:

    python3 /tmp/probe-link.py        (a symlink into /tmp/reallib)
      sys.path[0]                = /tmp/reallib      <- resolved
      dirname(abspath(__file__)) = /tmp              <- not resolved

The file's own `sys.path.insert(0, dirname(abspath(__file__)))` therefore adds
`~/.claude/bin`, which is where `dialects` is found, while `modes` was coming
from the interpreter's resolved entry. Two sibling imports, two different
mechanisms, one of them unintended. It worked for a reason the code does not
state, and that is not the same as working: an installation that copies rather
than symlinks, an older interpreter, or a run through `runpy` all break it.

So this file asks the only question that matters — for every module the
installer places in `~/.local/lib/llm-stack`, is every sibling it imports
placed there too? Reading install.sh rather than the live installation,
because the test has to fail in CI on a machine where nothing is installed.
"""
import ast
import re
import unittest

import common

REPO = common.REPO
INSTALL = REPO / "setup" / "install.sh"

# Modules the interpreter always has. Anything else a linked file imports at
# top level has to be linked beside it.
STDLIB = set(getattr(__import__("sys"), "stdlib_module_names", ()))


def linked():
    """{module name: source path} for every *.py install.sh puts in bin/."""
    out = {}
    for src, dst in re.findall(r'link_\s+"([^"]+)"\s+"([^"]+)"',
                               INSTALL.read_text(encoding="utf-8")):
        # $LIB is install.sh's name for ~/.local/lib/llm-stack — the one
        # directory whose modules import each other by neighbourhood.
        if not dst.endswith(".py") or not dst.startswith("$LIB/"):
            continue
        # install.sh addresses the tree through two variables: $SRC is
        # setup/, $REPO is the checkout root.
        rel = src.replace("$SRC/", "setup/").replace("$REPO/", "")
        out[dst.rsplit("/", 1)[-1][:-3]] = REPO / rel
    return out


def top_level_imports(path):
    """The plain `import X` / `from X import …` names at module level.

    Imports inside functions are deliberately ignored: `prewarm.py` imports
    urllib lazily, and a deferred import is not a packaging constraint.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


class TestEveryLinkedModuleFindsItsSiblings(unittest.TestCase):
    def setUp(self):
        self.linked = linked()
        self.assertTrue(self.linked, "install.sh links no python modules?")

    def test_the_gateway_and_its_siblings_are_all_installed(self):
        """The concrete case: gateway.py imports dialects AND modes."""
        self.assertIn("gateway", self.linked)
        for sibling in ("dialects", "modes"):
            self.assertIn(sibling, self.linked,
                          "%s is imported by gateway.py and not linked "
                          "beside it" % sibling)

    def test_no_linked_module_imports_a_sibling_that_is_not_linked(self):
        """The general rule, so the next one is caught on the day it is added."""
        missing = []
        for name, path in sorted(self.linked.items()):
            if not path.exists():
                self.fail("install.sh links a file that is gone: %s" % path)
            for imp in sorted(top_level_imports(path)):
                if imp in STDLIB or imp in self.linked:
                    continue
                # Third-party packages are pip's problem, not the installer's.
                if (path.parent / (imp + ".py")).exists():
                    missing.append("%s imports %s" % (name, imp))
        self.assertEqual(missing, [],
                         "a module found by directory has to BE in the "
                         "directory — see setup/install.sh")

    def test_the_rule_is_written_down_where_the_links_are(self):
        src = INSTALL.read_text(encoding="utf-8")
        self.assertIn("has to BE in the directory", src)


if __name__ == "__main__":
    unittest.main()
