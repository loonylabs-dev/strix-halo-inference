#!/usr/bin/env python3
"""The system unit, DERIVED from the user unit. There is no second file.

    python3 setup/lib/systemunit.py                  print it
    python3 setup/lib/systemunit.py --check          exit 1 if the installed
                                                     copy is not what this
                                                     would produce
    bash setup/install.sh --system-unit              generate and install it

Why this exists as a derivation and not as a file
-------------------------------------------------
There WAS a second file, `setup/systemd/llama@.service`, hand-maintained
beside `llama-user@.service`. It had never been started on this machine —
SELinux refuses it, see setup/README.md — and by 27.08. it had rotted in three
independent ways that nobody could have noticed, because nothing ran it:

    ExecStart      hard-wired to build-vulkan/bin/llama-server, so it ignored
                   LLAMA_BIN and would have served the WRONG BACKEND and, worse,
                   the build WITHOUT setup/patches/hip-integrated-off.patch —
                   which is the patch that stops the gfx1151 corruption. A
                   stranger enabling it would have got '////' and blamed the model.
    MemoryHigh     48G against the user unit's 96G
    MemoryMax      64G against the user unit's 108G
    TimeoutStartSec  absent, so systemd's 45 s default — while its own
                   ExecStartPre=llm-wait-for-model waits up to 120. On a slow
                   mount systemd would have killed it while the wait was doing
                   its job, three times, and StartLimitBurst=3 leaves it down
                   for good. Found by diffing this derivation against the file
                   it replaced.

That is the failure this repo keeps writing about: nothing breaks, an effect
simply fails to appear. And it is the same answer as everywhere else here —
one source of truth, everything else derived. `lib/models.sh` for which models
exist, `lib/systemdfile.py` for systemd syntax, `lib/budget.py` for the memory
budget, and now this for the second unit.

What the system unit is FOR
---------------------------
A Strix Halo box in a cupboard. Framework Desktop, GMKtec EVO-X2, Beelink
GTR9 — this hardware is sold as a headless inference server, and there a
system service is the right shape. `sudo loginctl enable-linger` covers the
same ground with the user unit and is what this machine uses; a system unit is
for a host with no user account to linger on.

Honest limitation: it is generated, installable and unit-tested against the
unit that actually runs — and it has still never been STARTED, because Fedora
refuses it. Proving it matches the tested unit is a different claim from
proving it works, and only the first one is made here.
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from systemdfile import directive                              # noqa: E402

SOURCE = os.path.join(HERE, "..", "systemd", "llama-user@.service")

# The whole mapping, in one list, in the order it is applied. Every line of the
# system unit is either this list applied to the user unit, or one of the
# ADDITIONS below. There is nothing else — which is the point.
RULES = [
    # A system unit may not read from a home directory, so every path that
    # %h would have expanded moves to a system location. install.sh puts the
    # files there; setup/README.md's table says which.
    ("-%h/.config/llm-stack.env",     "-/etc/llm-stack.env"),
    ("%h/.claude/env/%i.env",         "/etc/llm-profile/%i.env"),
    ("%h/.claude/bin/waitformodel",   "/usr/local/bin/llm-wait-for-model"),
    ("%h/.claude/bin/checkroom",      "/usr/local/bin/llm-check-room"),
    ("%h/.claude/bin/llamaexec",      "/usr/local/bin/llm-exec"),
    # Instance names. A template's Conflicts= has to name every instance by
    # hand because systemd has no wildcard for them — so the list is derived
    # here too rather than written twice.
    ("llama-user@",                   "llama@"),
    ("(%i, user service)",            "(%i)"),
    # A system unit is wanted by the system, not by a login session.
    ("WantedBy=default.target",       "WantedBy=multi-user.target"),
]

# What the user unit cannot carry, because a user service already runs as the
# user. Inserted after Type=, which is where they read naturally.
ADDITIONS = ("User=%(user)s",
             "SupplementaryGroups=render video")

HEADER = """# GENERATED — do not edit. Derived from setup/systemd/llama-user@.service
# by setup/lib/systemunit.py; tests/test_systemunit.py fails if the two
# disagree.
#
#     regenerate:   bash setup/install.sh --system-unit
#     compare:      python3 setup/lib/systemunit.py --check
#
# THE REASONING IS NOT HERE. Every line below has a paragraph explaining it in
# the user unit, and that is the only copy — carrying the comments across was
# tried and produced prose that was wrong in this context: the instance-name
# rule rewrote `systemctl --user enable --now llama-user@X` into `llama@X`,
# which is advice that does not work. A generated file with stale reasoning is
# the rot this generator exists to end.
#
# Edit the USER unit. The predecessor of this file was hand-maintained, never
# started here (SELinux refuses it), and had silently drifted to the Vulkan
# binary — ignoring LLAMA_BIN, so it would have served the build WITHOUT the
# gfx1151 corruption patch — with memory ceilings half the user unit's.
#
# Generated for a host with no user session to linger on. On a desktop use the
# user unit and `sudo loginctl enable-linger $USER`.
"""


def render(user=None, source=None):
    """The system unit as text. PURE apart from the default user name."""
    user = user or os.environ.get("SUDO_USER") or os.environ.get("USER") or ""
    with open(source or SOURCE, encoding="utf-8") as fh:
        raw = fh.read()
    # Comments are DROPPED, not translated. See HEADER: the substitutions are
    # right for directives and wrong for prose, and prose that is subtly wrong
    # in a file marked "do not edit" is worse than no prose at all.
    kept = [l for l in raw.splitlines() if not l.lstrip().startswith("#")]
    text = "\n".join(kept)
    for old, new in RULES:
        text = text.replace(old, new)
    out, blank = [], False
    for line in text.splitlines():
        # Collapse the runs of blank lines the comments left behind.
        if not line.strip():
            if blank or not out:
                continue
            blank = True
        else:
            blank = False
        out.append(line)
        if line.strip() == "Type=simple":
            out.append("# The two directives a user service cannot carry, because it "
                       "already runs")
            out.append("# as the user. Everything else on this page comes from the "
                       "user unit.")
            for a in ADDITIONS:
                out.append(a % {"user": user})
    return HEADER + "\n".join(out) + "\n"


def leftovers(text):
    """Anything the mapping failed to translate.

    A `%h` that survives into a system unit resolves to /root — not to the
    home directory of User=. That is a silent wrong answer, so it is an error
    here rather than a warning.
    """
    bad = []
    for n, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        if "%h" in line:
            bad.append("%d: %%h survives and would resolve to /root — %s" % (n, line.strip()))
        if "llama-user@" in line:
            bad.append("%d: a user-unit instance name survives — %s" % (n, line.strip()))
    return bad


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    installed = "/etc/systemd/system/llama@.service"
    text = render()
    bad = leftovers(text)
    if bad:
        print("the derivation is incomplete:\n  " + "\n  ".join(bad), file=sys.stderr)
        return 2
    if "--check" in argv:
        if not os.path.exists(installed):
            print("%s is not installed" % installed, file=sys.stderr)
            return 1
        with open(installed, encoding="utf-8") as fh:
            have = fh.read()
        if have == text:
            print("%s matches the user unit" % installed)
            return 0
        print("%s differs from what the user unit would produce.\n"
              "  bash setup/install.sh --system-unit" % installed, file=sys.stderr)
        return 1
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
