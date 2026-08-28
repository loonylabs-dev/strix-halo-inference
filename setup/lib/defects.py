#!/usr/bin/env python3
"""defects — the known defects of this hardware and build, evaluated.

    python3 setup/lib/defects.py              report against what is running
    python3 setup/lib/defects.py --json       the same, machine-readable
    python3 setup/lib/defects.py --list       everything, without evaluating

Why this exists as CODE and not as another section in a document: the same
defect was written down in six places — HANDOVER, setup/patches/README.md,
three .env files and a commit message — and they had already drifted. A
profile said "PR #27739 is open" hours after it was closed unmerged. Prose
cannot be asked whether the machine in front of you is affected.

What it does NOT do: it does not measure. A defect whose only honest answer
is a measurement says so and names the suite. The registry's job is to know
what to look for, not to pretend it can see everything from a command line.

The ordering is by `shows_as`, worst first, and that ordering is the point.
On this hardware the dangerous defects do not raise — they degrade the
output, and nothing in a normal stack notices. Listing crashes first would
put the harmless half at the top.
"""
import argparse, json, os, re, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from systemdfile import flag                    # noqa: E402  the one flag reader

REGISTRY = os.path.join(HERE, "..", "defects.json")
RAW = "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/%s"
DEFAULT_BUILD = os.path.expanduser("~/llama.cpp/build-rocm-patched")

# Worst first. See the module docstring.
SEVERITY = ["silent", "loud", "unrepeatable", "slow"]

EXPOSED, GUARDED, MANUAL, UNKNOWN, NA = "EXPOSED", "guarded", "manual", "unknown", "n/a"
WITHDRAWN = "withdrawn"


# --- reading the world ----------------------------------------------------

def load(path=REGISTRY):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["defects"]


def running_cmdline(proc="/proc"):
    """The argv of the running llama-server, or None.

    The running process, not the profile: a server started by hand is still
    a server, and the whole point is to describe the machine in front of you.
    """
    try:
        pids = [d for d in os.listdir(proc) if d.isdigit()]
    except OSError:
        return None
    for pid in pids:
        try:
            with open(os.path.join(proc, pid, "cmdline"), "rb") as fh:
                argv = fh.read().split(b"\0")
        except OSError:
            continue
        argv = [a.decode("utf-8", "replace") for a in argv if a]
        if argv and os.path.basename(argv[0]) == "llama-server":
            return argv
    return None


def build_stamp(build_dir=DEFAULT_BUILD):
    """The .build-stamp as a dict, or None. Written by build-llama.sh."""
    path = os.path.join(build_dir, ".build-stamp")
    out = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if "=" in line:
                    k, _, v = line.partition("=")
                    out[k.strip()] = v.strip()
    except OSError:
        return None
    return out


# --- the evaluation, pure so it can be tested without a machine -----------

def applies(defect, cmdline, gpu=None):
    """Is this defect about the thing that is running, ON THIS HARDWARE?

    An architecture-specific defect must not be reported against a model it
    cannot touch — a registry that cries about qwen4exp while qwen38 serves
    trains the reader to skip it, which is how a real warning gets missed.

    The GPU half of that was missing until 27.08. Nine of the twelve entries
    carry `applies_to: {"gpu": "gfx1151"}` and nothing read the field, so on
    any other card the registry reported all nine — the exact behaviour the
    paragraph above calls a failure, applied to the hardware axis instead of
    the model axis.

    `gpu` is passed in rather than read, so the decision stays pure. None
    means nobody looked, which is not the same as a mismatch: an unknown GPU
    gets the defect reported, because not knowing what you are on is a reason
    for MORE caution, not less.
    """
    want = (defect.get("applies_to") or {}).get("gpu")
    if want and gpu is not None and gpu != want:
        return False
    pat = defect.get("when_cmdline_matches")
    if not pat:
        return True
    if cmdline is None:
        return None                              # cannot tell
    return bool(re.search(pat, " ".join(cmdline)))


def evaluate(defect, cmdline=None, stamp=None, gpu=None):
    """(verdict, detail) for one defect against one machine state."""
    # A withdrawn entry is not an open question, and `manual` would keep
    # sending the reader off to run a measurement that has already been run
    # and has already answered. That is the crying-wolf failure this module's
    # own docstring names, and slot-restore-hangs-busy walked straight into
    # it on 27.08.: withdrawn as a defect, still printed by check.sh as
    # "only a measurement answers this".
    #
    # It stays in the registry rather than being deleted — it was the stated
    # reason -np 2 remained closed through three sessions, and a correction
    # that is removed takes the record of the mistake with it.
    if str(defect.get("status", "")).startswith("withdrawn"):
        return WITHDRAWN, "withdrawn — not a defect; see `measured`"
    want = (defect.get("applies_to") or {}).get("gpu")
    scope = applies(defect, cmdline, gpu)
    if scope is False:
        if want and gpu is not None and gpu != want:
            return NA, "about %s; this machine is %s" % (want, gpu)
        return NA, "not the model that is running"
    if scope is None:
        return UNKNOWN, "no llama-server running — cannot tell if it applies"

    d = defect.get("detect") or {}
    kind = d.get("kind")

    if kind == "manual":
        suite = defect.get("suite")
        return MANUAL, ("only a measurement answers this: %s" % suite) if suite \
            else "only a measurement answers this"

    if kind == "cmdline":
        if cmdline is None:
            return UNKNOWN, "no llama-server running"
        names = [n for n in (d.get("flag"), d.get("alias")) if n]
        if d.get("expect_absent"):
            present = any(n in cmdline for n in names)
            return (EXPOSED, "%s is present" % names[0]) if present \
                else (GUARDED, "%s absent" % names[0])
        got = flag(cmdline, *names, default=d.get("default"))
        want = d.get("expect")
        if got == want:
            return GUARDED, "%s %s" % (names[0], got)
        return EXPOSED, "%s is %s, wanted %s%s" % (
            names[0], got,
            want, " (nothing passed, and the default is %s)" % d["default"]
            if d.get("default") is not None and not any(n in cmdline for n in names) else "")

    if kind == "build-flag":
        if stamp is None:
            return UNKNOWN, "no .build-stamp — was this built by build-llama.sh?"
        forbidden = d.get("forbidden", "")
        if forbidden in stamp.get("cmake", ""):
            return EXPOSED, "%s is in the build" % forbidden
        return GUARDED, "%s not in the build" % forbidden

    if kind == "build-stamp":
        if stamp is None:
            return UNKNOWN, "no .build-stamp — was this built by build-llama.sh?"
        key = d.get("key", "")
        val = stamp.get(key, "")
        if d.get("must_be_set") and val:
            return GUARDED, "%s=%s" % (key, val[:12])
        return EXPOSED, "%s is not set in the build stamp" % key

    return UNKNOWN, "unknown detect kind %r" % kind


def report(defects, cmdline=None, stamp=None, gpu=None):
    """[(defect, verdict, detail)] sorted worst-shows-as first, exposed first."""
    rows = [(d,) + evaluate(d, cmdline, stamp, gpu) for d in defects]

    def key(row):
        d, verdict, _ = row
        sev = SEVERITY.index(d.get("shows_as")) if d.get("shows_as") in SEVERITY else len(SEVERITY)
        rank = {EXPOSED: 0, UNKNOWN: 1, MANUAL: 2, GUARDED: 3, NA: 4,
                WITHDRAWN: 5}.get(verdict, 6)
        return (rank, sev, d.get("id", ""))
    return sorted(rows, key=key)


# --- the retirement condition ---------------------------------------------

def check_upstream(defects, fetch=None):
    """[(defect, state, detail)] for every defect that names a retirement probe.

    `state` is "keep" while the mitigation is still needed, "RETIRE?" when the
    condition for dropping it is met, "unknown" when master could not be read.

    WHAT THE PATTERN IS matters, and `present_means` says which:

        "keep"    (default) the pattern names the CAUSE. Present -> keep.
        "retire"  the pattern names the FIX. Present -> RETIRE?

    The second exists because of gfx1151-hip-integrated, 28.08.2026. That
    probe watched the line `info.devices[id].integrated = prop.integrated`,
    on the reasonable assumption that the fix would delete it. llama.cpp PR
    #27311 does not: it leaves the line and makes the buffer it leads to safe,
    in a different file. Measured — a build containing that PR has the line at
    ggml-cuda.cu:306 and is 0 of 10 corrupt where master is 10 of 10.

    So the probe could only ever say "keep". A check that cannot reach its
    other answer is the shape this repository keeps finding, here in the one
    thing whose job is to tell you when you may stop carrying a patch.

    It reads the SOURCE, deliberately, and not an issue's state. This project
    learned that the expensive way on 26.08.: a watch that followed pull-request
    TITLES sat on #27739 for hours after it had been closed unmerged in favour
    of another PR, and would have reported "still nothing" straight through the
    merge. A closed issue is a story about a fix. The source is the fix.

    Retiring a mitigation is never automatic — the probe going green means
    "re-measure now", not "drop it". Cause and trigger are different things,
    and on 26.08. this very defect stopped reproducing while its cause stood
    unchanged in master.
    """
    if fetch is None:
        def fetch(path):
            req = urllib.request.Request(RAW % path,
                                         headers={"User-Agent": "defects.py"})
            with urllib.request.urlopen(req, timeout=25) as fh:
                return fh.read().decode("utf-8", "replace")
    out, cache = [], {}
    for d in defects:
        probe = d.get("upstream_check")
        if not probe:
            continue
        path = probe.get("path", "")
        if path not in cache:
            try:
                cache[path] = fetch(path)
            except Exception as e:
                cache[path] = e
        text = cache[path]
        if isinstance(text, Exception):
            out.append((d, UNKNOWN, "could not read %s: %s" % (path, text)))
        else:
            present = bool(re.search(probe.get("pattern", ""), text))
            retire_on_present = probe.get("present_means", "keep") == "retire"
            if present:
                state = "RETIRE?" if retire_on_present else "keep"
                detail = probe.get("while_present", "still present")
            else:
                state = "keep" if retire_on_present else "RETIRE?"
                detail = probe.get("when_gone", "gone from master")
            out.append((d, state, detail))
    return out


# --- output ---------------------------------------------------------------

def _c(code, text):
    return text if not sys.stdout.isatty() else "\033[%sm%s\033[0m" % (code, text)


MARK = {EXPOSED: ("31", "!"), GUARDED: ("32", "="), MANUAL: ("33", "?"),
        UNKNOWN: ("33", "?"), NA: ("90", "-"), WITHDRAWN: ("90", "~")}


def print_report(rows, verbose=False):
    exposed = 0
    for d, verdict, detail in rows:
        colour, sign = MARK.get(verdict, ("0", " "))
        if verdict == EXPOSED:
            exposed += 1
        print("  %s %-28s %-11s %s" % (_c(colour, sign), d["id"],
                                       d.get("shows_as", "?"), detail))
        if verdict == EXPOSED or verbose:
            print("      %s" % d["title"])
            print("      shows as: %s" % d.get("symptom", "")[:150])
            print("      do      : %s" % d.get("mitigation", "")[:150])
            for u in d.get("upstream", [])[:2]:
                print("      %s" % u)
    return exposed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true", help="machine-readable")
    ap.add_argument("--list", action="store_true", help="do not evaluate")
    ap.add_argument("--verbose", action="store_true", help="detail for every row")
    ap.add_argument("--upstream", action="store_true",
                    help="ask master whether a mitigation can be retired")
    ap.add_argument("--build", default=DEFAULT_BUILD)
    a = ap.parse_args(argv)

    defects = load()
    if a.upstream:
        rows = check_upstream(defects)
        if not rows:
            print("  no defect names a retirement condition")
            return 0
        for d, state, detail in rows:
            colour, sign = {"keep": ("32", "="), "RETIRE?": ("33", "***")}.get(
                state, ("33", "?"))
            print("  %s %-28s %s" % (_c(colour, sign), d["id"], state))
            print("      %s" % detail)
        return 0
    if a.list:
        for d in sorted(defects, key=lambda x: SEVERITY.index(x["shows_as"])
                        if x.get("shows_as") in SEVERITY else 9):
            print("%-28s %-12s %-9s %s" % (d["id"], d.get("shows_as"),
                                           d.get("status"), d["title"]))
        return 0

    cmdline = running_cmdline()
    stamp = build_stamp(a.build)
    # Which GPU this is. Read here rather than inside evaluate(), so the
    # decision stays pure and a test can hand it any machine.
    try:
        import hardware
        this_gpu = hardware.gpu()["gfx"]
    except Exception:
        this_gpu = None
    rows = report(defects, cmdline, stamp, this_gpu)

    if a.json:
        print(json.dumps([{"id": d["id"], "shows_as": d.get("shows_as"),
                           "verdict": v, "detail": t} for d, v, t in rows],
                         indent=2))
        return 1 if any(v == EXPOSED for _, v, _ in rows) else 0

    if cmdline is None:
        print("  no llama-server running — argument checks are skipped, not passed")
    exposed = print_report(rows, a.verbose)
    print()
    if exposed:
        print("  %d defect(s) EXPOSED. Silent ones do not announce themselves;" % exposed)
        print("  that is why they are listed first.")
    else:
        print("  Every known defect is guarded, withdrawn, unmeasurable from here,")
        print("  or not applicable.")
    return 1 if exposed else 0


if __name__ == "__main__":
    sys.exit(main())
