#!/usr/bin/env python3
"""Editing a kernel command line without losing anything on it.

    from kernelcmdline import set_params, remove_params, get_param
    python3 setup/lib/kernelcmdline.py set    <file> k=v [k=v …]
    python3 setup/lib/kernelcmdline.py remove <file> k [k …]
    python3 setup/lib/kernelcmdline.py get    <file> k

Why this is a module with tests and not a `sed`
-----------------------------------------------
The strings this edits decide whether the machine boots. `root=UUID=…` is on
them. A regex that is one character off does not produce a wrong GTT size, it
produces a system that stops at the initramfs prompt — and the way back is a
rescue USB stick, not an undo.

So the rule this file enforces is deliberately paranoid: **every token that
was on the line before is still on it afterwards, except the ones that were
explicitly named.** `set_params` verifies that on its own result and raises if
it does not hold. Nothing here trusts its own regex.

What it deliberately does NOT do
--------------------------------
Quoted values (`key="a b"`). The kernel supports them, this machine has none,
and a half-correct quote parser silently mangling `rd.luks.options=` is worse
than a refusal. If a quote turns up, it raises.

Three places carry the same command line on Fedora and all three have to agree,
or the setting survives exactly until the next kernel:

    /boot/loader/entries/*.conf   what BOOTS      — edited via `grubby`, which
                                                    does its own replace-or-add
    /etc/kernel/cmdline           what a NEW kernel inherits (kernel-install
                                  prefers this file over /proc/cmdline)
    /etc/default/grub             what grub2-mkconfig would regenerate from

grubby handles the first. This module handles the other two.
"""
import re
import sys

__all__ = ["parse", "get_param", "set_params", "remove_params",
           "grub_default_set", "grub_default_remove",
           "pages_for_gib", "gib_for_pages", "GRUB_LINE"]

GRUB_LINE = "GRUB_CMDLINE_LINUX"
PAGES_PER_GIB = 262144                      # a page is 4 KiB


def pages_for_gib(gib):
    """GiB -> ttm.pages_limit. The formula from the runbook, § 7."""
    if float(gib) != int(gib):
        raise ValueError("whole GiB only, got %r" % gib)
    return int(gib) * PAGES_PER_GIB


def gib_for_pages(pages):
    return int(pages) / PAGES_PER_GIB


def parse(cmdline):
    """The command line as a list of tokens, order and duplicates preserved.

    A token is a plain string; `key=value` is not split, because a value may
    itself contain '=' (`root=UUID=1f0a54c2-…`).
    """
    if '"' in cmdline or "'" in cmdline:
        raise ValueError("quoted values on the command line — refusing to "
                         "edit it rather than risk mangling one:\n  %s" % cmdline)
    return cmdline.split()


def _key(token):
    return token.split("=", 1)[0]


def get_param(cmdline, name):
    """The value of `name`, or None. The LAST one wins, as in the kernel."""
    found = None
    for token in parse(cmdline):
        if _key(token) == name:
            found = token.split("=", 1)[1] if "=" in token else ""
    return found


def set_params(cmdline, params):
    """Return `cmdline` with `params` set. Everything else stays byte-identical.

    An existing occurrence is replaced WHERE IT STANDS, so the order of the
    line does not shuffle between edits and a diff stays readable. Several
    occurrences of the same key collapse into the first one — the kernel would
    take the last, and leaving both is a trap for whoever reads the file next.
    A key that is not there yet is appended.
    """
    tokens = parse(cmdline)
    want = dict(params)
    out, done = [], set()
    for token in tokens:
        k = _key(token)
        if k in want:
            if k in done:
                continue                    # collapse a duplicate
            out.append("%s=%s" % (k, want[k]))
            done.add(k)
        else:
            out.append(token)
    for k, v in want.items():
        if k not in done:
            out.append("%s=%s" % (k, v))
    result = " ".join(out)
    _assert_nothing_lost(tokens, result, want)
    return result


def remove_params(cmdline, names):
    """Return `cmdline` without the named parameters."""
    names = set(names)
    tokens = parse(cmdline)
    out = [t for t in tokens if _key(t) not in names]
    result = " ".join(out)
    _assert_nothing_lost(tokens, result, {n: None for n in names})
    return result


def _assert_nothing_lost(before, after, touched):
    """The safety net. Every token that was there is still there, unless its
    key was one we were asked to change.

    This is not belt and braces for its own sake: the failure mode of getting
    it wrong is an unbootable machine, and the check costs nothing.
    """
    after_tokens = parse(after)
    after_keys = {_key(t) for t in after_tokens}
    for token in before:
        if _key(token) in touched:
            continue
        if token not in after_tokens:
            raise AssertionError(
                "editing the kernel command line would have dropped %r — "
                "refusing.\n  before: %s\n  after:  %s"
                % (token, " ".join(before), after))
    for k in touched:
        if touched[k] is None:
            assert k not in after_keys, "%s should have been removed" % k
        else:
            assert get_param(after, k) == str(touched[k]), \
                "%s did not take the new value" % k


# --- /etc/default/grub, which wraps the same line in a shell assignment ----

def grub_default_set(text, params, line=GRUB_LINE):
    """Rewrite GRUB_CMDLINE_LINUX="…" inside /etc/default/grub.

    Only that one assignment is touched; every other line of the file is
    returned unchanged, including comments.
    """
    pattern = re.compile(r'^(%s=)"([^"]*)"\s*$' % re.escape(line), re.M)
    m = pattern.search(text)
    if not m:
        raise ValueError('no %s="…" line in the file' % line)
    new = set_params(m.group(2), params)
    return pattern.sub(lambda _: '%s"%s"' % (m.group(1), new), text, count=1)


def grub_default_remove(text, names, line=GRUB_LINE):
    """The counterpart of grub_default_set: drop the named parameters."""
    pattern = re.compile(r'^(%s=)"([^"]*)"\s*$' % re.escape(line), re.M)
    m = pattern.search(text)
    if not m:
        raise ValueError('no %s="…" line in the file' % line)
    new = remove_params(m.group(2), names)
    return pattern.sub(lambda _: '%s"%s"' % (m.group(1), new), text, count=1)


def main(argv):
    if len(argv) < 3:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        print("usage: kernelcmdline.py {set|remove|get} <file> …", file=sys.stderr)
        return 2
    action, path = argv[1], argv[2]
    text = open(path, encoding="utf-8").read().strip("\n")
    if action == "get":
        v = get_param(text, argv[3])
        print(v if v is not None else "")
    elif action == "set":
        params = dict(a.split("=", 1) for a in argv[3:])
        print(set_params(text, params))
    elif action == "remove":
        print(remove_params(text, argv[3:]))
    elif action == "grub":
        params = dict(a.split("=", 1) for a in argv[3:])
        sys.stdout.write(grub_default_set(open(path, encoding="utf-8").read(), params))
    elif action == "grub-remove":
        sys.stdout.write(grub_default_remove(open(path, encoding="utf-8").read(), argv[3:]))
    else:
        print("unknown action: %s" % action, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
