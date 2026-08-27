#!/usr/bin/env python3
"""Reading systemd files the way systemd reads them. The only implementation.

Covers both kinds this repo has: EnvironmentFiles (setup/env/*.env) and unit
files (setup/systemd/*.service). They share the one property that trips
everything up — a line ending in a backslash continues on the next one.

    from systemdfile import llama_args, variable, directive     # as a module
    python3 setup/lib/systemdfile.py args      <file>           # as a command
    python3 setup/lib/systemdfile.py value     <file> <NAME>
    python3 setup/lib/systemdfile.py directive <unit> <NAME>
    python3 setup/lib/systemdfile.py models                     # the model directory
    python3 setup/lib/systemdfile.py conventions                # where it looks
    python3 setup/lib/systemdfile.py local     <NAME> [default] # this machine's answer

Why this is its own file
------------------------
`.env` files here are systemd EnvironmentFile syntax, NOT bash. `. file.env`
fails silently, because bash reads `VAR=value command args`. So everything
that wants to know what a profile says has to parse it — and by 26.08. three
places did, each slightly differently:

    setup/check.sh          a Python loop
    bench/run.py            a global re.sub over the whole file
    bench/suites/slot-scaling.sh   a regex with a lookahead, plus shlex.split

The third one was measurably wrong: on a profile whose LLAMA_ARGS is followed
by comment lines and then LLAMA_BIN, its lookahead `(?=^[A-Za-z_]+=|\\Z)` ran
past the comments and appended their words — `# Backend for this profile.
Measured 24.08. …` — to the server's command line. The other two had been
wrong before in their own ways: shlex.split strips the double quotes out of
`--chat-template-kwargs {"enable_thinking":false}`, so the server got invalid
JSON from the bench harness while the identical line worked under systemd.

A measurement harness that reads the profile differently from the service is
not measuring the service. That is why this is one file.

The two rules, both taken from systemd's own behaviour:

  * a line ending in `\\` continues on the next line;
  * the value is split on WHITESPACE and quotes are passed through as DATA —
    no shell quoting, no shlex.
"""
import glob, os, sys

__all__ = ["llama_args", "variable", "directive", "flag", "expand", "LLAMA_ARGS",
           "local_env_path", "local_var", "models_dir", "MODELS_CONVENTIONS"]

LLAMA_ARGS = "LLAMA_ARGS"

# --- what belongs to the MACHINE rather than to the repo ---------------------
#
# Until 27.08. the answer to "where do the models live" was the string
# `/mnt/shared/LLM`, written into eighteen files as a default. That directory
# exists on exactly one computer. The same was true of the repo's own path and
# of a gateway hostname that `smoketest.sh` used as its default — so a stranger
# who cloned this and ran the smoke test fired requests at somebody else's
# tunnel.
#
# A machine's answers now live in ONE file, outside the repo and gitignored:
#
#     ~/.config/llm-stack.env        (or $LLM_STACK_ENV)
#
# written once by setup/install.sh from setup/local.env.template. It is systemd
# EnvironmentFile syntax like everything else here, so this file can read it and
# the units can load it directly with `EnvironmentFile=-`.
#
# Note what is NOT in the list below. A fallback containing `/mnt/shared/LLM`
# would be the same hard-coding one level down — a directory that exists on one
# machine is not a convention. What IS here are places a person might plausibly
# keep GGUF files on ANY machine: three under $HOME, the FHS locations for
# machine-local data, and a glob for a mounted volume, which matches by SHAPE
# rather than by name.
#
# This list is THE list. setup/lib/models.sh and setup/install.sh read it from
# here (`systemdfile.py conventions`) rather than keeping their own — they each
# had one for a few hours on 27.08. and the three had already diverged, which
# is the failure this file's own docstring is about.
MODELS_CONVENTIONS = ("~/models", "~/.cache/llama.cpp", "~/.local/share/models",
                      "./models", "/srv/models", "/var/lib/llm",
                      "/mnt/*/LLM", "/mnt/*/models")


def local_env_path():
    """This machine's answers. Overridable for tests and for a second install."""
    return os.environ.get("LLM_STACK_ENV",
                          os.path.expanduser("~/.config/llm-stack.env"))


def local_var(name, default=None):
    """One value out of the local config, or `default` if it says nothing.

    Missing file is not an error: everything here has to work on a machine
    that has never run install.sh, or the first run of install.sh could not
    happen.
    """
    path = local_env_path()
    if not os.path.exists(path):
        return default
    return variable(path, name, default)


def models_dir(required=True):
    """Where the .gguf live, in this order:

        $LLAMA_MODELS               an explicit answer for this one command
        ~/.config/llm-stack.env     this machine's answer, written once
        the conventions above       a directory that actually holds .gguf files

    and then it gives up rather than guessing. `required=False` returns None
    instead, for callers that only want to know whether an answer exists.
    """
    explicit = os.environ.get("LLAMA_MODELS") or local_var("LLAMA_MODELS")
    if explicit:
        return os.path.expanduser(explicit)
    for d in MODELS_CONVENTIONS:
        # A conventional entry may be a glob (/mnt/*/LLM). Sorted, so two
        # matching volumes give the same answer on every run — an unordered
        # readdir would make the model directory depend on the filesystem.
        for full in sorted(glob.glob(os.path.abspath(os.path.expanduser(d)))):
            if glob.glob(os.path.join(full, "*.gguf")):
                return full
    if not required:
        return None
    raise SystemExit(
        "\nNo model directory. Three ways to say where the .gguf files are, in\n"
        "the order they are consulted:\n\n"
        "    LLAMA_MODELS=/path/to/models <command>      just this once\n"
        "    bash setup/install.sh                       writes %s\n"
        "    put them in one of %s\n"
        % (local_env_path(), ", ".join(MODELS_CONVENTIONS)))


def _assignment(path, name):
    """The raw value of `name`, continuation lines joined, or None.

    Stops at the end of the assignment — this is the part every previous
    version got wrong in a different way.
    """
    prefix = name + "="
    parts, collecting = [], False
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.rstrip("\n")
            if not collecting:
                if not stripped.startswith(prefix):
                    continue
                collecting = True
                stripped = stripped[len(prefix):]
            parts.append(stripped.rstrip("\\"))
            if not stripped.rstrip().endswith("\\"):
                break
    if not collecting:
        return None
    return " ".join(parts)


# Two tokens, and deliberately only two. A profile has to name a home
# directory and a model directory, and both are properties of the MACHINE, not
# of the model — hard-coding them is what made every profile in this repo
# unusable on anybody else's computer.
#
# Tokens rather than $HOME. systemd hands LLAMA_ARGS to `sh -c` as a variable,
# and a shell expands a variable's value once: a literal $HOME inside it stays
# literal. `eval` would expand it, and `eval` on a config file that users are
# invited to edit is a shell-injection surface, not a feature. A fixed token
# with a plain string replacement cannot execute anything.
#
# systemd's own %h is not available either: specifiers are expanded in unit
# files, not in EnvironmentFile values.
def expand(text, home=None, models=None):
    """Replace @HOME@ and @MODELS@. Pure, so the unit and the tools agree.

    The model directory is resolved ONLY when @MODELS@ is actually in the
    text. A missing answer is a problem for a profile that names a model, and
    no problem at all for a unit file that does not — raising in both cases
    would make a machine without models unable to read its own configuration.
    """
    home = home if home is not None else os.path.expanduser("~")
    if models is None and "@MODELS@" in text:
        models = models_dir()
    return text.replace("@HOME@", home).replace("@MODELS@", models or "")


def llama_args(path, home=None, models=None):
    """LLAMA_ARGS as an argv list, exactly as systemd would hand it over.

    Raises SystemExit if the profile has none — a profile without it would
    start llama-server with no arguments at all, which the unit deliberately
    refuses to do.
    """
    raw = _assignment(path, LLAMA_ARGS)
    if raw is None:
        raise SystemExit("no LLAMA_ARGS in %s" % path)
    # Expand BEFORE splitting, and split afterwards: a home directory with a
    # space in it would otherwise become two arguments. That is not this
    # machine's problem, but it is somebody's.
    return expand(raw, home, models).split()


def variable(path, name, default=None):
    """A plain scalar (MODEL_TITLE, MODEL_SWA, LLAMA_BIN), or `default`.

    One layer of surrounding double quotes is stripped, the way systemd does.
    """
    raw = _assignment(path, name)
    if raw is None:
        return default
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
        raw = raw[1:-1]
    return raw if raw else default


def directive(path, name):
    """Every value a unit file gives for `name`, continuation lines joined.

    Unit files carry the one list this stack cannot derive from anywhere:
    `Conflicts=`, which has to name every model instance by hand because
    systemd has no wildcard for template instances. Three places wanted to
    read it — switch-model.sh before a switch, tests/test_models.py to pin it
    against the registry, and tests/test_scripts.py for EnvironmentFile and
    ExecStart. Three readers of the same syntax is how the LLAMA_ARGS parsers
    got out of step, so this is one.

        directive(unit, "Conflicts")      -> ["llama-user@a.service llama-…"]
        directive(unit, "ExecStartPost")  -> one entry per occurrence
    """
    prefix = name + "="
    out, current = [], None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.rstrip("\n")
            if current is None:
                if not stripped.lstrip().startswith(prefix):
                    continue
                current = [stripped.lstrip()[len(prefix):].rstrip("\\")]
            else:
                current.append(stripped.rstrip("\\"))
            if not stripped.rstrip().endswith("\\"):
                out.append(" ".join(x.strip() for x in current).strip())
                current = None
    if current is not None:
        out.append(" ".join(x.strip() for x in current).strip())
    return out


def flag(argv, *names, default=None):
    """The value following the LAST occurrence of any of `names` in `argv`.

    llama.cpp takes the same option under a short and a long name (-c and
    --ctx-size, -np and --parallel), and every caller that wants to reason
    about the window has to know both.

    LAST, and that is the whole point of this function existing rather than
    being three lines at each call site. llama-server assigns as it walks its
    command line, so a repeated option keeps the value that comes LAST. This
    read the FIRST until 27.08.2026, which meant an override could not be seen
    at all: bench/sideserver.py appends `--extra`, so `-c 32768 … -c 131072`
    made the server allocate for 131072 while every reader here computed for
    32768. Measured: 27.2 GiB of GTT against a prediction built on 23.1, and
    budget.py reported a KV figure four times too high because it divided by
    the wrong window.

    The direction is the dangerous one. The memory guard under-predicts when
    an override raises the context, and under-prediction on a machine with
    pinned GTT is not an error message, it is a frozen desktop.

    A trailing name with no value after it is not a value: it cannot clobber
    an earlier occurrence that had one.
    """
    found = default
    for i, tok in enumerate(argv):
        if tok in names and i + 1 < len(argv):
            found = argv[i + 1]
    return found


def main(argv):
    if len(argv) >= 2 and argv[1] == "models":
        print(models_dir())
    elif len(argv) >= 2 and argv[1] == "conventions":
        # The one list, for the shell side. See MODELS_CONVENTIONS.
        for d in MODELS_CONVENTIONS:
            print(d)
    elif len(argv) >= 3 and argv[1] == "local":
        v = local_var(argv[2], argv[3] if len(argv) > 3 else "")
        print(v if v is not None else "")
    elif len(argv) >= 3 and argv[1] == "args":
        print(" ".join(llama_args(argv[2])))
    elif len(argv) >= 4 and argv[1] == "directive":
        for v in directive(argv[2], argv[3]):
            print(v)
    elif len(argv) >= 4 and argv[1] == "value":
        v = variable(argv[2], argv[3], argv[4] if len(argv) > 4 else "")
        print(v if v is not None else "")
    else:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        print("usage: systemdfile.py args <file>\n"
              "       systemdfile.py value <file> <NAME> [default]\n"
              "       systemdfile.py directive <unit> <NAME>\n"
              "       systemdfile.py models\n"
              "       systemdfile.py conventions\n"
              "       systemdfile.py local <NAME> [default]", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
