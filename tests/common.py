"""Shared helpers for the tests.

What is checked here are scripts, not packages: systemd starts cc-gateway.py
and cc-router.py, prewarm.py is called by hand. They are therefore loaded by
file path rather than imported.

The precondition for that is an import without consequences: no network, no
token file, no web.run_app. That is exactly what the main() guard in each of
the three files is for.
"""
import importlib.util, os, pathlib, sys

REPO = pathlib.Path(__file__).resolve().parent.parent

# --- the tests must not need this machine's model directory ------------------
#
# Reading a profile expands @MODELS@, and since 27.08. that resolves through
# setup/lib/systemdfile.py: $LLAMA_MODELS, then ~/.config/llm-stack.env, then
# the conventional locations — and then it gives up rather than guessing.
# Giving up is right for a server that is about to start and wrong for a test
# suite, which would turn "this machine has no models" into a dozen errors
# that say nothing about the code.
#
# So: if nothing resolves, point at a directory that certainly holds no
# weights. The tests that care already handle a model they cannot find, by
# NAMING it rather than skipping silently — see
# test_models.TestEveryProfileFitsTheMachineItIsOn. On a machine that does
# have models, nothing here changes.
if not os.environ.get("LLAMA_MODELS"):
    sys.path.insert(0, str(REPO / "setup" / "lib"))
    import systemdfile as _sf                                     # noqa: E402
    os.environ["LLAMA_MODELS"] = (_sf.models_dir(required=False)
                                  or str(REPO / "tests" / "no-such-model-dir"))

def load(path, name, env_=None):
    """Load a script file as a module.

    `env_` applies during loading — the modules read their constants from
    os.environ at import time, so setting them later would come too late.
    """
    before = {}
    for k, v in (env_ or {}).items():
        before[k] = os.environ.get(k)
        os.environ[k] = v
    try:
        spec = importlib.util.spec_from_file_location(name, REPO / path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for k, v in before.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

async def free_port():
    """Find a free port without holding on to it."""
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p

async def wait_until(condition, limit=2.0, tick=0.01):
    """Wait until `condition()` is true. Returns the result.

    Needed because the gateway has already written its answer before its
    finally runs: the client sees the end of the request earlier than the
    server finishes it. Without this, any statement about counters after a
    request would be a race.
    """
    import asyncio, time
    end = time.monotonic() + limit
    while time.monotonic() < end:
        if condition():
            return True
        await asyncio.sleep(tick)
    return condition()


# --- locale, and why a test may not simply ASK for one -----------------------
#
# Two tests here prove that a script pins LC_ALL, and both were silently
# vacuous in the wrong environment — in OPPOSITE directions, which is why
# neither was noticed:
#
#   test_gtt   sets LC_ALL=de_DE.UTF-8 to show that bash's printf rejects
#              awk's "8.9". If that locale is not INSTALLED, bash warns on
#              stderr, falls back to C, printf works, and the test passes —
#              it would pass with `export LC_ALL=C` deleted from gtt.sh.
#   test_prune shows that `sort -r` puts a line with an empty leading field
#              first, because collation ignores the leading tab. That is true
#              in EVERY UTF-8 locale and false under C. Run the suite under
#              LC_ALL=C and the test passes without exercising anything — and
#              it guards a command that deletes.
#
# Measured 27.08.:
#     printf "%.1f" 8.9   C: 8.9    ·  de_DE/fr_FR/es_ES: "invalid number"
#     sort -r  "1111" vs "\t9999"   C: 1111 first  ·  any UTF-8: \t9999 first
#
# So a test that needs a locale has to NAME the property it needs, take the
# first installed candidate, and SKIP LOUDLY when there is none. Passing is
# not an option: the whole point of these two is to fail when the pin is gone.

def _installed_locales():
    import subprocess
    try:
        out = subprocess.run(["locale", "-a"], capture_output=True, text=True,
                             timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    return {l.strip() for l in out.splitlines() if l.strip()}


def _pick(test, candidates, what):
    have = _installed_locales()
    norm = {l.lower().replace("-", ""): l for l in have}
    for c in candidates:
        for key in (c, c.lower().replace("-", ""), c.lower().replace(".utf-8", ".utf8")):
            if key in have:
                return key
            if key.lower().replace("-", "") in norm:
                return norm[key.lower().replace("-", "")]
    test.skipTest("no %s locale installed (tried %s) — this test cannot "
                  "prove anything here, and passing would be a lie"
                  % (what, ", ".join(candidates)))


# Locales whose LC_NUMERIC uses a comma. The bug is NOT German; it is every
# comma-decimal locale, which is most of continental Europe and Latin America.
COMMA_DECIMAL = ("de_DE.UTF-8", "fr_FR.UTF-8", "es_ES.UTF-8", "it_IT.UTF-8",
                 "pt_BR.UTF-8", "nl_NL.UTF-8", "pl_PL.UTF-8", "de_DE.utf8",
                 "fr_FR.utf8", "es_ES.utf8")

# Any locale with dictionary collation, where `sort` ignores leading
# punctuation. en_US is deliberately first: this property has nothing to do
# with a non-English language, and listing German first suggested it did.
COLLATING = ("en_US.UTF-8", "en_GB.UTF-8", "de_DE.UTF-8", "fr_FR.UTF-8",
             "en_US.utf8", "de_DE.utf8", "C.UTF-8")


def comma_decimal_locale(test):
    """A locale whose printf refuses "8.9", or skip loudly."""
    return _pick(test, COMMA_DECIMAL, "comma-decimal")


def collating_locale(test):
    """A locale whose sort ignores a leading tab, or skip loudly."""
    return _pick(test, COLLATING, "dictionary-collating")
