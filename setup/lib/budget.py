#!/usr/bin/env python3
"""budget — what a profile will cost in memory, and whether it fits. The only
implementation.

    from budget import plan, verdict, read_machine     # as a module
    python3 setup/lib/budget.py --profile qwen38       # the arithmetic
    python3 setup/lib/budget.py --check --profile qwen38     # exit 1 if it does not fit
    python3 setup/lib/budget.py --running              # what is serving right now
    python3 setup/lib/budget.py --observe              # what it ACTUALLY took
    python3 setup/lib/budget.py --cache                # what -cram buys
    python3 setup/lib/budget.py --json

Why this is its own file
------------------------
The same question — "does this fit?" — was answered in three places with three
different formulas, and none of them sat where a model actually gets started:

    bench/run.py       weights x 1.10, against GTT and MemAvailable, HOST 10
    bench/sideserver.py    its own ceiling, its own HOST_RESERVE 12
    tests/test_models.py   weights + cram + host, no KV at all

    switch-model.sh, llamaexec, systemd     nothing

That is the same failure class as the three LLAMA_ARGS parsers documented in
systemdfile.py, with a worse consequence: GTT is pinned, so a model that does
not fit does not page and does not get OOM-killed — it takes the machine down.
It has done so three times on this hardware.

And the formula that DID run was the wrong one. bench/run.py estimates
`weights x 1.10`, where the 10 % stands for "KV, compute buffers, the loader's
slack". At bench windows that is about right. At the production profile's
`-c 204800 -cram 32768` it is 18 GiB against a real 79 — it would wave qwen38
onto a 64 GB machine and freeze it.

The model
---------
Two quantities, and conflating them cost this machine a hang on 26.08.:

  GTT   what the GPU pins: the weights it offloads, plus the KV cache, plus
        compute buffers. Comes out of system RAM, is NOT swappable, and is
        bounded by ttm.pages_limit.
  HOST  everything resident: the GTT allocation itself, PLUS any part of the
        weights that does not land in GTT, PLUS the RAM prompt cache (-cram),
        PLUS what the desktop and the gateway need.

    gtt_need  = weights + kv + buffers         from the file size
              = gtt_base + kv x 1.10           where GTT itself was measured
    buffers   = max(6.0 GiB, (weights + kv) x 0.10)
    host_need = gtt_need + outside + cram
    fits      = gtt_need <= gtt_total - gtt_used
                and host_need <= mem_available - host_reserve

The buffer term is MEASURED, and the constant next to it is why: nine recorded
measurements put it at 3.1-4.6 GiB and show it barely moving when the KV
triples. It is not proportional. The floor sits above the worst of them; the
percentage takes over only for footprints far larger than anything measured
here, where a constant would be the optimistic answer.

Note which term the buffers are NOT added to: `gtt_base` is an OBSERVATION of
GTT and already contains them, so charging them again would count them twice —
8 GiB on a 100 GiB model, the difference between "tight" and "refused".
Overestimating is the safe direction here and the repo has said so since the
-cram audit; underestimating is how the machine goes down.

`outside` is what stays resident but never reaches GTT. Declared where it has
been measured, and otherwise derived as `weights - gtt_base` — which is a
LOWER bound and is marked as one, because gtt_base carries the buffers. For
Flash-Next that subtraction gives 16.3 GiB and the measurement was 27.1.

Where the KV number comes from, and why it is not derived
--------------------------------------------------------
KV cannot be read off the file system, and on this hardware it must not be
computed from the architecture either. qwen38's own profile records why: the
plain arithmetic (65 layers x 4 KV heads x 512 x 2 B) gives 256 KiB per token
and the MEASURED figure is 67.5, because the model is a hybrid — only about
every fourth layer keeps a full KV cache. A derived number would be 4x too
high and would refuse the one profile that is known to fit.

So the number is DECLARED in the profile, with its provenance:

    MODEL_KV_KIB_PER_TOKEN=74.3
    MODEL_KV_SOURCE=27.08.2026, from the saved prefixes, large context

and a profile that has none is estimated from a deliberately pessimistic
constant — which the output then says out loud, every time. An estimate that
does not announce itself is how `-cram 32768` got copied into five profiles.

This is not a number to be trusted forever: `--observe` reads what the running
server actually pinned and holds it against the declaration. A declared figure
with nothing checking it is an assertion; one that is re-checked after every
start is a measurement with a shelf life.
"""
import argparse, glob, json, os, re, sys
from collections import namedtuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from systemdfile import llama_args, variable, flag, expand   # noqa: E402

__all__ = ["Machine", "Plan", "Verdict", "Comparison", "compare",
           "plan", "verdict", "render", "brief", "cache_windows",
           "declared_kv", "declared_gtt", "declared_anon", "declared_lazy",
           "lazy_relief",
           "read_machine", "weights_gib", "kv_kib_per_token", "cram_gib",
           "running_argv", "observe", "HOST_RESERVE_GIB", "SLACK"]

# --- the constants, and what each one is standing in for ---------------------

# Desktop, gateway, this process, page cache — everything that is not the
# model. fits_the_machine() reads it as `model_host_footprint + reserve <=
# MemTotal`, so it is the share of RAM the model may not have.
#
# MEASURED 27.08.2026, and the word "conservative" did not survive it. The
# quantity is observable while production serves:
#
#     host = MemTotal - MemAvailable - GTT - RssAnon(llama-server)
#
# because GTT is pinned and belongs to the model without appearing in any
# process's RSS, and the server's anonymous half belongs to it too. Sampled
# every 5 s with qwen38 serving, twice on 27.08.:
#
#     min 11.1   median 11.1   MAX 11.2 GiB    300 s, with an unused Supabase
#                                              stack of ten containers running
#     min 10.9   median 10.9   MAX 11.0 GiB    180 s, after it was stopped
#
# So 12.0 sits 1.0 GiB above what the host occupies — 8 %, a floor with a
# margin rather than the generous figure the old comment implied. And the 10.0
# that bench/run.py charged until 27.08. was a full GiB BELOW it: a guard that
# would have waved through a model leaving the desktop short.
#
# THE SECOND READING IS ALSO A CORRECTION. `docker stats` reported 616 MiB for
# those containers and the obvious inference was that stopping them would give
# most of the margin back. Measured, it gave 0.25 GiB: most of what docker
# reports is page cache and shared pages, which MemAvailable already counts as
# available. A number from one tool, read as if it were a different quantity.
#
# WHAT THIS DOES NOT SAY. Five minutes on a quiet desktop is a FLOOR, not a
# peak, and a peak is what a reserve has to cover. A browser, a build, or the
# Claude Desktop that was OOM-killed here on 26.08. all push it up. Raising
# this number makes the guard refuse more profiles, so it is a trade rather
# than a free win — take the peak over a working day before touching it.
HOST_RESERVE_GIB = 12.0

# Compute buffers and the loader — the term that is neither the weights nor
# the KV, and the one number here that was NOT measured until 27.08.
#
# It was `(weights + KV) x 1.10`, and nine recorded measurements say that is
# the wrong SHAPE. The buffers are roughly CONSTANT for a model and backend;
# they scale with -ub and the embedding width, not with the footprint:
#
#     variant                  -c        weights   KV    GTT obs   buffers
#     rocm-*-spec           65,536         16.7    4.6      24.4       3.1
#     rocm-medium-spec-q8kv 65,536         16.7    2.3      22.7       3.6
#     vulkan-medium-spec    65,536         16.7    4.6      26.0       4.6
#     production (live)    204,800         17.6   14.5      35.6       3.5
#
# Tripling the KV moved the buffers by 0.4 GiB. Vulkan costs ~1.5 more than
# ROCm. And 10 % of weights+KV is 1.9-3.2 GiB against a measured 3.1-4.6 — so
# the old factor UNDER-predicted every one of those nine points, which is the
# dangerous direction. It looked right in production only because at the
# largest window the percentage happened to land near the constant.
#
# So: a FLOOR above the worst measurement, and the percentage kept as an upper
# branch for models far larger than anything measured here, where a constant
# would be the optimistic answer instead.
BUFFER_FLOOR_GIB = 6.0        # measured max 4.6 (Vulkan), plus ~30 % headroom
BUFFER_FRACTION = 0.10        # takes over above ~60 GiB of weights+KV


def buffers_gib(base_gib):
    """Compute buffers and the loader for a footprint of `base_gib`."""
    return max(BUFFER_FLOOR_GIB, base_gib * BUFFER_FRACTION)


# Kept as a name because bench/run.py's history and several messages refer to
# "the slack". It is no longer a multiplier applied to everything.
SLACK = 1.0 + BUFFER_FRACTION

# What a profile without a declared figure is charged per token of context, in
# KiB, for an f16 cache. Deliberately high: the measured figures on this
# hardware run 67-75 KiB/token for a hybrid 27B, and a dense model of the same
# size costs several times that. An estimate that is too low is indistinguish-
# able from no guard at all.
ESTIMATE_KV_KIB_PER_TOKEN = 128.0

# A quantised KV cache really is smaller, and that is physics rather than
# optimism, so the estimate scales with it. Anything unrecognised keeps the
# f16 figure — an unknown type is not a licence to charge less.
KV_TYPE_FACTOR = {"f32": 2.0, "f16": 1.0, "bf16": 1.0,
                  "q8_0": 0.5, "q5_1": 0.34, "q5_0": 0.32,
                  "q4_1": 0.28, "q4_0": 0.25}

# --swa-full removes the sliding-window cap, so every layer keeps a FULL KV
# cache instead of a windowed one. The estimate had no term for it and was
# wrong by 50 % on the one profile that carries it: laguna measures 96.0
# KiB/token at q8_0 where the unadjusted estimate charged 64.0. Under-
# prediction against pinned GTT is the failure this whole module exists to
# prevent, so that is not a rounding error.
#
# The contrast is measured, on two profiles that both declare MODEL_SWA=yes:
#
#     gemma31   --swa-full ABSENT    44.5 KiB/token   10 of 60 layers grow
#     laguna    --swa-full PRESENT   96.0             all of them
#
# 2.5 rather than 2.0: it has to COVER laguna, not merely reach it, and 2.0
# lands at 128 against a measured 96 with the same margin the windowed
# profiles get by accident rather than by choice. It is one data point, and
# the true factor is architectural — Gemma 3 alternates five local layers to
# one global, so removing the cap can cost up to sixfold. If a --swa-full
# profile is ever REFUSED by this term, measure it and declare the figure.
# Do not lower the factor to make a start succeed; that is the estimate being
# argued with instead of replaced.
#
# tests/test_budget.py holds the property this is here for: the estimate must
# cover EVERY profile that declares a measured figure.
SWA_FULL_FACTOR = 2.5

# How far the observation may drift from the declaration before it is worth
# saying so. 10 % is wide enough to absorb a build change and narrow enough to
# catch a number that was copied from another profile.
OBSERVE_TOLERANCE = 0.10

Machine = namedtuple("Machine", "mem_total mem_available gtt_total gtt_used")
Item = namedtuple("Item", "name gib source")
Plan = namedtuple("Plan", "what gtt_gib host_gib items estimated")
Verdict = namedtuple("Verdict", "fits problems notes")


# --- reading the machine -----------------------------------------------------

def _meminfo_gib(field):
    """A /proc/meminfo field BY NAME.

    Positionally is how sideserver.py once read MemTotal and got the label
    instead of the number: the ceiling then silently fell back to its
    conservative default and nobody saw it.
    """
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith(field + ":"):
                    return float(line.split()[1]) / 1048576.0
    except (OSError, ValueError, IndexError):
        pass
    return None


def _gtt_gib(which):
    """mem_info_gtt_total / mem_info_gtt_used, in GiB, or None off amdgpu."""
    for path in sorted(glob.glob("/sys/class/drm/card*/device/mem_info_gtt_" + which)):
        try:
            with open(path) as fh:
                return float(fh.read().split()[0]) / 1073741824.0
        except (OSError, ValueError, IndexError):
            pass
    return None


def read_machine():
    """The four facts, each independently allowed to be unknown.

    Unknown is NOT a refusal. On a machine without amdgpu — a laptop reading
    this repo, a CI runner — every GTT fact is None and the guard has no
    opinion about the cap while still checking what it can. A safety that
    blocks everything it cannot see gets switched off, and then it is gone.
    """
    return Machine(mem_total=_meminfo_gib("MemTotal"),
                   mem_available=_meminfo_gib("MemAvailable"),
                   gtt_total=_gtt_gib("total"),
                   gtt_used=_gtt_gib("used"))


# --- reading the profile -----------------------------------------------------

def weights_gib(argv, size_of=None):
    """Every byte the profile will load, in GiB, or None if it cannot be read.

    Shards and the mmproj are counted: a sharded GGUF names part one and the
    siblings are just as resident, and a vision projector is weights like any
    other. None means "not our call" — an unreadable path is not evidence that
    a model is small.
    """
    size_of = size_of or os.path.getsize
    names = []
    path = flag(argv, "-m", "--model")
    if not path:
        return None
    m = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", path)
    if m:
        found = sorted(glob.glob("%s-*-of-%s.gguf" % (m.group(1), m.group(3))))
        names.extend(found or [path])
    else:
        names.append(path)
    mmproj = flag(argv, "--mmproj", "--mmproj-file")
    if mmproj:
        names.append(mmproj)
    total = 0
    for n in names:
        try:
            total += size_of(n)
        except OSError:
            return None
    return total / 1073741824.0


def kv_kib_per_token(argv, declared=None):
    """(KiB per token of context, where the number came from).

    Order: an explicit override, then the profile's declaration, then an
    estimate scaled by the KV cache type. The source travels with the value
    because the caller has to be able to say which it was — see the docstring.
    """
    override = _num("KV_KIB_PER_TOKEN")
    if override is not None:
        return override, "stated"
    if declared is not None:
        return declared, "declared"
    ctk = (flag(argv, "-ctk", "--cache-type-k") or "f16").lower()
    ctv = (flag(argv, "-ctv", "--cache-type-v") or "f16").lower()
    factor = (KV_TYPE_FACTOR.get(ctk, 1.0) + KV_TYPE_FACTOR.get(ctv, 1.0)) / 2.0
    if "--swa-full" in argv:
        factor *= SWA_FULL_FACTOR
    return ESTIMATE_KV_KIB_PER_TOKEN * factor, "estimated"


def kv_gib(argv, declared=None):
    """(KV cache size in GiB, source). Zero when the profile names no window.

    `-c` is the TOTAL number of cells across all slots — `-np` divides it, it
    does not multiply the cost — so the window alone decides this.
    """
    ctx = flag(argv, "-c", "--ctx-size")
    if not ctx:
        return 0.0, "no -c"
    try:
        cells = int(ctx)
    except ValueError:
        return 0.0, "no -c"
    per_token, source = kv_kib_per_token(argv, declared)
    return cells * per_token / 1048576.0, source


def cram_gib(argv):
    """`-cram` is MEGABYTES. Reading it as GiB is a factor of 1024, and this
    is the flag that was copied unread into five profiles."""
    v = flag(argv, "-cram")
    if not v:
        return 0.0
    try:
        return int(v) / 1024.0
    except ValueError:
        return 0.0


def _declared_number(env_path, name):
    if not env_path or not os.path.exists(env_path):
        return None
    raw = variable(env_path, name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        raise SystemExit("%s in %s is not a number: %r" % (name, env_path, raw))


def cache_windows(argv, declared=None):
    """(GiB of RAM prompt cache, GiB per full window, how many windows).

    What `-cram` actually BUYS, which is the question nobody asked of it.
    The flag was computed once, for qwen38, and copied into every profile that
    came after — the -cram audit of 27.08. found it 30 GiB over on Flash-Next
    and stopped there. Measured across the registry the same day:

        qwen38      2.2 windows      the one that was computed
        laguna      1.3
        flashnext   1.7
        gptoss      7.1
        gemma31    16.0             on an ESTIMATED KV figure
        gemma26    24.6
        batch     196.9             8k of context, 16 GiB of cache for it

    A window is the worst case a single prefix can cost. Two to four of them
    is headroom; two hundred is a number nobody read.
    """
    cram = cram_gib(argv)
    kv, _ = kv_gib(argv, declared)
    if not cram or not kv:
        return cram, kv, None
    return cram, kv, cram / kv


def declared_kv(env_path):
    """MODEL_KV_KIB_PER_TOKEN from a profile, or None."""
    return _declared_number(env_path, "MODEL_KV_KIB_PER_TOKEN")


def declared_gtt(env_path):
    """MODEL_GTT_BASE_GIB from a profile, or None. See plan()."""
    return _declared_number(env_path, "MODEL_GTT_BASE_GIB")


def declared_anon(env_path):
    """MODEL_HOST_ANON_GIB from a profile, or None. See plan()."""
    return _declared_number(env_path, "MODEL_HOST_ANON_GIB")


def declared_lazy(env_path):
    """MODEL_LAZY_GIB from a profile, or None. See plan().

    The bytes llama.cpp reads ROW BY ROW from the mapped file instead of
    holding resident — `TENSOR_READ_LAZY`, upstream #27794, merged 27.08.2026.
    Until then every byte of a GGUF had to be resident somewhere, and this
    file's whole model rests on that. It stopped being true.

    IT IS AN OBSERVATION OF RssAnon, like MODEL_GTT_BASE_GIB is an observation
    of GTT, and that wording is the whole lesson of 28.08. This field was first
    written to hold the SIZE OF THE TENSOR, read out of the GGUF — a fact about
    the file, checkable, hard to get wrong. For Qwen3.8-Flash-Next that is
    `per_layer_token_embd.weight`, IQ4_NL, 26.82 GiB, against 0.83 for the next
    largest tensor in the model.

    The guard then granted 26.82 GiB of relief and the run measured `RssAnon
    26.83 GiB` with not one page of the GGUF mapped: the table was copied into
    anonymous memory exactly as before. The machine ended that run at 7.1 GiB
    available with 140 MiB of swap left.

    A tensor's size says what COULD be demand-paged. Only a measurement says
    what is. So this field takes the second, and a profile that has not run the
    measurement leaves it out and is refused instead — which is the correct
    answer for a model that genuinely does not fit.
    """
    return _declared_number(env_path, "MODEL_LAZY_GIB")


def lazy_relief(env_path, argv=None, binary=None):
    """The GiB plan() may subtract as demand-paged, after checking it is real.

    THREE conditions, and each one is a way this could take the machine down
    rather than protect it:

      1. the profile declares an OBSERVED figure  — nothing is inferred
      2. the binary is the one it was observed on — see below
      3. argv does not turn it off               — `--tensor-read-lazy off`
                                                    restores the old
                                                    behaviour, so the old
                                                    arithmetic applies

    Condition 2 was first written as "the binary knows --tensor-read-lazy",
    asked of `--help`, and the measurements of 28.08. refuted it in one
    afternoon. Three builds, all of which know the flag:

      master b10665-1, load_mode auto -> none   RssAnon 27.13 GiB   no relief
      master b10665-1, --load-mode mmap         lazy read enabled, but the
                                                whole model loads through the
                                                mapping and the server had not
                                                served /slots after 7 minutes
      + PR #27837, load_mode none               RssAnon  0.31 GiB   it works

    Knowing the flag says nothing, because on this machine `auto` resolves to
    `none`: ggml-cuda.cu reports `mmap_support = props->type != IGPU`, the
    type comes from `prop.integrated`, and the 8060S is integrated — so mmap
    is off for the whole model and TENSOR_READ_LAZY, which requires it, never
    fires. #27837 is what separates the two: it mmaps the lazy tensor alone
    and leaves the rest on the fast path.

    None of that is visible in `--help`. What IS knowable before starting is
    WHICH BUILD will run, and the profile already pins it and says why. So the
    figure and the binary travel together: an observation made on one build is
    not evidence about another, and a rebuild that changes LLAMA_BIN withdraws
    the relief until somebody measures again.

    Returns 0.0 rather than None when a condition fails, so a caller cannot
    accidentally read "unknown" as "granted".
    """
    lazy = declared_lazy(env_path)
    if not lazy:
        return 0.0
    # BOTH channels. llama.cpp registers this option with
    # .set_env("LLAMA_ARG_TENSOR_READ_LAZY") (common/arg.cpp:2743), so a
    # systemd Environment= line or a shell export turns it off just as an
    # argument does. Checking only argv would grant 26.82 GiB of relief the
    # server will not deliver — and the profile records what that costs:
    # "the machine finished that run at 7.1 GiB available with 140 MiB of swap".
    if os.environ.get("LLAMA_ARG_TENSOR_READ_LAZY") == "off":
        return 0.0
    for i, tok in enumerate(argv or []):
        if tok == "--tensor-read-lazy" and i + 1 < len(argv) and argv[i + 1] == "off":
            return 0.0
    if not binary or not _is_the_observed_binary(env_path, binary):
        return 0.0
    return lazy


def _is_the_observed_binary(env_path, binary):
    """Is `binary` the LLAMA_BIN this profile was measured with?

    Compared by resolved path, so a symlink that has been repointed since the
    measurement does not pass as the build behind it — which is the whole
    failure mode the profile's own pin was written against.
    """
    declared = variable(env_path, "LLAMA_BIN") if env_path else None
    if not declared:
        return False
    want = os.path.realpath(os.path.expanduser(
        declared if os.path.isabs(declared) else os.path.join("~", declared)))
    return os.path.realpath(binary) == want


# --- the escape hatches ------------------------------------------------------
#
# There has to be a way past a WRONG estimate that is not a way past the guard.
# Flash-Next is the case that proves it: 103.7 GiB of file, 87.4 in GTT,
# because llama.cpp keeps its 26 GiB n-gram table out ("mmap PLE offload").
# Without a narrow correction the only escape is the off switch — and a safety
# that is inconvenient at the wrong moment gets switched off entirely, which is
# how this machine hung twice in one day.
#
# So these correct the INPUT and every comparison still runs. The names are
# LLM_*; the BENCH_* names are the ones bench/run.py shipped with and are kept
# as aliases so a measurement in flight does not change meaning.

_ALIASES = {"MODEL_GIB": "BENCH_MODEL_GIB",
            "HOST_GIB": "BENCH_HOST_GIB",
            "HOST_RESERVE_GIB": "BENCH_HOST_RESERVE_GIB",
            "KV_KIB_PER_TOKEN": "BENCH_KV_KIB_PER_TOKEN",
            "NO_MEMORY_GUARD": "BENCH_NO_MEMORY_GUARD"}


def _raw(name, env=None):
    env = env if env is not None else os.environ
    v = env.get("LLM_" + name)
    if v is None:
        v = env.get(_ALIASES[name])
    return v


def _num(name, env=None):
    """A numeric knob, or None. A typo is refused rather than ignored:
    silently falling back would hide a mistake in the one place where a number
    is being trusted instead of measured."""
    v = _raw(name, env)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        raise SystemExit("%s is not a number: %r" % ("LLM_" + name, v))


def guard_disabled(env=None):
    return _raw("NO_MEMORY_GUARD", env) == "1"


def host_reserve_gib(env=None):
    v = _num("HOST_RESERVE_GIB", env)
    return HOST_RESERVE_GIB if v is None else v


# --- the plan (pure) ---------------------------------------------------------

def plan(argv, weights, declared=None, what="this server",
         gtt_base=None, host_anon=None, lazy=0.0):
    """What starting `argv` will cost. PURE: the machine is not consulted.

    `weights` is passed in rather than read so that the arithmetic can be
    tested without a model on disk, and so that a caller which has already
    measured them does not read the file twice.

    The two optional figures are MEASUREMENTS a profile may carry, and they
    exist because for one model here the file size answers the wrong question:

      gtt_base   what GTT shows for this model with the KV taken out — i.e.
                 the weights the GPU pins PLUS the compute buffers. Qwen3.8-
                 Flash-Next is the case: 103.7 GiB of file, and the two-point
                 measurement (load at 65536 and at 262144, take the difference)
                 gives 78.1 GiB of GTT before any KV, because llama.cpp keeps
                 the n-gram table out of GTT.
      host_anon  what is resident OUTSIDE GTT. Measured for the same model:
                 RssAnon 27.1 GiB, with no .gguf mapping at all — so it is
                 anonymous memory and not reclaimable, and subtracting
                 gtt_base from the file size would have under-counted it.

    Note which one gets the slack. A file size says nothing about compute
    buffers, so it is charged SLACK. A gtt_base is an OBSERVATION of GTT and
    already contains them, so charging it again would double-count — 8 GiB on
    a model this size, which is the difference between "tight" and "refused".
    The KV is slacked in both cases: it is derived from a per-token figure, not
    observed as part of a total.

    `lazy` is the third, and it is younger than the other two: the GiB that
    are DEMAND-PAGED and therefore need not be resident at all. Everything
    else here assumes the opposite — "ALL of it has to be resident somewhere"
    is what verdict() says and what the LLM_HOST_GIB floor enforces — and that
    assumption was simply true until llama.cpp #27794 landed on 27.08.2026.
    Pass it only through lazy_relief(), which checks that the binary can
    actually do it; plan() stays PURE and asks nothing.
    """
    items, estimated = [], False

    # An environment override is a different quantity from gtt_base and is
    # treated as one: LLM_MODEL_GIB corrects the WEIGHTS the GPU will pin —
    # the file size, not a GTT reading — so it still gets the slack.
    in_gtt = weights
    stated_gtt = _num("MODEL_GIB")
    if stated_gtt is not None and weights is not None:
        in_gtt = stated_gtt

    kv, kv_source = kv_gib(argv, declared)
    estimated = kv_source == "estimated"

    gtt_need = None
    if gtt_base is not None:
        # A measured gtt_base is an observation of GTT and already CONTAINS the
        # compute buffers, so they are not added again. Only the KV, which is
        # derived from a per-token figure, keeps a margin.
        items.append(Item("GTT with the KV taken out", gtt_base, "measured"))
        if kv:
            items.append(Item("KV cache", kv, kv_source))
            items.append(Item("margin on the KV (%.0f %%)" % (BUFFER_FRACTION * 100),
                              kv * BUFFER_FRACTION, "floor"))
        gtt_need = gtt_base + kv * (1.0 + BUFFER_FRACTION)
    elif in_gtt is not None:
        if stated_gtt is not None:
            items.append(Item("weights in GTT", in_gtt, "stated"))
        else:
            items.append(Item("weights", in_gtt, "file"))
        if kv:
            items.append(Item("KV cache", kv, kv_source))
        buf = buffers_gib(in_gtt + kv)
        how = "floor" if buf == BUFFER_FLOOR_GIB else "%.0f %%" % (BUFFER_FRACTION * 100)
        items.append(Item("compute buffers and loader", buf, how))
        gtt_need = in_gtt + kv + buf

    # HOST: the GTT allocation is system RAM too, so it is not a separate
    # column — it is the first term. What is added is whatever stays resident
    # outside GTT and the RAM prompt cache, which is charged in full because
    # worst case is what a guard is for.
    outside = None
    if host_anon is not None:
        outside = host_anon
        items.append(Item("resident outside GTT", outside, "measured"))
        if lazy:
            # A host_anon measured on a build that loaded the table is not
            # wrong, it is STALE — it recorded bytes that this build reads
            # from the file instead. Shown as its own line rather than folded
            # into the one above, because the two have different provenance
            # and a reader has to be able to see which one moved.
            take = min(lazy, outside)
            items.append(Item("of that, read on demand (not resident)",
                              -take, "lazy"))
            outside -= take
    elif weights is not None and gtt_base is not None:
        # A LOWER bound, and it is worth saying why: gtt_base contains the
        # compute buffers, so the weights actually in GTT are less than it and
        # the part outside is more than this difference. Measured for
        # Flash-Next: this subtraction gives 16.3 GiB and RssAnon was 27.1.
        # Declare MODEL_HOST_ANON_GIB rather than relying on it.
        outside = max(0.0, weights - gtt_base - lazy)
        items.append(Item("resident outside GTT (lower bound)", outside,
                          "derived, less demand-paged" if lazy else "derived"))
    elif weights is not None and stated_gtt is not None:
        outside = max(0.0, weights - stated_gtt)
        if outside:
            items.append(Item("weights held outside GTT", outside, "file"))
    else:
        outside = 0.0

    cram = cram_gib(argv)
    if cram:
        items.append(Item("RAM prompt cache (-cram)", cram, "profile"))

    host_need = None
    if gtt_need is not None:
        host_need = gtt_need + (outside or 0.0) + cram

    # A stated host figure may correct the estimate, but it has a FLOOR: it may
    # not fall below the bytes on disk. BENCH_HOST_GIB=88 was once used to
    # claim a model needed 88 GiB when 103.7 had to be resident. The server
    # started, the machine went to 100 %, and the kernel OOM-killed it. A
    # measurement can say where the bytes land. Nothing makes a file smaller.
    stated_host = _num("HOST_GIB")
    if stated_host is not None and weights is not None:
        # The floor is the bytes that must be RESIDENT, which used to be the
        # same thing as the bytes on disk. With demand-paged tensors it is not,
        # so the floor moves down by exactly what lazy_relief() has already
        # vouched for — and by nothing else. A caller who wants to claim more
        # than that still hits the same refusal that took this machine down on
        # 26.08.
        floor = weights - lazy
        if stated_host < floor:
            raise SystemExit(
                "\nLLM_HOST_GIB=%.1f is below the %.1f GiB the model must keep "
                "resident\n  (%.1f GiB on disk%s).\n  All of it has to be "
                "resident somewhere — GTT or anonymous memory.\n  That exact "
                "claim took this machine down on 26.08."
                % (stated_host, floor, weights,
                   ", less %.1f read on demand" % lazy if lazy else ""))
        host_need = stated_host + cram
        items.append(Item("host footprint", stated_host, "stated"))

    return Plan(what=what, gtt_gib=gtt_need, host_gib=host_need,
                items=items, estimated=estimated)


def verdict(p, machine, reserve=None):
    """Does it fit? PURE: the machine's facts are handed in.

    `problems` is what refuses. `notes` is what the reader has to know to
    judge the refusal — above all whether it rests on an estimate.
    """
    reserve = host_reserve_gib() if reserve is None else reserve
    problems, notes = [], []

    if p.gtt_gib is None:
        notes.append("the weights could not be read, so this is not our call")
        return Verdict(True, problems, notes)

    if machine.gtt_total is not None and machine.gtt_used is not None:
        free = machine.gtt_total - machine.gtt_used
        if p.gtt_gib > free:
            problems.append("GTT has %.1f GiB left of %.1f (%.1f in use), and "
                            "this needs about %.1f"
                            % (free, machine.gtt_total, machine.gtt_used, p.gtt_gib))
    else:
        notes.append("GTT is not readable here, so the cap was not checked")

    if p.host_gib is not None and machine.mem_available is not None:
        if p.host_gib > machine.mem_available - reserve:
            problems.append("the host has %.1f GiB available, %.0f must stay "
                            "free, and ALL %.1f GiB has to be resident "
                            "somewhere — GTT or anonymous memory"
                            % (machine.mem_available, reserve, p.host_gib))
    elif machine.mem_available is None:
        notes.append("MemAvailable is not readable here, so the host was not checked")

    if p.estimated:
        notes.append("the KV figure is an ESTIMATE, not a measurement — "
                     "declare MODEL_KV_KIB_PER_TOKEN in the profile once you "
                     "have run `budget.py --observe` against it")
    return Verdict(not problems, problems, notes)


def fits_the_machine(p, machine, reserve=None):
    """The STATIC question: would this profile fit an idle machine of this size?

    Different from verdict(), which asks whether it fits what is left RIGHT
    NOW. A profile can be sound and still not startable beside a running one,
    and a profile can be unsound on a machine that happens to be empty.
    """
    reserve = host_reserve_gib() if reserve is None else reserve
    if p.host_gib is None or machine.mem_total is None:
        return None
    return p.host_gib + reserve <= machine.mem_total


# --- rendering ---------------------------------------------------------------

def render(p, machine, v):
    out = ["  %s" % p.what]
    for it in p.items:
        out.append("    %-38s %7.1f GiB   %s" % (it.name, it.gib, it.source))
    if p.gtt_gib is not None:
        out.append("    %-38s %7.1f GiB" % ("= pinned in GTT", p.gtt_gib))
    if p.host_gib is not None:
        out.append("    %-38s %7.1f GiB" % ("= resident in host RAM", p.host_gib))
    if machine.mem_total is not None:
        out.append("    %-38s %7.1f GiB" % ("machine has", machine.mem_total))
    for n in v.notes:
        out.append("    note: %s" % n)
    for pr in v.problems:
        out.append("    PROBLEM: %s" % pr)
    return "\n".join(out)


def brief(p, machine, v):
    """One line, for a journal. See --brief.

    It names the two numbers a reader would otherwise have to go and compute,
    and it says when the KV figure was guessed — because "it started" and "it
    was checked and fits" look identical in a log otherwise, and the second is
    the only one that means anything.
    """
    if p.gtt_gib is None:
        return "%s: the weights could not be read — not weighed" % p.what
    total = ("of %.1f" % machine.mem_total) if machine.mem_total else ""
    return ("%s %s: %.1f GiB in GTT, %.1f resident %s%s"
            % (p.what, "fits" if v.fits else "DOES NOT FIT",
               p.gtt_gib, p.host_gib if p.host_gib is not None else p.gtt_gib,
               total,
               "  (KV ESTIMATED — measure it and declare "
               "MODEL_KV_KIB_PER_TOKEN)" if p.estimated else ""))


def refusal(p, machine, v):
    """The message a guard raises. One shape, wherever it is raised from."""
    return (
        "\nREFUSING TO START %s: it needs about %.1f GiB and it does not fit.\n"
        "    %s\n"
        "\n%s"
        "\n  GTT is not swappable, so starting anyway does not page — it takes\n"
        "  the machine down. That happened on 26.08.2026.\n"
        "\n  Correct a wrong estimate with LLM_MODEL_GIB / LLM_HOST_GIB /\n"
        "  LLM_KV_KIB_PER_TOKEN — every check still runs. LLM_NO_MEMORY_GUARD=1\n"
        "  switches the guard off, which is a different and worse thing.\n"
        % (p.what, p.host_gib if p.host_gib is not None else p.gtt_gib,
           "\n    ".join(v.problems),
           # The table WITHOUT the problems: they are already above, and a
           # refusal that states its reasons twice reads like two refusals.
           render(p, machine, Verdict(v.fits, [], v.notes))))


# --- observation: what it ACTUALLY took --------------------------------------

# "not passed" is a different thing from "measured as unavailable", and a test
# has to be able to state the second. Same distinction machine_ram_gib() needed.
_UNSET = object()


def server_pid(proc="/proc"):
    """The pid of the running llama-server, or None.

    One scan, because there were three: running_argv(), _rss_anon_gib() and —
    very nearly — the GTT reader below each walked /proc looking for the same
    process by the same test. Three readers of one thing is how this repo has
    found most of its bugs; see setup/lib/systemdfile.py.

    The running process rather than a profile: a server started by hand is
    still a server, and the point is to describe the machine in front of you.
    """
    try:
        pids = [d for d in os.listdir(proc) if d.isdigit()]
    except OSError:
        return None
    for pid in sorted(pids, key=int):
        try:
            with open(os.path.join(proc, pid, "cmdline"), "rb") as fh:
                if fh.read().split(b"\0")[0].endswith(b"llama-server"):
                    return pid
        except OSError:
            continue
    return None


def running_argv(proc="/proc"):
    """The argv of the running llama-server, or None."""
    pid = server_pid(proc)
    if pid is None:
        return None
    try:
        with open(os.path.join(proc, pid, "cmdline"), "rb") as fh:
            argv = fh.read().decode("utf-8", "replace").split("\0")
    except OSError:
        return None
    return [a for a in argv if a] or None


def _rss_anon_gib(proc="/proc"):
    """RssAnon of the running llama-server. Measured 27.08.: for Flash-Next
    this is 27.1 GiB and there is no .gguf mapping at all, so the part of a
    model that stays out of GTT is anonymous and NOT reclaimable."""
    pid = server_pid(proc)
    if pid is None:
        return None
    try:
        with open(os.path.join(proc, pid, "status")) as fh:
            for line in fh:
                if line.startswith("RssAnon:"):
                    return float(line.split()[1]) / 1048576.0
    except OSError:
        pass
    return None


def server_gtt_gib(proc="/proc"):
    """GTT held by the llama-server ITSELF, or None if the kernel will not say.

    THE OBSERVATION USED TO BE SYSTEM-WIDE, and that was the open item. GTT
    read from mem_info_gtt_used counts the desktop too, so an under-prediction
    of two or three GiB — exactly the size that matters — disappeared into
    whatever the compositor happened to be holding. Measured 27.08. while
    qwen38 served: 35.64 GiB system-wide against 34.78 the process itself, so
    the slack being hidden was 0.86 GiB on an idle desktop and more on a busy
    one.

    amdgpu accounts per DRM client in /proc/PID/fdinfo. Summed over DISTINCT
    drm-client-id, because a process may hold both the card and the render
    node open and the same allocation would otherwise be counted twice.
    `drm-resident-gtt` rather than `drm-total-gtt`: resident is what is
    actually occupying memory now, which is the quantity the guard predicts.

    None when the kernel is too old for these keys, when the process belongs
    to another user, or when there is no server. None means the caller falls
    back to the system-wide figure and SAYS which one it used — an observation
    whose source is unknown is worse than a coarse one.
    """
    pid = server_pid(proc)
    if pid is None:
        return None
    d = os.path.join(proc, pid, "fdinfo")
    try:
        names = os.listdir(d)
    except OSError:
        return None
    per_client = {}
    for name in names:
        client, kib = None, None
        try:
            with open(os.path.join(d, name)) as fh:
                for line in fh:
                    if line.startswith("drm-client-id:"):
                        client = line.split(":", 1)[1].strip()
                    elif line.startswith("drm-resident-gtt:"):
                        parts = line.split(":", 1)[1].split()
                        if parts and parts[0].isdigit():
                            kib = float(parts[0])
        except OSError:
            continue
        if client is not None and kib is not None:
            per_client[client] = kib
    if not per_client:
        return None
    return sum(per_client.values()) / 1048576.0


def observe(argv=None, machine=None, weights=None, gtt_process=_UNSET):
    """What the running server actually costs, and how that compares.

    This is the half that keeps the declaration honest. The GPU pins the
    weights it offloads plus the KV, and llama.cpp allocates the whole KV at
    load — so with the weights known, GTT tells us the KV, and the KV tells us
    the KiB per token. That number is the one the profile should carry.

    It reports; it does not write. A profile is a file people are invited to
    edit, and a tool that edits it behind them is how a measured number turns
    back into a copied one.
    """
    argv = argv if argv is not None else running_argv()
    if not argv:
        return None
    machine = machine or read_machine()
    weights = weights if weights is not None else weights_gib(argv)
    ctx = flag(argv, "-c", "--ctx-size")
    mine = server_gtt_gib() if gtt_process is _UNSET else gtt_process
    # The server's own GTT when the kernel will say, the system-wide figure
    # when it will not — and the answer records WHICH, because the two are not
    # the same measurement and a reader cannot tell them apart by their value.
    gtt = mine if mine is not None else machine.gtt_used
    out = {"argv_model": flag(argv, "-m", "--model"), "ctx": ctx,
           "weights_gib": weights,
           "gtt_used_gib": machine.gtt_used,
           "gtt_process_gib": mine,
           "gtt_observed_gib": gtt,
           "gtt_source": "the server's own, per DRM client" if mine is not None
                         else "system-wide — the desktop is in it too",
           "rss_anon_gib": _rss_anon_gib()}
    if gtt is not None and weights is not None and ctx:
        try:
            cells = int(ctx)
        except ValueError:
            cells = 0
        if cells:
            kv = gtt - weights
            out["kv_gib_observed"] = kv
            out["kv_kib_per_token_observed"] = kv * 1048576.0 / cells
    return out


Comparison = namedtuple("Comparison", "ok predicted observed margin kv_upper note")


def compare(p, got):
    """Predicted GTT against observed GTT. The honest comparison, and it took
    one wrong version of this function to see why.

    The tempting comparison is the KV figure: derive KiB per token from
    (GTT - weights) and hold it against the declaration. It reads 24 % high on
    qwen38 and the profile is fine — because a single reading cannot separate
    the KV from the compute buffers, so that whole term lands on the KV. It
    flagged a correct profile red the first time it ran.

    What CAN be compared is the total, and it is also the number the guard
    actually uses. Two caveats travel with it, both in the safe direction:

      * mem_info_gtt_used is SYSTEM-WIDE, not per process. The desktop's
        compositor is in there too, so the observation is an upper bound on
        what the model holds.
      * the prediction carries the 10 % slack, so it is an upper bound too.

    Only ONE direction is a defect. Observed comfortably below predicted means
    the guard is conservative, which is what it is for. Observed ABOVE it means
    the guard under-predicts — and under-predicting is how a machine freezes.
    """
    observed = None if not got else got.get("gtt_observed_gib",
                                             got.get("gtt_used_gib"))
    if p.gtt_gib is None or observed is None:
        return None
    margin = (observed - p.gtt_gib) / p.gtt_gib
    kv_upper = got.get("kv_kib_per_token_observed")
    if got.get("gtt_process_gib") is not None:
        note = ("the server's own GTT, per DRM client; the prediction carries "
                "its compute-buffer floor, so the prediction still overstates "
                "and the observation no longer does.")
    else:
        note = ("system-wide GTT, so the desktop is counted in it too; the "
                "prediction carries its compute-buffer floor. Both overstate.")
    return Comparison(ok=margin <= OBSERVE_TOLERANCE, predicted=p.gtt_gib,
                      observed=observed, margin=margin, kv_upper=kv_upper,
                      note=note)


# --- CLI ---------------------------------------------------------------------

def _profile_path(name):
    if os.path.exists(name):
        return name
    for base in (os.path.join(HERE, "..", "env"),
                 os.path.expanduser("~/.config/llm-profile"), "/etc/llm-profile"):
        p = os.path.join(base, name if name.endswith(".env") else name + ".env")
        if os.path.exists(p):
            return os.path.abspath(p)
    raise SystemExit("no such profile: %s" % name)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--profile", "--env", dest="profile",
                   help="a profile name or a setup/env/*.env path")
    g.add_argument("--running", action="store_true",
                   help="the llama-server that is serving now")
    g.add_argument("--from-env", dest="from_env", action="store_true",
                   help="LLAMA_ARGS and the MODEL_* declarations out of this "
                        "process's environment — which is exactly what systemd "
                        "hands a unit, so ExecStartPre needs no path at all")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if it does not fit (this is the guard)")
    ap.add_argument("--static", action="store_true",
                    help="ask whether it fits an IDLE machine of this size, "
                         "not what is left right now")
    ap.add_argument("--cache", action="store_true",
                    help="what -cram buys, in full windows, for every profile")
    ap.add_argument("--observe", action="store_true",
                    help="what the running server actually took")
    ap.add_argument("--brief", action="store_true",
                    help="one line instead of the table — for a journal, where "
                         "a guard that passes silently cannot be told from a "
                         "guard that is not there")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    machine = read_machine()

    if a.cache:
        import glob as _g
        here = os.path.join(HERE, "..", "env")
        print("  %-11s %9s %11s %8s %9s" % ("profile", "-c", "KV/window", "-cram", "windows"))
        for env in sorted(_g.glob(os.path.join(here, "*.env"))):
            argv = llama_args(env)
            cram, kv, n = cache_windows(argv, declared_kv(env))
            if not cram:
                continue
            print("  %-11s %9s %9.2f G %6.0f G %9s"
                  % (os.path.basename(env)[:-4],
                     flag(argv, "-c", "--ctx-size") or "-", kv, cram,
                     "%.1f" % n if n else "?"))
        return 0

    if a.observe:
        got = observe(machine=machine)
        if got is None:
            print("no llama-server is running", file=sys.stderr)
            return 1
        if a.json:
            print(json.dumps(got, indent=2))
            return 0
        print("  observed, from the running server:")
        for k, val in got.items():
            print("    %-30s %s" % (k, ("%.2f" % val) if isinstance(val, float) else val))
        return 0

    declared, in_gtt, anon, what, args = None, None, None, "this server", None
    if a.from_env:
        # The unit's EnvironmentFile is already IN the environment here. Not
        # guessing a path is the point: the user unit reads %h/.claude/env/%i.env
        # and the system unit reads /etc/llm-profile/%i.env, and a guard that
        # had to know which would be a fourth place that knows about profiles.
        raw = os.environ.get("LLAMA_ARGS", "")
        if not raw.strip():
            print("no LLAMA_ARGS in the environment — nothing to weigh",
                  file=sys.stderr)
            return 0
        args = expand(raw).split()
        what = flag(args, "--alias") or os.environ.get("LLM_PROFILE", "this server")
        for name, setter in (("MODEL_KV_KIB_PER_TOKEN", "kv"),
                             ("MODEL_GTT_BASE_GIB", "gtt"),
                             ("MODEL_HOST_ANON_GIB", "anon")):
            v = os.environ.get(name)
            if v in (None, ""):
                continue
            try:
                val = float(v)
            except ValueError:
                raise SystemExit("%s is not a number: %r" % (name, v))
            declared, in_gtt, anon = ((val, in_gtt, anon) if setter == "kv" else
                                      (declared, val, anon) if setter == "gtt" else
                                      (declared, in_gtt, val))
    elif a.running:
        args = running_argv()
        if not args:
            print("no llama-server is running", file=sys.stderr)
            return 1
        what = flag(args, "--alias") or "the running server"
    else:
        name = a.profile or "qwen38"
        path = _profile_path(name)
        args = llama_args(path)
        declared = declared_kv(path)
        in_gtt, anon = declared_gtt(path), declared_anon(path)
        what = os.path.basename(path)[:-4]

    p = plan(args, weights_gib(args), declared, what,
             gtt_base=in_gtt, host_anon=anon)
    v = verdict(p, machine)
    if a.static:
        ok = fits_the_machine(p, machine)
        v = Verdict(True if ok is None else ok, [] if ok is not False else [
            "this profile needs %.1f GiB plus %.0f for the host, and the "
            "machine has %.1f" % (p.host_gib, host_reserve_gib(), machine.mem_total)],
            v.notes)

    # With --check the refusal below already carries the whole table, so
    # printing it here as well would state the case twice — once on stdout and
    # once on stderr, which is what a caller capturing 2>&1 then shows.
    show = not (a.check and not v.fits)

    if a.brief and not a.json:
        print(brief(p, machine, v))
        show = False

    if a.json:
        print(json.dumps({"what": p.what, "gtt_gib": p.gtt_gib,
                          "host_gib": p.host_gib, "estimated": p.estimated,
                          "fits": v.fits, "problems": v.problems,
                          "notes": v.notes,
                          "items": [i._asdict() for i in p.items],
                          "machine": machine._asdict()}, indent=2))
    elif show:
        print(render(p, machine, v))

    if a.check and not v.fits:
        if guard_disabled():
            print("  LLM_NO_MEMORY_GUARD=1 — starting anyway", file=sys.stderr)
            return 0
        print(refusal(p, machine, v), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
