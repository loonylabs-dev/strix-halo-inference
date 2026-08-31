# Which model, and which settings

The short answer. Everything here was measured on one machine (Ryzen AI Max+
395, 128 GB, Fedora) and the numbers say nothing about any other.

    bash setup/get-model.sh --list       what exists, what is already here
    bash setup/get-model.sh qwen38       fetch it
    bash setup/switch-model.sh qwen38    serve it

## Start with qwen38

| | |
|---|---|
| **Model** | Qwen 3.8 27B, `UD-Q4_K_XL`, 16.7 GiB |
| **Backend** | ROCm, with MTP/n-gram speculation |
| **Thinking** | off by default |
| **Why** | it fits with room to spare, and it is the fastest thing here that also answers correctly at depth |

### It replaced Laguna S 2.1 on 25 August — and on what

The whole task battery, one machine, one night. Raw data in
`bench/reports/2026-08-24_2251_sweep_qwen38/` (adjudication in its `NOTES.md`)
and `…_2350_sweep_qwen38/`:

| | battery total | output tokens | weights |
|---|---|---|---|
| **qwen38** (ROCm, nothink, spec) | **360.8 s** | 4,801 | **16.7 GiB** |
| Laguna S 2.1 (production flags) | 405.3 s | 6,396 | 68.4 GiB |

**It is the time and the memory that decide it, and not the pass rate.** The
decision log used to lead with "9 of 9 against 8 of 9", and that margin does
not hold: `bench/README.md` says why — a single run with speculation on is not
a statement about correctness — and the log's own footnote already recorded
that Laguna's one failure was a CHECKER artefact, adjudicated and fixed. The
margin was never real. The 45 seconds and the 51 GiB are.

**And the instrument is gone.** `bench/quality.py` and the coding battery were
removed on 26 August (see below), so this comparison cannot be re-run or
extended to a new model. The numbers above stand as a dated record of one
night; the raw data behind them stays in `bench/reports/`.

**What this repo does not do is benchmark model intelligence.** Other people do
that, at more scale and with more scrutiny than a home-grown battery could
survive, and such a battery ages badly the week a new model lands. What is
measured here is the STACK on this hardware: what fits, what it costs in
memory, how fast it prefills and decodes at a given window — and whether the
answers stay *correct* as that window fills, which is a property of the build
and the flags, not of the model.

## Q4 is the floor, and that is measured

| | KL divergence | top-1 | what it costs |
|---|---|---|---|
| **UD-Q4_K_XL** | **0.0096** | 96 % | "indistinguishable from the full model" |
| the Q3 tier | ~0.03+ | — | visible loss |
| Q3_K_XL | — | — | ~2x as verbose thinking → **35-50 % slower to a finished answer** |

**A smaller quant is not "worse but cheaper" on a reasoning model.** It talks
itself to the same answer more slowly, and the metric that decides here is
seconds until something CORRECT comes out — so Q3_K_XL is the smaller file and
the slower model at once.

No tool in this repository enforces that floor — `setup/scripts/scout.py`
answers whether a model loads, how big it is and whether it fits, which are
different questions. The floor is a finding you apply yourself when you pick a
quant.

## The backend choice belongs to the model, not to the machine

* **Qwen + speculation → ROCm.** Vulkan doubles the time for identical token
  counts.
* **Anything without speculation → Vulkan.** It wins decode by 4–31 %; ROCm
  wins prefill by 16–31 %.

Five of the seven profiles here run Vulkan for that reason. This is set per
profile in `LLAMA_BIN`, so you do not choose once for the machine.

## Thinking: `low` is the sweet spot, and more is not better

Measured on the hard battery, five tasks:

| Mode | Solved | Median | Note |
|---|---|---|---|
| nothink | 4/5 | 21.8 s | fails one SQL window function — a real model error |
| **low** | **5/5** | 53.6 s | solves everything, +47 s over the whole battery |
| medium | 4/5 | — | talked itself out of an 8,192-token budget on a task nothink solved |

So: **nothink for everyday work, `low` when something is actually hard.**
`medium` is not "more careful", it is a different failure mode.

Switching costs nothing. In Claude Code pick the model name — `qwen38`,
`qwen38-think` (low), `qwen38-deep` (medium). One model stays loaded, and the
prompt cache survives the switch **100 % warm**: 71 s cold becomes 0.2–0.3 s.

## What the speed depends on

Decode spreads by how repetitive the output is, because that is what
speculation feeds on. Measured 27.08. with `bench/speed.py`, medians of three
runs, at three KV depths:

    free prose              9.3 – 7.2 tok/s     shallow -> ~36k deep
    predictable output    124.3 – 78.6
    warm repeated edits   160.0 – 98.8

**An agent editing a file it has already seen is in the bottom row, and that
is the workload this stack is tuned for.** The spread between top and bottom
is what speculation is worth here: **13.7 – 17.2x**.

Read all three or none. A single "decode rate" for this machine does not
exist — it is a property of the configuration AND the workload together, and
these three differ by a factor of seventeen. The earlier version of this table
gave 7.9-12 / 20-41 / 86.8 from a probe that measured tokens over TOTAL
request time, so prefill sat inside every decode number.

## The other profiles

| Name | For |
|---|---|
| `qwen38` | coding agent, vision, judge — production |
| `gemma26` | fast sidekick, 13.5 GiB |
| `gemma31` | prose, 16.4 GiB |
| `gptoss` | judge and evals, 59.0 GiB, very cheap KV |
| `laguna` | the predecessor, kept as the way back |
| `batch` | batch classification, small window |
| `flashnext` | Qwen3.8-Flash-Next — **runs, not served.** See below |

### Why Flash-Next runs and is not served

It is not slow, and since 31 August it no longer runs out of memory either.
Both blockers of the first week fell to measurements:

**The memory wall is gone.** The ~27 GiB of anonymous host memory the n-gram
table used to occupy became demand-paged page cache when llama.cpp **#27837**
merged (30.08.): RssAnon 0.31 GiB instead of 27.1, re-measured here on the
build that serves. On a current master build the model answers anchor and
tool questions correctly through a 51.7k-token window, 16 of 16 cells
(`bench/reports/2026-08-31_1943_depth-correctness_flashnext-masterA/`).

**The decode gap closes when the MTP head drafts.** The checkpoint ships a
speculation head that public GGUFs strip; with it exported as a sidecar and
llama.cpp PR **#27836** (open, previewed here as a pinned build), a
one-variable pair on the same build measured decode 16→100 t/s on copy at
shallow depth, 12→49 at 37k, prose +6 to +17 % and never negative —
acceptance 42-100 % by workload
(`bench/reports/2026-08-31_2149…/` and `…_2158…/`).

**What stops it today is a hang, not a wall.** The exact serving shape —
that build plus `draft-mtp,ngram-mod` — generated zero tokens in 39 minutes
at GPU 97 % on one depth-correctness question, found by the last gate before
switching. Until that is isolated (suspects and the two experiments are in
`setup/env/flashnext.env`), the profile says DO NOT SERVE, and

    python3 setup/lib/budget.py --profile flashnext

still prints the arithmetic that guards every start. qwen38 keeps serving.

---

**The raw data behind all of this** is under
[measurements/](measurements/README.md) and in `bench/reports/` — every task,
pass or fail with its reason, wall time and token count, per variant, in JSON.

There used to be a 294-line German decision log in front of it. It was deleted
on 27 August: its headline margin was retracted, its instrument
(`bench/quality.py`) no longer exists, and its opponent is a disabled profile
kept as the way back. What survived is the table above. The evidence never
lived in that prose — it lives in `bench/reports/`, and it is still there.

To measure a different model or a different setup on this hardware, see
[../bench/README.md](../bench/README.md): windows, token speeds, and the
correctness probes that guard the stack.
