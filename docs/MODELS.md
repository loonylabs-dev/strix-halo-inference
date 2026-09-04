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
`qwen38-low`, `qwen38-medium` (the level suffixes are the API's standard
vocabulary; whichever profile serves uses the same scheme, e.g.
`flashnext-low`). One model stays loaded, and the
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
| `qwen36` | Qwen3.6-35B-A3B — **production since 04.09.2026.** Speed measured, quality not. See below |
| `flashnext` | Qwen3.8-Flash-Next — production 01.-04.09.2026, kept as the way back (`switch-model.sh flashnext`) |
| `qwen38` | coding agent, vision, judge — production until 01.09.2026, kept as the way back |
| `glm47flash` | GLM-4.7-Flash-30B-A3B — measured 04.09.2026 as a second-family sparring partner. 28.9 GiB, depth-correctness clean. See below |
| `gemma26` | Gemma 4 26B-A4B — measured 04.09.2026. The fastest thing here below 32k and **capped there**, because it loses the middle of a longer context. See below |
| `gemma31` | prose, 16.4 GiB |
| `gptoss` | judge and evals, 59.0 GiB, very cheap KV — never measured with bench/speed.py |
| `laguna` | the predecessor, kept as the way back |
| `batch` | batch classification, small window |

### Qwen3.6-35B-A3B: measurably faster, and that is only half a verdict

Asked for on 04.09.2026 as a faster model at sufficient quality. The speed
half came out decisively and the quality half was never in this repo's gift,
so it stays a candidate rather than a replacement.

**It cost nothing to support.** The GGUF declares `general.architecture =
qwen35moe` — Qwen3.6 reuses Qwen3.5's graph and changes the weights and the
post-training, not the tensors — so the builds already here load it. That was
checked against the built `libllama.so`, not against the source tree, because
a tree and a binary are different things.

**Against the incumbent, same morning, same three workloads, same depths.**
flashnext measured live in its own production shape (MTP head, `-c 240640`),
qwen36 behind `sideserver` at its own window. Decode, t/s —
qwen36 / flashnext:

| | d512 | d8192 | d36k |
|---|---|---|---|
| prose | 44.9 / 18.6 | 37.8 / 19.4 | 37.3 / 13.9 |
| count | 265.0 / 112.3 | 261.8 / 104.8 | 212.7 / 85.1 |
| copy | 171.7 / 60.7 | 263.8 / 103.4 | 215.1 / 86.6 |
| prefill | 636.6 / 219.5 | 854.5 / 290.4 | 589.2 / 213.9 |

**1.9× to 2.9× on all twelve cells, with no weak column**, at **27.7 GiB of
GTT against 91.0** — one token of context costs 21.0 KiB here rather than
37.3, measured two-point with both points agreeing on the base.

**Two findings did that, and neither was the obvious one.**

*The model's own MTP head.* unsloth converts with `--no-nextn` and ships every
quant without it, which left an n-gram drafter as the only speculation
available — and an n-gram drafts *from the prompt*, so on novel text its
acceptance measured 4.7–5.5 % and prose sat at 29–35 t/s while copy was
already past 200. A publisher who keeps the head (`havenoammo`) closes exactly
that gap: acceptance on prose rises to 45–49 %, prose gains 15–29 %, count
31–36 %. It is inside the main GGUF, so no `-md` is involved. **The price is
provenance** — that quant publishes no imatrix, so the file is trusted for its
structure, which was read off it, and not for its calibration, which cannot be
checked.

*The quant is nearly free.* Every other profile here serves Q4 on the received
wisdom that decode is weight-bandwidth-bound. Measured on this model, one
variable: **Q8 costs 5–8 %, not half** — 3B of 35B parameters are active per
token, so weight traffic is not what decode waits for. Q4 is what runs, because
the operator asked for throughput; the swap is one line and both files are on
disk.

Six of those eighteen cells carry `speed.py`'s own spread warning, and decode
on this machine has a 19 % coefficient of variation. The *directions* — two-
to eight-fold — are far outside that. The individual numbers are not, and
`prose d8192 = 44.2` in particular should be read as "no loss" rather than as
a gain, since it sits above the no-speculation figure it should at best have
matched.

**What makes it fast is also what makes it suspect.** 3B active parameters per
token against Flash-Next's 6B is the reason decode is twice as quick — and
published agentic-coding figures put the model below the incumbent on the same
axis: SWE-bench Multilingual 67.2 against 81.0, SWE-bench Pro 49.5 against
62.5 (vendor and third-party numbers, read 04.09.2026, **not** measured here,
and not strictly comparable across sources). This repo removed its model
battery on 26.08. rather than repair it, and that decision stands: whether the
answers are good enough is an operator judgement on real work.

So the honest summary is a trade, not an upgrade: **roughly twice the speed at
a quarter of the memory, for a model the published evidence says is weaker at
the job.** `switch-model.sh qwen36` is one command, and `switch-model.sh
flashnext` is the way back.

## Not only language models any more

Since 01.09.2026 the same machine also renders images, speaks, and films,
under the same memory authority: `setup/workloads/*.env`, started only
through `bench/sideserver.py --workload`, benched and judged by the
`bench/*bench.py` / `bench/*check.py` pairs. Measured that day (n=3 each,
idle machine): text-to-image — flux-schnell 56 s, sdxl 112 s, qwen-image
409 s per 1024×1024 image; text-to-speech — qwen3-tts 2.65× realtime on
Vulkan (German), chatterbox-multilingual 0.29× realtime on CPU torch with
voice cloning; text-to-video — wan2.1-1.3b ~9 min per 2 s clip at 480p
(the 5B figures are in its profile). See the workload-registry section of
[../setup/README.md](../setup/README.md) — and `media/README.md` for the
border that keeps the base install torch-free.

### A second model beside production, and why the fast one lost (04.09.2026)

qwen36 serves; what was wanted beside it is a sparring partner from a DIFFERENT
family that criticises code and arguments. It does not carry the interactive
load, so "slower than production" is not disqualifying. Two candidates were
measured the same day, both behind `bench/sideserver.py`, and the verdict is
`glm47flash`.

| | glm47flash | gemma26 |
|---|---|---|
| prose t/s, d512 / d8192 / d36k | 63.3 / 36.1 / 21.7 | **91.5 / 65.9 / 49.8** |
| GTT | 28.9 GiB | **17.2 GiB** |
| depth-correctness | **48 / 48** | 40 / 48 |
| published GPQA-D / LCB v6 | 75.2 / 64.0 | **82.3 / 77.1** |

**gemma26 wins everything except the row that decides the role.** Its
long-context failure is measured four ways — with the drafter, without it, with
`--swa-full`, and on the other backend — and it survives all four, first wrong
at 51,728 tokens (35,843 on ROCm). qwen36 and glm47flash are 48/48 on the same
suite the same day, so this is not a test everything fails. The failing anchor
is almost always the one planted in the MIDDLE; the first one never is.

A critic is handed a large artifact and asked to judge it. One that silently
loses the middle of what you gave it produces noise instead of disagreement —
so the seat goes to the model with the measured correctness, not the one with
the better vendor scores. `setup/defects.json` →
`gemma26-loses-the-middle-of-a-long-context` carries the numbers;
`bench/reports/2026-09-04_1905_gemma26-vs-qwen36/REPORT.md` carries the day.

**gemma26 stays registered and capped at `-c 32768`**, below the shallowest
depth at which it has been observed wrong. There it is measured green and it is
the fastest model on this machine — Google publishes a purpose-built 0.30 GiB
draft model for it which is worth 1.6x and, unlike GLM's MTP head, does not
allocate a second KV cache.

**Neither is served yet.** Both are registered profiles; whether "beside" means
`switch-model.sh` between them or two servers on two ports is a separate
decision that has not been taken.

### How Flash-Next became production (and why it took a week)

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

**The hang that stopped it is resolved — it belonged to the old base.** The
exact serving shape — `draft-mtp,ngram-mod` — generated zero tokens in 39
minutes at GPU 97 % on one depth-correctness question, found by the last
gate before switching. Five isolation experiments on 01.09. pinned it to
`--spec-type draft-mtp` on the pre-#27941 memory path: on master b10743 +
the same PR commits (build `b10743-15-g62850522e`) the identical shape
answers that question, correctness holds 16/16 to 52k, decode at depth is
2-3x the old build's, and the answer-keeping patch holds (the full matrix:
`setup/defects.json` → `flashnext-mtp-serving-shape-hangs`).

**It serves since 01.09.2026 ~21:4x**, on the operator's go: pin moved to
the m2 build, RssAnon re-measured on exactly that binary (0.31 GiB — the
#27837 lazy path is native), then `switch-model.sh flashnext` — which on
its first real run surfaced and got fixed a preflight bug of its own (it
weighed the file size instead of the measured figures; red→green in
tests/test_models.py). Verified serving: the m2 binary behind the unit,
smoke through the gateway, RssAnon 0.53 GiB warm.

    python3 setup/lib/budget.py --profile flashnext

still prints the arithmetic that guards every start. qwen38 stays one
`switch-model.sh qwen38` away.

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
