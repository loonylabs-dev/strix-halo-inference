# Qwen3.6-35B-A3B is 1.9 to 2.9 times faster than the incumbent on every cell measured, in under a third of the memory — and that settles less than it looks like

04.09.2026, 09:24–12:05. Ten `bench/speed.py` runs and one four-cell sweep,
three workloads at three depths each, `--reps 3`. Qwen3.6-35B-A3B behind
`bench/sideserver.py` on port 8081; Qwen3.8-Flash-Next measured live on the
production port in the shape it had been serving in all night. Production was
never switched: sideserver stopped it and put it back for every run.

**Read the sections in order — they were written in order, and each one moved
the answer.** The morning's shape (unsloth Q4, n-gram drafter) had one weak
column and the headline read "two to eight times", a range wide enough to be
suspicious. What closed it was the model's own MTP head, in the last section,
and what the head cost is a second unknown on the quality side.

## The question

The operator asked for a faster model at sufficient quality, having been
pointed at this one as the sweet spot; the priority was later stated as
throughput. Two things had to be found out, and only one of them is answerable
here: **is it faster on this machine**, and **is it good enough**. This report
answers the first and says as precisely as it can why it cannot answer the
second — and by the end it has made the second question harder rather than
easier.

## What it cost to support: nothing

The GGUF declares `general.architecture = qwen35moe` and the checkpoint
declares `model_type: qwen3_5_moe`. Qwen3.6 reuses Qwen3.5's graph — the
weights and the post-training changed, the tensors did not. Both builds in
service here already carry that architecture, checked on the **built
artifact** rather than on the source tree:

    strings build-rocm-patched-b10702-11-gc799f1014/bin/libllama.so | grep '^qwen35moe$'   -> 1
    strings build-rocm-cachelookup-b10750-10-g6bbb3eecf/bin/libllama.so | grep '^qwen35moe$' -> 1

A tree and a binary are different things, and the tree here sits on a feature
branch. That is why the check was run on the `.so`.

## Memory, two-point

`sideserver` at two windows, reading the server's own GTT per DRM client:

    -c  65536     GTT 21.90 GiB     RssAnon 0.24 GiB
    -c 262144     GTT 25.84 GiB     RssAnon 0.25 GiB
    -----------------------------------------------
    3.94 GiB over 196,608 tokens  =  21.01 KiB per token

The base — GTT with the KV taken out — computes to **20.59 GiB from each
point independently** (21.90 − 1.31 and 25.84 − 5.25). That agreement is the
check that neither point was taken in a disturbed state, and it is the reason
the figure went into the profile rather than a single-point estimate.

Two things worth keeping:

* The arithmetic done over `config.json` **before** the download said 20.0
  KiB/token — 5 % low. On this machine that is unusual: the same exercise for
  flashnext was 56 % low. The difference is that this architecture states its
  KV layout plainly (10 of 40 layers keep a real KV, `head_count_kv 2`,
  `key_length 256`), and the other thirty carry a per-sequence recurrent
  state that does not grow with the window.
* The base is **below the 20.82 GiB file size**, so about 0.23 GiB of the
  file is never pinned. Not investigated. It is the harmless direction and it
  is what the two-point method exists to absorb.

## Speed

Decode, t/s, median of three. Prefill in the second table.

| depth | workload | qwen36 nospec | qwen36 ngram | qwen36 ngram+p-min | flashnext (MTP) |
|---|---|---|---|---|---|
| 512 | prose | 35.7 | 35.4 | 35.2 | 18.6 |
| 512 | count | 35.4 | 197.9 | 202.9 | 112.3 |
| 512 | copy | 36.2 | 113.2 | **291.3** | 60.7 |
| 8192 | prose | 34.2 | 32.2 | 44.2 | 19.4 |
| 8192 | count | 34.2 | 188.3 | 219.7 | 104.8 |
| 8192 | copy | 34.2 | 268.0 | **281.5** | 103.4 |
| 36k | prose | 30.6 | 28.3 | 29.5 | 13.9 |
| 36k | count | 30.7 | 157.4 | 161.6 | 85.1 |
| 36k | copy | 30.7 | 94.6 | **220.3** | 86.6 |

Prefill, t/s — qwen36 (ngram+p-min) against flashnext:

| depth | prose | count | copy |
|---|---|---|---|
| 512 | 683.3 / 219.5 | 833.0 / 258.4 | 667.4 / 197.8 |
| 8192 | 935.2 / 290.4 | 750.1 / 229.0 | 587.9 / 191.4 |
| 36k | 662.1 / 213.9 | 443.4 / 159.7 | 387.3 / 140.6 |

GTT while serving: **26.61 GiB against 90.99.**

### The n-gram drafter is what makes the decode column true

Read down the *nospec* column: 30–36 t/s on all three workloads, flat. That is
the hardware with nothing hiding behind it, and against flashnext's *prose*
figure — the one workload where no drafter of any kind can help — it is
already 1.9–2.2×. Everything above that comes from `--spec-type ngram-mod`,
which drafts from the prompt and needs no second model. The model's own MTP
head would have been the better drafter, and it is not in the file: unsloth
converts with `--no-nextn`, confirmed here on the file itself (`gguf_dump
--no-tensors`, not one key matching `nextn` or `mtp`).

**Which column decides this is not a matter of taste.** `copy` is a block
reproduced with one substitution — the shape of most of what a coding agent
emits — and it is the workload an n-gram drafter can help. `count` is
predictable but absent from the prompt. `prose` is the floor. Reading only one
of them is how this repo once recorded *"the ngram drafters give nothing"*
from a probe that could not have shown one working (bench/README.md).

`--spec-draft-p-min 0.75` is what keeps prose from paying for the drafter.
Without it the drafter also runs where it cannot win — acceptance 3.9 % at
d8192, 0.0 % at d36k — and prose loses 1–7 %. With it, prose acceptance rises
to 7–14 % and the column is level with nospec, while `copy` at 36k goes from
94.6 to 220.3.

## How much of this to believe

Six of the eighteen qwen36 cells and five of the nine flashnext cells carry
`speed.py`'s own spread warning; the widest is `copy` at d512, which ranged
87.3–294.3 t/s over three reps. Decode on this machine has a **19 %
coefficient of variation**, so nothing under ~22 % is resolvable at n=3.

The *directions* here are two- to eight-fold and safely outside that. The
individual numbers are not. One cell to distrust specifically: **prose d8192 =
44.2**, which sits above the no-speculation figure it should at best have
matched — read it as "no loss", not as a gain.

### Three ways this is not a clean A/B, all of them stated rather than fixed

1. **Different binaries.** qwen36 ran on `build-rocm-patched-b10702-11-gc799f1014`;
   flashnext runs on its pinned `build-rocm-cachelookup-b10750-10-g6bbb3eecf`.
   That is deliberate — the comparison asked is "what the machine delivers
   today against what it would deliver", not "these two models on one build" —
   but it is a confound and a build difference has been worth 25 % here before.
2. **Different surroundings.** qwen36 sat behind sideserver with `-cram 32768`
   and no gateway attached; flashnext was measured live with `-cram 12288`,
   `--slot-save-path` active and the gateway in front of it. The prompt-cache
   settings differ; the workloads are sent cold, so this should not reach the
   numbers, and it was not controlled for.
3. **Different windows**, 262,144 against 240,640. Both near their own
   maximum, and the depths measured are far below either.

### And one measurement was contaminated by this session itself

The four-cell backend sweep started at 09:41 had `bash tests/run.sh` run
against it while its first variant was measuring — the contention this repo's
own rules forbid. The gate took 23.2 s where it takes 17.6 s idle, which is
the evidence that the contention was real. The affected cell is
`rocm-nospec` at d8192 and d36k. It is recorded rather than repeated.

## What is NOT measured here, and it is the question that matters

Nothing above is a statement about quality. This repo removed its model
battery on 26.08.2026 rather than repair it, and that decision stands (see the
note at the top of bench/README.md): other people benchmark model quality at
more scale than a home-grown battery survives.

What the published figures say points the **other way**:

| | Qwen3.6-35B-A3B | Qwen3.8-Flash-Next |
|---|---|---|
| SWE-bench Multilingual | 67.2 | 81.0 |
| SWE-bench Pro | 49.5 | 62.5 |
| active parameters / token | 3B | 6B |

Vendor and third-party numbers, read 04.09.2026, **not measured here**, and
gathered from different sources — treat the gap as a direction, not as a
margin. The 3B against 6B is the same fact from both sides: it is why this
model is twice as fast, and it is plausibly why it is weaker at the job.

So the finding is a **trade, not an upgrade**: roughly twice the speed at a
quarter of the memory, for a model the published evidence says is the weaker
coder. Whether that trade is worth taking is an operator judgement on real
work, and this report deliberately does not make it.

## The backend, measured after this report was first written

`bench/variants/qwen36.json`, 2×2 at `-c 65536`, `platform_profile=performance`
verified unchanged at both ends —
`bench/reports/2026-09-04_0942_sweep_qwen36`. Decode, t/s:

| | prose d512 / 8k / 36k | count d512 / 8k / 36k | copy d512 / 8k / 36k |
|---|---|---|---|
| rocm-nospec | 38.1 / 35.3 / 30.8 | 47.6 / 35.2 / 31.6 | 36.7 / 34.3 / 30.7 |
| vulkan-nospec | 55.1 / 51.5 / 44.9 | 54.1 / 51.3 / 44.9 | 54.0 / 51.3 / 45.0 |
| rocm-ngram | 39.0 / 39.4 / 36.0 | 223.3 / 212.2 / 178.2 | 105.3 / 285.6 / 233.0 |
| vulkan-ngram | 57.0 / 48.8 / 41.4 | 213.0 / 203.4 / 172.6 | 108.9 / 226.8 / 112.1 |

**The answer is split, and both halves are worth keeping.**

*Without speculation Vulkan wins by 1.45×*, and that is as solid as anything
in this report: the factor repeats to two decimals at all three depths
(1.45 / 1.46 / 1.46), neither cell carries a spread warning, and 45 % is twice
the resolution floor. Prefill ties at d512 and Vulkan is +16 % at d36k.

*With the drafter the order inverts on the column that matters.* `copy` at
depth goes to ROCm — 285.6 against 226.8 at d8192, 233.0 against 112.1 at
d36k — and that is also the more trustworthy half, since neither ROCm cell
carries a warning while Vulkan's d36k copy ranged 90.1–197.8 t/s. Vulkan keeps
`prose` and prefill.

So ROCm stays in the profile, on the workload the machine is for. It is the
weakest-held line in `setup/env/qwen36.env`, and **two things handicap the
Vulkan arm, both in the same direction**: `build-vulkan` is commit `54ee5ee`
of 22.08. against the ROCm arm's 30.08. build, and the RADV APU heap option is
not set on this machine — there is no `drirc` anywhere, checked, on Mesa
26.1.7 where the option exists. Setting it changes GPU memory behaviour for
every Vulkan client including the desktop, on a machine that has frozen three
times over GTT, so it is an operator decision.

### One cell of that sweep was contaminated by this session

`rocm-nospec` had `bash tests/run.sh` running against it at d8192 and d36k —
the contention this repo forbids. The gate took 23.2 s where it takes 17.6 s
idle, which is the evidence that it was real. Recorded rather than repeated.
It is the cell Vulkan is compared *against*, so if it is depressed, the 1.45×
is an overstatement — but the same 1.45× appears at d512, which was measured
before the gate started.

## What the operator's preparatory handover got right, and where it does not carry here

The operator brought a Strix Halo integration handover written by another
model. Checked against this morning's measurements, claim by claim.

**Right, and it changed what this repo does.** Vulkan over ROCm — its table
said 53.5 against 44.6 t/s and the direction is confirmed here at 1.45× on the
unspeculated floor. Without it the backend question would have stayed open.
Right too: 3B active parameters make decode bandwidth-bound; and its quality
placement of the model *below* Flash-Next, which matches the published figures
above.

**Its performance targets were already met before it was read.** It asks for
decode 25–50+ t/s and prefill 600–1000+; this morning's Q4 + n-gram + ROCm
measured 29.5–291 decode and 387–935 prefill.

**Where it understates the incumbent.** It describes Flash-Next as "nur 10–20
tok/s Decode". Measured here the same morning that is true only of `prose`
(13.9–19.4); with its MTP head it does 85–112 on `count` and `copy`. The case
for switching is real, and smaller than that framing makes it.

**What must not be copied.** Its launch line carries `-np 3 --kv-unified`.
Two slots on this GPU are a **measured defect**, not a preference: the gfx1151
HIP race (llama.cpp #27579, root cause #27572) degenerates every answer to
`////`, and slot restore at two slots was corrupt 4/4 with a populated store.
The matrix is in `setup/env/qwen38.env`.

**Where it contradicts this repo, and neither side is settled for this model.**
It recommends `-b 4096 -ub 4096`; this repo measured `-ub 512` beating 2048 by
25 % at d32768 on qwen38. Different model, and their figure comes from
`llama-bench` rather than from a serving profile. It reports q8_0 KV at +151 %
at 64k; this repo's objection to q8 KV is a *correctness* one — it dropped the
long-context task on qwen38 — so the speed claim cannot be acted on before
that question is re-asked here. Both are one `flag-ab` run away.

**Not applicable rather than wrong.** Its warning against an external draft
model (−60 % on this platform) is about a second loaded model; `ngram-mod`
loads none, and it is what produced the 2–8× above.

**The one experiment it names that this report cannot yet answer** is
`UD-Q8_K_XL` as the quality/speed sweet spot. Its Vulkan Q8 figure (53.5 t/s)
sits almost exactly on the Q4 Vulkan figure measured here (55.1), which would
mean the larger quant costs very little — and that is the single thing that
could answer the quality objection without leaving this machine. Fetched
04.09.2026; measurement pending.

## The quant, measured — and it changes the verdict above

Written after the rest of this report. The one experiment that could reach the
quality objection without leaving this machine was the larger quant, and it
came out better than the hypothesis.

**One variable, same window, same flags, same build** —
`bench/reports/2026-09-04_1048_speed_qwen36-q8-rocm` against
`…_0937_speed_qwen36-ngram-pmin075`:

| | Q4 | Q8 | Q8/Q4 |
|---|---|---|---|
| prose d512 | 35.2 | 33.4 | 0.95 |
| prose d36k | 29.5 | 27.5 | 0.93 |
| count d36k | 161.6 | 155.3 | 0.96 |
| copy d8192 | 281.5 | 258.9 | 0.92 |
| prefill d512 | 683.3 | 635.5 | 0.93 |
| prefill d36k | 662.1 | 609.5 | 0.92 |

**Doubling the weights costs 5–8 %, not half.** The received wisdom on this
machine — decode is weight-bandwidth-bound, so the quant is the speed knob —
is not wrong in general and does not reach this model: 3B of 35B parameters
are active per token, so the weight traffic is not what decode waits for here,
and where speculation accepts 100 % several tokens leave one forward pass
anyway.

One cell disagrees loudly and is not explained: `copy` at d36k reads 220.3 at
Q4 and 91.1 at Q8. Neither carries a spread warning, but the same Q4 cell read
94.6 in the ngram-only run an hour earlier, so it is a volatile cell rather
than a quant effect. Not resolved; recorded.

So the profile serves **Q8_K_XL**, and at that quant it is still 1.6–4.6×
the incumbent's decode and ~2.9× its prefill, in 40.8 GiB of GTT against 91.0:

| depth | prose | count | copy | prefill |
|---|---|---|---|---|
| 512 | 33.4 / 18.6 | 189.1 / 112.3 | 276.7 / 60.7 | 635.5 / 219.5 |
| 8192 | 31.9 / 19.4 | 180.0 / 104.8 | 258.9 / 103.4 | 855.9 / 290.4 |
| 36k | 27.5 / 13.9 | 155.3 / 85.1 | 91.1 / 86.6 | 609.5 / 213.9 |

**What this does and does not settle.** It removes the quantisation from the
quality question: the model now runs at near-checkpoint fidelity, so whatever
it gets wrong is the model's and not this machine's storage of it. It does
**not** close the gap in the published table above — that is about 3B active
parameters against 6B, and no quant reaches it.

Vulkan at Q8 repeats the sweep's split rather than resolving it: prose 43.4 /
34.4 / 30.2 against ROCm's 33.4 / 31.9 / 27.5, and count and copy decisively
to ROCm (189.1 against 103.0, 276.7 against 78.0 at d512). Second quant, same
verdict.

## The MTP head, found and measured — and it is what closes the last gap

Written last. `havenoammo/Qwen3.6-35B-A3B-MTP-GGUF` turned out to publish two
different things, and only one of them works.

**The sidecar does not.** `35BA3B-MTP.gguf`, 0.90 GB, needed ten metadata keys
added before it would even parse its hyperparameters and then died on a
missing `token_embd.weight` — which no metadata repair reaches, because
`common_speculative_init_from_params` loads a draft GGUF as a standalone
model. The contrast that proves the mechanism rather than arguing it: this
repo's one working head, `mtp-Qwen3.8-Flash-Next-Q8_0-drluoto.gguf`, is
4.14 GB with 37 tensors and *does* carry `token_embd.weight`. Registered as
`qwen36-sidecar-mtp-head-has-no-embeddings`, with the one-line check that
answers it in seconds.

**The full model does.** `Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf` carries the head
inside the main GGUF, so `--spec-type draft-mtp` finds it in the model it is
already serving and no `-md` is involved — the shape qwen38 has used since
August. Structure verified from the file's first 16 MB before spending 23 GB
on it: 753 tensors, `nextn_predict_layers 1`, `block_count 41`, four
`blk.40.nextn.*`.

**What it is worth**, one variable — same file, same binary, same window,
`draft-mtp` added and nothing else
(`…_1152_speed_qwen36-mtpq4-rocm-mtp` against `…_1154_…-ngramonly`):

| | ngram only | + MTP head | draft accepted |
|---|---|---|---|
| prose d512 | 36.0 | 44.9 | — → 83.3 % |
| prose d8192 | 33.0 | 37.8 | 5.5 % → 45.2 % |
| prose d36k | 29.0 | 37.3 | 4.7 % → 49.2 % |
| count d36k | 162.1 | 212.7 | |
| copy d8192 | 265.7 | 263.8 | |

**The acceptance column is the finding, not the t/s.** An n-gram drafts *from
the prompt*, so on novel text it has nothing to offer; the model's own head
predicts. That is why `prose` — the floor of everything this machine does, and
the one column speculation could not previously touch — gains 15–29 %, and
`count` 31–36 %. `copy` is a wash where its cells are trustworthy; both its
other cells carry a spread warning on the control side, so no factor should be
read off them. The head costs about 1.9 GiB of GTT and about 4 % of prefill.

**It did not hang**, which is not a formality: `setup/defects.json` carries
`flashnext-mtp-serving-shape-hangs`, where this kind of shape produced zero
tokens in 39 minutes at 97 % GPU. That entry is scoped to `qwen4exp` and to
the pre-#27941 memory path, so it proves nothing here — it is a reason not to
find out with a serving profile, and why the measurements used the post-#27941
build flashnext already serves MTP from.

### So the final shape, against the incumbent

| depth | prose | count | copy | prefill |
|---|---|---|---|---|
| 512 | 44.9 / 18.6 | 265.0 / 112.3 | 171.7 / 60.7 | 636.6 / 219.5 |
| 8192 | 37.8 / 19.4 | 261.8 / 104.8 | 263.8 / 103.4 | 854.5 / 290.4 |
| 36k | 37.3 / 13.9 | 212.7 / 85.1 | 215.1 / 86.6 | 589.2 / 213.9 |

**1.9× to 2.9× on all twelve cells, at 27.7 GiB of GTT against 91.0**, and
with no weak column left.

Vulkan was measured with the head too, and the split holds a third time: it
takes prose at shallow and mid depth (53.5 and 45.0 against 44.9 and 37.8) and
prefill at d36k, ROCm takes count and copy everywhere. Same verdict as the Q4
sweep and the Q8 pair.

**And the quality question is now worse rather than better.** The file that
made this possible is a third-party quant with no published imatrix, stacked
on a model whose published agentic-coding scores already sit below the
incumbent's. `suites/depth-correctness.py` has still never run against this
model, and that is now the first thing to spend time on rather than the last.
The clean way out is to convert the checkpoint here: `conversion/qwen.py:636`
registers this exact architecture and its chain carries
`supports_mtp_export = True`, so the head unsloth drops can be kept locally,
with provenance. Neither the conversion nor the serving has been tried.

## Sources

    bench/reports/2026-09-04_0924_speed_qwen36-c262144        nospec, and the memory two-point
    bench/reports/2026-09-04_0927_speed_flashnext-live-04.09  the incumbent, same morning
    bench/reports/2026-09-04_0934_speed_qwen36-ngram          ngram-mod alone
    bench/reports/2026-09-04_0937_speed_qwen36-ngram-pmin075  ngram-mod + p-min 0.75
    setup/env/qwen36.env                                      every figure above, beside the flag it justifies
