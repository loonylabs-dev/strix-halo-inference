# Gemma 4 26B-A4B as the second model beside qwen36 — measured for the first time

**Measured 04.09.2026, 18:11-19:36**, all of it on this machine, all of it the
same evening, `platform_profile=performance` verified at both ends of the
sweep.

> **The recommendation is at the bottom and it is: `glm47flash`, not
> `gemma26`.** That is the opposite of what the previous report and the baton
> both expected, and it does not rest on speed — gemma26 is faster than
> production qwen36 on every prose cell, at two thirds of its memory, with
> Google's own drafter working exactly as hoped. It rests on one measurement:
> gemma26 loses facts from the middle of a long context, in four
> configurations, while both alternatives are perfect on the same suite the
> same day.

## What was asked

qwen36 is production since 17:44 on 04.09. and stays. Wanted beside it: a
sparring partner from a DIFFERENT family that criticises code and arguments.
It does not carry the interactive load, so "slower than qwen36" is not
disqualifying. The two candidates are `glm47flash` (measured yesterday, viable,
not chosen) and `gemma26` — installed since 17.08.2026 and, until this evening,
**never measured with this instrument at all**.

Production was used for reading only. Every measurement ran behind
`bench/sideserver.py`; nothing was switched.

## 0 · The profile was repaired before it was measured, and that is why

`setup/env/gemma26.env` was written in August, before this repo measured
anything, and carried six flags nobody had checked. Four were wrong in ways
that had nothing to do with Gemma:

| carried | why it is gone |
|---|---|
| `--port 8081` | `switch-model.sh` reads the port out of `LLAMA_ARGS` and **aborts** when it disagrees with the gateway's. The profile could not be switched to at all. `sweep.py` polls 8080 for the same reason. |
| no `-np 1` | two independent defects measured on this hardware at two slots (gfx1151 HIP race #27579, slot-restore poisoning). The default was a known-corrupting configuration. |
| `--kv-unified` | `docs/HANDOVER.md` → *do not try again*. |
| `-ctk q8_0 -ctv q8_0` | this repo's objection to q8 KV is a CORRECTNESS objection; it dropped qwen38's long-context task. |

The profile was **repaired rather than duplicated**. It serves nothing, so the
corrected serving shape changed nothing that was running, and a profile that is
simultaneously candidate and measurement object proved handy on 04.09. The two
open questions — `--swa-full` and the draft model — were measured rather than
decided.

## 1 · Google's draft model works, and it does NOT double the KV cache

This was the lever the whole exercise was about, and the one that decided
qwen36's morning for 0.3 GiB instead of 17.

**The route is not guessable from the flag name.** `--spec-type` has no
`gemma4-assistant` value — `common/speculative.cpp` lists eleven names and that
is not one of them. The assistant is driven by the `draft-mtp` implementation,
which carries three modes and picks the gemma one by `is_mem_shared`. Two
checks say the wiring took, and neither alone would:

* `Gemma4Assistant requires ctx_other to be set` (`src/llama-context.cpp:146`)
  appears in **no** load;
* GTT rose by **0.69 GiB**, not by a second copy of the 13.45 GiB target.

The second check matters because the first is ambiguous: the assertion only
fires for `LLM_ARCH_GEMMA4_ASSISTANT`, so a wrongly-loaded target model would
also be silent. **A source read that looked like an upstream bug was wrong and
is recorded as wrong**: `speculative.cpp:2545` loads `params.model.path` in the
`has_draft` branch, which reads like the target — but the caller builds
`params_dft` through `common_base_params_to_speculative()`, so that field *is*
the draft path. Measurement and source agree; there is no defect.

### Memory, three two-point pairs and one prediction

Every base computed from BOTH points of its pair independently, agreeing to two
decimals — that agreement is the check, not a coincidence.

| arm | -c 65536 | -c 262144 | KiB/token | base GiB |
|---|---:|---:|---:|---:|
| no drafter | 17.04 | 21.16 | **21.97** | 15.67 |
| **with the drafter** | 17.73 | 22.23 | **24.00** | 16.23 |
| `--swa-full`, no drafter | 29.37 | 43.37 (@131072) | **224.00** | 15.37 |

**The drafter costs +2.03 KiB/token and +0.56 GiB — not a second cache.**
GLM-4.7-Flash's MTP head measured 55.0 against 110.0 on this same machine seven
hours earlier, exactly double, because that path allocates a second full-size
KV. This one shares the target's, which `common/speculative.cpp:1337` documents
as `is_mem_shared`.

A seventh load checked the model rather than repeating it: `--swa-full` AND the
drafter together were **predicted at 30.06 GiB before the load** and measured at
**30.18**, +0.4 %.

The derivation from the GGUF config came closer than any before it on this
machine: 20.00 KiB/token predicted against 21.97 measured (+9.9 %), and 220.00
against 224.00 with `--swa-full` (+1.8 %). The old profile's declared 10.4 was
out by 2.1x and its note had the layer roles inverted — the 5 GLOBAL layers
carry 2 KV heads and cost 20.00 KiB/token; the 25 SLIDING ones carry 8 and would
cost 200.00 if they were ever allocated at full length.

## 2 · Backend and speculation: six cells, and the profile was accidentally right

`bench/reports/2026-09-04_1823_sweep_gemma26`, -c 65536, decode t/s on `prose`
— the workload no drafter can help, and the **only column with no spread
warning in any of the seven cells**.

| variant | d512 | d8192 | d36k | GTT |
|---|---:|---:|---:|---:|
| rocm-nospec | 42.0 | 37.7 | 34.7 | 15.9 |
| vulkan-nospec | 57.1 | 50.9 | 44.7 | 16.4 |
| rocm-ngram | 40.3 | 37.8 | 35.4 | 16.0 |
| vulkan-ngram | 55.1 | 51.2 | 41.5 | 16.5 |
| rocm-assistant | 86.8 | 64.4 | 47.9 | 16.4 |
| **vulkan-assistant** | **91.5** | **65.9** | **49.8** | 17.2 |
| qwen36 (production, live) | 42.7 | 39.0 | 32.9 | 28.4 |

* **Vulkan wins all six cells** — 1.36/1.35/1.29x on the floor and still ahead
  with speculation. Unlike qwen36 the answer does not split; like glm47flash it
  is one-sided. The profile has served build-vulkan since 17.08. on no
  measurement whatsoever and turns out to have been right.
* **The assistant is worth 1.60/1.29/1.11x** over the Vulkan floor at 67-86 %
  acceptance and 0.8 GiB. Read with yesterday's lesson: GLM's head accepted
  59-82 % and *cost* up to 42 %. Acceptance is not a speed prediction; this one
  earns its place on the t/s column.
* **The n-gram drafter is rejected** — -3.5 % / +0.6 % / -7.2 % on Vulkan.
* Against production qwen36 on prose: **2.14 / 1.69 / 1.51x, at 17.2 GiB of GTT
  against 28.4.**

The Vulkan arm is handicapped twice, both understating it: build-vulkan is
commit 54ee5ee of 22.08. against the ROCm arm's b10750-10, and no RADV APU heap
option is set on this machine. So that margin is a floor.

**`count` and `copy` are NOT comparable across the two models and the gap is
large** — qwen36 reads 213-282 t/s there against gemma26's 47-67. Every gemma26
cell in those columns carries speed.py's *"the model answered in the thinking
channel"* warning, so they are decode rates and not copy rates. The cause was
only found afterwards (section 3), and re-running the sweep with the flag is
what would settle whether qwen36 really is 3-4x on copy-heavy output — which is
the shape most agent OUTPUT takes.

## 3 · The model reasons past any budget unless it is told not to

Three requests, one after another, one server, temperature 0, max_tokens 4096:

| chat_template_kwargs | tokens | wall | visible answer |
|---|---:|---:|---|
| none — *what the profile did* | 4096 (cap) | **56.0 s** | **empty** |
| `enable_thinking=false` | **32** | **0.7 s** | correct |
| `enable_thinking=true` | 4096 (cap) | 54.5 s | empty |

**Unset behaves like `true`, not like `false`.** The profile's own note of
28.08. says the opposite — that unset and `enable_thinking=false` render the
same prefix, which is why no `none` mode exists. Two byte-identical prefixes
cannot produce 0.7 s against 56.0 s at temperature 0, so one of the two
measurements is wrong about what it measured. **Both are kept in the profile**;
the behavioural one is what a user experiences and is what the launch line acts
on. `/apply-template` would settle which.

The profile now carries `--chat-template-kwargs {"enable_thinking":false}`,
which qwen36 and glm47flash both carry, with `MODES` turning thinking back on
per request.

## 4 · `--swa-full` was measured and rejected — twice, for two different reasons

`setup/README.md`'s 100.2 s → 10.4 s was measured on **laguna**. The MECHANISM
transfers (gemma4's window is 1024 and Claude Code appends ~1624 tokens behind
the question, so the divergence is always outside it); the MAGNITUDE does not,
and here it would cost 24.96 GiB. Four requests on the real Claude Code body
(`tools/synthetic.py`), same head, changed question:

| | new | cache | reused | wall |
|---|---:|---:|---:|---:|
| without `--swa-full` | 1043 | 13354 | **92.8 %** | 2.2 s |
| with `--swa-full` | 971 | 13426 | **93.3 %** | 1.8 s |
| cold, either arm | 14397 | 0 | 0.0 % | 17.6 s |

**72 tokens and 0.4 seconds, for 24.96 GiB.** The work is already being done by
`-ctxcp 64 -cms 4096`, which this profile has carried since August. So gemma26
stays in `tests/test_models.py`'s `KNOWN_MISSING` — and now for a measured
reason rather than as an open gap.

It was rejected a second time in section 5, where it does not buy correctness
either.

*(What this does not say: the test repeats a single turn. Whether the reuse
survives a multi-turn conversation is untested, and `defects.json` →
`the-previous-answer-is-sometimes-not-reused` describes a checkpoint search
that would bear on it.)*

## 5 · depth-correctness — and this is the finding

Anchors planted every 4,000 tokens, each asked twice, six depths to 99,398 —
48 answers per run. **Four configurations**, because the suite's own docstring
says a WRONG is not proof until the other backend has been tried.

| configuration | ok/48 | first wrong at |
|---|---:|---:|
| Vulkan + assistant *(the profile's shape)* | 40 | 51,728 |
| Vulkan + assistant + `--swa-full` | 38 | 51,728 |
| Vulkan, **no drafter at all** | 40 | 51,728 |
| ROCm + assistant | 36 | **35,843** |
| — | | |
| **qwen36** (11:52, same suite, same day) | **48** | — |
| **glm47flash** (16:46, same suite, same day) | **48** | — |

**Every arm fails and both obvious suspects are exonerated.** Removing the
drafter changes nothing — 40/48 either way, same depths, same anchors.
`--swa-full` changes nothing and is one cell worse. ROCm is worse still and
fails a whole step earlier. So this is not speculation, not the sliding window,
and not the backend.

**The control was a profile copy, not an `--extra` override**, and that
mattered: `--spec-type` accumulates, so `--extra "--spec-type none"` would have
left `draft-mtp` running and produced a control that was not one. The copy's
argv names neither `-md` nor `--spec-type`.

**What makes it a defect rather than a hard test:** the same suite, the same
machine, the same day, is 48/48 clean on BOTH alternatives. This is not a test
everything fails.

**The failing anchor is almost always `middle`, and `first` is never wrong** —
the model keeps the start of the context and the end and loses the part in
between. Five global layers of 30 would predict that; nothing here establishes
it as the cause.

**Not separable from the quant.** This is Google's QAT-q4_0 and no second quant
of this model is on the disk, so "the model" and "this file" cannot be told
apart. QAT provenance makes the file the *less* likely of the two.

Filed as `setup/defects.json` → `gemma26-loses-the-middle-of-a-long-context`,
`shows_as: silent`. The profile caps `-c` at 32768, below the shallowest depth
at which the model has been observed wrong.

## 6 · What is NOT measured here

Quality. This repo removed its model battery on 26.08. and that decision
stands. The figures below are **published vendor numbers, read 04.09.2026, not
measured here, and not strictly comparable across sources**:

| (published) | active | GPQA-D | LiveCodeBench v6 | MMLU-Pro | AIME |
|---|---|---:|---:|---:|---:|
| qwen36 (production) | 3B | 86.0 | 80.4 | 85.6* | 92.7 |
| **Gemma 4 26B-A4B** | 3.8B | **82.3** | **77.1** | **82.6** | **88.3** |
| GLM-4.7-Flash | 3B | 75.2 | 64.0 | — | 91.6 |

\* from NVIDIA's own harness, the only like-for-like row.

**On this table gemma26 is the better critic**, and that was the whole argument
for measuring it first. The table is not what decides below.

## The recommendation

**`glm47flash` as the second model beside qwen36 — not `gemma26`.**

One reason, and it outweighs everything in gemma26's favour:

**gemma26 silently loses the middle of a long context and the alternative does
not.** 40/48 against 48/48, on the same suite, the same machine, the same day,
with the failure surviving every configuration change tried. For a sparring
partner this is not a performance characteristic, it is a correctness one: the
role is *hand it a large artifact and ask it to judge*. A critic that
confabulates what was in the middle of what you gave it produces exactly the
noise the role exists to avoid — and it does so **silently**, in well-formed
confident prose, which is the failure mode this repo hunts hardest.

What that verdict costs, stated plainly, because it is not nothing:

* gemma26 is **1.5-2.1x** production qwen36 on prose and **2.4-4.2x**
  glm47flash at the shallow end, at **17.2 GiB** of GTT against glm's 28.9;
* Google's drafter is a genuine, cheap, working lever — the only speculation
  win of the day;
* its published judgement scores beat GLM's on every shared axis, which is the
  axis the role actually cares about.

**Why that is still not enough:** the published gap is vendor numbers this repo
cannot verify, while the depth failure is this repo's own measurement, taken
four ways. Choosing a model that is better on unverifiable numbers and worse on
a measured correctness property is the wrong trade for a role whose entire
value is being right about something you hand it.

### What gemma26 IS good for, and it is not nothing

**Below ~32k tokens it is measured green and it is the fastest thing on this
machine.** The profile is capped there and stays registered. For focused
review — a file, a diff, a design note — that window is enough, and 91.5 t/s at
17.2 GiB is a better deal than anything else here. If the operator wants a fast
short-context critic in addition to a long-context one, this is it.

### What would change the verdict

* **A second quant of Gemma 4 26B.** The defect cannot be separated from the
  file. A different quant coming back 48/48 would move this from "the model" to
  "that file" and change everything.
* **A depth-correctness run on glm47flash and qwen36 at 32k-only**, to check
  that the comparison is not flattering them at a depth nobody uses.
* **Re-running the sweep with `enable_thinking=false`**, which would settle the
  `count`/`copy` columns that this report has to leave incomparable.

### Unchanged and still parked

**How "beside" is served** — switchable via `switch-model.sh`, or simultaneous
on two ports — remains the operator's decision and is untouched here. Both
candidates are registered profiles either way. gemma26's port is now 8080,
which is what makes it switchable at all; that was a repair, not a choice about
the arrangement.
