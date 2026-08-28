# Defect 1 on gfx1151: reproduced, isolated, and located in one upstream commit

Night of 27./28.08.2026. Started as "find any configuration in which defect 1
appears at all", because two sessions had concluded it no longer reproduced.
It does. Every run below is on a side server; production served throughout
except where noted.

## The answer in four lines

* **It reproduces deterministically** — and the recipe that hid it had hidden
  it once before, two days earlier.
* **Necessary: two slots AND two concurrent requests.** The tool block is an
  amplifier, not a requirement. Speculation, `--no-kv-unified` and `-cram` are
  irrelevant.
* **`setup/patches/hip-integrated-off.patch` fixes it, and so does llama.cpp
  PR #27311** — separately, each against the same upstream commit.
* **Inside that PR it is ONE commit**, `3181ed701`, and switching just that
  commit's behaviour off with the author's own environment variable brings the
  corruption straight back on the same binary.

## Configuration

Qwen3.8-27B UD-Q4_K_XL · `-ngl 999 -fa on -c 32768 -np 2 --no-kv-unified
-b 2048 -ub 2048 -cram 32768` · speculation on · AMD Ryzen AI Max+ 395
(Strix Halo, gfx1151), 128 GB unified, ROCm, GTT cap 108 GiB, Fedora 44,
kernel 6.19.10. Detector: `////` in the answer, or more than eight slashes.

`bench/suites/slot-corruption.py par-two-prefixes` — two distinct ~2,700-token
prefixes, each with ten tool definitions, sent CONCURRENTLY by two threads,
two requests each. Fresh server per start.

## 1 · It reproduces, and what hid it

    stock 54ee5ee, 10 fresh starts
      par-two-prefixes    10 of 10 starts CORRUPT   every answer, '////' x70
      seq-two-prefixes     0 of 10 starts CORRUPT

`bench/suites/np2-candidates.py` and `stock-vs-patched.py`'s `-orig` cases
send their requests SEQUENTIALLY and carry no tool block. `-orig` describes
itself as "the exact shape the defect was found in". It is not: it produced
the sentence *it does not reproduce* on 26.08. (18 of 18 clean) and again on
28.08. (30 of 30 clean), on the very binaries that corrupt 10 of 10 here.

## 2 · The ingredient, one variable at a time

All on stock `54ee5ee`, five starts each unless noted.

| variant | CORRUPT | |
|---|---|---|
| baseline — `par`, `-np 2`, 10 tools | **10 of 10** | |
| `--np 1` | 0 of 5 | **necessary: two slots** |
| `seq-two-prefixes` | 0 of 10 | **necessary: concurrency** |
| `seq-no-tools` | 0 of 5 | |
| `--tools 0` | 1 of 5 | amplifier, not a requirement — but see §6: the amplifier is mostly SIZE |
| `--no-spec` | 5 of 5 | irrelevant |
| `--kv-unified` | 5 of 5 | irrelevant |
| `--no-cram` | 5 of 5 | irrelevant |
| `--np 4` | 5 of 5 | more slots, same |
| **Vulkan backend** | **0 of 5** | negative control holds |
| `--ctx 204800` (production's own window) | **5 of 5** | with production stopped |

This corrects the original finding, which reads *"it is the SECOND SLOT, not
concurrency — serialising every request in the gateway did not help"*. Direct
at the server, with no gateway in the way, sequential is 0 of 15 and
concurrent is 10 of 10. Both statements can stand: the gateway serialises
ADMISSION, not batching inside llama-server.

`--ctx 204800`, production's own window, was first REFUSED by the memory guard
five times out of five (79.7 GiB needed, 77.2 available beside a serving
qwen38) and recorded as an error rather than as a clean cell. **Measured 28.08.
09:00 with production stopped: 5 of 5 CORRUPT.** So the window production
actually runs is not a shelter — `-np 1` is.

## 3 · The build matrix, ten fresh starts each

| build | patch | upstream | CORRUPT |
|---|---|---|---|
| `build-rocm` | no | `54ee5ee` | **10 of 10** |
| `build-rocm-unpatched-b10631` | no | `5d5cb4c` | **10 of 10** |
| `build-rocm-patched-b10631` | **yes** | `5d5cb4c` | 0 of 10 |
| `build-rocm-unpatched-b10631-18-gc1dcd9825` | no | `5d5cb4c` + #27311 | 0 of 10 |
| `build-rocm-patched-b10631-18-gc1dcd9825` | yes | `5d5cb4c` + #27311 | 0 of 10 |

Two one-variable comparisons on the same upstream commit `5d5cb4c`: the patch
fixes it, and #27311 fixes it with no patch on either side.

A clean column means something here, which it did not in earlier attempts on
this defect: the unpatched builds are deterministic — 20 of 20 starts across
two upstream commits, every answer of every start, never a partial.

## 4 · Which commit of #27311, and the mechanism

Four of the PR's 18 commits touch `integrated` in `ggml-cuda.cu`. One is
named like the PR itself, and its message describes the defect exactly:

> `3181ed701` — *ggml-backend: ring buffer graph inputs when a backend
> computes on host memory*
>
> "Graph inputs are pinned to the last (CPU) backend, and llama.cpp hands the
> scheduler a device host buffer type for that slot. On an integrated GPU the
> device also accepts that buffer type … no split input copy is made, and the
> device reads the very memory the host thread writes. Nothing then stops the
> host from writing the next ubatch's inputs while the device is still reading
> the previous one."

Two builds rather than a blind bisect, ten starts each:

    b10631-5-gd3488aebe   the commit BEFORE it    10 of 10 CORRUPT
    b10631-6-g3181ed701   the commit WITH it       0 of 10

**And the same binary both ways.** That commit ships an escape hatch,
`GGML_SCHED_UMA_RING`. On the FULL PR build, which is 0 of 10 clean:

    GGML_SCHED_UMA_RING=1   (ring off)   10 of 10 CORRUPT

Same binary, same recipe, same hour, one variable — upstream's own switch.

That mechanism accounts for every row above: it needs two sequences in flight
(two slots, concurrent), it gets worse with more and larger graph inputs
(tools), it is indifferent to speculation, KV layout and the RAM prompt cache,
it cannot happen on Vulkan, and it is removed by our patch — which sets
`integrated = false`, so the device never accepts the host buffer type, a
split input copy IS made, and the shared buffer never exists.

## 5 · Both defects are the same bug

The same switch, on the same binary, against the OTHER defect — the
restore-during-prefill corruption that `bench/suites/restore-safety.py`
measures, and that this stack has treated as a separate mechanism since
25.08.:

    build-rocm-unpatched-b10631-18-gc1dcd9825   (0 of 10 on defect 1)
      ring ON    prefill-spec CLEAN   prefill-nospec CLEAN
      ring OFF   prefill-spec DIRTY   prefill-nospec DIRTY

The dirty probes leak fragments of other contexts where `391` belongs,
the same signature as everywhere else:

    '收到，将执行git' · '2\n\nThe user is asking for a number…' · '22'
    'The user":\nUser: tools.assistant.execute\n\n{"status": "success' · '27'

**So there is one cause, not two.** A slot restore was never a separate
mechanism — `llama_state_seq_set_data`'s tensor writes are simply another
concurrent writer into the shared graph-input buffer, which is why a restore
during another slot's PROMPT PROCESSING poisons it and a restore into an idle
server does not. Everything this stack has recorded as "defect 1" and "defect
2" since 25.08. collapses into the race `3181ed701` describes.

**Re-run as a pair on 28.08. 08:46 and 08:50**, four minutes apart, because
the first version of this measurement did not record its own environment —
the suite could not yet, and the value was the session asserting it rather
than the run capturing it. Both sides record it now:

    env {"GGML_SCHED_UMA_RING": "1"}   prefill-spec DIRTY   prefill-nospec DIRTY
    env {}                             prefill-spec CLEAN   prefill-nospec CLEAN

Reports: `2026-08-28_0846_…` and `2026-08-28_0850_…` (the pair), and
`2026-08-28_0603_…` (the first, which carries a field saying what it could
not capture).

## 6 · How small it gets — and a prediction that failed

All on stock `54ee5ee`, five starts each, `par-two-prefixes`.

| tools | bulk | sys | body | CORRUPT |
|---|---|---|---|---|
| 10 | 700 | 300 | ~6,800 tok | **10 of 10** |
| 3 | 700 | 300 | ~6,600 | 5 of 5 |
| 1 | 700 | 300 | ~6,300 | 3 of 5 |
| 0 | 700 | 300 | ~6,200 | 1 of 5 |
| 10 | 50 | 20 | ~1,000 | **0 of 5** |
| 3 | 50 | 20 | ~900 | **0 of 5** |

**The dominant factor is PROMPT SIZE, not the tool block.** A short prompt is
clean with ten tools; a long one corrupts with none. Tools raise the rate
sharply at a given length — 20 % to 100 % for about 600 extra tokens — so they
do something beyond their own length, and what that is was not established.

### The prediction, and why it was wrong

The fixing commit speaks of the host writing *"the NEXT UBATCH's inputs while
the device is still reading the previous one"*. With `-ub 2048`, a
6,800-token prompt is four ubatches and a 1,000-token one is a single ubatch —
which would explain the table exactly. **Prediction: raise `-ub` to 8192 so
the whole prompt is one ubatch, hold everything else, and it should go clean.**

    -ub 8192, same ~6,800-token prompt    2 of 5 CORRUPT

**It did not.** The rate falls — 10 of 10 to 2 of 5 — and the defect does not
disappear. So the overlap is not between consecutive ubatches of one request.
With two requests in flight, the *other* request's inputs are the ones being
written while these are read, and that happens at any ubatch count.

Which fits the commit message better than the reading that produced the
prediction, and re-reads the size effect: a longer prompt means longer in
prompt processing, so a WIDER WINDOW for two sequences to overlap — not a
discrete ubatch threshold. It is a rate, not a switch, and every clean cell
above is a low rate rather than an immunity.

## What this does not say

* **Nothing about whether #27311 is the right fix upstream**, only that it
  removes both symptoms here.
* The single-cause claim rests on one binary and one switch. It is strong —
  the switch turns both symptoms on and off together — and it is not the same
  as having shown that the original observations of 25./26.08., on other
  builds, had that cause.
* One machine, one model, one quant.
* Production is unaffected either way: it runs `-np 1`, which is 0 of 5 here.

## Reproducer

https://github.com/loonylabs-dev/strix-halo-inference/blob/main/bench/suites/slot-corruption.py

    bash setup/scripts/build-llama.sh --ref <ref> --no-patch
    python3 bench/suites/slot-corruption.py par-two-prefixes \
        --binary <the build id it prints> --starts 10

Every run has its own report directory under `bench/reports/2026-08-28_*`,
with the build stamp, the server argv, the environment and every answer.

## For an upstream post

Numbers, tables and the link above only. **No prose here is written for
posting.** llama.cpp `CONTRIBUTING.md` line 25 prohibits AI-written
contributions outright; the words have to be the operator's.
