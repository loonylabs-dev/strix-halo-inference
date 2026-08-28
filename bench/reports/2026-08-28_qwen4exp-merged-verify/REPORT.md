# The three qwen4exp defects, measured on the build we actually run

28.08.2026, evening. Until this run the three entries said *fixed upstream, in
the build we pin, unverified here* — the fixes were merged (#27742, squashed as
`6c84c7d5d`) and the pinned build has that commit as an ancestor, but nothing
on this machine had ever tried the configuration each defect describes. A
source reading says what the code does. It does not say what this GPU does with
it.

## Setup, identical for every run

    build     ~/llama.cpp/build-rocm-patched-b10665-3-gc93d460b6
              master 28.08. + the gfx1151 patch + PR #27837, cherry-picked
              server fingerprint b288-c93d460b6
    model     Qwen3.8-Flash-Next-UD-Q4_K_XL (4 files, 103.7 GiB)
    profile   setup/env/flashnext.env, unchanged
    start     bench/sideserver.py, production stopped and restarted by it,
              dead man's switch armed, memory guard in front of every start
    variable  ONE per run, appended to the profile's LLAMA_ARGS

Every run came up at GTT 81–82 GiB in about 20 seconds, which is the #27837
cherry-pick doing its job: without it this model does not fit on this machine
at all.

## 1. `-fit on` — the fit probe no longer asserts

`GGML_ASSERT(obj_new)` in `common_params_fit_impl` was the defect. It is gone,
and the trace shows the probe ran rather than being skipped:

    common_init_: fitting params to device memory ...
    common_params_fit_impl: getting device memory data for initial parameters:
    | memory breakdown [MiB]  |  total     free     self   model  context  compute  unaccounted |
    |  - ROCm0 (8060S)        | 110592 = 110337 + (82525 = 78056 +   2224 +   2244) +    -82271 |
    |  - Host                 |                    29027 = 28110 +      0 +    916             |
    common_params_fit_impl: projected to use 82525 MiB vs 110337 MiB of free device memory
    common_params_fit_impl: will leave 27812 >= 1024 MiB free, no changes needed
    common_fit_params: successfully fit params to free device memory  (0.54 s)

Two things worth keeping from that table.

**The negative `unaccounted` is not the fault.** The bug report on #27742
quoted `unaccounted -47656 MiB` right before the assert, which reads like the
symptom. It is not: the probe runs BEFORE anything is allocated, so `free` is
still the whole device and `self` is a projection — the row balances at
110337 + 82525 − 82271 = 110591. Our run prints the same shape and fits
successfully. What crashed was `LLM_ARCH_QWEN4EXP` missing from the
`graph_max_nodes` list, and that line is in master now
(`llama-context.cpp:2312`).

**The estimator does not know about demand paging.** It projects 28110 MiB on
the host — the PLE table, at full size. With #27837 the measured RssAnon is
0.31 GiB. So `-fit on` reasons about a host cost that this build does not pay.
It fits anyway here, with 27.8 GiB of headroom, but a tighter case would be
judged on the wrong number.

Argument precedence, checked in the source rather than assumed: the `-fit`
handler (`common/arg.cpp:2890`) is a plain setter the parser calls per
occurrence, so the appended `-fit on` overrides the profile's `-fit off`. And
`common_fit_params()` is called unconditionally when the flag is set
(`common/common.cpp:1295`) — only its output depends on verbosity, which is why
the first run's log looked empty and this one was repeated with `-lv 4`.

## 2. `-ctk q8_0 -ctv q8_0` — the quantised KV no longer asserts

`GGML_ASSERT(inp->self_k_rot == nullptr && inp->self_v_rot == nullptr)` fired
at graph build, so one request is enough to answer it. The server came up and
answered `eins` with HTTP 200. No assert, no 500.

The profile keeps f16 regardless. That was never a workaround for this defect —
the one q8_0-KV cell measured on 24.08. dropped a long-context task that f16
solved, and that reason is untouched by anything upstream merged.

## 3. `-np 2` and `-np 4` — the cross-slot desync does not reproduce

The assert fires *the first time a request lands on a different slot than the
previous one*, so the measurement has to put work on two slots. **Sequential
requests do not.** The first attempt sent three requests one after another with
`-np 2`, got three 200s, and proved nothing: `/slots` shows all three landed on
slot 1 while slot 0 stayed empty. That run is kept in `np-2.out` as the
negative example.

Concurrent requests do. With `-np 2`:

    initializing, n_slots = 2, n_ctx_slot = 32768, kv_unified = 'false'
    slot 1 | task 2  | processing task     ) overlapping
    slot 0 | task 4  | processing task     )
    slot 0 | task 31 | selected slot by LCP similarity, f_sim_best = 0.125

And with `-np 4`, which is the shape the defect was REPORTED in — llama-server
defaults to four slots, so anyone who does not pass `-np` has this
configuration:

    initializing, n_slots = 4, n_ctx_slot = 16384
    slot 3 | task 2 | slot 2 | task 4 | slot 1 | task 3 | slot 0 | task 5
    slot 0 | task 32   (the fifth request, after the four)

Four requests in flight on four distinct slots, then a fifth on a slot another
request had used. No assert, `health=200`, and the answers are coherent
(`Rhein, Donau, Elbe, …`, `42`). The server prompt cache — the announced cause,
not the scheduler — was active throughout: the profile carries `-cram 4096`.

`-np 1` stays in the profile. Its two other reasons are properties of this GPU
(`gfx1151-hip-integrated`, `gfx1151-two-slots`) and nothing upstream merged
touches them.

## What this does NOT settle

* One quant (UD-Q4_K_XL), one machine, one build. The reports these entries
  came from were RTX 3090/5090 and a 1-bit quant.
* Short prompts and 24-token answers. This is a crash test, not a correctness
  or a stability test — none of these runs says anything about output quality
  under load, which is what `gfx1151-two-slots` is about.
* The fit probe was run once, in a configuration where every memory parameter
  is set by hand. A profile that leaves them unset gives the estimator more to
  do and is not covered here.

## A second finding, in the harness rather than in llama.cpp

The first three runs all failed with *GTT did not fall within 180 s* while
production was already stopped and GTT was already at 1.5 GiB.
`bench/sideserver.py` waited with `wait_for_gtt_release(0.0)`, whose tolerance
is +1.0 GiB — so it waited for GTT to fall below 1 GiB, and the DESKTOP on this
machine holds 1.5 GiB and never gives it back. The condition could not be met
while anyone was logged in, and the answer it defaulted to was *refuse*.

A guard that cannot be satisfied refuses everything, and it does it silently:
nothing was wrong with the memory, and nothing in the message said the
question had no answer here. `wait_for_gtt_to_settle()` now waits for the
reading to STOP FALLING, which is the question that has an answer without a
baseline — whether there is room is still `check_room_for()`'s job, three lines
later, with the profile and the machine in hand.

## Files

    fit-on.out              first run, no server log (the flag needs -lv 4)
    fit-on-trace.llama.log  the run above, with the fit trace
    kv-q8.out               q8_0 K and V
    np-2.out                THE NEGATIVE EXAMPLE: sequential, one slot used
    np-2-parallel.out       two slots, concurrent
    np-4-parallel.out       four slots, concurrent, plus a fifth request
    *.llama.log             llama-server's own log where it was captured
