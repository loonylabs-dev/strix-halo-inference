# ROCm 10.1 against ROCm 7.1 — it trades nothing for decode, up to 32k

**31 August 2026.** Two runs, four interleaved rounds each, Qwen3.8-27B-UD-Q4_K_XL.
Same llama.cpp commit `c799f1014`, same two patches, same cmake line. The only
difference is the ROCm the binary loads: Fedora's **7.1.52802** against
**10.1.0a20260830** (TheRock nightly for gfx1151, unpacked under `~/rocm-sdks`).

## The answer

| | run 1 (A,B) | run 2 (counterbalanced) |
|---|---|---|
| **decode @ 0** | +9.9 % | **+10.0 %** |
| **decode @ 16k** | +7.1 % | **+7.4 %** |
| **decode @ 32k** | +3.9 % | **+4.5 %** |
| decode @ 64k | −1.2 % | −1.0 % |
| prefill @ 0 | −2.4 % | −0.7 % |
| prefill @ 16k | −2.8 % | +1.0 % |
| prefill @ 32k | −2.2 % | +1.7 % |
| prefill @ 64k | −1.8 % | −1.8 % |

**Decode is faster on 10.1, and the gain shrinks with depth** — ten percent on
an empty context, gone by 64k. It reproduced to within half a point across two
runs on different orderings, which is what makes it a finding rather than a
number.

**There is no prefill penalty.** Run 1 said there was, consistently, at every
depth. Run 2 says there is not. The difference between the runs is the whole
lesson below. The one cell that survives both is prefill @ 64k, −1.8 % twice.

## The lesson: A,B,A,B is not enough

Run 1 alternated arms and still put the variant SECOND in every round — on a
machine one pass warmer each time. The operator noticed package power drifting
from 51 W down to 49 W and asked whether that was fair. It was not, and the
size of the unfairness was already on disk:

The unroll-flag run of the same morning compared two builds that turned out to
be identical in speed. Anything it measured as a difference IS the method's
systematic offset:

| | A→B offset | drift over 4 rounds |
|---|---|---|
| prefill @ 0 | −0.56 % | −3.0 % |
| prefill @ 16k | −0.73 % | −4.1 % |
| prefill @ 32k | −1.24 % | −3.7 % |
| decode @ 0 | −0.49 % | −2.3 % |
| decode @ 32k | −0.26 % | −1.5 % |

So the machine loses 3–4 % over a run, interleaving hides most of it, and what
is left — half a percent to one and a quarter — lands entirely on whichever arm
goes second. That is small next to a 10 % decode gain and the same size as the
prefill difference run 1 reported. Which is exactly what happened: the prefill
penalty was the instrument.

Run 2 fixes it properly. Each round alternates which arm leads, so over four
rounds each is first twice and the offset cancels instead of accumulating. A
discarded warm-up pass runs first, and GPU temperature and power are recorded
per round:

    warm-up             59.0 °C  47.1 W
    round 1 reference   59.0 °C  47.1 W
    round 1 variant     61.0 °C  43.0 W
    round 2 variant     61.0 °C  40.0 W
    round 2 reference   61.0 °C  39.0 W
    ...
    round 4 reference   61.0 °C  37.1 W

Temperature settles at 61 °C after the warm-up and stays there; power keeps
falling to 37 W. So this is not thermal throttling in the usual sense — the
part is inside its temperature budget and reducing draw anyway. Recorded as an
observation; the mechanism was not investigated.

## What this does NOT say

**Nothing about the profile that actually serves.** llama-bench runs without
speculation; qwen38.env serves with `--spec-type draft-mtp,ngram-mod`.
Speculation changes decode fundamentally — that is its purpose — so whether a
10 % raw-decode gain survives it, grows, or disappears is not measured here.
That needs bench/speed.py behind bench/sideserver.py with the real profile.

**Nothing about correctness.** No slot-corruption or restore-safety run was
made against the 10.1 build. On this hardware the dangerous defects do not
raise, and a faster binary that degenerates to `////` is worse than a slower
one. setup/defects.json knows nothing about ROCm 10.

**One model, one quantisation, one nightly.** 10.1.0a20260830 is an alpha of a
version AMD has not released; the stable line is 10.0. Whether 10.0 behaves
the same is unknown.

**Nothing about the 3.3× AMD advertises.** That figure is 8× MI355X under
vLLM/SGLang with ROCm.AI's tuning. Nothing in it was ever going to appear on an
APU running llama.cpp, and nothing in it did.

## What it would mean for the stack

The stack's own operating point is deep: `-c 204800`, a prefix cache, and long
agent conversations. At 64k the decode gain is gone and prefill is 1.8 % worse
— reproduced twice. On short exchanges 10.1 would be noticeably quicker.

So this is not a reason to move, and it is a reason to look again when the
stable 10.x reaches Fedora — with the speculation profile, and with the
correctness suites, before any of it is believed.

## Reproducing

    bash setup/scripts/build-llama.sh --ref master-2patches \
        --rocm-path ~/rocm-sdks/rocm-10.1.0a20260830 --with-bench
    python3 bench/suites/speed-ab.py --variant-family rocm-altsdk --reps 4

Two things that cost an hour and are now guarded by the script:

- ROCm 10.1's `libamdhip64` carries the SAME soname as Fedora's 7.1 —
  `.so.7`, pointing at 7.16.26344 and 7.1.52802. A build against the new SDK
  loads the OLD runtime through the system search path unless something
  prevents it, and reports numbers either way. `-DCMAKE_BUILD_RPATH_USE_ORIGIN`
  plus the SDK's lib in RUNPATH is what prevents it here; speed-ab.py checks
  with `ldd` which one each arm will actually load and refuses a pair that
  loads the same one.
- `CMAKE_PREFIX_PATH` is not enough to change which HEADERS are used. clang
  keeps finding `/usr/include/hip` because it is a default search path, and
  ROCm 10.1's clang against ROCm 7.1's headers dies on `use of undeclared
  identifier '__ocml_log2_f32'`. Both `--rocm-path=` and `-isystem <sdk>/include`
  are needed.
