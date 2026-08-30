# patches — changes to llama.cpp that this machine needs

> This file holds the NARRATIVE: what the defect is, how it was found,
> what it cost. Whether **your** build is currently exposed is a
> different question and has a different home:
>
>     python3 setup/lib/defects.py
>
> Status belongs in `setup/defects.json`, not here — this file said
> things about open pull requests that stopped being true the same day.

The stack runs a **patched llama.cpp build**. Not a fork: one line, kept here
as the canonical text and carried in `~/llama.cpp` as a one-commit branch
(`gfx1151-patched`), because losing it does not break anything visibly — it
just makes the answers wrong.

## hip-integrated-off.patch

**What it does:** stops the HIP backend from trusting `prop.integrated`.
Upstream already disables that flag for CUDA with the comment "Temporarily
disabled due to issues with corrupted output (e.g. #15034)" and re-enabled
it for HIP in PR #24233. On this hardware (Strix Halo, gfx1151, ROCm) that
path corrupts the KV state as soon as a **second slot** is used: every
answer afterwards degenerates to `////` until the server restarts.

**Measured** (`bench/suites/slot-corruption.py`, `bench/suites/np2-candidates.py`,
fresh server per cell):

    stock build,  two slots, empty store   CORRUPT 6/6
    patched,      two slots, empty store   clean — and CORRUPT 3/6 on the
                                           NEXT start, same configuration
    patched,      one slot                 clean in every cell

**Cost**, measured side by side on identical bodies: prefill −6.0 %, decode
unchanged.

**Note what this does NOT buy:** the patch alone does not make two slots
safe — see the second line above. Production runs `-np 1` for that reason.
The patch is still applied because it removes one of the two failure modes,
and because two slots must not become tempting again while it is missing.

Upstream: llama.cpp #27579 (same hardware, same symptom, trigger
`--parallel >= 2`), #27572 (root cause), #27506 (a gfx1151 regression
bisected to the commit this reverts).

## Building it

    cd ~/llama.cpp
    # 1 · apply (the patch is against ggml/src/ggml-cuda/ggml-cuda.cu)
    git apply "$REPO/setup/patches/hip-integrated-off.patch"   # $REPO = where you cloned this

    # 2 · configure a SEPARATE build dir, so the stock build stays usable
    cmake -S . -B build-rocm-patched -DCMAKE_BUILD_TYPE=Release \
          -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DGGML_HIP_GRAPHS=ON \
          -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON -DBUILD_SHARED_LIBS=ON \
          -DCMAKE_HIP_COMPILER=/usr/lib64/rocm/llvm/bin/clang \
          -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF

    # 3 · build (nice, so a running session keeps its CPU)
    nice -n 10 cmake --build build-rocm-patched --target llama-server -j 12

Do NOT enable `GGML_HIP_ROCWMMA_FATTN`: measured elsewhere as a 41 %
prefill regression on gfx1151.

## After every llama.cpp update

The patch is gone the moment the source is updated or reset, and nothing
says so. `bash setup/check.sh` checks all three conditions — binary
present, source still patched, binary newer than source — and complains
loudly if one fails. Run it after every `git pull` in `~/llama.cpp`.

## When can this be dropped?

Both issues are still **open** upstream (checked 26.08.2026). But the
behaviour changed anyway: remeasured on b10631 with
`bench/suites/np2-candidates.py rocm-patched+cram+mmproj` over **four fresh
server starts**, 24 of 24 answers came back clean. On b10577 the same cell was
clean on one start and CORRUPT 3/6 on the next.

That is not yet a reason to drop anything:

* the cell runs `-c 32768`; production runs `-c 204800`, and a race is
  timing-dependent;
* it covers defect 1 only. Defect 2 — the slot RESTORE with two slots — was
  CORRUPT 4/4 with a populated store and is a different code path;
* four clean starts against an intermittent, SILENT failure raise the odds
  without making an argument.

**A candidate fix for defect 2 is measured, 27.08.2026: llama.cpp PR #27311**
("Scheduler UMA ring buffer"), open and unmerged. On builds WITHOUT this
patch, the restore-during-prefill cells are DIRTY 6 of 6 on upstream master
`5d5cb4c` and on stock `54ee5ee`, and CLEAN 6 of 6 with #27311 applied to the
same tree — three runs each, interleaved. Synthesis and every caveat:
`bench/reports/2026-08-27_2143_restore-safety-rocm-patched_b10631-18-gc1dcd9825/COMPARISON.md`.

**That does not retire this patch, and the reason is the second bullet above.**
Every cell in that suite involves a restore. Defect 1 — two slots, an EMPTY
store, no restore at all, CORRUPT 6/6 on stock — is what this patch was
written for.

**It was attempted on 28.08. and the attempt is a NULL RESULT, not a green
light.** `np2-candidates.py` on three builds, one fresh server start each:

    unpatched master 5d5cb4c            clean 6/6
    unpatched master + #27311           clean 6/6
    stock 54ee5ee — THE POSITIVE CONTROL clean 6/6   <- and it must not be

The third line is the finding. `54ee5ee` is the exact binary defect 1 was
measured CORRUPT 6/6 on, and it came back clean. So the suite reproduced
nothing on any build, and the two lines above it are therefore about nothing.

That is not a surprise once the first paragraph of this file is read again:
**the unit of risk is the START, not the answer.** Same configuration, one
start clean and the next CORRUPT 3/6; four fresh starts gave 24 of 24 clean on
26.08. while the defect was known to be present. Three starts is a sample of
three from a per-start rate bounded at about 10 % — getting three clean is the
ordinary outcome whether or not the defect is there.

**Rule (a) was then run on the control, 28.08. 00:25-00:56 — and it did not
fire either.** 30 fresh server starts on the stock `54ee5ee` build, the shape
the defect was found in (`np2-orig`: `-np 2 --no-kv-unified -cram 32768
--mmproj`, speculation on), on a side server with production up:

    30 starts · 180 answers · ZERO corrupt · zero failed starts
    bench/reports/2026-08-28_0056_stock-vs-patched_b200-54ee5ee

That looked like "defect 1 does not reproduce on this machine any more", and
**it was written down as such at 00:56 and was WRONG.** The second reading of
the two below was the right one: the recipe was not the one the 6/6 came from.

**REPRODUCED 28.08. 01:07, on the first attempt, deterministically.**
`bench/suites/slot-corruption.py par-two-prefixes --binary rocm --starts 10`,
stock `54ee5ee`, side server, production up:

    par-two-prefixes   10 of 10 starts CORRUPT   '////' x70, all four answers
    seq-two-prefixes    0 of 10 starts CORRUPT

The ingredient the 30-start control was missing is in that pair. `np2-orig`
and `np2-candidates` send **sequential** requests with **no tool block**;
`par-two-prefixes` sends two distinct prefixes **CONCURRENTLY**, each carrying
ten tools. Sequential is clean 10 of 10 on the same build in the same hour.

**And it refines the original finding.** This file's own docstring says *"it
is the SECOND SLOT, not concurrency — serialising every request in the gateway
did not help"*. Going directly at the server, with no gateway in the way,
concurrency is an ingredient: 10/10 against 0/10. Both can be true — the
gateway serialises ADMISSION, not batching inside llama-server — but the
cleaner experiment is the one without it.

**So a build comparison for defect 1 was possible after all, and it was run.**
`par-two-prefixes`, ten fresh starts per build, side server, production up,
28.08. 01:07-03:20:

| build | patch | upstream | CORRUPT |
|---|---|---|---|
| `build-rocm` (stock) | no | `54ee5ee` | **10 of 10** |
| `build-rocm-unpatched-b10631` | no | `5d5cb4c` | **10 of 10** |
| `build-rocm-patched-b10631` | **yes** | `5d5cb4c` | 0 of 10 |
| `build-rocm-unpatched-b10631-18-gc1dcd9825` | no | `5d5cb4c` **+ #27311** | 0 of 10 |
| `build-rocm-patched-b10631-18-gc1dcd9825` | yes | `5d5cb4c` + #27311 | 0 of 10 |

Two one-variable comparisons on the same upstream commit `5d5cb4c`:

* **this patch fixes defect 1** — 10 of 10 without it, 0 of 10 with it;
* **llama.cpp PR #27311 fixes defect 1 as well** — 10 of 10 without it, 0 of
  10 with it, and with no patch on either side.

**And a clean column means something here, which it did not before.** On the
unpatched builds this is DETERMINISTIC: 20 of 20 starts across two different
upstream commits, every answer of every start, never a partial. Against a
deterministic reproducer, ten clean starts is a signal rather than the absence
that ten clean starts of an intermittent fault would be.

**Defect 1 is still in upstream master.** `5d5cb4c` unpatched corrupts 10 of
10, and `info.devices[id].integrated = prop.integrated` for HIP is still in
master's `ggml-cuda.cu` — checked, not assumed.

### Verify a build instead of trusting it

`setup/env/qwen38.env` has said *"rebuild the patched binary after every
llama.cpp update, or defect 1 returns silently"* since the patch existed, and
until 28.08. there was no way to check the result — the defect was believed to
be an intermittent per-start gamble. It is not. It is deterministic in the
right shape, so a build is now checkable in about a minute:

    python3 bench/suites/slot-corruption.py par-two-prefixes \
        --binary <the build id> --starts 3

Side server, production untouched. Ten of ten corrupt without the patch and
zero of ten with it, on the same upstream commit — three starts is already a
decisive signal against that.

### It is ONE defect, not two, and it is one commit of #27311

Four of the PR's 18 commits touch `integrated` in `ggml-cuda.cu`. One is named
like the PR itself and its message describes this defect exactly:

> `3181ed701` — *ggml-backend: ring buffer graph inputs when a backend
> computes on host memory*
>
> "Graph inputs are pinned to the last (CPU) backend, and llama.cpp hands the
> scheduler a device host buffer type for that slot. On an integrated GPU the
> device also accepts that buffer type … no split input copy is made, and the
> device reads the very memory the host thread writes. Nothing then stops the
> host from writing the next ubatch's inputs while the device is still reading
> the previous one."

Two builds rather than a blind bisect, ten fresh starts each:

    b10631-5-gd3488aebe   the commit BEFORE it    10 of 10 CORRUPT
    b10631-6-g3181ed701   the commit WITH it       0 of 10

**And the same binary both ways.** That commit ships an escape hatch,
`GGML_SCHED_UMA_RING`. On the FULL PR build, which is 0 of 10 clean:

    GGML_SCHED_UMA_RING=1   (ring off)   10 of 10 CORRUPT

**The same switch brings the RESTORE corruption back too**, on the same
binary — `bench/suites/restore-safety.py --cells prefill`:

    ring ON    prefill-spec CLEAN   prefill-nospec CLEAN
    ring OFF   prefill-spec DIRTY   prefill-nospec DIRTY

**So defect 1 and defect 2 are the same bug.** A slot restore was never a
separate mechanism: `llama_state_seq_set_data`'s tensor writes are simply
another concurrent writer into the shared graph-input buffer, which is exactly
why a restore during another slot's PROMPT PROCESSING poisons it while a
restore into an idle server does not.

That mechanism accounts for every ingredient measured: it needs two sequences
in flight (two slots AND concurrency), it is indifferent to speculation, KV
layout and `-cram`, it cannot happen on Vulkan, and **this patch removes it by
making the shared buffer never exist** — `integrated = false` means the device
does not accept the host buffer type, so a split input copy IS made.

**It also corrupts at production's own window.** `-c 204800`, 5 of 5, measured
with production stopped. The window is not what keeps this stack safe; `-np 1`
is — 0 of 5 on the same build.

**And it is a RATE, not a switch.** Prompt size dominates: ~6,800 tokens is 10
of 10, ~1,000 tokens is 0 of 5 with the same ten tools. A prediction that the
ubatch boundary was the mechanism — raise `-ub` so the whole prompt is one
ubatch and it should go clean — was tested and FAILED: 2 of 5, down from 10 of
10 but not gone. With two requests in flight the other request's inputs are
the ones being written while these are read, at any ubatch count. So a longer
prompt is a wider window for overlap, and every clean cell in the hunt is a
low rate rather than an immunity. Details in `HUNT.md` § 6.

Full evidence, every run and every caveat:
`bench/reports/2026-08-28_defect1-hunt/HUNT.md`.

### So: can this patch be dropped?

**Not yet, and the reason is no longer about evidence.** #27311 is OPEN and
UNMERGED. Dropping the patch means running either an unmerged pull request in
production or plain master, which corrupts 10 of 10. What has changed is that
the wait now has a defined end and a defined test:

    when 3181ed701 reaches master, rebuild WITHOUT the patch and re-run
      python3 bench/suites/slot-corruption.py par-two-prefixes --binary <id> --starts 10
      python3 bench/suites/restore-safety.py --binary <id> --spec both
    both clean -> this patch can go.

The `upstream_check` probe in `setup/defects.json` still watches for the
`integrated` assignment disappearing, and that is now the WRONG condition:
#27311 does not remove that line, it makes the buffer it leads to safe. The
probe to watch for is the ring buffer arriving — `n_copies_uma`, or the
`GGML_SCHED_UMA_RING` environment variable, in `ggml/src/ggml-backend.cpp`.

**And there is a second mitigation now, which is upstream's own.** On any
build that contains `3181ed701`, `GGML_SCHED_UMA_RING=2` turns the ring on
explicitly. That is worth knowing for anyone who cannot carry a patch.

The open sequence, in order:

1. **An unwritten suite** — two slots WITH a populated store and real
   restores, several server starts. That is defect 2.
2. the same at the production window, `-c 204800`.
3. only then the STOCK build, to find out whether the patch itself is still
   earning its keep (it costs 6 % prefill).

Until all three are measured, `-np 1` and the patch both stay.

---

## draft-tail-past-stop.patch

**MEASURED AND REFUTED, 30.08.2026.** It is kept, unapplied, because being
wrong for a reason is worth more here than being deleted: the next person to
reach for this fix should find out in one minute that it has already been
tried and why it cannot work.

**What it tried:** the accepted draft tokens that sit BEHIND the stop token are
inserted into the slot in one block and never removed, so the slot keeps 1-2
tokens no re-rendered history contains. The patch removed them again.

**What happened:** the server aborted on the second round.

    common.cpp:1628: failed to remove sequence 0 with p0=324, p1=-1
    ggml_abort -> SIGABRT, code=dumped, status=6/ABRT

**Two things were wrong, and only the first one was mine.**

1. The guard read `COMMON_CONTEXT_SEQ_RM_TYPE_FULL` as permission. It is the
   opposite: "can seq_rm full sequences ONLY", i.e. no partial rollback at
   all. The code twenty lines above uses the same field correctly, to decide
   that a CHECKPOINT is needed. An inverted predicate, caused by a name that
   reads like a capability.

2. That was not the cause here. Measured afterwards with `-lv 4`: the line
   "speculative decoding will use checkpoints", which is printed only for
   FULL, DOES NOT APPEAR. So the type is PART or RS — a partial rollback is
   advertised — and `seq_rm` refused anyway.

**Which makes the failure the interesting part.** `common_context_can_seq_rm()`
promises a capability the recurrent half then declines at runtime, and
`common_context_seq_rm` turns that refusal into `GGML_ABORT`. That is exactly
what llama.cpp PR #28007 describes, and this is an independent reproduction of
it at a DIFFERENT call site — the speculative draft path rather than prompt
processing.

**So the approach is dead, not just the patch.** The draft tokens are already
DECODED when the stop is found: they sit in the KV cache and in the recurrent
state, and the recurrent state is what cannot be wound back. Inserting fewer
tokens instead of removing them afterwards does not help for the same reason —
the memory would still hold them. A real fix has to be lower down: either the
recurrent memory gains genuine rollback (which is what `n_rs_seq` is for), or
speculation stops decoding past an end-of-generation token in the first place.

**What this cost:** one crash on a service with `Restart=on-failure`, which
came back by itself, plus one restart to put the untouched library back.
Nothing else. The measurement was worth its price — the alternative was
sending a guess upstream.
