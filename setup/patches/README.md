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
written for, and #27311 was not tested against it. The nearest measurement is
`bench/suites/restore-safety.py --cells parallel` on an unpatched build with
and without the PR, and it has not been run.

The open sequence, in order:

1. **An unwritten suite** — two slots WITH a populated store and real
   restores, several server starts. That is defect 2.
2. the same at the production window, `-c 204800`.
3. only then the STOCK build, to find out whether the patch itself is still
   earning its keep (it costs 6 % prefill).

Until all three are measured, `-np 1` and the patch both stay.
