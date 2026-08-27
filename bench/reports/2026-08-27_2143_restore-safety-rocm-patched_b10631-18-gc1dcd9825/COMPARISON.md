# Does llama.cpp PR #27311 fix the restore-during-prefill corruption on gfx1151?

**On this machine, in eight runs on the evening of 27.08.2026: the corruption
is present in upstream master and does not reproduce with PR #27311 applied.**

This file synthesises all eight runs of that evening. Each has its own report
directory beside this one, named by family and build.

## The measurement

`bench/suites/restore-safety.py` restores a saved slot state while another
slot is mid-prompt-processing, then asks the server `17*23` three times. `391`
or garbage. Fresh server per cell.

Only the two `prefill` cells have ever been dirty on any build, so the repeats
measured those:

| build (all **without** `hip-integrated-off.patch`) | runs | `prefill-spec` | `prefill-nospec` |
|---|---|---|---|
| stock `54ee5ee` (b200) | 1 | DIRTY | DIRTY |
| master `5d5cb4c` (b251) | 2 | DIRTY, DIRTY | DIRTY, DIRTY |
| master `5d5cb4c` **+ PR #27311** (b269) | 3 | CLEAN ×3 | CLEAN ×3 |

**6 of 6 dirty without the PR. 6 of 6 clean with it.** The runs were
interleaved — PR, base, PR, base, PR — so a drift over the evening cannot line
up with the build.

The dirty probes return fragments of other contexts where `391` belongs:

    'This is a long independent_files:!user\n!\n!\n!' · '242' · '198'
    'To find": {"diff' · '39' · '339'
    '```toml: {"message!' · '2' · '1980\n</think>\n\n1980'
    '<tool_call>\nfunction' · '{" this is a simple response …": "alpha",' · '288'

## The full six cells

| cell | stock `54ee5ee` | master `5d5cb4c` | `5d5cb4c` + #27311 | + patch, no PR | + patch + PR |
|---|---|---|---|---|---|
| `idle-spec` | CLEAN | CLEAN | CLEAN | CLEAN | CLEAN |
| `busy-spec` | CLEAN | CLEAN | CLEAN | CLEAN | CLEAN |
| **`prefill-spec`** | **DIRTY** | **DIRTY** | CLEAN | CLEAN | CLEAN |
| `idle-nospec` | CLEAN | CLEAN | CLEAN | CLEAN | CLEAN |
| `busy-nospec` | CLEAN | CLEAN | CLEAN | bound* | bound* |
| **`prefill-nospec`** | **DIRTY** | **DIRTY** | CLEAN | CLEAN | CLEAN |

\* not a result — see *The cell that was never a defect* below.

## What was built

One variable separates the two builds that answer the question: PR #27311's
18 commits, rebased onto the same master, with no other difference.

| | `b251-5d5cb4c` | `b269-c1dcd98` |
|---|---|---|
| upstream | `5d5cb4c` | `5d5cb4c` **+ #27311, 18 commits, rebased, no conflicts** |
| patch | none, marker verified absent | none, marker verified absent |
| binary reports | `build 251, commit 5d5cb4c3a` | `build 269, commit c1dcd9825` |
| stamp | `patched=no` | `patched=no` |

PR #27311's own base is `bb4caa754`, 65 commits behind `5d5cb4c`. It was
rebased forward rather than built as-is, so that the only difference from the
control is the PR itself.

## Configuration, identical in every run

Qwen3.8-27B UD-Q4_K_XL · `-ngl 999 -fa on -c 65536 -np 2 --no-kv-unified
-b 2048 -ub 2048` · f16 KV · AMD Ryzen AI Max+ 395 (Strix Halo, gfx1151),
128 GB unified, ROCm, GTT cap 108 GiB, Fedora 44, kernel 6.19.10.
`spec` = `--spec-type draft-mtp,ngram-mod --spec-draft-n-max 12
--spec-ngram-mod-n-min 24`; `nospec` = none.

## What this establishes, and what it does not

**Establishes.** The restore-during-prefill corruption is in upstream master
at `5d5cb4c`, reproduces on the first attempt every time, and does not
reproduce on the same tree with #27311 applied — three runs, six cells,
interleaved with the control.

**Does not establish that #27311 is the fix.** Six clean cells against an
intermittent fault is an absence, not an observation. The dirty side is the
stronger half of this evidence: it reproduced 6 of 6, immediately, on two
different unpatched builds.

**Says nothing about the other gfx1151 defect.** `setup/patches/hip-integrated-off.patch`
exists for the plain two-slot corruption — `-np 2` with an *empty* prefix
store, no restore involved, CORRUPT 6/6 on stock. Every cell here involves a
restore.

That was attempted separately on 28.08. with `np2-candidates.py`, one fresh
start per build, and it is a **null result**: unpatched master clean 6/6,
unpatched master + #27311 clean 6/6, and **the positive control — stock
`54ee5ee`, the binary that defect measured CORRUPT 6/6 on — also clean 6/6.**
The control did not fire, so nothing was reproduced on any build and the two
lines above it are about nothing.

The reason is in that defect's own record: the unit of risk is the START. One
start clean and the next CORRUPT 3/6 in the same configuration; 24 of 24 clean
over four starts while the defect was known present. Three starts cannot see a
per-start rate bounded at ~10 %. Reports:
`bench/reports/2026-08-28_001{2,3,5}_np2-candidates_*`.

**One machine, one model, one quant.**

## The cell that was never a defect

`busy-nospec` was recorded as the defect `slot-restore-hangs-busy` and
reproduced four times. It is **withdrawn**. A restore queues behind the slot
it targets, and the cell fills that slot with a 2,500-token generation before
asking for one:

| cell | what occupies the slot | how long | restore returned after |
|---|---|---|---|
| `idle-*` | nothing | — | 0.0 s |
| `prefill-spec` | 14,941-token prefill | 77.1 s | 6.2 s |
| `prefill-nospec` | 14,945-token prefill | 78.1 s | 16.3 s |
| `busy-spec` | 2,500 tokens at 33.9 t/s | 73.7 s | 68.8 s |
| `busy-nospec` | 2,500 tokens at 7.45 t/s | **335.5 s** | never — bound was 300 s |

Confirmed by raising the bound: `--restore-timeout 900`, same build, same
cell — generations 325.8 s, restore **returned after 319.8 s**, probes
`391 391 391`. And confirmed twice more without being asked: on the unpatched
builds the same cell is CLEAN with the restore returning at 278.8 s, 279.4 s
and 333.0 s. The asymmetry — speculation on, always clean — is the drafter,
which runs the same generation 4.5× faster.

## A comparison this file made and withdrew

Its first version said the corruption "is already gone", holding the patched
runs against a six-cell run of 25.08. where both `prefill` cells were DIRTY.
Those are not the same binary: `BINARY` was hard-wired to `~/llama.cpp/build-rocm`
— the stock build — until commit `fcaf6bb` on 26.08. So DIRTY-then against
CLEAN-tonight differed by the patch AND 52 upstream builds, and the report of
25.08. could not say so, because it recorded no provenance at all. Everything
above was re-measured with the control in the same evening.

## Where each number comes from

| number | source | in the repository? |
|---|---|---|
| every cell verdict, and the probe strings | `result.json` per run | yes |
| `restore_seconds`, the 300 s / 900 s bound | `result.json`, `_meta` | yes |
| build identity, `patched=`, the commits | `result.json`, `_meta.stamp` | yes |
| the workload durations (77.1, 78.1, 73.7, 335.5 s) | llama-server's own `print_timing` lines | **no — `.gitignore` excludes `*.log`** |

## Reproducer

https://github.com/loonylabs-dev/strix-halo-inference/blob/main/bench/suites/restore-safety.py

    bash setup/scripts/build-llama.sh --ref <ref> --no-patch
    python3 bench/suites/restore-safety.py --binary <the build id it prints> \
        --spec both --restore-timeout 900

## For an upstream post

Numbers, tables and the link above only. **No prose here is written for
posting.** llama.cpp `CONTRIBUTING.md` line 25 prohibits AI-written
contributions outright; the words have to be the operator's.

The older draft `bench/reports/2026-08-25_1845_restore-safety/UPSTREAM-COMMENT.md`
is superseded: it reports a build it does not identify, and its conclusion
about which cells are dirty holds only for a build without the patch.
