# Does llama.cpp PR #27311 fix the restore-during-prefill corruption?

**Not answerable from these runs, and the reason is a mistake in how they
were paired.** Both builds carry our gfx1151 patch, which is what suppresses
the corruption — so #27311 was tested on top of the thing that already
prevents the symptom. What the runs do establish: the PATCHED build is clean
in all six cells, and the one cell that had been failing was our own
measurement bound.

Measured 27.08.2026, three runs, production stopped for each.

## What was built, and why not what the instruction said

The instruction in `docs/NEXT-SESSION.md` was
`bash setup/scripts/build-llama.sh --ref pr/27311`. That would have rebased
`gfx1151-patched` onto the PR — and `gfx1151-patched` has not been a
one-commit branch since 26.08., when it was rebased onto PR #27742 to build
Flash-Next. It carries 26 commits over `origin/master`, and PR #27311's base
(`bb4caa754`) is 65 commits behind that master.

Executed in a throwaway worktree rather than predicted: the rebase replays
**91 commits and succeeds, exit 0.** The result would have been PR #27311 plus
all 25 qwen4exp commits plus 65 master commits, stamped
`upstream_ref=pr-27311`. No conflict, no warning, and a measurement attributed
to the wrong change.

What was built instead is one variable against the baseline:

| | baseline | comparison |
|---|---|---|
| build id | `b10631` | `b10631-18-gc1dcd9825` |
| upstream | `5d5cb4c` (master) | `5d5cb4c` **+ PR #27311, 18 commits, rebased, no conflicts** |
| patch | `6b39dd5d5` gfx1151 `prop.integrated` | `545de45e5`, the same commit |
| binary reports | `build 252, commit 6b39dd5d5` | `build 270, commit 545de45e5` |

## Configuration, identical in every run

Qwen3.8-27B UD-Q4_K_XL · `-ngl 999 -fa on -c 65536 -np 2 --no-kv-unified
-b 2048 -ub 2048` · f16 KV · fresh server per cell · AMD Ryzen AI Max+ 395
(Strix Halo, gfx1151), 128 GB unified, ROCm, GTT cap 108 GiB.
`spec` = `--spec-type draft-mtp,ngram-mod --spec-draft-n-max 12
--spec-ngram-mod-n-min 24`; `nospec` = none.

Detector: three arithmetic probes after the restore (`17*23`, answer `391`),
plus the surviving generation's tail.

## The six cells, both builds

| cell | b10631 | b10631 + PR #27311 |
|---|---|---|
| `idle-spec` | CLEAN | CLEAN |
| `busy-spec` | CLEAN | CLEAN |
| `prefill-spec` | CLEAN | CLEAN |
| `idle-nospec` | CLEAN | CLEAN |
| `busy-nospec` | restore hit the 300 s bound | restore hit the 300 s bound |
| `prefill-nospec` | CLEAN | CLEAN |

`prefill-spec` is the cell that matched the production incident of 25.08.
exactly. It is clean on both. `prefill-nospec` had **never been measured on
the patched build** — three earlier runs (26.08. twice, 27.08. once) aborted
at `busy-nospec` before reaching it.

### The comparison this table CANNOT support

The first version of this file said the corruption "is already gone". That
compared these runs against the six-cell run of 25.08., where both `prefill`
cells were DIRTY — and those are not the same binary. Settled from the
pre-publication history rather than from memory:

    25.08.  BINARY = ~/llama.cpp/build-rocm/bin/llama-server   the STOCK build
    26.08.  commit fcaf6bb makes the default rocm-patched

`~/llama.cpp/build-rocm/bin/llama-server` still reports `build 200, commit
54ee5ee` today, which is what the 25.08. draft names. So DIRTY-on-25.08. and
CLEAN-tonight differ by **the patch AND 52 upstream builds**, and the report
of 25.08. could not say so itself: it carried no provenance at all. That is
the field `_meta` was added for, one night too late to help.

**So the honest scope of everything above is: on the patched build.** Whether
stock gfx1151/ROCm still corrupts is unmeasured on any current binary, and
whether #27311 fixes it there is exactly the question nobody has answered —
including this file.

## Why `busy-nospec` was never a defect

A restore queues behind the slot it targets. The cell fills slot 0 with a
2,500-token generation and then asks for a restore into it. The bound the
client gave that restore was 300 s. What it was waiting for:

| cell | what occupies the slot | how long that takes | restore returned after |
|---|---|---|---|
| `idle-spec` / `idle-nospec` | nothing | — | 0.0 s / 0.0 s |
| `prefill-spec` | 14,941-token prefill | 77.1 s | 6.2 s |
| `prefill-nospec` | 14,945-token prefill | 78.1 s | 16.3 s |
| `busy-spec` | 2,500 tokens at 33.9 t/s | **73.7 s** | **68.8 s** |
| `busy-nospec` | 2,500 tokens at 7.45 t/s | **335.5 s** | never — bound 300 s |

Same on the PR build: `busy-spec` 81.6 s of work, restore at 76.7 s;
`busy-nospec` 341.6 s of work, bound 300 s.

The restore returns when its slot frees. The asymmetry recorded in
`setup/defects.json` as unexplained — "busy-SPEC, the same cell with
speculation ON, was clean both times" — is the drafter: speculation runs the
same counting generation 4.5x faster, so the restore's wait fits inside the
bound.

**Confirmed by raising the bound**, run `2026-08-27_2159`, same build,
same cell, `--restore-timeout 900`:

    generations   325.8 s each (7.68 t/s)
    restore       RETURNED after 319.8 s
    probes        391 · 391 · 391
    generations   tails 648-652, both, intact

319.8 s against 325.8 s of work started ~6 s earlier: the restore came back
in the moment the slot freed.

## What this settles, and what it does not

**Settled.** `slot-restore-hangs-busy` is not a property of llama.cpp, of
gfx1151 or of ROCm. It is a client timeout below the workload the same cell
created. Three reports filed it as a defect and a fourth reproduced it before
the bound and the work were ever compared.

**Settled.** With the bound above the work, `restore-safety` is CLEAN in all
six cells on `b10631` at `-np 2`.

**Not settled by anything here.** Whether `-np 2` may come back. Rule (b) of
the decision in `bench/suites/stock-vs-patched.py` is the only condition this
touches; (a) and (c) are separate and were decided elsewhere.

**Not settled by anything here, and this is the important one.** Whether
#27311 fixes the corruption. Both builds measured here carry
`setup/patches/hip-integrated-off.patch`, which is the thing that suppresses
the symptom — so the PR was tested on top of its own competitor. A build with
no corruption in it cannot show a fix.

The experiment that would answer it, and that nobody has run:

    1  the STOCK build as it is today            does the 25.08. DIRTY still
       (`--binary rocm`, already on disk,        reproduce? A positive control
       commit 54ee5ee)                           for the whole instrument
    2  current master, NO patch                  is it still there upstream?
    3  current master + #27311, NO patch         does the PR fix it?

Step 1 needs no build at all. Without it, nothing below it means anything:
if the stale stock binary no longer reproduces the corruption either, then
what changed is not the patch and not the PR.

## The positive control, run 22:33 — and it fires

`--backend rocm`, the stock build already on disk (`build 200, commit
54ee5ee`, no `hip-integrated-off.patch`). Report:
`bench/reports/2026-08-27_2233_restore-safety-rocm_b200-54ee5ee`.

| cell | stock `b200-54ee5ee` | patched `b10631` | patched + #27311 |
|---|---|---|---|
| `idle-spec` | CLEAN | CLEAN | CLEAN |
| `busy-spec` | CLEAN | CLEAN | CLEAN |
| **`prefill-spec`** | **DIRTY** | CLEAN | CLEAN |
| `idle-nospec` | CLEAN | CLEAN | CLEAN |
| `busy-nospec` | CLEAN — restore at 278.8 s | bound hit, 335.5 s of work | bound hit, 341.6 s of work |
| **`prefill-nospec`** | **DIRTY** | CLEAN | CLEAN |

**The suite can still see the corruption.** The dirty probes return fragments
of other contexts, which is the 25.08. signature verbatim:

    prefill-spec     'This is a long independent_files:!user\n!\n!\n!' · '242' · '198'
    prefill-nospec   'To find": {"diff' · '39' · '339'

So the clean sheet on the patched build is a real result and not an
instrument that has gone blind. That was the one thing capable of
invalidating everything above, and it is now excluded.

**And `busy-nospec` is CLEAN here, at a restore of 278.8 s** — under the same
300 s bound that the patched build ran past. Same cell, same code, same bound;
the generation simply finished sooner. That is the bound theory confirmed a
second time, by a run that was not designed to test it.

### What is still not one variable

Stock is `54ee5ee` and the patched build is `5d5cb4c` — 52 upstream builds
apart. So "DIRTY on stock, CLEAN on patched" is still the patch AND 52 builds,
and this file does not claim otherwise. What it now does claim, because it was
measured today rather than inferred from a report of 25.08.:

**the restore-during-prefill corruption reproduces on a gfx1151/ROCm build
without our patch, on this machine, with this harness.**

Which is what makes the two remaining steps worth the GPU time:

    2  current master, NO patch            is it still there upstream?
    3  current master + #27311, NO patch   does the PR fix it?

## Where each number in this file comes from

| number | source | in the repository? |
|---|---|---|
| the six verdicts | `result.json` of each run | yes |
| `restore_seconds`, and the 300 s / 900 s bound | `result.json`, `_meta.restore_timeout` | yes |
| the confirmation run's 325.8 s generations | `result.json`, `gen_seconds` | yes |
| **the workload durations of the first two runs** — 77.1, 78.1, 73.7, 335.5, 84.5, 76.0, 81.6, 341.6 s | llama-server's own `print_timing` lines in `*-busy.log` / `*-prefill.log` | **no** |

`.gitignore` excludes `*.log`, and no report in this repository has ever
carried one. Those eight figures therefore live only on the machine that
produced them. They are `eval time` and `prompt eval time` as the server
reports them, one line each, and the `gen_seconds` / `prefill_seconds` fields
that would have put them in `result.json` were added to the suite between the
second run and the third.

## Reproducer

https://github.com/loonylabs-dev/strix-halo-inference/blob/main/bench/suites/restore-safety.py

    python3 bench/suites/restore-safety.py --binary <build id> --spec both
    python3 bench/suites/restore-safety.py --binary <build id> --cells busy \
        --spec nospec --restore-timeout 900

## For an upstream post

Numbers, tables and the link above only. **No prose here is written for
posting.** llama.cpp `CONTRIBUTING.md` line 25 prohibits AI-written posts
outright; the words have to be the operator's.

The neighbouring draft `bench/reports/2026-08-25_1845_restore-safety/
UPSTREAM-COMMENT.md` is stale in substance as of tonight: it reports both
`prefill` cells DIRTY, and neither reproduces on the patched build.
