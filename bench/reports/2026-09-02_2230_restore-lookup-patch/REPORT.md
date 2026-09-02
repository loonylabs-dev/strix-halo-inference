# Two gates, not one — 111.5 s to 0.53 s, and the first patch alone did nothing

02.09.2026, 21:30–22:30. flashnext profile on a side server, `--slot-save-path`
and the profile's own `-cram 14336`. Three builds through the SAME reproducer
(`bench/suites/restore-vs-cram.py`), differing by exactly the commits under
test. Nothing was activated; the profiles pointed at the serving build
throughout.

## The result

    build                                       arm 1        arm 2
                                             (no restore)  (restore first)
    b10743-15-g62850522e   serving, unpatched  40000/0.32s   20000/111.51s
    b10750-9-gf8d3b3c91    + patch 1           40000/0.32s   20000/111.83s
    b10750-10-g6bbb3eecf   + patch 2           40000/0.35s   40000/ 0.53s

**Two gates sit between a restored slot and the longer state in the cache, and
opening only the first one changes nothing.** That is not a guess about the
second patch; patch 1 was built and measured on its own first, and it moved
neither number.

Arm 1 is the control throughout: the same continuation without a restore in
front of it, warm from the cache in every build. Its stability across all three
(0.32 / 0.32 / 0.35 s) is what makes arm 2's change attributable to the patches
rather than to the machine.

## Gate 1 — the lookup is gated on the save condition

`update_cache` decides two things in one flag: whether the outgoing state is
SAVED into the prompt cache, and whether a cached state is LOOKED FOR. It is
set when `f_keep < 0.5f`, a condition whose comment describes only the saving
half ("if we are about to lose a large portion of the existing context"). A
slot that keeps most of its context therefore never reaches the lookup.

After a restore that is always the case: the slot holds exactly what was
loaded, so `f_keep` is 1.0.

Patch 1 adds a second reason to enter the same block — the cache holding
something better — and exposes `server_prompt_cache::has_better()` for it. The
existing search loop is split out as `find_best()` so both callers share one
implementation; the selection rule itself is untouched.

## Gate 2 — the selection rule cannot be satisfied at all

This is the part the first measurement exposed, and it was NOT visible from
reading the code. `find_best` requires a candidate to improve on both metrics:

```
if (f_keep_best < f_keep_cur && f_sim_best < f_sim_cur)
```

With this run's numbers:

    slot  (restored, 20000)   f_keep = 1.00000   f_sim = 0.49990
    cache (B, 40001)          f_keep = 0.99998   f_sim = 0.99980

    1.00000 < 0.99998  =  FALSE      →  rejected

The cached entry covers **twice as much of the prompt** and loses over
**0.00002** of f_keep. And this is not a near-miss that better tuning would
fix: once the incumbent is a complete prefix of the new prompt its f_keep is
1.0 by construction, and any entry reaching further is necessarily below it —
because reaching further past the common prefix is exactly what lowers f_keep.
Such an incumbent is unbeatable however much more of the prompt an entry
covers.

Patch 2 lets f_sim decide alone in that one case. The incumbent is saved to the
cache before a load replaces it, so the trade costs no context.

## What patch 2 does NOT change

A rule tested only on the case it was written for is not a tested rule. The
condition was evaluated against six cases before building — the measured one
and five that must keep their previous outcome:

    case                                            before   after
    the measured one: slot is a complete prefix     False    True    <- the fix
    empty slot (baseline -1)                        True     True
    candidate covers LESS -> reject                 False    False
    slot not full, candidate worse in f_keep        False    False
    slot not full, candidate better in both         True     True
    candidate under the 0.25 guard                  False    False

Exactly one outcome flips. The `f_keep_cur < 0.25f` guard against trashing large
prompts and the rejection of entries covering less are both intact.

## What this run does not settle

* **Behaviour with a full cache over time.** The table above tests the rule's
  logic, not what many entries do to it across hours of real traffic. This run
  had two entries and a probe.
* **Whether gate 1 is needed at all if gate 2 is fixed.** Not measured — patch 2
  was never built alone. The reasoning says yes (the lookup still has to be
  reached), but that is an argument, not a measurement.
* **Any effect on throughput.** `find_best` now runs on slot selections it
  previously skipped, once per candidate over `get_common_prefix`. Expected to
  be negligible against a prefill and not measured.
* **Other backends and models.** One machine, one model, gfx1151/ROCm.

## The instrument said so itself

The suite's verdict text for arm 2 reads: *"this is the cell most likely to be
true for a reason other than the one being tested."* That warning is correct
and stays worth heeding — the attribution here rests on arm 1 being unchanged
across all three builds and on the two binaries differing by exactly the two
commits (`git log 62850522e..6bbb3eecf` is two lines, `nm` finds `find_best`
only in the patched one).

## Upstream

The patch is a proposal, not a submission. llama.cpp's CONTRIBUTING forbids
AI-written issues, PRs and comments; the measurements, the table and the
reproducer may be handed over, the prose of any contribution has to be the
operator's own. The commits here carry no `Co-Authored-By` for that reason and
would need to be re-authored for a real PR.

Related upstream, neither of them this: **#24746** (closed) — explicit
`id_slot` requests bypass the same block, same effect from a different cause.
**#26676** (open) — slot restore reported as a no-op.

## Reproduce

    D=~/.cache/slot-save-cost && mkdir -p $D
    python3 bench/sideserver.py --env setup/env/flashnext.env --port 8081 \
        --stop "llama-user@$(bash setup/lib/models.sh serving)" --deadline 45 \
        --bin <absolute path to the llama-server under test> \
        --extra "--slot-save-path $D/" -- \
        python3 bench/suites/restore-vs-cram.py --url http://127.0.0.1:8081 \
            --dir $D --out $D/rows.json

`--bin` is taken literally by sideserver.py, so it needs an ABSOLUTE path — a
relative one fails at `systemd-run` with "Failed to find executable", which is
at least a clean refusal rather than a silent start of the wrong binary.

Build the subject into a family of its own, never activated:

    PATCH_BRANCH=<branch> bash setup/scripts/build-llama.sh \
        --ref <branch> --family cachelookup

## Files

    0001-0002-restore-lookup.patch   both commits, format-patch
    ab-unpatched.json / ab-patched.json / ab-patched2.json
    (the three run logs stay local — *.log is gitignored repo-wide)
    (the suite: bench/suites/restore-vs-cram.py)
