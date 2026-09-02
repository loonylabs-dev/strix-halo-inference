# A restored state DOES serve the continuation of its own conversation

02.09.2026, 18:15–18:28. flashnext (b10743-15-g62850522e) on a side server,
port 8081, `--slot-save-path` and `-cram 0`, production stopped and restored
around it. `bench/suites/restore-continuation.py`, one run, nine cells.

## Why this was run at all

`bench/reports/2026-08-29_restore-semantics/` concluded:

> A restored state is only reused when it is a PREFIX of the incoming prompt.
> A state carrying anything beyond that — even the question it was saved with
> — is discarded whole, not trimmed back to the common part.

That verdict was read since as "saving a live session is pointless", and the
save policy was built around isolating a bare prefix instead. But the report's
own scripts never sent a continuation. `postanswer.py` saves `prefix+Q1+A1`
and then posts `SYN.body(question=Q2)` — a fresh body with a different first
question, which the saved state is not a prefix of. `turnproof.py` is three
separate one-question bodies. Both are the report's control B seen from the
wrong side.

A Claude Code turn N+1 is turn N plus the model's answer plus a new user
message. The saved state IS a true prefix of it — the exact condition the
report names — and no cell had ever tested that shape.

## The measurement

    cell                                      s   reused   computed
    1  build turn 1 (cold)                 71.7        0     14942
       build turn 2                         0.8    14943        25
       build turn 3                         1.0    14970        25
       settle — the slot now holds S        0.9    14997        25
    2  save S                               0.1   n_saved=15024, 627 MB, 129 ms
    3  displace with a foreign prefix      71.3        0     14944
    4  CONTROL cold: continuation, no restore
                                           68.3        0     15049
    5  displace again                      67.2        0     14944
    6  restore S                            0.1   n_restored=15024, 627 MB, 142 ms
    7  MEASUREMENT: continuation, restored  0.8    15024        25
    8  CONTROL warm: ordinary follow-up     0.8    15051        25
    9  CONTROL: the 29.08. shape
       displace                            67.1        0     14944
       restore S                            0.1   n_restored=15024
       fresh body, other question          67.4        0     14942

**Cell 7 reused 15,024 of the 15,024 tokens it restored — all of them —
and ran in 0.8 s where the same continuation without a restore took 68.3 s.
Factor 85.**

## What makes the cells admissible

Each control answers "what would this have reported if the thing were fine?"
(bench/README).

* **Cell 4** is cell 7 with the restore removed: 0 reused, 68.3 s. So the
  displacement in cell 3 really displaced, and cell 7's warmth cannot be
  left-over residency. The suite aborts here if this cell comes back warm.
* **Cell 8** is an ordinary follow-up: 15,051 reused. The instrument can see
  warm, so a zero in cell 7 would have meant something.
* **Cell 9** rebuilds the 29.08. shape on today's binary: 0 reused against a
  head of 14,942 tokens. The old rule is intact and unchanged — the state was
  discarded WHOLE, not trimmed back to the common head. Had it been trimmed,
  cell 9 would have returned roughly the head; it returned nothing.
* **`-cram 0`** removes llama.cpp's own RAM prompt cache from the experiment,
  so everything cell 7 reused came from the file. Passed through sideserver's
  `--extra`, it lands after the profile's `-cram 14336` and wins: `arg.cpp`
  assigns `cache_ram_mib` rather than accumulating it. Cell 4 is the check
  that this actually took — with a cache in play it would have been warm.

## What this settles

The 29.08. rule stands, and a session continuation satisfies it. Persisting a
live session state is therefore mechanically possible on this build, which
the previous reading of that report had ruled out.

It does NOT settle whether it is worth doing. Three things stand between this
result and a design, and the first is the one that decides the rest.

## What is NOT measured here

* **Depth.** This ran at 15,024 tokens. Everything that matters for the real
  case — save duration, file size, whether the reuse survives — is unmeasured
  above that. The one number this run gives is **41,700 bytes per token**
  (626,512,312 bytes for 15,024 tokens), and one point is not a line.
  Extrapolated to a 180k session that is ~6.8 GiB per file — notably SMALLER
  than the 11,440–11,498 MiB the same session costs as a RAM cache entry
  (`journalctl … 'making room'`, 02.09.), so the two are not the same
  quantity and the file must be measured on its own.
* **The write never went near a disk — CORRECTED 02.09. the same evening.**
  This first said "into the page cache, not onto the platter", which is the
  right conclusion for the wrong reason and would have left a reader expecting
  the bytes on disk a moment later. `--slot-save-path` was `/tmp/restore-cont/`
  and **`/tmp` is tmpfs here** (`findmnt -no FSTYPE /tmp` → tmpfs, 62.5 GiB).
  So 627 MB in 129 ms is RAM speed, and those bytes reach no disk at all,
  ever. What the cell does measure stands: how long the slot is BLOCKED. The
  SSD cost is a separate quantity and was not measured here — and a state
  written to tmpfs also sits in the RAM the model is pinned in, which on this
  machine is its own hazard. `bench/suites/slot-save-cost.py` refuses a
  `--slot-save-path` without a block device for both reasons.
* **Interaction with `-cram`.** Production runs at 14336, this ran at 0. A
  restore is known to blind llama.cpp's own cache lookup (`f_keep < 0.5f`,
  server-context.cpp; measured 30.08., 56.4 s against 1.0 s), so a restore
  policy on a server WITH a cache is a different experiment.
* **Real Claude Code bodies.** These are `tools/synthetic.py` shapes with 24
  tools and a stable volatile counter. A real session's counter drifts every
  turn; whether that breaks the prefix is a separate question the trace can
  answer without a server.

## Next

Measure the save duration and file size over depth — 2k / 20k / 80k / 180k on
a side server with `--slot-save-path`. That produces the line this report has
one point of, and with it the two numbers the design needs: how long the slot
is blocked per save, and how many bytes a cooldown interval costs.

## Reproduce

    mkdir -p /tmp/restore-cont
    python3 bench/sideserver.py --env setup/env/flashnext.env --port 8081 \
        --stop "llama-user@$(bash setup/lib/models.sh serving)" --deadline 60 \
        --extra "--slot-save-path /tmp/restore-cont/ -cram 0" -- \
        python3 bench/suites/restore-continuation.py \
            --url http://127.0.0.1:8081 --out /tmp/restore-cont/rows.json

## Files

    rows.json   the cells, machine-readable
    (run.log stays local — *.log is gitignored repo-wide, so the numbers
     that matter are carried in the table above and in rows.json)
    (the suite: bench/suites/restore-continuation.py)
