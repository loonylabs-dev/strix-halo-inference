# What writing a session state costs — a straight line, and 1.9 s at full depth

02.09.2026, 18:56–19:23. flashnext (b10743-15-g62850522e) on a side server,
port 8081, `--slot-save-path ~/.cache/slot-save-cost/` and `-cram 0`,
production stopped and restored around it. `bench/suites/slot-save-cost.py`,
one run, five points, all reached.

## The line

        point   n_saved   reported MB   on disk MB   B/token   save ms
         2000      2000           186          188     92864        49
        20000     20000           795          799     39746       232
        80000     80000          2826         2843     35320       817
       140000    140000          4856         4870     34687      1364
       180000    180000          6210         6227     34500      1889

Point to point, the slope does not move:

       2000 → 20000      33833 B/token
      20000 → 80000      33850 B/token
      80000 → 140000     33833 B/token
     140000 → 180000     33850 B/token

**A saved state is 118 MB + 33,844 bytes per token, and the four intervals
agree to within 0.05 %.** The B/token column falls only because the fixed
118 MB is being amortised; there is no super-linear term anywhere in this
range. The 92,864 B/token at the 2,000-token point is that offset and nothing
else.

## The two numbers this was run for

**The slot is blocked for 1,889 ms at 180,000 tokens** — 49 ms at 2,000, so
roughly 10 µs per token. `SLOT_SAVE` runs in the main task loop
(server-context.cpp:2517) and defers while the slot is processing, so nothing
generates or prefills anywhere on the server for that time.

**6,227 MB reach the disk for one 180k state** — measured from
/proc/diskstats around a forced `sync`, not from the server. It sits 0.3 %
above the 6,210 MB the server reports writing, which is btrfs metadata.

The idle baseline — the same sync window with no save in it — was 12.0 MB.
Against the smallest measured point (188 MB) that is 6.4 %, and against the
deepest (6,227 MB) 0.2 %. The disk column is usable; at the 2,000-token point
it is the loosest.

## The comparison that decides it

    writing a 180k state       1.9 s   (this run)
    recomputing a 180k state   ~1800 s (this run's own prefill, see below)

**A factor of about 950.** A save would have to be wasted 900 times over
before it costs what one lost state costs. Latency is not what a cooldown has
to protect against; SSD wear is the only remaining argument for spacing the
writes out.

At 6.21 GB per save that is 596 GB/day at a 5-minute cooldown over an
eight-hour day, 199 GB at 15 minutes, 99 GB at 30 — and 224 GB at 5 minutes
over the three active hours a day this machine actually sees.

What the drive says about that, read the same evening (`sudo smartctl -A
/dev/nvme0n1`, Sandisk PC SN5100S 2 TB): **Percentage Used 0 %** after
**2.73 TB** written, Available Spare 100 %, zero integrity errors. A counter
still on zero bounds the endurance from BELOW and not from above — it says
2.73 TB has not consumed a measurable percent, and nothing more. The rated
TBW is not in the SMART log and is not guessed at here. Two hundred GB a day
would double the drive's lifetime write volume in under a fortnight, which is
the honest way to state the scale; turning that into years needs a second
reading of the same counter weeks from now.

## The restore holds at full depth

    displaced (tiny unrelated prompt, LRU path, -cram 0 so no RAM cache)
    restore                1250 ms   n_restored=180000   6210 MB
    continuation            0.6 s    cache_n=180000      prompt_n=9

180,000 of 180,000 tokens carried, 9 computed — the prompt plus the token the
model generated plus 8 new ones. This closes the depth question the
15,024-token run of 18:15 left open: the 02.09. restore-continuation result
is not an artefact of a small state.

The continuation is built as a true SUPERSET of the saved state (prompt +
generated tokens via `return_tokens` + new tokens). Without the generated
token the state would carry something the prompt does not, and llama.cpp
discards such a state whole — the check would have said NO for a reason
having nothing to do with depth.

## The file is not the cache entry

    180k as a FILE            6,210 MB   (5,923 MiB, this run)
    180k as a RAM cache entry 11,440 and 11,498 MiB
                              (journalctl 'making room', 02.09. 14:42 and 17:03)

The file is 52 % of the entry. They are different quantities and neither
extrapolates to the other: the cache entry carries a surcharge that GROWS with
depth (6.8 → 16.6 → 24.6 KiB/token at 80k/148k/179k, HANDOVER), the file does
not grow beyond its straight line at all.

This also corrects an extrapolation made earlier the same evening. The
restore-continuation run's single point gave 41,700 B/token and a predicted
~6.8 GiB at 180k; the real figure is 34,500 B/token and 6.21 GB. That point
sat inside the 118 MB offset, which one point cannot separate from a slope.

## Prefill, measured in passing

Each step computed only its delta, so the run also traced the prefill rate
down the context:

         0 →   2000    7.9 s     253 t/s
      2000 →  20000   76.6 s     235 t/s
     20000 →  80000  426.8 s     141 t/s
     80000 → 140000  687.3 s      87 t/s
    140000 → 180000  601.6 s      66 t/s

The 66 t/s at the deep end matches the production trace of the same day
independently — 62.8 t/s over 44,062 tokens at 14:37 and 61.6 over 47,650 at
17:17 — which is worth having, because those two figures came from a
completely different path (real Claude Code bodies through the gateway).

## What is NOT measured

* **The save under contention — ANSWERED the same evening**, in
  `bench/reports/2026-09-02_2010_save-under-load/`: the colliding turn keeps
  its state and pays only the wait, and the save is deferred rather than
  interleaved. Neither defect of 28.08. reaches this path. That report also
  shows the save time here is the bottom of a range — the same 180k state
  wrote in 1996, 3070 and 2971 ms, 54 % spread — so the 1,889 ms above must
  not be quoted as a constant.
* **Drive endurance beyond "0 % used".** See the SMART reading above: it
  bounds from below only.
* **`-cram` interaction.** This ran at 0; production runs at 14336. A restore
  blinds llama.cpp's own cache lookup (`f_keep < 0.5f`), so a policy on a
  server WITH a cache is a different experiment.
* **Beyond 180k.** `-c` is 204800 and the run stopped short of it.
* **Real Claude Code bodies.** These are random-token prompts of exact
  length. Nothing suggests the KV state cares, but nothing here shows it.

## Reproduce

    D=~/.cache/slot-save-cost && mkdir -p $D
    python3 bench/sideserver.py --env setup/env/flashnext.env --port 8081 \
        --stop "llama-user@$(bash setup/lib/models.sh serving)" --deadline 90 \
        --extra "--slot-save-path $D/ -cram 0" -- \
        python3 bench/suites/slot-save-cost.py --url http://127.0.0.1:8081 \
            --dir $D --out $D/rows.json

About 30 minutes, almost all of it the one 180k prefill — the intermediate
points are prefixes of the deepest and cost nothing. NOT `/tmp`: it is tmpfs
here, and the suite refuses a --slot-save-path without a block device.

## Files

    rows.json   the points, machine-readable
    (run.log stays local — *.log is gitignored repo-wide; every figure it
     carries is in the table above)
    (the suite: bench/suites/slot-save-cost.py)
    The 6.2 GB state of the deepest point is left at
    ~/.cache/slot-save-cost/slot-save-cost-4.bin — delete it when done.
