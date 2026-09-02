# A restore hides a longer state the cache already holds — 330x, in production configuration

02.09.2026, 20:50–20:57. flashnext (b10743-15-g62850522e) on a side server,
`--slot-save-path` and the profile's own **`-cram 14336`** — the one suite of
the four that measures the RAM cache instead of switching it off.
`bench/suites/restore-vs-cram.py`, one run.

## Why this had to be run before anything is designed

Every earlier result of 02.09. — that a restored state serves its own
continuation, what a save costs, that a collision costs only waiting — was
measured at `-cram 0`, so that nothing but the file could answer. Production
runs 14336. And llama.cpp consults its RAM prompt cache only `if (f_keep <
0.5f)`: a restore fills the slot, f_keep goes large, the lookup is switched
off. Measured 30.08. (2026-08-30_restore-blinds-cache): 56.4 s against 1.0 s.

For a session-persistence design that is the sharp end. The FILE holds turn N;
the CACHE may hold turn N+k; and a restore that puts turn N into the slot can
stop the longer state from ever being found. Then persistence makes production
slower rather than faster, and every figure of this day would be true and
useless.

## The measurement

A is a shallow state written to disk, B the same conversation grown deeper and
left to the cache. `A = toks[:20000]`, `B = toks[:40000]`, so A is a true
prefix of B by construction and the save captures exactly `toks[:20000]`
(n_saved 20000, n_predict 1).

    1  prefill A                        85.58 s   cache_n=0       prompt_n=20000
       save A                                     n_saved=20000   795 MB
    2  prefill B                       111.79 s   cache_n=20000   prompt_n=20000
       displace (LRU path saves B)       0.78 s   cache_n=0       prompt_n=24
    3  ARM 1: continuation of B          0.33 s   cache_n=40000   prompt_n=4
    4  re-establish B                    3.38 s   cache_n=39484   prompt_n=516
       displace again                    0.82 s   cache_n=0       prompt_n=24
       restore A                         0.10 s   n_restored=20000
       ARM 2: continuation of B        108.97 s   cache_n=20000   prompt_n=20008

**Arm 1 got 40,000 tokens back from the cache in 0.33 s. Arm 2, identical
except for a restore in front of it, got 20,000 and spent 108.97 s
recomputing the rest. A factor of 330.**

Arm 1 is what makes arm 2 admissible: had the cache not returned B there,
nothing in this run would be about restores — the state would simply have been
gone — and the suite aborts with that reason rather than reporting a
difference.

## What it means

The 30.08. finding stands on this build, in the configuration production
actually runs, and larger than it was measured then (330x against 56x).

The consequence for a design is not "restore carefully". It is that **the
gateway cannot make this decision at all**: deciding correctly requires
knowing whether the cache holds a LONGER state than the file, and llama.cpp
exposes no way to ask. `/slots` reports the slot, not the cache. So a restore
in a running server is a bet, and this run prices the losing side at 330x.

Which is why `RESTORE_ONLY_WHEN_SERVER_COLD=1` is what runs today, and why it
should stay: after a server restart the cache is empty, the file is the only
copy, and the bet cannot be lost.

**Session persistence is therefore worth building for the restart case and not
for the running one.** That is a smaller prize than the day's earlier results
suggested — a restart is rare — but it is a real one: today a restart throws
away every resident state, and the deepest of them costs ~30 minutes to
rebuild (2026-09-02_1856_slot-save-cost).

## What this does NOT decide

* **The eviction case.** When the cache evicts a state — the 02.09. 14:42 and
  17:03 incidents, `making room ... 11440 MiB` — the cache no longer holds
  anything and a restore would be right. The gateway could in principle learn
  that from the same journal line check.sh already counts. Whether that is
  timely enough to act on is unmeasured, and it is the one path that could
  extend persistence past the restart case.
* **Where the f_keep threshold actually falls.** This run shows the effect,
  not the boundary. A restore of a state that is a small fraction of the
  incoming prompt might leave f_keep under 0.5 and keep the lookup alive; that
  would be a different, narrower policy and is untested.
* **Whether the cache would have held B in a real session.** Here B was put
  there deliberately by an LRU takeover. Production's takeovers are the health
  probe and second sessions, and 02.09. showed both evicting rather than
  preserving at depth.

## Reproduce

    D=~/.cache/slot-save-cost && mkdir -p $D
    python3 bench/sideserver.py --env setup/env/flashnext.env --port 8081 \
        --stop "llama-user@$(bash setup/lib/models.sh serving)" --deadline 45 \
        --extra "--slot-save-path $D/" -- \
        python3 bench/suites/restore-vs-cram.py --url http://127.0.0.1:8081 \
            --dir $D --out $D/restore-vs-cram.json

About seven minutes. Note the ABSENCE of `-cram 0` — the profile's budget is
the subject here, and passing 0 would measure the arrangement this suite
exists to leave behind.

## Files

    rows.json   the two arms, machine-readable
    (run.log stays local — *.log is gitignored repo-wide)
    (the suite: bench/suites/restore-vs-cram.py)
