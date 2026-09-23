# Halogen 0.8.1 vs 0.12.3, and the setting that matters more than either

**Measured 21.09.2026** on this machine (Ryzen AI MAX+ 395, 128 GiB, gfx1151),
`bench/sessionload.py`, `platform_profile=performance` verified at both ends of
every run. Four runs, identical documents throughout (fixed salt `ab2609`, so
every run built the same bytes), thinking off on both releases so the two do the
same work. `HALOGEN_KV_POOL_POSITIONS=524288`, `slot_ctx=262144`, 4 slots,
vision tower on — the shape production serves.

## Why this was run

`bench/logstats.py` over the 4,423 turns 0.8.1 served between 17.09. 01:59 and
21.09. 14:00 showed the served load was 90% turns under 2,000 tokens, where
prefill p50 is 1.45 s and p99 is 1.54 s. **Nothing can be won there.** The bands
above it read p99 17.8 s / 44.3 s / 240.5 s on 20 to 250 incidental turns, so
the case that decides an upgrade had no measurement at all. The operator's
recalled problem shape was two sessions at 180k–230k beside one at 80k. That is
what these runs reproduce.

## The runs

| run | release | docs (tokens) | `max_tokens` | reservation | wall |
|---|---|---|---|---|---|
| A | 0.12.3 | 29k × 3 | 32,000 | 35% of pool | 95 s |
| B | 0.12.3 | 193k / 183k / 77k | 32,000 | **129%** | 2,713 s |
| C | 0.12.3 | 193k / 183k / 77k | 8,000 | 94% | **466 s** |
| D | 0.8.1 | 193k / 183k / 77k | 8,000 | 94% | 1,191 s |

Six turns per session, round robin (`--mode pingpong`), so every session's next
turn lands after every other session has taken a region. Turn 1 is cold by
construction and is excluded from the follow-up figures.

## Finding 1 — the reservation, not the release, decides whether the cache works

A KV reservation is **prompt plus `max_tokens`**, and Claude Code sends
`max_tokens=32000` while using a few hundred. Three deep sessions therefore
reserve 675k positions in a 524,288 pool. Runs B and C are the same release and
the same prompts:

| | B, `max_tokens` 32,000 | C, `max_tokens` 8,000 |
|---|---|---|
| reservation | 129% of the pool | 94% |
| follow-up turns p50 | **184 s** | **0.95 s** |
| follow-up turns p90 | 194 s | 1.20 s |
| cache hit rate | **0.0** | 0.83 (every follow-up) |
| entries evicted | 32 | 0 |
| regions held | 2 of 3 | **3 of 3** |

**Every single turn in run B was a full re-prefill**, and the server said so at
every one of them:

    kv pool: no room for 225024 positions; forgot the region at 0
             (2 cache entries, 0 chunks; the least recently used)

225,024 is one deep session's reservation (193k prompt + 32,000 budget, rounded
to block boundaries). Three of those do not fit a pool that holds two, and no
placement algorithm can change that: `moved`, `relocated`, `packed` and
`cold_resorts` all read 0 in run B — 0.11.5's and 0.11.7's machinery never even
engaged. **This is capacity, not a defect.**

Filed as `halogen-kv-reservation-is-prompt-plus-max-tokens` in
`setup/defects.json`.

## Finding 2 — below the line, the release is worth 5 lost turns out of 15

Runs C and D are the same prompts, the same 94% reservation, thinking off on
both. Only the server differs:

| | D, 0.8.1 | C, 0.12.3 |
|---|---|---|
| follow-up turns p50 | 1.17 s | 0.95 s |
| **follow-up turns p90** | **187.48 s** | **1.20 s** |
| follow-up turns max | 197.68 s | 1.22 s |
| turns that lost their cache | **5 of 15** | **0 of 15** |
| cache hit rate | 0.556 | 0.833 |
| entries evicted | 8 | 0 |
| wall for the same work | 1,191 s | **466 s** |

**The median hides this and the p90 shows it.** On 0.8.1 five follow-up turns
re-prefilled from scratch: two at turn 2 (sessions 1 and 2) and then all three
at turn 5. The same turns on 0.12.3 cost under 1.3 s.

What 0.12.3 does instead is visible, and it is 0.11.7's fix verbatim — three
times, each `(no loss)`, each in tens of milliseconds:

    kv pool: no room for 191232 positions; moved 1 held regions (76934 rows,
             20.1 ms) up against the next busy region and the region at 200960
             this request resumes from grows in place (no loss)

    kv pool: no room for 85248 positions; moved the 76972 rows this request
             resumes from, region 439296 -> 392192 (overlapping, copied in
             chunks) in 20.3 ms (no loss)

    kv pool: no room for 201216 positions; moved 2 held regions (260099 rows,
             66.8 ms) up against the next busy region and the region at 0 this
             request resumes from grows in place (no loss)

The second one is the overlapping-copy case 0.11.7 was written for. On 0.8.1
that situation is one of the five losses.

## Finding 3 — on 0.8.1 the loss is invisible

0.8.1 printed **no line at all** for its five evictions: no `no room`, no
`forgot`, and `/cache` returns `pool: null` because the pool counters arrived in
0.11.5. A turn that suddenly costs 187 s instead of 1.2 s looked like the model
being slow. This is what the 0.11.0 changelog means by "which read as a cold
prefill with no line in the log", and it is the reason the 17.09. investigation
could measure the *consequence* of the KV pool being too small without ever
seeing the *event*.

For an operator this is the most consequential of the three findings: the stack
could not previously tell "your session lost its cache" from "the model is
slow", and now it can.

## What does NOT follow from these runs

* **No statement about the flat band.** Runs A–D all use deep sessions. The
  served baseline says the flat band is at 1.45 s p50 / 1.54 s p99 on 0.8.1 and
  nothing here improves on that or needs to.
* **No throughput claim.** These runs reproduce one shape under contention; they
  do not look for a maximum rate.
* **Turn-1 prefill is unchanged between releases** (198/187/74 s on 0.8.1 vs
  195/184/74 s on 0.12.3, within 2%). 0.12.0's indexer work pays at 262k and
  above, and these prompts sit at 193k — consistent with upstream saying 32k is
  unchanged, but not a test of it.
* **n=1 per cell.** Each configuration ran once; the effect sizes here (194x,
  and 5 losses against 0) are far outside any plausible run-to-run variation,
  but a 20% difference measured this way would not be trustworthy.
* **`max_tokens=8000` is not a recommendation with a number behind it.** 8,000
  was chosen to land under 100% of the pool for THIS session mix. The rule is
  the arithmetic, not the constant: `sum over live sessions of (prompt +
  max_tokens) <= HALOGEN_KV_POOL_POSITIONS`.

## Consequences

1. **Clamp `max_tokens` before it reaches Halogen.** At 94% the pool holds three
   deep sessions; at 129% it holds two and thrashes. The lever is in the
   gateway, needs no upstream change, and is worth more than the upgrade on its
   own. Related to `HALOGEN_FIT_TO_ROOM` but at the other end: fit-to-room
   clamps against the context WINDOW, this would clamp against the POOL.
2. **The upgrade is justified independently**: 5 lost turns out of 15 become 0,
   and the losses become visible when they do happen.
3. **Do not raise the pool to fix this.** 786,432 positions would hold three
   32,000-budget sessions, and this repo has that measurement already: it leaves
   5.6 GiB of host RAM, the 47.7 GiB PLE table falls out of the page cache, and
   the result was six watchdog shutdowns
   (`halogen-pagecache-starvation-watchdog-kill`). Lowering the reservation
   costs nothing; raising the pool costs the page cache.

## Files

* `../2026-09-21_1400_halogen-0.8.1-served-baseline/` — the 4,423 served turns
  and what they do and do not cover
* `../2026-09-21_1437_concurrent_0.12.3-A/` — run A
* `../2026-09-21_1525_concurrent_0.12.3-B-overbooked/` — run B
* `../2026-09-21_1537_concurrent_0.12.3-C-maxtok8000/` — run C
* `../2026-09-21_1602_concurrent_0.8.1-C-maxtok8000/` — run D
* each carries `context.json` (server identity, pool, profile at both ends),
  `turns.json` (every turn) and `summary.json`
