# Halogen 0.14.0 against 0.13.8 — side measurements, 26.09.2026

Three questions, one sitting (08:16–09:29, production down for all of it,
the machine otherwise idle, `platform_profile` performance at every arm's
start and end):

1. The new draft head (upstream: depth 2 by default, 2.47 committed tokens a
   round against 1.68, +15-18 % tokens/s on their prompts). What does it do on
   this machine's traffic, and is depth 3 worth setting?
2. Public issue #106: does a request that `</block>` ends report its real
   timings now?
3. Public issue #107: do `timings.disk_restore_n` / `disk_restore_ms` say
   what the disk tier did, per request?

**Verdicts.** (1) Depth 2 decodes 35.4-35.9 → 39.0-40.3 tok/s on long
prose answers (+10-12 %, not upstream's +15-18 %) and cuts the classifier
replay's wall sum by 8-11 %. Depth 3 is no better than 0.13.8 on prose and
no better than depth 2 on the classifier. Leave `HALOGEN_MTP_DEPTH` unset.
(2) Fixed: 23/23 calls reported no `cached_tokens` on 0.13.8, 0/23 on 0.14.0.
The end-token stop strings the gateway stopped sending on 25.09. still empty
the turn on 0.14.0 (6/6), so that gateway change stays; and #106's fix had
made `stopstring_ab`'s detector blind to it (fixed here). (3) Works as
described.

## Method

Five arms in the order P, N2, N3, N2b, Pb (ABCBA, so a drift over the
morning would show as P ≠ Pb or N2 ≠ N2b; neither moved), each a fresh side
server in production's configuration (4 slots, pool 524288, sidecar, disk
tier on) with its OWN empty cache directory:

| arm | image | serve_api | `HALOGEN_MTP_DEPTH` |
|---|---|---|---|
| P, Pb | 0.13.8 | the 0.13.8 cut | unset (0.13.8 drafts depth 1) |
| N2, N2b | 0.14.0 | the 0.14.0 cut | 2 |
| N3 | 0.14.0 | the 0.14.0 cut | 3 |

Per arm, in this order:

- `bench/classreplay.py`, session `753f92836b67` of 24.09. (24 recorded
  auto-mode classifier calls, 32-53k tokens, thinking on, `</block>` stop,
  the client's cache marks — the agent shape, and #106's case). Call 0 is
  cold and left out of the sums below.
- `bench/sessionload.py --long-answer --sessions 1 --turns 4 --max-tokens
  1200` at `--doc-tokens` 2000 and 32000 (prose; turns 2-4 are cached, so
  what is left is the decode rate at that depth).
- N2 and N3 only: `bench/stopstring_ab.py --n 3` (does a stop string undo
  the #84 end-of-turn guard?).

Then D: one 0.14.0 server (depth unset) with an empty cache directory, a
28,316-token prompt cold, SIGTERM (the flush), a new server on the same
directory, the same prompt again, then once more.

**The first attempt (08:14) was aborted after one request.** The side unit
carried `EnvironmentFile=` for the local env file, and systemd lets that
override `-E`: the side server used production's cache directory despite the
`-E HALOGEN_CACHE_DIR`. One ~1.1 GiB record was written there; nothing was
evicted. The runner now passes no env file (halogenexec reads it itself for
every variable left unset) and refuses a side server whose `/cache` mount is
not its own before it serves. The older runners of 24.09. have the same
shape; it was invisible then because production's disk tier was not on yet.

## 1. The draft head

Long-answer decode, turns 2-4, the server's own `timings`
(completion tokens / `predicted_ms`, pooled over the three turns; per-turn
rates in brackets). Every turn ended on `stop`, not on the 1200 cap.

| arm | 2k | 32k |
|---|---|---|
| P 0.13.8 | 35.7 (35.7 35.2 36.2) | 35.8 (35.7 35.6 36.2) |
| **N2 0.14.0 d2** | **39.0** (40.0 37.3 39.6) | **40.2** (41.4 39.7 39.6) |
| N3 0.14.0 d3 | 35.4 (37.1 33.8 35.2) | 36.7 (38.3 35.9 36.2) |
| **N2b 0.14.0 d2** | **39.2** (40.3 37.6 39.8) | **40.3** (41.8 40.1 39.2) |
| Pb 0.13.8 | 35.4 (35.5 35.1 35.7) | 35.9 (35.8 35.7 36.1) |

Classifier replay, calls 1-23:

| arm | verdicts (no / yes / unusable) | wall p50 | wall sum | output tokens | decode tok/s | prefill p50 |
|---|---|---|---|---|---|---|
| P 0.13.8 | 22 / 0 / 1 | 22.8 s | 509 s | 16,810 | — (#106) | — (#106) |
| N2 0.14.0 d2 | 22 / 0 / 1 | 21.0 s | 470 s | 16,884 | 41.6 | 2.18 s |
| N3 0.14.0 d3 | 22 / 1 / 0 | 19.9 s | 462 s | 16,778 | 42.2 | 2.24 s |
| N2b 0.14.0 d2 | 23 / 0 / 0 | 19.1 s | 453 s | 16,411 | 42.1 | 2.12 s |
| Pb 0.13.8 | 23 / 0 / 0 | 24.4 s | 507 s | 16,945 | — (#106) | — (#106) |

Committed tokens per draft round, read off the server logs' `mtp … commit
x/round` lines: 0.13.8 **1.52**, depth 2 **1.89 / 1.92**, depth 3 **1.96**
(upstream: 1.68 → 2.47 on their prompts). WEAK: 0.13.8 logs no line for a
request a stop string ends (#106 again), so its figure comes from the 8
prose turns alone, while the 0.14.0 figures mix prose and classifier.

Why +10-12 % here where upstream measured +15-18 %: NOT measured. The
operator's guess, plausible but untested: this machine is a laptop whose GPU
holds ~70 W sustained (global machine notes, 03.09.2026), upstream's
reference machine a desktop Strix Halo. Their prompts differ from ours as
well, and the committed-per-round gap (1.9 here against their 2.47) points
at the traffic at least as much as at the watts.

What it means: depth 2 is the gain and it holds across both repeats; depth
3 proposes one more token and gets 0.04-0.07 more accepted a round, which
does not pay for proposing it on this traffic. Upstream says the same for
agent and prose traffic ("3 is faster on code generation"); no code shape
was measured here. The wall sums fall less than the decode rate rises
because prefill and the per-call overhead did not change.

The yes verdicts on these benign calls: N3 one in calls 1-23, and N3 and
N2b one each on the cold call 0 (left out of the table); the 24.09. baseline
on 0.13.8 had one in 24. With n=24 per arm that is not a difference between
the releases or the depths.

## 2. #106 — `</block>` requests report their figures

| | calls with `cached_tokens` absent | `prompt_ms` p50 |
|---|---|---|
| 0.13.8 (P, Pb) | 23/23, 23/23 | 0.0 (the made-up record) |
| 0.14.0 (N2, N3, N2b) | 0/23 each | 2.12-2.24 s |

The cached share is the one change 5 (`setup/halogen/serve_api.py`) exists
for, e.g. call 23 on N2: 51,328 of 53,483 tokens. The gateway's trace can
count these requests as warm now.

The second bug upstream found while fixing it (a multi-token stop string
left its first pieces in the text) is in these rows too: every `no` on
0.13.8 was 16 characters, `<block>no` plus a trailing `</block` (47/47);
on 0.14.0 it is 9, `<block>no` (68/68; the three `yes` are 10). The verdict
parser read both, which is why the verdict columns never showed it.

`stopstring_ab` (the gateway's former end-token stop strings against the
#84 guard), n=3 per arm on N2 and N3:

| arm | runs | empty (no content, no tool call) | of them timings zeroed | reasoning kept the end token |
|---|---|---|---|---|
| gateway-stops, N2 / N3 | 3 / 3 | **3 / 3** | 0 / 0 | 0 / 0 |
| no-stops, N2 / N3 | 3 / 3 | 1 / 1 | 0 / 0 | 3 / 3 |
| gateway-stops, 25.09. (0.13.8) | 5 | 5 | 5 | 0 |

The stop string still fires on the end token the guard keeps in the
reasoning, and ends the turn empty — 25.09.'s finding, unchanged on 0.14.0;
only the zeroed timings are gone. Production is not exposed: the gateway has
sent no end-token stop strings since `6652c19`. **The tool's own summary
printed 0 here** — it counted an empty turn only with zeroed timings, the
0.13.8 signature — and read as "guard intact" at first. `summarize()` now
counts empty turns and zeroed ones apart (`tests/test_stopstring_ab.py`).
The empty no-stops turns (1 per arm) are the model ending after its
reasoning without an answer; the 25.09. report has the same side finding.

## 3. #107 — disk restore per request

| request | `cached_tokens` | `disk_restore_n` | `disk_restore_ms` | `prompt_ms` | wall |
|---|---|---|---|---|---|
| cold | — | 0 | 0.0 | 28,735.9 | 30.0 s |
| after a restart | 28,288 | **28,288** | **213.5** | 711.5 | 2.3 s |
| again (RAM) | 28,316 | **0** | 0.0 | 0.3 | 1.3 s |

The record after the flush: 0.83 GiB for 28,316 tokens. `cache_n` still
counts the whole reused prefix; the RAM part is `cache_n - disk_restore_n`,
as the maintainer wrote.

## What an upgrade of production costs — found here, not measured on it

- **The disk tier's key changes with the build AND with the configuration.**
  Same directory, three keys: 0.13.8 `45453a85…`, 0.14.0 with
  `HALOGEN_MTP_DEPTH` unset `1eab201a…`, set to 2 `eab19302…`, set to 3
  `5b593ede…` — so setting the default value explicitly is a different
  configuration to the cache. Production runs `HALOGEN_CACHE_PRUNE_OLD=1`,
  whose startup line reads "removed another configuration's subtree": the
  first 0.14.0 start deletes the 0.13.8 records (200 GiB) and every session
  prefills cold once. A rollback to 0.13.8 would find a cold cache the same
  way.
- The pool is 524288 in every arm: `halogenexec` passes
  `HALOGEN_KV_POOL_POSITIONS` explicitly, so 0.14.0's own fit (262144 with
  the vision tower on 128 GB) never applies here.
- The side arms' cache directories hold 167 GiB after the run.

## Files

`classreplay-cls-*.jsonl` (per call), `decode-*/` (sessionload reports),
`stopstrings-*.jsonl`, `disk-restore.jsonl`, `cache-D2.json`, and each side
server's log (`server-halogen-bench-*.log`). Component versions: Halogen
images 0.13.8 (`sha256:6e626c97…`) and 0.14.0 (`sha256:ec7ec0c6…`),
checkpoint `qwen38-flash-next-w4b.hgn` with the quality sidecar, ROCm as
shipped in each image, host kernel 6.19.10.
