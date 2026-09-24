# Prompt cache on disk (`HALOGEN_CACHE_DIR`) — 0.13.8, 24.09.2026

**Question.** Deep sessions that do not fit the KV pool together evict each
other, and an evicted session re-reads its whole prompt on its next turn
(21.09.2026: p50 184 s, 0.12.3). Upstream 0.10.0 added a disk tier under the
prompt cache: every turn's new rows are written behind the request, and a
request whose rows are no longer in memory is restored from disk. Upstream
measured it across a restart only. Does it also catch an eviction in a
running server, on this machine, at production's pool size?

**Setup.** Halogen 0.13.8 with the quality sidecar (production's
configuration: `~/.config/llm-stack.env`, pool 524288, `HALOGEN_FIT_TO_ROOM=1`),
started as a side unit from the night worktree's `halogenexec` — which is
what mounts the directory; see "What it needed here". `bench/sessionload.py
--mode pingpong --sessions 3 --turns 4 --doc-tokens 193000,183000,77000
--max-tokens 32000`: three conversations taking turns, reserving ~549k
positions of 524288 (104.7 %), so every turn evicts a neighbour. Same salt,
so all three arms send byte-identical prompts.

- **A** memory only (production today)
- **B** + `HALOGEN_CACHE_DIR` on btrfs (`/mnt/shared`, NVMe), fresh directory
- **C** B's directory kept, server stopped and started again, same prompts

Conditions: `platform_profile` performance at both ends of every arm.
Halogen warned in every arm that 10.9-11.1 GiB of host RAM were in use
before it started (the desktop) — equal across arms. Arms B and C ran with
the worktree's `serve_api.py`, which carries vendored change 5 (a snapshot
place at a client's `cache_control` mark); sessionload sends plain string
messages, for which change 5 does nothing.

## Result

| | turn 1 (cold) | turns 2-4, p50 | p90 | disk hits | restored |
|---|---|---|---|---|---|
| A memory only | 192 / 176 / 71 s | **175.4 s** | 186.8 s | — (RAM hits 0 of 12) | — |
| B + disk | 188 / 177 / 71 s | **2.32 s** | 2.81 s | 9 of 9 | 37.0 GB in 7.0 s |
| C restart, same dir | **2.7 / 2.1 / 0.7 s** | **1.84 s** | 5.32 s | 12 of 12 | 49.3 GB in 9.3 s |

The server's own log shows the same ten `no room … forgot the region` lines
in A and B: the eviction still happens, and with the disk tier it costs a
restore instead of a prefill. Per turn: `prompt 186129 (186048 cached)` in B
where A reads 186129 new tokens.

- Restore rate: 37.0 GB in 7.0 s is ~5.3 GB/s, 49.3 GB in 9.3 s the same.
- Size on disk: 13 GB after B for three conversations of 186k + 176k + 74k
  tokens, ~30 KiB per token (upstream: ~27 KiB).
- Shutdown: the server exited on its own 10 s after SIGTERM, `shutting down`
  in its log; C's 12/12 hits say the last turns had been written.

## What it needed here

`HALOGEN_CACHE_DIR` names a directory INSIDE the container, and
`setup/halogenexec` forwards an allowlist of variables only — so the
variable set in `llm-stack.env` would have reached nothing, silently.
halogenexec now mounts the host directory at `/cache` and passes
`HALOGEN_CACHE_DIR=/cache`, and forwards `HALOGEN_CACHE_DISK_GIB` and
`HALOGEN_CACHE_PRUNE_OLD` (`tests/test_halogen.py`,
`TestTheDiskCacheReachesTheContainer`). The directory must accept direct
I/O: btrfs on `/mnt/shared` does (the startup line says `prompt cache on
DISK`); `/tmp` is tmpfs here and would be refused.

## What this does not show

- The WRITE side under a nearly full pool. Upstream warns the row copy slows
  to a few hundred MB/s near the ceiling. In this shape a session's next turn
  came minutes after its last, and every restore hit — a turn that follows
  its own previous turn within seconds, beside a full pool, is not measured.
- Real traffic. This is the synthetic eviction case; the operator's shape
  (a main session idle while two subagents and Claude Code's side requests
  fill the pool, 24.09. 13:23, a 53k main turn back cold in 51.7 s) runs
  through the same restore path but was not replayed.
- Disk wear: not measured. A new 190k conversation writes ~5.5 GB once, a
  follow-up turn only its new rows; `HALOGEN_CACHE_DISK_GIB` (default 64)
  bounds the directory, least recently used first.
