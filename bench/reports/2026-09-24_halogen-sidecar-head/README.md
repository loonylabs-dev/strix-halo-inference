# Sidecar draft head at 8 bits vs 4 bits — 24.09.2026

**Verdict: +0.6 % decode at 193k, +1.8 % at 2k — real (every run of the new
file beats every run of the old one at 2k, and both pairs agree at 193k),
and small. Upstream's ~4 % on prose did not reproduce here at this size.**

The quality sidecar on disk (`qwen38-flash-next-w4b.overlay.hgn`, 2.31 GiB,
downloaded 11.09.) predates Halogen 0.6.0: it carries the 723 quality tensors
but not the MTP draft head's 18 projections at 8 bits, so the head runs at the
base file's 4 bits (the container says "predates 0.6.0"). The current file on
Hugging Face (`peonist-ai/halogen-qwen3.8-flash-next`, 2,572,466,560 bytes,
sha256 `1cdfc3a9…49b39ab`, checked after download) was put beside it as
`…overlay-v06.hgn` and loaded with `HALOGEN_CK_OVERLAY=<path>`: the log said
`741 tensors, 2.40 GiB` and `30 tensors upgraded to q8g64` (old: 723 / 2.31 /
12). Upstream says the 723 shared tensors are byte-identical — read, not
measured; greedy answers were token-for-token the same length in every turn.

## Method

Halogen 0.13.8, production's environment (`llm-stack.env`,
`HALOGEN_FIT_TO_ROOM=1`), sidecar on in both arms. `bench/sessionload.py
--long-answer --sessions 1 --turns 4 --max-tokens 1200` — four different
~700-word prose questions the document cannot supply (the 22.09. decode
method, so prompt lookup has nothing to quote), greedy, thinking off — at
`--doc-tokens 193000` (cold start) and then `2000` on the same server. Order
old, new, new, old; a fresh start per arm; platform_profile performance at
every arm's end. Decode rate = completion tokens / the server's
`predicted_ms`.

## Result (tok/s per turn, median)

| arm | 193k | 2k | commit/round (median, 8 requests) |
|---|---|---|---|
| old 1 | 32.7 33.5 33.1 34.1 — **33.29** | 35.1 35.8 34.3 35.0 — **35.07** | 1.485 |
| new 1 | 32.0 33.8 33.6 33.5 — **33.52** | 35.7 35.8 35.2 35.9 — **35.75** | 1.500 |
| new 2 | 31.9 33.5 33.6 33.5 — **33.50** | 35.5 35.8 35.2 35.9 — **35.64** | 1.500 |
| old 2 | 32.6 33.6 33.1 34.2 — **33.34** | 35.1 35.8 34.8 35.0 — **35.05** | 1.485 |

## Weaknesses

- **Greedy only.** Claude Code samples at temperature 1.0, where upstream
  says the head drafts alone (no prompt lookup); acceptance there is not
  measured.
- **Prose, not code.** Upstream reports the gain "within noise on code".
- Per-request logs of the served lines are beside each run (`*-requests.log`).
