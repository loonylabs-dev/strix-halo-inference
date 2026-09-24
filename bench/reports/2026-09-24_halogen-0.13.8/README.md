# Halogen 0.13.8 — side measurements, 24.09.2026

Two questions, one sitting (07:40–11:13, production down 07:40–10:00, the
machine otherwise idle, `platform_profile` performance at every phase end):

1. Is 0.13.8 better than production (0.12.3) in the deep side-turn shape that
   0.13.4 lost on 23.09.?
2. Upstream #89's objection to our sidecar verdict: the five stalls were picked
   where the sidecar stalled, which favours bare — and do earlier stall
   episodes drive the next one?

**Verdicts.** (1) Yes: 0/7 main turns lose their history against 1/7, a
third of the time. (2) The objection holds. At positions the sidecar did
NOT pick, the bare checkpoint stalls and the sidecar mostly does not. Neither
checkpoint is free of it; each has its own positions. The 23.09. line
"the sidecar causes it" was a selection effect in its strong form.

## 1. sessionload, side turns beside live neighbours

`bench/sessionload.py --mode pingpong --sessions 1 --turns 8 --doc-tokens 40000
--max-tokens 12000 --side-turns --turn-pad 4000 --fillers 200000,200000
--live-fillers`, salt `ab2609` — the 22./23.09. shape, identical prompts. Both
runs today are BARE, the production config; the 23.09. rows had the sidecar
on, which changes the answers and so the history's growth.

| run | main turns lost | recomputed (main + side) | wall | max_tokens cut |
|---|---|---|---|---|
| 0.12.3 bare, today | 1/7 | 79k + 97k | 1412 s | 3x: 8166, **3093**, 8088 |
| **0.13.8 bare, today** | **0/7** | **29k + 42k** | **947 s** | 4x: 6995, 6964, **3002**, **2961** |
| 0.12.3 + sidecar, 23.09. | 1/7 | 78k + 94k | 1394 s | — |
| 0.13.4 + sidecar, 23.09. | 3/7 | 199k + 153k | 2330 s | — |

n=1 per row. The budget cut ("the region cannot grow and nothing holds a
move; the turn runs in the N positions it has left") is NOT new in 0.13.x:
0.12.3 cuts in the same shape, down to 3093. It is upstream's missing
two-sided pack (#97, "stays on our list"). This shape fills 91.6 % of the
pool with live fillers — an extreme, but it is the one where an answer to
Claude Code (max_tokens up to 32000) would end at ~3000 tokens.
`kvpool-0.13.8-bare.log` holds 0.13.8's `kv pool` lines (sessionload first,
then the colonstop replays).

Render check: 0.13.8's front-end renders all 16 replayed histories
byte-identically to 0.12.3's. 0.13.6's merge of consecutive assistant
messages does not touch this client's prompts (0 such pairs, 23.09.).

## 2. colonstop — stalls, stalls without earlier episodes, and the counter-sample

`bench/colonstop.py`, n=10 per cell, 120 tokens, the server's sampling
defaults, byte-identical prompts in every column (rendered on 0.12.3; the five
original stalls are the 23.09. bytes). Cells = continuations that ended the
turn again.

| set | position | 0.13.8 bare | 0.13.8 + sidecar | 0.12.3 bare |
|---|---|---|---|---|
| stall (picked: sidecar stalled) | 776999:17 | 0 | 2 | 0 (23.09.) |
| | :23 | 0 | 8 | 0 (23.09.) |
| | :29 | 0 | 0 | 0 (23.09.) |
| | :35 | 0 | 4 | 0 (23.09.) |
| | :47 | 0 | 0 | 0 (23.09.) |
| | **sum** | **0/50** | **14/50** | 0/50; sidecar 0.12.3: 17/50 |
| same, earlier stall episodes removed (1/2/3/4 of them) | :23 | 0 | 6 | 0 |
| | :29 | 0 | 0 | 0 |
| | :35 | 0 | 1 | 0 |
| | :47 | 0 | 0 | 0 |
| | **sum** | **0/40** | **7/40** | **0/40** |
| announcement (picked: sidecar went on to a call) | 776999:8 | **6** | 0 | **9** |
| | :11 | 0 | 0 | 0 |
| | :26 | 0 | 1 | 0 |
| | a84e:101 | 0 | 2 | 1 |
| | :116 / :131 / :146 / :161 / :191 | 0 | 0 | 0 |
| | :176 | **4** | 0 | **5** |
| | **sum** | **10/100** | **3/100** | **15/100** |
| stall ending in '.' | a84e:152 | 4 | 8 | 5 |

Every continuation that did not end opened a `<tool_call>` (0 "other").

What this says:
- **Each checkpoint has its own positions.** Bare stalls where the sidecar
  went on (776999:8 at 6 and 9 of 10 on two releases, a84e:176 at 4 and 5);
  the sidecar stalls where bare goes on. The positions are the checkpoint's,
  not the release's: bare 0.12.3 and bare 0.13.8 agree cell for cell.
- **Selection decides the sum.** Stalls picked by the sidecar: 14 vs 0.
  Positions picked by the sidecar NOT stalling: 3 vs 10. Both samples are
  conditioned on the sidecar, in opposite directions; the redraws at a fixed
  position are unbiased, the choice of positions is not.
- **A rough population estimate, not a measurement:** these two sessions hold
  ~6 stall positions and ~58 announcements that went on. Weighted by that,
  ~6 % of announcements end with the sidecar and ~10 % bare (each stall
  position at its 0.13.8 rate, each announcement at the set's mean — a
  weighting, with every weakness below in it). Consistent with
  upstream's finding that bare is at least as prone, on far fewer draws.
- **Earlier stall episodes amplify, they do not cause.** Removing them lowers
  the sidecar's count 12 -> 7 of 40, but :23 still ends 6 of 10 with no
  earlier stall left in its history. Bare stays at 0 either way.

## Method notes

Side servers ran this branch's `setup/halogenexec` (the 0.13.8 re-cut,
hash-verified at mount) from a transient unit with production's environment
(`llm-stack.env`, `HALOGEN_FIT_TO_ROOM=1`, `HALOGEN_CK_OVERLAY=none` for the
bare arm), each verified from its container log before measuring
(`halogen-flash-server 0.13.8`; `overlay: DISABLED` resp. `overlay: 723
tensors`). A trap restored production on any exit. Phase C ran on production
itself after a fresh restart. Rows carry `kind` and `stalls_removed`.

## Weaknesses

- **Two sessions of one client**, 10 announcement positions from a84e chosen
  by stride (every fifth candidate from k=101), three from 776999 (all its
  ':' announcements). The population estimate rests on that.
- **Continuations, not whole turns**, as on 23.09.
- **The sidecar on disk is the pre-0.6.0 file** (2.31 GiB, no 8-bit draft
  head; the container says "predates 0.6.0"). Upstream's changelog says the
  723 quality tensors are byte-identical in the current file — read, not
  measured. It changes the draft head, i.e. speed, not these outcomes.
- **sessionload is n=1 per release.**
- The 0.12.3 bare column ran on production after the 0.13.8 phases; the
  a84e:152 cell there paid a 228 s cold prefill for its first draw (0.12.3
  did not reuse the longer a84e:191 entry for the shorter prefix).
