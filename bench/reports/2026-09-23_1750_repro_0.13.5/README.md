# repro_evict_instead_of_move.py on 0.13.5 — both findings gone

23.09.2026 17:50-17:58, asked for in upstream #97. Image
`ghcr.io/peonist-ai/halogen-flash-server:0.13.5`, digest `sha256:6aaf7d82…`.
Same setup as the 0.12.3 and 0.13.4 runs in `../2026-09-23_0130_repro-evict-instead-of-move/`:
pool 524288, 4 slots, freshly started, shipped quality sidecar (production
itself served the bare checkpoint by then — the sidecar was kept here so the
layout matches the earlier runs). ONE difference: the image's OWN
`serve_api.py`, because this branch's re-cut is pinned to 0.13.4's hash.

| | 0.12.3 | 0.13.4 | 0.13.5 |
|---|---|---|---|
| finding 1 (round 3: live 192k neighbour forgotten) | reproduced | reproduced | **not reproduced** |
| finding 2 (round 5: neighbour instead of own stale entries) | not reproduced | reproduced | **not reproduced** |
| script wall time | ~17 min | 21.8 min | **8.2 min** |

Every F1/F2 touch stayed warm (0.7 s). Where 0.13.4 forgot F1, 0.13.5 prints
`forgot the 2 entries longer than the hit … (the least recently used region,
at 0, is newer and would cost 192398 of its 192398 rows to rebuild)` —
SIDE's own superseded entries, the fix the maintainer described.

NEW, and a trade to know before upgrading: `the region at 385024 this request
resumes from cannot grow and nothing holds a move; the turn runs in the 8093
positions it has left (max_tokens 12000 -> 8093)`. Instead of evicting a
neighbour, 0.13.5 shrinks the turn's OUTPUT budget. Harmless here (short
answers); for Claude Code (max_tokens 32000) it means a long answer in a tight
pool ends early. What Claude Code then does is NOT measured. The maintainer
names the missing piece: a pack when free space sits on both sides of a region.

Posted: https://github.com/peonist-ai/halogen-flash-server/issues/97#issuecomment-5798228493

Files: `repro-output.txt` (the script's stdout), `kv-pool-lines.txt` (every
`^kv pool` line of the container log, kept before `--rm` removed it).
