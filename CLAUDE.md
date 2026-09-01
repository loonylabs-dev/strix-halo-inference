# CLAUDE.md — strix-halo-inference

Session rules for working ON this repository. The published docs describe how
to RUN the stack; this file binds how sessions CHANGE it. It stays small:
rules that bind every session and were paid for once. Everything else has a
home named below.

## Start of session

- Read `docs/HANDOVER.md` — the baton, gitignored, head first. The tree wins
  over the baton: run `git log <its sha>..HEAD --oneline` before acting on it.
- `python3 setup/lib/defects.py --upstream` — not optional. It was right on
  30.08.2026 and walked past twice.
- The gate is `bash tests/run.sh` (~20 s, no GPU). Green before AND after a
  change; nothing gets committed on a red or unrun gate.

## Hard rules, each one paid for

- **Never start a second model by hand.** `python3 bench/sideserver.py` is
  the only way — it stops production, guards memory, and puts production
  back. Direct starts froze the machine three times on 26.08.2026.
- **Numbers are measured, not derived**, and every figure in a profile
  carries its source (date + method) in the comment beside it. Repointing
  `LLAMA_BIN` turns every observed figure back into a claim until it is
  re-measured on that build — RssAnon of the RUNNING server is the one-line
  check (28.6 GiB measured where 0.31 was declared, 31.08.2026).
- **A foreign tree builds into its own `--family` and is never served.** The
  one sanctioned exception is an explicit `LLAMA_BIN` pin with a written
  retirement condition — setup/README.md, family table.
- **Production changes only on the operator's explicit go**: `--activate`,
  `switch-model.sh`, restarting a unit with a new binary.
- **The tree IS the installation.** install.sh symlinks `$HOME` (the user
  units, `~/.local/lib/llm-stack`, `~/.config/llm-profile`) into this
  checkout, so a checked-out branch that moves files breaks every service
  restart. Move files in a `git worktree`; merging to main STARTS the
  machine migration — install.sh and the unit switch belong in the same
  sitting (01.09.2026).
- **One load on the machine at a time.** No compiling while a measurement
  runs, no measurement while a build runs — contention contaminates both.
- **The repo is public.** `docs/HANDOVER.md`, `docs/HANDOVER-LOG.md`,
  `docs/FLASHNEXT-PLAN.md` and the German sources are gitignored and stay so. Scan every new artifact
  (reports, copied server logs) for identifying values before committing —
  paths fold to `@HOME@`.
- **Upstream posts are written by the human.** llama.cpp's CONTRIBUTING
  forbids AI-written issues/PRs/comments; measurements, tables and
  reproducers may be handed over, prose may not.
- **A guard's refusal must not be piped away.** build-llama.sh's MAX_REPLAY
  and the memory guard say no through stderr and a non-zero exit; a
  background `cmd 2>&1 | tail` reports tail's 0 and the refusal reads as
  "completed" — a build that never ran passed as success twice on
  31.08.2026. Redirect full output to a file and read that; pipe nothing
  whose exit code decides anything.

## Language

Everything in this repo is English — code, comments, docs, commit messages.
(Repo-level confirmation of the global rule; conversation language is
independent of it.)

## Where learnings get filed — not here, and not in the global CLAUDE.md

| Learning about | Goes to |
|---|---|
| an instrument, flag or figure of this stack | comment beside the code/env value it concerns, with date + report path |
| a defect, hang or corruption | `setup/defects.json` (copy an existing entry; tests validate the schema) |
| a measured dead end | `docs/HANDOVER.md` → *Do not try again* |
| build/family mechanics | `setup/README.md` |
| measurement discipline | `bench/README.md` |
| Claude-Code/tool behaviour that holds in EVERY project | global `~/.claude/CLAUDE.md` — and only that |

When `/update-claude-md` runs in this repo, THIS file and the table above are
the target; the global file takes only what survives the last row's test.

## Commit style

Thematic commits. The subject is a claim with an em-dash, the body carries
the reasoning and the measurement — read `git log` for the pattern.
