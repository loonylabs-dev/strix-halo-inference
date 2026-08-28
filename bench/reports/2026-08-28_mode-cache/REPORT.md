# Does a thinking mode get its own cache, and is `warm` true?

28.08.2026, 22:14 and 22:18. Two runs of `bench/suites/mode-cache.py` against
the live gateway and the serving qwen38, with `AUTO_SAVE=0` in
`~/.config/cc-gateway.env` for the duration — otherwise the automatic save
evicts the working slot mid-measurement (`autosave-evicts-the-working-slot`)
and every warm/cold time is somebody else's.

The subject is not speed. It is an honesty property:

    reported warm  =>  actually fast

A label that says warm while the wall clock says cold is the defect the prefix
id fix was written against.

## Run 2, the clean one — `--project /tmp/mode-cache-clean-1`

Fresh prefix ids, so nothing could be answered from the saved-prefix store and
every step is what it says it is.

    step                          seconds   gateway verdict   prefix id
    same-mode  1st  qwen38-low      13.03   COLD              61409ac6ccfd
    same-mode  2nd  qwen38-low       1.42   warm              61409ac6ccfd
    mode-switch     qwen38-medium   75.69   COLD              d6952a21be89
    mode-switch 2nd qwen38-medium    1.42   warm              d6952a21be89
    stale-name      qwen38-think     0.74   COLD              363e4b4d51e8
    bare            qwen38           0.57   warm              363e4b4d51e8
    back-again      qwen38-low       2.42   warm              61409ac6ccfd

All four conditions hold.

* **The modes have separate ids.** `low` and `medium` are `61409ac6ccfd` and
  `d6952a21be89`. Before 28.08. they were one id, and a switch was reported
  warm while the server re-prefilled — 75.7 s of work behind a label that said
  there was none.
* **The switch costs what it claims.** COLD, and 75.69 s of actual prefill. The
  mode text sits at character 19, so nothing of the other mode's prompt can be
  reused, and the number says exactly that.
* **The stale name falls through to the bare alias and shares its id.**
  `qwen38-think` and `qwen38` are both `363e4b4d51e8` — not a third id, not an
  error. The gateway also says so once per name:
  `'qwen38-think' matches no mode of qwen38 — serving it as the bare alias.
  Offered: qwen38 qwen38-low qwen38-medium qwen38-high`.
* **The modes do not evict each other.** Returning to `low` after two other
  renderings is warm at 2.42 s, on a server with ONE slot — that is `-cram
  32768`, the server's RAM prompt cache, doing what it is provisioned for.

`warm => fast` holds in every row. The converse does not, and that is fine: two
COLD steps were fast (13.03 s and 0.74 s) because the server's RAM cache held
something close enough. Cold is a statement about what the GATEWAY knows, and
it is allowed to be pessimistic; warm is a promise, and it is the one that must
not lie.

## Run 1 — the same measurement on a store with history

The first run used the suite's hard-wired project name, which earlier work
that day had also used. Two of its seven steps were answered from `.bin` files
on disk, and one of those two is the finding:

    RESTORED prefix 8774f83a80be from 8774f83a80be.bin -> slot 0, 14957 tokens, 285 ms
    START ... prefix=8774f83a80be warm
    DONE  ... took=75.3s

A disk restore, reported warm, followed by what looks like a full prefill. The
other restore in the same run behaved: `b4124abc721a`, 14838 tokens, 1.3 s.
Both `.bin` files were written at 14:56 and 14:58 that day — before the id
definition and the injection order changed — so a name that still matches a
state that no longer does is the obvious suspect, and it is a suspect rather
than a finding: one observation, not isolated, and the honest reading is that
the file, the id and the request need to be compared before anything is
claimed. To settle it: clear the store (`prewarm.py cleanup --purge`) or move
it aside, then repeat run 1's project name.

The suite hard-wired that project name, which is why the two runs could differ
at all. It takes `--project` now, and the flag's help says what it is for: a
value an earlier run used can be answered from disk instead of prefilled, and a
measurement that inherits what is lying around is not repeatable.

## Files

    run.log / after.json    run 1, hard-wired project, two disk restores
    clean.log / clean.json  run 2, fresh project, no disk involved
