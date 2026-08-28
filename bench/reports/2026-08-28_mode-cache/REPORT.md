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

## Run 1 — and the defect it walked into

The first run used the suite's hard-wired project name, which earlier work
that day had also used, so two of its seven steps were answered from `.bin`
files on disk. One of them is the finding:

    RESTORED prefix 8774f83a80be from 8774f83a80be.bin -> slot 0, 14957 tokens, 285 ms
    START ... prefix=8774f83a80be warm
    DONE  ... took=75.3s

llama-server's own log says what happened: `selected slot by LRU` and
`prompt eval time = 73523 ms / 14960 tokens`. The restored state was not used
at all. The gateway said warm.

**Two wrong explanations, both refuted by measurement, both worth keeping**
because each looked sufficient:

1. *The file is stale — saved before the id definition changed today.* No. The
   ids recompute exactly: `b4124abc721a` is this body with no kwargs,
   `8774f83a80be` is this body in `low`. And the sidecar's `render_id`
   (`8b698a91deb2`, 70892 chars) is byte-for-byte what `low` renders now.
2. *The gateway sends something else than prewarm saved — MID_SYSTEM_TO_USER
   rewrites the body after the id is taken.* No. Rendered both ways through
   `/apply-template`: same 70892 chars, same hash.

**What it actually is**, measured in a controlled run — fill the slot with a
foreign prefix, restore the file the way cc-gateway does, then send the body
the file is named for:

    8774f83a80be (saved for qwen38-low)   restored 14957 tokens in 174 ms
      -> the request it was saved for      87.0 s   full prefill
    b4124abc721a (saved for the bare alias) restored 14838 tokens in 179 ms
      -> the request it was saved for       1.9 s   the restore carried it

The mechanism works. One file does not. And the file holds none of the four
renderings its name could stand for — restored again before each probe:

    qwen38-low     74.9 s        qwen38        73.7 s
    qwen38-high    74.5 s        qwen38-medium 75.4 s

So `8774f83a80be.bin` contains 14957 tokens of SOMETHING ELSE, under a correct
name, with a sidecar that describes the state that was meant to be saved.

The likely writer is the defect already in the register: `auto_save` runs
asynchronously and takes ~102 s on this machine, and there is ONE slot. A
request arriving during the save puts its own prefix into the slot being
written out. The file then carries that prefix under our name — the mirror
image of `autosave-evicts-the-working-slot`, which is the same collision seen
from the serving side.

**Nothing notices, and that is the part worth fixing.** The sidecar already
carries `render_id`, written by prewarm at save time — and no line of code
reads it back. The gateway matches on the name alone, reports warm, and pays a
full prefill. Worse, the truth arrives in every upstream response and is
thrown away: llama-server returns `timings.cache_n` and `prompt_n`, which say
exactly how many tokens were reused and how many were computed. A warm/cold
label could be a measurement instead of a claim, at no cost.

Registered as `saved-prefix-holds-a-foreign-state`, and fixed at both ends the
same evening.

## The fix, and its proof on the file that started it

**Read side.** `dialects.reuse_from_text()` reads what the server actually did
out of the answer the gateway is already proxying — `timings.cache_n` /
`prompt_n` from llama.cpp, `cache_read_input_tokens` / `input_tokens` from the
Anthropic route, `prompt_tokens_details.cached_tokens` from the OpenAI shape —
sniffed from the first and last 8 KiB, because an Anthropic stream carries the
accounting in its FIRST event and an OAI stream in its last. Every `DONE` line
now ends in `reused=N computed=M`, and a restore that reused less than half of
what it loaded quarantines its own file.

**Write side.** `auto_save` counts the requests that reached the model before
and after the save and drops its own file if anything was served in that
window — with one slot, that request took the slot being written out. The race
is not prevented; its result is no longer published.

Live, on the production gateway, against `8774f83a80be`:

    RESTORED    prefix 8774f83a80be ... 14957 tokens, 122 ms
    START       ... warm
    DONE        ... took=74.3s  reused=0 computed=14960
    QUARANTINED 8774f83a80be — restored 14957 tokens, the server then reused 0
                and computed 14960 — the file does not hold what its name says
    START       ... warm
    DONE        ... took=1.1s   reused=14956 computed=4

The first line is the defect, the fourth is the system noticing it for the
first time, and the last is what a working restore looks like when it is
measured rather than claimed. The store now cleans itself, one prefill per bad
file, and `ls ~/.cache/llama-slots/*.unusable` says what went.

## Files

    run.log / after.json    run 1, hard-wired project, two disk restores
    clean.log / clean.json  run 2, fresh project, no disk involved
    restore-probe.py        fill the slot, restore, ask — does the file carry?
    whats-in-it.py          restore, then ask each candidate: what IS in it?
    restore-probe.log       ] not in git (*.log), the numbers are quoted above
    whats-in-it.log         ]
