# A save colliding with a turn costs waiting, and nothing else

02.09.2026, 19:55 and 20:10. flashnext (b10743-15-g62850522e) on a side
server, `--slot-save-path` and `-cram 0`, production stopped and restored
around each run. `bench/suites/save-under-load.py`, two runs: one at 180,000
tokens (cells 2–4 valid, cell 5 void — see below) and one at 20,000 with the
probe rebuilt.

## The question

`bench/reports/2026-09-02_1856_slot-save-cost/` measured the write at ~1.9 s
for a 180k state against ~1800 s to recompute one — cheap enough to do often.
But every save there found an IDLE slot, while in service it would fire right
after an answer, with the next turn a measured median of 1.0 s away
(save-policy, 28.08.). The two overlap. Two registered defects are about
exactly that overlap on the PREWARM path:

    autosave-evicts-the-working-slot     the turn lost: 0.7 s -> 13.6 s, 19x
    saved-prefix-holds-a-foreign-state   the file carried another prefix

## The measurement

At 20,000 tokens, all five cells:

    baseline: a warm turn                0.31 s   cache_n=20000
    turn 200 ms into a save              0.30 s   cache_n=20008   (-0.01 s)
      the save itself                    0.22 s   n_saved=20008   save_ms=215
    save 200 ms into a long generation
      the generation                     3.07 s   cache_n=20016
      the save                           3.09 s   n_saved=20273   save_ms=220
                                                  waited 2.87 s
    file integrity
      under-load-3.bin  n_saved=20008  -> holds our state (cache_n 20008)
      under-load-4.bin  n_saved=20273  -> holds our state (cache_n 20273)

At 180,000 tokens, cells 2–4 (cell 5 of that run is void):

    baseline: a warm turn                0.62 s   cache_n=180000
    turn 200 ms into a save              3.42 s   cache_n=180008  (+2.80 s)
      the save itself                    3.07 s   n_saved=180008  save_ms=3070
    save 200 ms into a long generation
      the generation                     9.66 s   cache_n=180016
      the save                          12.41 s   n_saved=180096  save_ms=2971
                                                  waited 9.44 s

**The turn keeps its state in every cell.** `cache_n` comes back at the full
slot length whether a save is running or not. **Both files hold the state
they should**, verified by restoring each and continuing from it.

**The save is deferred, not interleaved.** Fired into a running generation it
waited 2.87 s (20k) and 9.44 s (180k) — in both cases until the generation
finished — and then wrote in its normal time. That is
`queue_tasks.defer(std::move(task))` (server-context.cpp:2525) seen from
outside.

Neither defect of 28.08. reaches this path, and the source says why: a
`SLOT_SAVE` only READS the slot (`slot->prompt.tokens.serialize()`) and never
puts anything into it. The prewarm path had to CREATE the prefix it wanted to
save, which is where both defects came from.

## What a save actually captures

`n_saved` is the prompt plus (n_gen - 1) generated tokens, on all four files:

    prompt 20008,  n_gen   1  -> n_saved  20008
    prompt 20024,  n_gen 250  -> n_saved  20273
    prompt 180008, n_gen   1  -> n_saved 180008
    prompt 180024, n_gen  73  -> n_saved 180096

Not a curiosity: it is what lets a probe be reconstructed exactly rather than
guessed, and cell 5 depends on it.

## The save time is a range, not a value

The same 180k state, three writes: **1996, 3070 and 2971 ms** — 54 % spread on
an identical amount of data. The 1,889 ms in the slot-save-cost report is the
bottom of that range, not a constant. It changes no decision (3 s against
~1800 s is still nothing) but the figure must not be quoted as exact.

## A finding retracted, and how it happened

The 180k run reported `under-load-4.bin` as **NOT OURS — the 28.08.
foreign-state shape**. That is wrong. It was the probe, not the file.

Two rules meet in cell 5 and each of the first two versions got one backwards:

* **In the slot**, reuse is trimmed to the common prefix. Cell 3 shows it: the
  slot held `toks+toks[:8]+gen`, the turn sent `toks+toks[:16]`, they diverge
  at 180008, and exactly 180008 came back.
* **After a restore**, a state carrying anything past the prompt is discarded
  WHOLE (2026-08-29_restore-semantics). So a probe that is not a true superset
  pays a full re-prefill of the depth.

Version 1 tried candidates one at a time — an hour of wrong guesses at 180k,
which is why the first run was killed mid-flight. Version 2 CONCATENATED the
candidates and called that a superset of all of them. It is not: candidates
BRANCH, they do not nest. `toks[:8]+gen` and `toks[:24]+gen4` share only their
first 8 tokens, so the chain supersets the shortest branch and nothing else.

    state in under-load-4.bin :  toks + toks[:24] + gen4[:72]   = 180096
    the probe began           :  toks + toks[:8]  + base_gen …
    common prefix 180008, then toks[8] against base_gen[0] -> diverge

The state was therefore not a prefix of the probe, was discarded whole, and
1831 s of prefill later the run accused a registered defect of being live.
`under-load-3.bin` passed only because its state happened to BE the common
prefix of the chain.

Fixed by reconstructing one probe per file from the turn that produced the
state, sized by `n_saved`; a length matching no candidate is now reported
without paying for a probe at all. And the suite's default depth is 20,000,
not 180,000: the mechanism is depth-independent, while a missed probe costs
80 s there against 1831 s.

## What this settles, and what it does not

Settled: the overlap is a WAITING cost. A cooldown is therefore a wear
decision, not a correctness one — which is what the SSD figures have to
answer, not this report.

Not settled:

* **`-cram` interaction.** Both runs used 0; production runs 14336. A restore
  blinds llama.cpp's own cache lookup (`f_keep < 0.5f`, measured 30.08.:
  56.4 s against 1.0 s), so a policy on a server WITH a cache is a different
  experiment.
* **Two clients at once.** One conversation collided with its own save here.
  Two sessions plus the health probe is the shape the 02.09. incident had.
* **`under-load-4.bin` at 180k specifically.** Its length matched the expected
  state exactly (180,024 prompt + 72), which is an indication and not a proof
   — the 28.08. defect also had the right length. Only the 20k pair is proven.

## Reproduce

    D=~/.cache/slot-save-cost && mkdir -p $D
    python3 bench/sideserver.py --env setup/env/flashnext.env --port 8081 \
        --stop "llama-user@$(bash setup/lib/models.sh serving)" --deadline 45 \
        --extra "--slot-save-path $D/ -cram 0" -- \
        python3 bench/suites/save-under-load.py --url http://127.0.0.1:8081 \
            --dir $D --depth 20000 --out $D/under-load-20k.json

About six minutes. `--from-file <state> --seed <n>` restores an existing state
instead of prefilling one, which is how the 180k run avoided a second
30-minute setup.

## Files

    rows-20k.json   the valid run, all five cells
    rows-180k.json  cells 2-4 valid, cell 5 void — the retraction above is
                    checkable against it: its integrity cell carries
                    cache_n 0 against n_restored 180096
    (both run logs stay local — *.log is gitignored repo-wide)
    (the suite: bench/suites/save-under-load.py)
