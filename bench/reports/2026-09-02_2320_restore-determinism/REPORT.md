# A restored state is the SAME state — byte-identical, and the index still finds the needle

02.09.2026, 23:20. flashnext (b10743-15-g62850522e) on a side server,
`--slot-save-path` and `-cram 0`. `bench/suites/restore-determinism.py`,
14,187 tokens of context, needle at fact 300.

## The question this had to answer

`bench/reports/2026-09-02_2247_restore-safety-…` found the idle cell CLEAN for
this model: after a restore the arithmetic probes still answer. That settles
"is it broken" — the shape of the 25.08. incident, where output degenerated
into `////`.

It does not settle what setup/env/flashnext.env actually objects to. Two
upstream reviewers rejected save/restore on #27742 because the QSA indexer
carries state the restore path does not know about; ngxson's wording is "a
context save/load will corrupt it". A corrupted learned INDEX does not look
like `////`. It looks like fluent output that attended to the wrong 2048
tokens — and three sums cannot see that, because 391 is 391 however the index
was chosen.

## The measurement

    cell                                    cache_n     answer
    fresh answer 1        62.12 s                 0     '424242'
    fresh answer 2        62.22 s                 0     '424242'
    warm arm               0.87 s            14,187     '424242'
    restored arm           0.79 s            14,187     '424242'

    fresh vs fresh      IDENTICAL      the instrument can resolve
    fresh vs warm       IDENTICAL
    warm  vs restored   IDENTICAL
    fresh vs restored   IDENTICAL
    needle found        both arms

**Byte-identical output from a state read off disk and one computed in place,
on a prompt long enough that the index had to choose.** The #27742 objection
does not reach this path on this build, in this shape.

## The three controls, and why each is load-bearing

* **Two fresh arms.** Byte equality proves nothing unless the server is
  byte-deterministic for this prompt. This is that check, and it is not a
  formality — see the retraction below.
* **The warm arm.** The fresh arm computes all 14,187 tokens; the restored one
  computes 30 and takes the rest off disk. Different batch shapes give
  bit-different logits. The warm arm has the restored arm's SHAPE (context
  already in the slot) without a file anywhere, so it separates "the file
  changed the state" from "fewer tokens were evaluated this pass". It matched
  both others, so neither effect is present at this answer length.
* **The needle.** The context is numbered facts and the question asks for one
  from the middle, so the answer DEPENDS on the context. Without that, a wrong
  index would produce the same reply as a right one and the comparison would
  pass on a broken state. 14,187 tokens is far past the 2,048-token index
  budget, so the index had to pick.

## A finding retracted, and it is the reason the run above is trustworthy

An earlier run of this suite reported **"THE RESTORE CHANGES THE OUTPUT"** and
printed two differing answers side by side, attributing them to #27742. That
was withdrawn before it left the session, and the reason is instructive:

With thinking enabled the model writes a paragraph of reasoning before the
answer. bench/README records what that costs an equality test — a reasoning
step is full of two-way near-ties, a last-bit difference decides one, and the
runs diverge from there. Measured on this very prompt: **one run had the two
FRESH arms byte-identical, the next run had them differing.** The instrument
resolved on one day and not the next, and the "finding" had been built on its
good day.

Two changes make the comparison mean something:

* thinking suppressed with an empty `<think>` block, leaving a six-character
  answer that sits on no near-tie
* `--predict 24`, for the same reason

Both runs are kept — `run.log` is the admissible one, `run-inadmissible.log`
is the one whose own guard stopped it with "the server is not byte-
deterministic for this prompt".

## What is NOT settled

* **Long generations.** The answer here is six characters. A state difference
  that only shows after hundreds of tokens would not appear — and cannot be
  tested this way, because that regime is not byte-deterministic in the first
  place. This is a limit of the method, not an oversight.
* **One needle, one position.** Middle of the context, one run. Not swept.
* **Idle only.** No concurrency; flashnext serves `-np 1` and the busy/prefill
  cells of restore-safety need two slots.
* **`-cram 0`.** No RAM cache in play, deliberately, so that only the file
  could answer.
* **The indexer is not proven correct** — it is proven not to differ HERE. The
  upstream objection is about a mechanism, and one shape of one prompt does not
  clear a mechanism.

## What it unblocks

setup/env/flashnext.env says the flag stays out "until
`bench/suites/restore-safety.py` has run against THIS model. Nobody has run
it." It has now, green, and this run adds the stricter question on top. The
condition the profile set for itself is met.

That is a decision for the operator, not a conclusion of this report.

## Reproduce

    D=~/.cache/slot-save-cost && mkdir -p $D
    python3 bench/sideserver.py --env setup/env/flashnext.env --port 8081 \
        --stop "llama-user@$(bash setup/lib/models.sh serving)" --deadline 60 \
        --extra "--slot-save-path $D/ -cram 0" -- \
        python3 bench/suites/restore-determinism.py --url http://127.0.0.1:8081 \
            --dir $D --predict 24 --out $D/restore-determinism.json

About five minutes, three quarters of it the three cold prefills.

## Files

    rows.json               the four cells and the verdict
    (run.log and run-inadmissible.log stay local — *.log is gitignored)
    (the suite: bench/suites/restore-determinism.py)
