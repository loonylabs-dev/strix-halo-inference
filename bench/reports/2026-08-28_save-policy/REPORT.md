# When to write a prefix to disk — simulated against real traffic first

28.08.2026. Before touching cc-gateway, the proposed rule was replayed through
seven days of this stack's own journal: every START and DONE the gateway
logged, every restart of llama-server. No GPU, no server, milliseconds per
run.

The rule has three numbers in it, and each is a trade:

    min_sightings   how often a prefix must be seen before it earns a gigabyte
    debounce_s      how long a gap has to be before a write may use it
    max_defers      how often a write may be postponed before it happens anyway

## What the traffic looks like

    all traffic     558 requests, 48 prefixes, 86 restarts, 98.9 hours
    only sessions   227 requests, 23 prefixes, 86 restarts, 91.5 hours
                    (who=martin-*; the rest is this repo's own measurement runs)

Two properties of that traffic decided more than any parameter:

* **Follow-ups are immediate.** Median gap between an answer and the next
  request: 1.0 s for martin-pc2, 13.0 s for martin-mobil. Claude Code does not
  wait for a human between turns — it runs tool loops. So a write started
  right after an answer lands in the middle of a burst, which is exactly what
  today's rule does.
* **One-off prefixes are rare here.** Only 2 of 23 session prefixes were seen
  exactly once. The whole point of `min_sightings=2` was to skip those.

## What the simulation says

Sessions only, every restart, 2 s per write:

    rule                          writes    collisions           after a restart
    today (first sighting)        22        5 (4 evicted,        20 warm / 21 cold
                                             3 files at risk)
    min=1 debounce=10s defers=3   23        0                    17 warm / 20 cold
    min=2 debounce=10s defers=3   21        0                    16 warm / 21 cold
    min=3 debounce=10s defers=3   19        0                    15 warm / 22 cold

And the same with the restarts thinned to a fifth and a twentieth — because
86 restarts in 91 hours is this repo benchmarking, not a machine's normal life:

    every 5th restart    today 16 warm / 18 cold      min=2 12 warm / 18 cold
    every 20th restart   today  9 warm / 12 cold      min=2  6 warm / 11 cold

Three findings, and the second one refutes what I proposed.

**1. "Exclusivity is free" — NOT SUPPORTED by this model, corrected in review.**
The zero in the collision column is DEFINITIONAL: `simulate_policy` contains no
statement that can increment it. And the waiting column is clairvoyant — the
model decides whether to start a write from `gap = t0 - prev_end`, the arrival
time of a request that has not happened yet, where a real gateway must commit
at `prev_end + debounce` and cannot know. Under a causal model the exposed
window is the gaps in [debounce, debounce + write cost): 3 of 226 at a 2 s
write, and 48 of 86 eligible gaps if the write costs ~100 s because the prefix
was no longer resident.

What the trace DOES support is narrower and still useful: gaps long enough to
hold a 2 s write are plentiful, and writing the instant an answer is out (a
debounce of 0) is the one setting that visibly collides with the next turn.

**2. `min_sightings` — WITHDRAWN 28.08., the same evening, in review.** This
report first said the parameter "saves one write in a week and costs three to
four warm starts, at every restart rate tested". That is wrong, and wrong in a
way worth keeping visible: the three-to-four figure is the TODAY-vs-policy gap
(20/17, 16/13, 9/6), not the min=1-vs-min=2 gap. Read off the same sweep, the
actual difference is:

    all 86 restarts    min=1 23 writes / 17 warm    min=2 21 / 16    min=3 19 / 15
    every 5th          min=1 23 / 13               min=2 21 / 12    min=3 19 / 11
    every 20th         min=1 23 /  6               min=2 21 /  6    min=3 19 /  6

Two writes and one warm start at the full restart rate, and at the most
defensible rate — every twentieth, i.e. closest to a machine that is not being
benchmarked — min=1, 2 and 3 are IDENTICAL on the warm column. And the model
has an infinite disk: it knows nothing of AUTO_MAX_GB or of prefix-cleanup's
LRU pruning, so the one cost `min_sightings` exists to avoid (a marginal write
evicting an older prefix) is not modelled at all.

The honest conclusion is therefore NOT "min=2 is a bad deal" but "this
simulation cannot tell min=1 from min=2, and the untested direction favours
2". The default stays 1 because it is what runs today, not because the
simulation chose it.

**3. `max_defers` is inert.** Every value from 1 to 5 produces identical rows:
on this traffic a quiet stretch always arrives before the deferrals run out.
It is insurance for a busier machine, not a tuned parameter, and pretending
the simulation chose it would be dishonest.

## What the simulation does NOT settle

The warm-start column consistently favours today's rule (20 against 17), and
the reason is timing rather than correctness: today writes the moment the
first answer is out, the policy waits for a gap. Whether that gap costs
anything in real life depends on how often the server restarts — and this
trace's restart count is dominated by measurement runs, so the column is
inflated in the direction that flatters writing early. The ranking survives
thinning, but neither the magnitude nor the ranking should be quoted as a
user-visible number — see finding 2 for what happened when this report tried.

Against it stands what the same column does not show: today's 5 collisions
cost 4 evictions — each a turn that went from 0.7 s to 13.6 s, measured — and
put 3 files at risk, of which two were later proven wrong on disk. A warm
start that never happens is a 75 s prefill once; a poisoned file is a 75 s
prefill on every request until somebody notices, and until 28.08. nobody could.

## The numbers that go into the build

    min_sightings   1     (unchanged from today; the model cannot tell 1 from
                           2, so nothing here justifies moving it)
    debounce_s      ?     (a debounce is what breaks the one property that
                           makes a write cheap — that the prefix is still in
                           the slot. See the review's first finding. This
                           number is NOT settled and must not be shipped.)
    max_defers      ?     (inert on this workload, and it latches: once
                           deferred enough, a prefix is due at quiet == 0
                           forever, which re-arms the very defect this is
                           meant to remove.)

None of these is ready to become a default. What this exercise produced is a
better question, not an answer: see the review findings recorded with the
commit that follows this report.

## Files

    trace.tsv        the replayed journal, so a rerun is exact
    sim.json         the rows above, machine-readable
    (the policy itself: setup/claude/savepolicy.py, tests/test_savepolicy.py)
