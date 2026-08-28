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

**1. Exclusivity is free here.** Every variant with the gate held reaches zero
collisions, and at a debounce of 5 s or more the added waiting is 0.0 s across
the whole week — the gaps are long enough that writes fit inside them. At
debounce 0 the cost appears (23 s of waiting in total, never more than 2 s at
once), which is the price of writing the instant an answer is out.

**2. `min_sightings=2` is a bad deal ON THIS WORKLOAD.** It saves one write in
a week — 1 GB — and costs three to four warm starts, at every restart rate
tested. The idea was sound and the data says it does not pay here, because
this machine has almost no one-off prefixes. It stays in the module as a knob,
with 1 as the default.

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
thinning (the gap stays at 3–4 warm starts), but the magnitude should not be
quoted as a user-visible number.

Against it stands what the same column does not show: today's 5 collisions
cost 4 evictions — each a turn that went from 0.7 s to 13.6 s, measured — and
put 3 files at risk, of which two were later proven wrong on disk. A warm
start that never happens is a 75 s prefill once; a poisoned file is a 75 s
prefill on every request until somebody notices, and until 28.08. nobody could.

## The numbers that go into the build

    min_sightings   1     (the data refuted 2; the knob stays)
    debounce_s      10    (5 already costs nothing; 10 for margin)
    max_defers      3     (untested by this workload — insurance)

## Files

    trace.tsv        the replayed journal, so a rerun is exact
    sim.json         the rows above, machine-readable
    (the policy itself: setup/claude/savepolicy.py, tests/test_savepolicy.py)
