# The hoist buys nothing it was built for and costs everything else

30.08.2026. cc-gateway moves the stable part of every system message that sits
INSIDE the conversation to the front of the prompt. This is the measurement of
whether that helps.

## What it was for, in its own words

`dialects.hoist_system_messages`:

    Claude Code appends a system block BEHIND the user question and glues a
    counter to its end. Left in place, the counter changes every turn, the
    prefix id changes with it, and every request runs cold.

## That reason does not hold

`system_head()` reads `body["system"]` on the Anthropic route and `messages[0]`
on the OpenAI one. A system message inside the conversation is in neither, so
it never entered the prefix id at all. Checked directly, same bodies, counter
incremented between turns:

    without the hoist   turn 1 c4406e7777b3   turn 2 c4406e7777b3   SAME
    with the hoist      turn 1 951c98a9e9a0   turn 2 951c98a9e9a0   SAME

And the case the hoist creates:

    with the hoist, a NEW block appears
                        turn 2 951c98a9e9a0   turn 3 da478be5a256   DIFFERENT
    without it          turn 2 ...            turn 3 ...            SAME

So the counter was never the problem, and hoisting is what makes a new block
one.

## What it costs, at the prompt level

`bench/suites/hoist-cost.py`, three turns of the same conversation rendered
both ways, driven straight at llama-server so no gateway behaviour is in the
way. Reused tokens of the prompt:

                     turn 1       turn 2       turn 3
    A hoisted       0/33920  33916/33920      0/34336     203.2 s
    B left alone    0/33920  33916/33920  31872/34336      20.4 s

Turn 2 is the case the hoist was built for, and **both keep 100 %** — because
the counter sits at the end of the prompt either way. Turn 3 is a new system
block appearing mid-conversation, and hoisting loses everything: 10x.

## The same three turns through cc-gateway

`bench/suites/hoist-live.py` sets the switch each way with a systemd drop-in,
restarts the gateway and runs the same three turns as Claude-Code-shaped
Anthropic bodies — so the gateway's own correction, id and restore logic are
all in the measurement:

                           turn 1         turn 2         turn 3
    hoist on             0 cached    4278 cached       0 cached   31.3 s
    hoist off            0 cached    4278 cached    4278 cached   12.3 s

Turn 2 is identical, again. Turn 3 is 2.5x at these small sizes and 10x at the
prompt-level sizes above; the ratio grows with the conversation, because what
is lost is everything behind the front.

## How often the front actually moved

Read out of the trace for `who=martin-pc2`, one day, main prompt type only
(the toolless second type alternates and is a separate strand):

    90 turns, 6 of them with a moved front

    07:57:13   35811 -> 60006   reused 14483  computed   233     76.0 s
    08:14:44   60006 -> 60428   reused 12435  computed 10253    119.7 s
    10:03:08   60428 -> 60006   reused 14483  computed   224      4.1 s
    10:26:05   60006 -> 60428   reused     0  computed 44072    305.9 s
    00:01:32   60428 -> 73404   reused 17784  computed 55856    655.8 s
    00:12:43   73404 -> 73738   reused     0  computed 73877    668.9 s

    184,515 tokens recomputed on six turns

TWO OF THE SIX ARE NOT THE HOIST. 07:57:13 and 00:01:32 are tool-count changes
(25 -> 13 and 13 -> 21) — the client sent a different tool list, which moves
the front whatever the gateway does. The other four are consistent with the
hoist: the prefix moves by 422 or 334 characters and back, which is the size
of a system block appearing and disappearing. `consistent with`, not proven —
the trace records the prefix's LENGTH and hash, not its text, so which block
moved cannot be read out of it.

Note also what the middle two rows are: 60428 -> 60006 -> 60428, a block that
comes and goes. The return trip cost 44,072 tokens.

A FIRST READING OF THIS TABLE WAS WRONG and is worth recording. Counting
`volatile_moved` increments gave "55 of 93 turns moved the front" — but that
field counts volatile fragments left INSIDE the conversation, and it rises
whenever another system message appears, whether or not the hoisted set
changed. `prefix_chars` is the field that says the front moved. The wrong
reading inflated the frequency by nine times.

## The live incident this predicts

30.08. 00:12:43, one Claude Code session, prefix 7ff6bcd1f1de:
`volatile_moved` 25 -> 26, the hoisted prefix 73404 -> 73738 characters, reuse
0, computed 73,877, **668.9 s** — while 72 of the 74 previous messages were
unchanged.

It cost the WHOLE conversation rather than the part behind the change because
of a second threshold: `server-task.cpp:1813` skips a cache entry whose
`f_keep_cur < 0.25f`, "don't trash large prompts". 17784/73678 = 0.2414, just
under. Arithmetic and source, not observation — llama.cpp does not log its
per-entry comparison at this verbosity.

## What is NOT settled

The hoist has been in place since 24.08. and there is history behind it these
measurements do not see — an earlier id definition, a template that rejected
something, a case not modelled here. The synthetic body places the system
block near the END of the conversation, which is where the hoist's own
docstring says Claude Code puts it; a block that arrived EARLY would change
the arithmetic, and nothing here measures that.

That is why `HOIST_SYSTEM` defaults to 1. One afternoon's numbers do not get
to overturn a mechanism silently.

## Reproduce

    python3 bench/suites/hoist-cost.py          # the two prompt shapes
    python3 bench/suites/hoist-live.py          # the switch, through cc-gateway
