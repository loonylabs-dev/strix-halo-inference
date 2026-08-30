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
