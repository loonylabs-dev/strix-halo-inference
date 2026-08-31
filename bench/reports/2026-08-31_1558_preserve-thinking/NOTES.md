# preserve_thinking — no calm at this depth, only context space

**Question** (from kyuz0/amd-strix-halo-toolboxes#127): a user of the same
model runs thinking permanently on and credits `preserve_thinking: false`
with ending the "Wait, Actually…" overthinking loops. Our template keeps all
past thinking in the prompt by default (render shas in the suite docstring),
so the hypothesis was: the overthinking is history feedback, and stripping
the history flattens it.

**Setup**: `bench/suites/preserve-thinking.py`, effort low, 3 sessions per
arm, 4 fixed turns, interleaved, against the production server (b10702
patched, -np 1). Both arms resend `reasoning_content` in the history — the
production shape — and differ only in the kwarg.

**Result — the arms look alike:**

    mean think tokens        turn1  turn2  turn3  turn4   session total
    default (keep history)     535   1357    408    840    3141 (2864-3455)
    strip   (kwarg false)      614    902    381    873    2769 (1998-3323)

The 12 % gap on totals is inside the spread of either arm alone (strip's
own sessions range 2000-3300). Think volume follows the QUESTION, not the
history: turn 3 ("write three tests") is the calmest turn in both arms,
turn 2 the heaviest. No monotonic growth in the default arm. No truncated
thinks (all finish=stop).

**What the run showed instead, systematically, all 3 sessions:** the cache
cost of stripping on a hybrid model. The default arm stays a prefix of the
slot (cached grows 30 → 3818); the strip arm makes the slot a non-prefix at
every turn's previous `<think>`, and since qwen38 cannot be trimmed it rolls
back to a checkpoint and re-prefills 150-700 tokens per turn (cached stuck
at ~30-1200). Harmless at this scale; at real session depth the price is
set by checkpoint placement — the thing checkpoint-grid.py measures.

**Verdict**: at shallow depth the overthinking is the effort level itself,
not history feedback. `preserve_thinking:false` is not the fix for "low and
medium think too much"; nothink stays the production default. The kwarg's
real value would be CONTEXT (thinking no longer accumulates in the window
of deep sessions) traded against cache warmth — unmeasured here, and only
worth measuring if think modes ever see real use.

**Limits, stated plainly**: n=3 per arm, 4 turns, ~4k context. The issue
author's experience was 26k-110k coding sessions; a feedback effect that
only compounds at that depth is not ruled out — a large shallow effect is.
