# The restore that hides the cache it was meant to replace

30.08.2026. A prefix restore makes the slot a perfect prefix of the incoming
request, and llama.cpp reads that as "nothing to gain from the cache" — so it
never looks. When the cache was holding the whole conversation, the restore
costs everything behind the prefix.

Measured on the production server, both directions, twice.

## The mechanism

`server-context.cpp:1587-1596`:

    const float f_keep = (f_sim_best*task.tokens.size()) / ret->prompt.tokens.size();
    ...
    if (f_keep < 0.5f) {
        update_cache = true;
    }

`f_keep` is how much of what the SLOT currently holds the new prompt keeps.
The RAM prompt cache — `prompt_save()` then `prompt_load()` — runs only inside
`if (update_cache)`. A slot holding exactly the prefix of the incoming prompt
scores f_keep = 1.0, so the lookup is skipped, and a longer state of the same
conversation sitting in the cache is never found.

The gateway restores a prefix file precisely into that condition.

## A/B, same request, one step different

`bench/suites/restore-blinds-cache.py`. Both rounds: prefill a 2290-token
prefix and save it, prefill a 13780-token conversation, hand the slot to an
unrelated tiny prompt (an LRU takeover, which saves the conversation into the
RAM cache), then send the conversation plus a short delta. Round A restores
the prefix file first; round B does not. Nothing else differs.

    round    restore      prefix      conv   cache_n    took s
    A0       ON             2290     13780      2290      56.3
    B0       OFF            2290     13780     13780       0.9

Repeated with the order reversed and fresh filler text, so neither a warmer
cache in the second half nor a leftover entry from the first run can explain
it:

    BX0      OFF            2290     13780     13780       1.0
    AX0      ON             2290     13780      2290      56.4

**62x.** With the restore, `cache_n` is exactly the file's 2290 tokens and the
remaining 11,580 are computed again. Without it, the cache hands back all
13,780 and the request is done in a second.

## The live incident this came from

29.08., one Claude Code session, martin-pc2:

    21:29:35  release n_tokens = 69939            the session's last turn
    21:30:22  selected slot by LRU                the watchdog takes the slot
                                                  -> prompt_save() of those 69939
    ... two hours of probes, each selecting by LCP similarity against its own
        34 tokens, f_keep = 0.794 — the entry is never touched
    ... no `removing oldest entry` and no `exceeds cache size limit` all day

    23:41:50  gateway restores the prefix file: 14568 tokens, 124 ms
    23:41:50  selected slot by LCP similarity, f_sim_best = 0.208, f_keep = 1.000
    23:50:16  reused 14568, computed 55452, took 506.4 s

0.208 x 70020 = 14564: the slot held exactly the restored prefix and all of it
matched. Eight and a half minutes for a turn the A/B above says would have
taken about a second.

## How often, over eight days

`bench/suites/restore-cost.py 8`, reading both journals. A hit is a restore
whose request reused EXACTLY the restored token count — the fingerprint of
f_keep = 1.0 — while a longer state of the same prefix had been served earlier
in the same llama-server life.

    8 days, 34 restores, 87 llama-server starts, 26 cache-pressure lines

    when           prefix         in file recomputed    hidden   took s
    29.08 23:50    4c911caefaaf     14568     55452     54699    506.4
    29.08 07:02    52f3f095ca67      6676        99        99      1.6
    29.08 00:21    b4124abc721a     14838        98        98      1.4

    3 incidents, at most 54896 tokens hidden from the cache.

So: rare, and almost all of the cost is in one event. Three hits out of 34
restores, and two of them cost under two seconds. This is a tail risk, not a
steady tax — which also means a fix must not make the common case worse.

**The number is an upper bound**, and it rests on an inference llama.cpp does
not let anyone check: that the earlier state was still resident. What supports
it is the absence of every eviction line since 26.08. and the fact that no
request in the relevant window was large enough to consume the entry (each
would need f_keep >= 0.25 against 69,939 tokens).

## The scan was wrong first, and silently

`SERVER_UP` originally matched llama.cpp's own wording, `server is listening`,
which this build does not print at the journal's verbosity. The scan found
**0 restarts across 8 days that contain 87**, put every sighting in one epoch,
and credited two further incidents with an earlier state that a restart had
already erased. It produced a plausible table and a larger total. What is
always there is systemd's own `Started llama-user@` line, because systemd
writes it rather than the program.

Left in the source as a comment, because the failure mode — a log-scraping
regex that matches nothing and reports zero instead of failing — has no
symptom of its own.

## What a fix has to weigh

Skipping the restore is not free. If the cache does NOT hold the conversation,
the restore is what saves the prefix: without it the slot holds something
foreign, f_keep falls below 0.5, the lookup runs, finds nothing, and the
prefix is prefilled too. On this machine that is 14,568 tokens, roughly 75 s.

    restore, cache warm    costs everything behind the prefix   (measured: 55 s)
    restore, cache cold    saves the prefix                     (~75 s)
    no restore, cache warm saves everything                     (measured: 1 s)
    no restore, cache cold costs the prefix                     (~75 s)

So the decision is entirely about whether the cache holds the conversation,
and the gateway cannot see the cache — nothing exposes it. What it CAN see is
whether it has served this prefix since llama-server started: if it has not,
the cache cannot hold anything for it, and the restore is safe. That is the
narrow rule implemented behind `RESTORE_WHEN_UNSEEN`, default off, and it is
NOT yet measured against real traffic.

## Reproduce

    python3 bench/suites/restore-blinds-cache.py                  # A then B
    python3 bench/suites/restore-blinds-cache.py --salt X --restore-first 0
    python3 bench/suites/restore-cost.py 8                        # the history
