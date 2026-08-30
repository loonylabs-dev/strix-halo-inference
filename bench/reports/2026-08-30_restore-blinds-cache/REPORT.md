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

## The third case, measured: the file is redundant while the cache is warm

`restore-blinds-cache.py --mode redundant`. Prefill a 2690-token prefix, hand
the slot to an unrelated tiny prompt (so the prefix goes into the RAM cache),
restore NOTHING, then send that prefix plus a long new conversation:

    0 prefill PREFIX             12.9 s   cache_n=0     prompt_n=2690
    1 unrelated tiny              0.7 s   cache_n=0     prompt_n=7
    3 PREFIX + new conversation  66.8 s   cache_n=2690  prompt_n=13290

`cache_n = 2690` with no file involved: llama.cpp found the prefix in its own
RAM cache and handed it back. The .bin added nothing.

## What a fix has to weigh

Skipping the restore is not free. If the cache does NOT hold the conversation,
the restore is what saves the prefix: without it the slot holds something
foreign, f_keep falls below 0.5, the lookup runs, finds nothing, and the
prefix is prefilled too. On this machine that is 14,568 tokens, roughly 75 s.

    the RAM cache holds     with the restore        without it
    ---------------------   ---------------------   ---------------------
    the conversation        56.4 s   (measured)     1.0 s   (measured)
    only the prefix         no different            no different (measured:
                                                    the cache returns it)
    nothing                 saves the prefix        prefilling the prefix,
                            (~75 s here)            ~75 s

Two of the three rows are now measured, and they say something sharper than a
threshold: **the disk restore adds value in exactly one situation — when
llama.cpp's RAM cache holds nothing for this prefix.** With `-cram 32768` that
means after a llama-server restart, or after the entry was evicted under
pressure (26 such lines in eight days, all on 26.08.).

So the better rule is not "how long is the tail" but "is the server cold for
this prefix", and it needs no magic number. What it needs is a way to know,
and there is none directly: `/props` carries no uptime and no boot id, and
`/health` is one field. Two ways round it, neither built:

  * derive it — persist the last-seen llama-server start beside the prefix
    ledger, and treat a restart as invalidating every "recently served" flag;
  * measure it — skip the restore, let the request run, and read `reused`.
    Poor reuse means the server is cold, so restore for the NEXT request. That
    costs exactly one slow turn per server restart, against a possible 500 s
    on any long conversation today.

Until one of them exists, what ships is the blunter `RESTORE_WHEN_UNSEEN`, default off, and it is
NOT yet measured against real traffic.

## What measuring it cost the machine it was measured on

The runs above evicted the operator's live session from the RAM cache:

    30.08. 00:12  removing oldest entry (327, 381, 360, 360, 942 MiB)
    30.08. 00:19  removing oldest entry (6570.640 MiB)   <- the session
    30.08. 00:25  removing oldest entry (301, 1134 MiB)

6570 MiB is a 66,826-token conversation, which is what Claude Code had running
at 00:01. At `-cram 32768` the cache holds roughly five states that size, and
each A/B round adds two of its own — so a few minutes of benchmarking is
enough to push a real session out. Their next turn is a cold start.

This is worth more than an apology in a report. It is the same arithmetic as
the defect itself: the cache is the thing that makes long conversations cheap,
it is small in units of real sessions, and nothing warns when it fills. The
suite's docstring now says so, and the honest place to run it is a side server
or an idle machine.

## The fix, measured through the real gateway

Everything above drives llama-server's slot API by hand. `restore-guard-live.py`
drives **cc-gateway** instead, so what is measured is the decision code that
would ship — the `cold` flag, the prefix ledger, the /slots reading.

It reproduces the 29.08. situation deliberately: warm the prefix, put a
conversation behind it, **restart the gateway** (so its ledger is empty and the
next request takes the restore path), then send the next turn.

    guard   after the gateway restart              took
    off     cached  7 298   computed 28 936      187.4 s
    on      cached 36 211   computed     22        1.6 s

**117x**, and the trace shows the decision itself:

    08:32:43  restore-skipped  c90c269ce2b2
    08:32:44  request          reused 36211  took 1.59

`RESTORE_ONLY_WHEN_SERVER_COLD` reads `id_task` from /slots — a counter that
only rises within one llama-server life. It restores when the counter fell (the
server restarted under a running gateway) or is still tiny (both restarted
together), and skips otherwise. That is the case the gateway's own `cold` flag
gets wrong: `cold` means "I have not served this since I started", and on
29.08. the gateway had started at 23:38 beside a server up since 09:48.

WHAT IT COSTS WHEN IT IS WRONG. If the server is warm but its cache no longer
holds the conversation — an eviction, 26 of them in eight days — the skipped
restore means the prefix is prefilled too: measured 8.9 s for 1890 tokens,
91.7 s for the 17,784-token production prefix. That is the whole downside, it
is bounded, and it is one turn.

It ships OFF. Turning it on is one line in `~/.config/cc-gateway.env`:

    RESTORE_ONLY_WHEN_SERVER_COLD=1

## Reproduce

    python3 bench/suites/restore-blinds-cache.py                  # A then B
    python3 bench/suites/restore-blinds-cache.py --salt X --restore-first 0
    python3 bench/suites/restore-blinds-cache.py --mode persistence --rounds 5
    python3 bench/suites/restore-cost.py 8                        # the history
    python3 bench/suites/restore-guard-live.py --guard off         # the fix,
    python3 bench/suites/restore-guard-live.py --guard on          # end to end
