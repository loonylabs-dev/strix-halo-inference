# What a restored slot state can and cannot do

29.08.2026, 00:04–00:11, production qwen38, one slot, measured through the
gateway so the rendering is the one that really arrives.

The question came out of a review of the save-policy design: the race for the
single slot exists only because `prewarm` RE-CREATES the prefix before saving
it. If the slot could simply be written as it stands the instant an answer is
out — prefix plus question plus answer — there would be no prefill, no
residency question, no policy, and no race. It is the cheapest possible fix,
so it had to be tried before anything else was built.

## The measurement

    1. Q1 asked                          14.3 s   reused 12912  computed 2055
       slot now holds prefix+Q1+A1       14998 tokens
    2. save the slot AS IT STANDS        14998 tokens, 1140 MB, 342 ms
    3. displace it with another prefix   76.8 s
    4. restore the post-answer state     14998 tokens, 941 ms
    5. ask the SAME prefix, NEW question 76.8 s   reused 0      computed 14969
    6. control, next question, no restore 2.5 s   reused 14860  computed 107
    7. restore again, ask again          77.5 s   reused 0      computed 14966

Step 6 is what makes steps 5 and 7 admissible: the same measurement, taken
between two ordinary requests, reports 14860 reused. The instrument works. The
restored state carried nothing, twice.

## Two controls that pin the rule

    A  post-answer state + the SAME question it was saved with
       state = prefix+Q1+A1, prompt = prefix+Q1     76.7 s  reused 0     computed 14967
    B  prefix-only state + a DIFFERENT question
       state = prefix,       prompt = prefix+Qx      1.5 s  reused 14838 computed 106

A is the decisive one. The saved state's first 14967 tokens ARE the incoming
prompt, exactly — and llama.cpp still recomputed all of them. So the rule is
not "the state must match the request"; it is stricter:

> **A restored state is only reused when it is a PREFIX of the incoming
> prompt. A state carrying anything beyond that — even the question it was
> saved with — is discarded whole, not trimmed back to the common part.**

B shows the other side: a state that IS a true prefix serves a completely
different question at 1.5 s, which is what the whole store exists for.

Worth recording: llama-server logged step 5 as `selected slot by LCP
similarity, f_sim_best = 0.997` and then evaluated 14969 tokens. The
similarity check that picks the slot and the reuse that follows are not the
same mechanism, and only the second one decides what a restore is worth. A
verdict read off the log line would have been wrong.

## What this settles

* **The cheap fix does not exist.** `prewarm` isolating the prefix — render,
  prefill it alone, save — is not accidental complexity. It is the only shape
  a saved state may have. Its existing guard ("slot holds more than the
  prefix — erasing it and recomputing") is load-bearing, and this measurement
  is why.
* **The save policy question stands** unchanged: the race for the one slot is
  real and has to be handled where it happens.
* **Correctness was never the risk here.** Steps 5 and 7 answered the NEW
  question cleanly; nothing of the saved turn leaked in. The state is dropped,
  not misused — so the failure is slow, not silent. `prewarm.py`'s warning
  about answering "the old question" describes a different mechanism (saving a
  session and restoring it as a prefix), not this one.
* **Writing the slot costs 342 ms.** For 1.1 GB, i.e. page cache. Everything
  above that in today's numbers (0.5–4.3 s, and 101.9 s when contended) is
  prewarm's preparation, not the write.

## What it makes possible

A file whose token count differs from the rendered prefix's is now KNOWN to be
worthless — too long is discarded whole (measured here), too short cannot
cover the prompt (the 34-token file found on 28.08.). prewarm can therefore
refuse to publish such a file at the moment it writes it, which is the local,
cheap guard the review asked for and which no policy is needed for.

## The follow-up turn, and what a save really costs

Same night, after the gateway learned to defer a save (nothing in flight, and
the slot still holding this prefix):

    turn 1  cold, a fresh prefix          13.0 s   reused 12919  computed 2066
            NOTE  not saved yet: requests in flight (strike 1 of 3)
    turn 2  one second later              12.4 s   reused 12937  computed 2048
    turn 3  eight seconds later            7.7 s   reused 14886  computed 99
            SAVED  automatically, 87.3 s

The follow-up turn is no longer evicted — that is the 0.7 s -> 13.6 s defect
not happening, and it is why the deferral exists.

THE 87 SECONDS ARE NOT WHAT THEY LOOK LIKE, and the first explanation written
here was wrong. It said prewarm always erases the slot after a turn (because
it then holds prefix+Q+A, i.e. more than the prefix) and pays a full prefill.
Measured directly:

    slot holds prefix+Q+A                14995 tokens
    /completion with the prefix, NO erase 11.0 s   cache_n 12940  prompt_n 1949
    slot now holds                       14889 tokens = the prefix exactly

So llama.cpp truncates the slot back to the prefix on its own, prewarm's erase
branch does not fire in this case, and the preparation costs 11 s rather than
75. The 87 s save had a different cause: it started between turn 2 and turn 3
and turn 3 arrived while it ran. Its file was still correct — the request was
for the SAME prefix, which both the gateway's window check and prewarm's
token-count check treat as harmless, and both were right to.

What remains true after that correction: a save is not a 342 ms write in
practice. Truncating the slot back to the prefix costs ~11 s of recomputation
(~1949 tokens), for reasons this measurement does not explain — the first
12940 tokens are reused and the tail is not. Worth a look, not tonight.

## The prefix computed FIRST — the whole problem dissolves

The idea is the operator's, and it turns the question around. Today the save
happens after the answer, when the slot holds prefix+question+answer, and
getting back to a prefix-only state costs a recomputation and cuts the session
out of the slot. But the prefix-only state DOES exist at one moment — right
before the first real message, if the prefix is prefilled on its own.

Measured end to end, one prefix, nothing else on the machine:

    prefix                                14866 tokens
    A  prefill the prefix ALONE   12.0 s  slot now holds 14866 = the prefix
    B  save it                     314 ms n_saved 14866, matches exactly
    C  the first real request       1.6 s reused 14866  computed 99
    D  displace the slot           22.8 s reused 0      computed 4730
    E  restore the file                   14866 tokens back
    F  a question it has NEVER seen  1.7 s reused 14866  computed 106

F is the one that matters: a file written this way serves a LATER session, not
just the request that created it.

WHAT IT COSTS. Step A is not extra work — it is the prefill the first request
had to do anyway, moved in front of it. (12.0 s rather than ~75 s here only
because the RAM cache still held most of this prefix from the evening's other
runs; on a cold machine A is the full prefill and C is still ~1.6 s.) The
genuine addition is B: 314 ms, once per prefix, ever.

WHAT IT REMOVES:

    the ~11 s truncation before every save          gone: nothing to truncate
    the session context cut out of the slot         gone: nothing is cut
    the background task racing the next turn        gone: it is not a task
    the deferral, the strikes, the owed retries     unnecessary
    the debounce and every parameter around it      unnecessary

The save becomes a half-step inside the first cold request for a prefix, while
that request holds the admission gate — so nothing else can take the slot, by
construction rather than by policy. That is the freeze the operator asked for,
and it is free because the state it freezes is the one we wanted anyway.

## Built, and measured again on the live stack

The ordering is in cc-gateway since 29.08. 00:47. A cold request whose prefix
is not on disk saves it BEFORE the request is forwarded:

    [ 2.9s] START  prefix=34576f74398b COLD
    [16.1s] SAVED  prefix 34576f74398b automatically, 13.2 s
    [17.5s] DONE   took=14.6s  reused=14885 computed=99
    [18.5s] START  warm
    [20.3s] DONE   took=1.8s   reused=14885 computed=99
    [25.3s] START  warm
    [27.0s] DONE   took=1.8s   reused=14885 computed=99

The first turn carries the save (13.2 s of it is the prefill it had to do
anyway; the write is 0.3 s of that). The request itself then computed 99
tokens. The follow-up turns are 1.8 s and undisturbed — no eviction, nothing
in the background to collide with.

What came out of the code with it: the background task, the deferral, the
strikes, the owed retries and the debounce. They existed to manage a race that
the order does not have.

Three guards went in instead, all of them about the save now sitting in the
request path: a timeout (a strange server must not turn the save into a hung
answer), an exception that costs a cold prefill rather than a reply, and a
SHIELD — a client that leaves mid-write must not leave a `.bin` without its
sidecar, which would be invisible to the store and undeletable by a cleanup
that prunes by sidecar.

One contract changed and is worth stating: "an abort before the answer saves
nothing" no longer holds, because the file is no longer whatever the slot
happened to hold. prewarm renders the prefix, prefills that alone and refuses
to publish a file whose token count is not the prefix's — so an abort decides
whether the work was worth doing, not whether the file is valid.

## Files

    postanswer.py / postanswer.log   the seven steps
    controls.py / controls.log       A and B
    (the logs are .gitignored; every number quoted above is in this file)
