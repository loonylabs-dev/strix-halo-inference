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

## Files

    postanswer.py / postanswer.log   the seven steps
    controls.py / controls.log       A and B
    (the logs are .gitignored; every number quoted above is in this file)
