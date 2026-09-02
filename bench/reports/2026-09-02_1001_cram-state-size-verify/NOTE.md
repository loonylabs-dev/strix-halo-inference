# A run that was aborted, kept because it agrees

This is the first attempt of the verification, abandoned mid-way and finishing
on its own afterwards. Two of its three points report
`selected_by_lru: false`: the seeds were taken from the clock, so their decimal
digits agreed for several characters, the generated prompts shared a prefix,
and three common tokens out of a 24-token takeover is `f_sim` 0.125 — past the
0.1 threshold, so the slot was chosen by LCP similarity and `prompt_save` never
ran on the takeover. Its `stored: true` on those points is wrong; it predates
the check that now drops such a point loudly.

The sizes came out right anyway, and the reason is worth writing down: the
state was written by the NEXT point's prefill instead, which did take the slot
by LRU. So the entries existed, just not where the run thought they did.

It is kept because that makes it a third independent pass over the same
question, with different prompts again, and it lands on the same line:
**39.12 KiB/token + 336.7 MiB fixed.** Read
`2026-09-02_1002_cram-state-size-verify/` for the clean run.
