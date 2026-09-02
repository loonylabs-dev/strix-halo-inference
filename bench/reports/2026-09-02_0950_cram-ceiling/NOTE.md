# What this run answered, and what it did not

**The question it was sent to answer — is a 90,000-token state too large for
`-cram 4096`? — came back NO.** The state was stored, and no
`exceeds cache size limit` line appeared. The ceiling is real but sits higher
than the probe: computed from the measured line (336.7 MiB + 39.12 KiB/token,
`2026-09-02_1002_cram-state-size-verify/`), 4096 MiB is exceeded at ~98,400
tokens for a bare prompt state, and at ~84,400 for a *served* session, which
carries a further ~534 MiB. That second figure is inside Claude Code's working
range, so the ceiling matters — it just was not reached here.

**Its RssAnon numbers are not usable** for the same reason as the run before
it; see `../2026-09-02_0938_cram-state-size/NOTE.md`. `rss_delta_mib: -674.5`
is the sum of one insertion and ten evictions, not the size of anything.

**The number worth keeping was taken afterwards, by hand.** The 90,000-token
entry was still resident, so one more 24-token takeover was sent to push it
out, and llama-server named it on the way:

    09:52:21  making room for prompt cache entry, removing oldest entry
              (size = 3774.721 MiB)

90,000 tokens = **3774.721 MiB**, which lands on the fitted line to 0.03 MiB
and is the highest point that line rests on.
