# Superseded — the sizes in this run are 21–30 % low

`entry_mib_est` and `kib_per_token` in `summary.json` were derived from the
llama-server process's **RssAnon delta** across the moment a state is written
into the prompt cache. That path is wrong, and wrong in the comfortable
direction: it reports entries as smaller than they are, which makes a `-cram`
budget look sufficient.

Why it fails: entries allocated after an eviction are served out of the arena
the eviction freed, and RssAnon does not move for them at all. The first point
of the morning — taken while the allocator had nothing free — agreed with the
server to four digits (226.1 against 226.004 MiB). Everything after it drifted.

| tokens | this run (RssAnon) | llama-server's own figure |
|---|---|---|
| 2,000 | 289.8 MiB | 413.087 MiB |
| 8,000 | 509.8 MiB | 642.290 MiB |
| 20,000 | 890.3 MiB | 1100.694 MiB |

**Use `2026-09-02_1002_cram-state-size-verify/` instead.** It reads the size
llama-server itself names when it evicts an entry, and its three points were
reproduced by a second run with different prompts to three decimal places.

What survives from this run: the RssAnon *purge* reading, which is the proof
that the prompt cache is host memory at all — evicting a 3949 MiB entry moved
the server from 4928.4 to 1205.3 MiB, this profile's declared idle figure.
