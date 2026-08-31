#!/usr/bin/env python3
"""How often did a prefix restore hide a conversation the server still had?

`restore-blinds-cache.py` proves the mechanism on demand: with the restore
56.4 s, without it 1.0 s, same request. This one asks the other question —
how often it actually fired — by reading the journals both services already
wrote. No GPU, no requests, seconds per run.

WHAT COUNTS AS A HIT, and why it is an UPPER BOUND.

    RESTORED prefix P ... N tokens          the gateway put the file in the slot
    DONE     ... prefix=P reused=N computed=C    reuse equals the restore EXACTLY

`reused == N` is the fingerprint: everything the request kept came from the
file and nothing from the slot, i.e. llama.cpp's own prompt cache contributed
zero. That is what f_keep = 1.0 does.

For it to have COST anything, the cache must have held something longer. This
script requires that the same prefix was served earlier — since the last
llama-server start, because a restart empties the cache — with a total larger
than N, and counts the difference. What it cannot check is whether that entry
was still resident: llama.cpp does not log its cache contents, and under
memory pressure it drops the oldest. So every figure here is an upper bound,
and the pressure lines are printed beside it so the reader can see how tight
the bound is.

    python3 bench/suites/restore-cost.py            the last 7 days
    python3 bench/suites/restore-cost.py 14
"""
import re, subprocess, sys, time
from collections import defaultdict

# llm-gateway since 09/2026 — and the pre-rename unit beside it, because the
# measured days live in ITS journal. History is read as it was written; the
# same reasoning as the German field names accepted further down.
GATEWAY_UNITS = ("llm-gateway.service", "cc-gateway.service")
SERVER_UNIT = "llama-user@qwen38"

RESTORED = re.compile(r"RESTORED\s+prefix (\w+) from \S+ -> slot \d+, (\d+) tokens")
DONE = re.compile(r"DONE\s+\S+\s+\S+\s+who=(\S+)\s+prefix=(\w+) took=([\d.]+)s"
                  r"(?:\s+reused=(\d+) computed=(\d+))?")
# The gateway prints its own banner on start; llama-server's start is what
# empties the RAM cache, so that is the one that matters here.
#
# MEASURED WRONG FIRST, 30.08.2026: this matched llama.cpp's own wording
# ("server is listening"), which this build does not print at the journal's
# verbosity. The scan then found 0 restarts across 8 days that contained one,
# put every sighting in the same epoch, and credited two incidents with an
# earlier state that a restart had already erased. It failed silently and in
# the direction that inflates the finding. What IS always there is systemd's
# own line, because systemd writes it, not the program.
# `Started` only: matching `Starting` as well counts every start twice.
SERVER_UP = re.compile(r"Started llama-user@")
PRESSURE = re.compile(r"removing oldest entry|exceeds cache size limit|"
                      r"cache size limit reached")


def journal(unit, days):
    """`unit` is one name or a tuple — journalctl merges repeated -u by time."""
    cmd = ["journalctl", "--user"]
    for u in (unit if isinstance(unit, tuple) else (unit,)):
        cmd += ["-u", u]
    out = subprocess.run(
        cmd + ["--since", "-%d days" % days, "--no-pager", "-o", "short-unix"],
        capture_output=True, text=True)
    return out.stdout.splitlines()


def stamp(line):
    try:
        return float(line.split(None, 1)[0])
    except Exception:
        return None


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    gw = journal(GATEWAY_UNITS, days)
    srv = journal(SERVER_UNIT, days)
    if not gw:
        print("  nothing to read: no journal for %s in %d days"
              % ("/".join(GATEWAY_UNITS), days))
        return 1

    restarts = sorted(t for t in (stamp(l) for l in srv if SERVER_UP.search(l))
                      if t)
    pressure = [stamp(l) for l in srv if PRESSURE.search(l)]
    pressure = sorted(t for t in pressure if t)

    def epoch_of(t):
        """Which llama-server life this moment belongs to. The RAM cache does
        not survive a restart, so a sighting from an earlier life proves
        nothing about what is cached now."""
        n = 0
        for r in restarts:
            if r <= t:
                n += 1
        return n

    # Walk the gateway journal once: remember the largest total seen per
    # (prefix, epoch), and flag every restore whose request reused exactly
    # what the file carried.
    biggest = defaultdict(int)
    pending, hits, restores, events = None, [], 0, []
    for line in gw:
        t = stamp(line)
        if t is None:
            continue
        m = RESTORED.search(line)
        if m:
            restores += 1
            pending = (t, m.group(1), int(m.group(2)))
            continue
        m = DONE.search(line)
        if not m:
            continue
        who, prefix, took = m.group(1), m.group(2), float(m.group(3))
        reused = int(m.group(4)) if m.group(4) else None
        computed = int(m.group(5)) if m.group(5) else None
        key = (prefix, epoch_of(t))
        if pending and pending[1] == prefix and reused is not None:
            _, _, n_file = pending
            seen = biggest[key]
            if computed is not None:
                # Every restore, not only the harmful ones: the sweep below
                # needs the ones a threshold would WRONGLY skip just as much.
                events.append({"t": t, "prefix": prefix, "n_file": n_file,
                               "total": reused + computed,
                               "tail": reused + computed - n_file,
                               "hidden": (min(seen, reused + computed) - n_file
                                          if reused == n_file and seen > n_file
                                          else 0)})
            if reused == n_file and computed:
                if seen > n_file:
                    hits.append({"t": t, "who": who, "prefix": prefix,
                                 "file_tokens": n_file, "computed": computed,
                                 "took_s": took, "best_earlier": seen,
                                 "hidden": min(seen, reused + computed) - n_file})
        pending = None
        if reused is not None and computed is not None:
            biggest[key] = max(biggest[key], reused + computed)

    print("  %d days, %d restores, %d llama-server starts, %d cache-pressure lines"
          % (days, restores, len(restarts), len(pressure)))
    if pressure:
        print("  pressure at: %s"
              % ", ".join(time.strftime("%d.%m %H:%M", time.localtime(p))
                          for p in pressure[:8]))
        print("  -> on those days the upper bound below is loose: the entry "
              "may already have been dropped.")
    if not hits:
        print("  no restore was followed by a request that reused ONLY the "
              "file while a longer state of the same prefix had been served "
              "in the same server life.")
        return 0

    hits.sort(key=lambda h: -h["hidden"])
    print("\n  %-14s %-13s %8s %9s %9s %8s"
          % ("when", "prefix", "in file", "recomputed", "hidden", "took s"))
    for h in hits[:20]:
        print("  %-14s %-13s %8d %9d %9d %8.1f"
              % (time.strftime("%d.%m %H:%M", time.localtime(h["t"])),
                 h["prefix"], h["file_tokens"], h["computed"], h["hidden"],
                 h["took_s"]))
    total = sum(h["hidden"] for h in hits)
    print("\n  %d incidents, at most %d tokens hidden from the cache." % (len(hits), total))
    print("  UPPER BOUND. It assumes the earlier state was still resident, "
          "which llama.cpp does not log.")
    sweep(events)
    return 0


def sweep(events):
    """What a RESTORE_MAX_TAIL_CHARS threshold would have done to this traffic.

    The switch skips the restore when the conversation BEHIND the prefix is
    long. Each restore in the journal falls into one of two buckets:

      harmful   the request reused only the file while a longer state of the
                same prefix had been served in the same server life -- the
                cache almost certainly had it, and the restore hid it
      useful    everything else. Here the restore is what put the prefix in
                the slot, so skipping it means prefilling the prefix too

    A threshold saves the hidden tokens of the harmful ones it skips and costs
    the prefix of the useful ones it skips. Both columns are in TOKENS; the
    switch is in characters, so the last column converts at the ratio this
    traffic actually shows.
    """
    if not events:
        return
    chars_per_token = 4.3      # this stack's own prefixes: 60428 chars / 14568
                               # tokens = 4.15, 73404 / 17784 = 4.13; rounded
                               # up so a threshold errs towards restoring
    print("\n  What a RESTORE_MAX_TAIL_CHARS threshold would have done")
    print("  %d restores with a request behind them, %d of them harmful"
          % (len(events), sum(1 for e in events if e["hidden"])))
    rows = []
    for t_tok in (500, 1000, 2000, 5000, 10000, 20000, 50000):
        skipped = [e for e in events if e["tail"] > t_tok]
        saved = sum(e["hidden"] for e in skipped)
        cost = sum(e["n_file"] for e in skipped if not e["hidden"])
        rows.append((t_tok, len(skipped), sum(1 for e in skipped if e["hidden"]),
                     saved, cost))
    print("\n  %10s %10s %8s %10s %10s %12s"
          % ("tail >", "skipped", "harmful", "saved tok", "cost tok", "net tok"))
    for t_tok, n, harm, saved, cost in rows:
        print("  %10d %10d %8d %10d %10d %12d"
              % (t_tok, n, harm, saved, cost, saved - cost))
    if len(set(r[1:] for r in rows)) == 1:
        # A table whose every row agrees is not a consensus, it is an absence
        # of evidence -- and printed without saying so it reads as the former.
        print("\n  EVERY ROW IS IDENTICAL, so this traffic CANNOT CHOOSE a")
        print("  threshold. Only %d restore(s) have a tail large enough to be"
              % rows[0][1])
        print("  affected at all, and the same one is affected at every value")
        print("  tested. Nothing here justifies a default; leave the switch off")
        print("  until a period with more of them has been recorded.")
        return
    print("\n  In characters, at ~%.1f characters a token: multiply the first"
          % chars_per_token)
    print("  column by that. THIN DATA -- read the counts, not the totals: a")
    print("  single incident dominates every row it appears in.")


if __name__ == "__main__":
    sys.exit(main())
