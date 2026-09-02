#!/usr/bin/env python3
"""cram-state-size — what one prompt-cache entry costs, in MiB per token.

WHY THIS EXISTS. `-cram` is a MiB budget for llama.cpp's RAM prompt cache, and
every decision about its value needs one number the server never prints at
INFO level: how large the entry for a conversation of N tokens actually is.
Without it the budget is guessed, and a guessed budget produced the 02.09.2026
incident — two Claude Code sessions, ~80k tokens each, the second one prefilled
from zero because the first one's entry had been evicted by a 30-token probe.

WHAT IT MEASURES, AND WHY NOT THE OBVIOUS WAY. The size of an entry is read
from llama-server itself: the `making room for prompt cache entry, removing
oldest entry (size = X MiB)` line names it exactly. To get that line for a
chosen entry the run FLUSHES the cache afterwards with tiny unrelated prompts,
which is possible at all because the cache is a deque — push_back on insert,
pop_front on eviction (server-task.cpp:1748) — so a long enough flush walks
every entry past that line in insertion order.

Which line belongs to which point is then decided by SIZE, not by position.
Position alone would be wrong whenever the cache also held entries from
before the run: those leave first and shift everything after them. Every
takeover this suite sends is the same 24-token prompt, so its entry size is
the value that repeats in the eviction list, and the points are the lines
differing from it by more than 10 % — the last such lines, since anything
older left earlier. A point small enough to fall inside that 10 % is reported
as unmatched instead of being guessed at.

The obvious way was tried first and is wrong. Reading the llama-server
process's RssAnon delta across the moment a state is written LOOKS like a
direct measurement, and it agrees with the eviction line while the allocator
has no free arena — 226.1 against 226.004 MiB on the first point of
02.09.2026. It then drifts low, and silently: once an eviction had freed
3.7 GiB, the following entries were served out of that arena without RssAnon
moving for them at all. Measured the same morning, RssAnon against the
server's own figure: 2,000 tokens 289.8 vs 413.087, 8,000 tokens 509.8 vs
642.29, 20,000 tokens 890.3 vs 1100.694 — 21 to 30 % low, in the direction
that makes a budget look sufficient. RssAnon is still recorded, because a
gross disagreement between the two is worth seeing, but it is a LOWER BOUND
and never the answer.

HOW AN ENTRY IS CREATED AT ALL. server-context.cpp:1631-1640: the outgoing
prompt is saved into the cache only on the LRU path of slot selection — i.e.
when the incoming request has nothing in common with what the slot holds. That
is what the health probe does five times a day, and what this suite does on
purpose with a tiny unrelated prompt after each measured prefill.

WHAT IT COSTS. One prefill per point at the machine's prefill rate — the
default points total 30,000 tokens, about three minutes on Flash-Next — plus
the flush, which is one ~2 s request per 226 MiB of cache to clear. It occupies
the one production slot for that time and empties whatever the cache holds.

    python3 bench/suites/cram-state-size.py
    python3 bench/suites/cram-state-size.py --points 2000,8000,20000,40000

The probe timer is stopped for the duration and restarted afterwards — a probe
landing mid-run would insert an entry of its own and break the ordering the
flush matching depends on. `--keep-probe` leaves it alone for a run that must
not touch the watchdog; the matching is then a guess and the report says so.
"""
import argparse, json, os, random, re, subprocess, time, urllib.request

SRV = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The eviction line names the size of the entry that was thrown out. It is the
# only place llama-server prints an entry size at WARN level, which is why the
# whole measurement hangs on it.
EVICT = re.compile(r"removing oldest entry \(size = ([0-9.]+) MiB\)")
LRU = re.compile(r"selected slot by LRU")
# The other end of the budget, and the one that fails silently: an entry larger
# than the whole limit is never stored at all (server-task.cpp:1727). The line
# names the size it refused, which makes it a measurement point above what the
# cache can hold.
TOOBIG = re.compile(r"prompt state size ([0-9.]+) MiB exceeds cache size limit ([0-9.]+) MiB")


def req(path, payload=None, method=None, t=3600):
    d = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(SRV + path, data=d, method=method,
                               headers={"content-type": "application/json"})
    with urllib.request.urlopen(r, timeout=t) as x:
        b = x.read().decode()
        return json.loads(b) if b.strip().startswith(("{", "[")) else b


def server_pid():
    """The llama-server PID, read from /proc rather than from pgrep.

    `pgrep -f llama-server` matches this script's own command line as readily
    as the server's, and a measurement keyed on the wrong PID reads a constant
    RssAnon and calls it "the cache costs nothing".
    """
    found = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open("/proc/%s/comm" % name) as f:
                comm = f.read().strip()
        except OSError:
            continue
        if comm in ("llama-server", "llamaexec"):
            found.append(int(name))
    if len(found) != 1:
        raise SystemExit("expected exactly one llama-server, found %r" % (found,))
    return found[0]


def rss_anon_mib(pid):
    with open("/proc/%d/status" % pid) as f:
        for line in f:
            if line.startswith("RssAnon:"):
                return int(line.split()[1]) / 1024.0
    raise SystemExit("RssAnon missing from /proc/%d/status" % pid)


def tokens_for(n, seed):
    """A prompt of EXACTLY n tokens, sharing no beginning with any other one.

    The shared beginning matters twice over: a prompt that starts like a cached
    one gets picked up by LCP similarity (no LRU, so no save), and a cached
    entry fully contained in a new prompt is deleted outright by
    server_prompt_cache::alloc. Both would measure something other than what
    this suite claims to measure.
    """
    rnd = random.Random(seed)
    words = ["%s%d" % (rnd.choice("qxzjvkwfy"), rnd.randrange(10 ** 6))
             for _ in range(n * 2 + 40)]
    # NOT "%d " % seed in front. Seeds that differ by a little — which is what
    # they do once the default is the clock — write decimal digits that agree
    # for the first several characters, and llama.cpp measures similarity as
    # common prefix over the NEW prompt's length: three shared tokens out of a
    # 24-token takeover is f_sim 0.125, past the 0.1 threshold, so the slot is
    # chosen by LCP similarity and prompt_save never runs. Measured 02.09.2026,
    # a whole run reporting `takeover by LRU: False`. The generated words carry
    # the seed's entropy from the first character instead.
    text = " ".join(words)
    toks = req("/tokenize", {"content": text})["tokens"]
    if len(toks) < n:
        raise SystemExit("tokenizer returned %d < %d tokens" % (len(toks), n))
    return toks[:n]


def prefill(toks, npredict=1):
    t0 = time.time()
    r = req("/completion", {"prompt": toks, "n_predict": npredict,
                            "cache_prompt": True, "stream": False})
    tm = r.get("timings") or {}
    return {"wall_s": round(time.time() - t0, 2),
            "prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n")}


def journal_since(unit_glob, since_iso):
    out = subprocess.run(
        ["journalctl", "--user", "-u", unit_glob, "--since", since_iso,
         "--no-pager", "-o", "short-iso"],
        capture_output=True, text=True, timeout=120)
    return out.stdout.splitlines()


def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default="2000,8000,20000",
                    help="token counts to measure, comma separated")
    ap.add_argument("--unit", default="llama-user@*")
    ap.add_argument("--label", default="cram-state-size")
    ap.add_argument("--keep-probe", action="store_true",
                    help="do not stop llama-probe.timer for the run")
    ap.add_argument("--flush-max", type=int, default=40,
                    help="most tiny requests to spend emptying the cache")
    # Every prompt this suite sends has to be new to the server. A fixed seed
    # made the SECOND run of the same points measure nothing: alloc() skips a
    # prompt already in the cache, and a takeover prompt the cache recognises
    # is picked up by LCP similarity instead of LRU — and only the LRU path
    # writes the outgoing state. The first re-run showed it as a purge that
    # freed 0.2 MiB. Pass --seed to repeat a specific run exactly.
    ap.add_argument("--seed", type=int, default=int(time.time()),
                    help="base seed for the generated prompts (default: now)")
    a = ap.parse_args()

    points = [int(x) for x in a.points.split(",") if x.strip()]
    pid = server_pid()
    print("llama-server pid %d, RssAnon %.1f MiB, seed %d"
          % (pid, rss_anon_mib(pid), a.seed))

    probe_stopped = False
    if not a.keep_probe:
        r = subprocess.run(["systemctl", "--user", "stop", "llama-probe.timer"],
                           capture_output=True, text=True)
        probe_stopped = (r.returncode == 0)
        print("probe timer stopped: %s%s"
              % (probe_stopped, "" if probe_stopped else " (%s)" % r.stderr.strip()))

    result = {"label": a.label, "url": SRV, "started": stamp(), "pid": pid,
              "seed": a.seed, "probe_stopped": probe_stopped,
              "points": [], "flush": None, "refused": [], "fit": None}
    # Insertion order, which is what the flush's eviction lines are matched
    # against. Every tiny takeover inserts an entry too, so it is listed here
    # as well — leaving those out is what would shift every following match by
    # one and turn a correct measurement into a plausible wrong one.
    inserted = []
    tiny = [0]

    def takeover():
        """A tiny unrelated prompt: sends slot selection down the LRU path,
        which is what writes the outgoing state into the cache."""
        tiny[0] += 1
        t0 = stamp()
        time.sleep(1)
        r = prefill(tokens_for(24, a.seed + 700000 + tiny[0]))
        time.sleep(1)
        lines = journal_since(a.unit, t0)
        return r, any(LRU.search(L) for L in lines), lines

    sizes = []
    try:
        for k in points:
            print("\npoint: %d tokens" % k)
            before = rss_anon_mib(pid)
            pf = prefill(tokens_for(k, a.seed + k))
            print("  prefilled: prompt_n=%s cache_n=%s in %.1f s"
                  % (pf["prompt_n"], pf["cache_n"], pf["wall_s"]))
            take, by_lru, lines = takeover()
            after = rss_anon_mib(pid)
            refused = [(float(m.group(1)), float(m.group(2)))
                       for L in lines for m in [TOOBIG.search(L)] if m]
            for s_mib, l_mib in refused:
                print("  REFUSED: state %.3f MiB exceeds the %.0f MiB limit — never cached"
                      % (s_mib, l_mib))
                result["refused"].append({"tokens": k, "state_mib": s_mib,
                                          "limit_mib": l_mib})
            # A takeover that did NOT go down the LRU path saved nothing, so
            # this point is not in the cache and must not be counted as if it
            # were — that is how the matching would silently name one point's
            # size for another. Loud, and the point is dropped.
            if not by_lru:
                print("  NOT MEASURED: the takeover was picked by LCP similarity, "
                      "so no state was saved.\n    The takeover prompt shares a "
                      "prefix with this point — see tokens_for().")
            elif not refused:
                inserted.append({"what": "point", "tokens": k})
                inserted.append({"what": "takeover", "tokens": 24})
            else:
                inserted.append({"what": "takeover", "tokens": 24})
            result["points"].append({
                "tokens": k, "prefill": pf, "takeover": take,
                "selected_by_lru": by_lru,
                "rss_delta_mib": round(after - before, 1),
                "stored": bool(by_lru and not refused)})
            print("  takeover by LRU: %s, RssAnon %+.1f MiB (lower bound only)"
                  % (by_lru, after - before))

        # THE FLUSH. Tiny requests until every entry inserted above has been
        # thrown out, reading the sizes the server names on the way. One
        # 226 MiB entry per request, so a full 4 GiB cache costs about 18.
        print("\nflush: emptying the cache to read the sizes back")
        t_flush = stamp()
        time.sleep(1)
        spent = 0
        n_points = len([i for i in inserted if i["what"] == "point"])

        def enough(seen):
            """Every point out of the cache, not merely every insertion.

            Counting insertions alone stops too early whenever the cache also
            held older entries: those are evicted first and use up the count,
            leaving the run's own last point still resident and unmeasured."""
            if len(seen) < len(inserted):
                return False
            r = [round(x, 1) for x in seen]
            t = max(set(r), key=r.count)
            return len([x for x in seen if abs(x - t) > 0.10 * t]) >= n_points

        while not enough(sizes) and spent < a.flush_max:
            spent += 1
            tiny[0] += 1
            prefill(tokens_for(24, a.seed + 800000 + tiny[0]))
            time.sleep(1)
            lines = journal_since(a.unit, t_flush)
            sizes = [float(m.group(1)) for L in lines for m in [EVICT.search(L)] if m]
        print("  %d requests, %d eviction lines for %d insertions"
              % (spent, len(sizes), len(inserted)))
        result["flush"] = {"requests": spent, "evicted_mib": sizes,
                           "inserted": inserted, "complete": enough(sizes)}
    finally:
        if probe_stopped:
            subprocess.run(["systemctl", "--user", "start", "llama-probe.timer"],
                           capture_output=True, text=True)
            print("\nprobe timer restarted")

    # MATCHING. Position alone is not enough: the cache also holds entries from
    # BEFORE this run, they are evicted first, and counting backwards from the
    # end would silently shift every point by however many those were. What
    # separates them reliably is size — every takeover in this run is the same
    # 24-token prompt, so its entry size is the most common value in the list,
    # and anything differing from it by more than 10 % is one of the measured
    # points, in order. A point small enough to land within that 10 % cannot be
    # told apart and is reported as unmatched rather than guessed at.
    points_in = [i for i in inserted if i["what"] == "point"]
    if result["flush"] and result["flush"]["complete"] and sizes:
        rounded = [round(s, 1) for s in sizes]
        tiny_mib = max(set(rounded), key=rounded.count)
        big = [s for s in sizes if abs(s - tiny_mib) > 0.10 * tiny_mib]
        result["flush"]["takeover_entry_mib"] = tiny_mib
        # Entries older than this run are evicted BEFORE it, so where there
        # are more large evictions than points, the run's own are the LAST
        # ones — never the first.
        if len(big) > len(points_in):
            print("  (%d large evictions belonged to entries older than this run)"
                  % (len(big) - len(points_in)))
            big = big[-len(points_in):]
        result["flush"]["matched"] = len(big) == len(points_in)
        print("  takeover entry: %.3f MiB (the repeated value in the eviction list)"
              % tiny_mib)
        if len(big) != len(points_in):
            print("  NOT MATCHED: %d eviction lines differ from it, %d points were "
                  "inserted — sizes not assigned" % (len(big), len(points_in)))
        else:
            print("\n  tokens        MiB      KiB/token")
            for ins, mib in zip(points_in, big):
                ins["mib"] = mib
                print("  %7d   %8.3f   %8.2f"
                      % (ins["tokens"], mib, mib * 1024.0 / ins["tokens"]))
        got = [(i["tokens"], i["mib"]) for i in points_in if "mib" in i]
        if len(got) >= 2:
            (t_lo, m_lo), (t_hi, m_hi) = got[0], got[-1]
            slope = (m_hi - m_lo) / (t_hi - t_lo)
            result["fit"] = {"from_tokens": t_lo, "to_tokens": t_hi,
                             "kib_per_token": round(slope * 1024.0, 2),
                             "fixed_mib": round(m_lo - slope * t_lo, 1),
                             "n_points": len(got)}
            print("\n  fit over %d points: %.2f KiB/token + %.1f MiB fixed"
                  % (len(got), slope * 1024.0, m_lo - slope * t_lo))
    else:
        print("\n  no sizes: the flush did not empty the cache within --flush-max")

    d = os.path.join(REPO, "bench", "reports",
                     time.strftime("%Y-%m-%d_%H%M_") + a.label)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("report: %s" % d)


if __name__ == "__main__":
    main()
