#!/usr/bin/env python3
"""save-policy-sim — replay real traffic through a save policy, before building it.

    python3 bench/suites/save-policy-sim.py --days 7
    python3 bench/suites/save-policy-sim.py --trace bench/reports/.../trace.tsv --sweep

Needs NO GPU, NO server and NO model. The input is what this stack has already
done: every START and DONE cc-gateway has logged, and every restart of
llama-server. The output is what a different rule WOULD have done with the
same traffic.

Why simulate rather than try it
-------------------------------
The rule about to be built has three numbers in it — how often a prefix must
be seen before it is worth a gigabyte, how long a gap counts as quiet, how
often a write may be deferred before it happens anyway. Every one of them is a
trade, and picking them by taste is how a stack ends up with a constant nobody
can defend six months later (`bench/sideserver.py --slots-timeout`, whose 420
was hard-wired and decided a measurement).

Seven days of real traffic answer all three in milliseconds, and the answer
comes with the workload it was derived from.

What is modelled, and what is not
---------------------------------
MODELLED:
  * every request, at its real time, for its real prefix
  * llama-server restarts — the RAM cache dies with them, which is the whole
    reason the disk store exists
  * saving occupies the ONE slot, so under today's rule a request arriving
    during a save is a collision, and under the new rule it waits
  * the two costs that collision has: the turn is re-prefilled (measured 0.7 s
    -> 13.6 s) and the file may hold the wrong state (measured twice)

NOT MODELLED:
  * llama.cpp's own RAM prompt cache (`-cram`) beyond "a prefix seen in this
    server's life is warm". It holds several prefixes, so the simulation
    UNDERSTATES how warm reality is — in the same direction for both rules.
  * how long a cold prefill takes for a given prefix. Durations come from the
    trace where the trace has them; nothing is invented.

So the numbers below are a comparison between rules on one workload, not a
prediction of wall-clock time. That is what they are for.
"""
import argparse, json, os, subprocess, sys, datetime, re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "setup", "claude"))
import savepolicy                                             # noqa: E402

# Measured on this machine, 28.08.2026, from the gateway's own SAVED lines:
# 0.5, 1.7, 1.9 and 4.3 s for saves that had the prefix in the slot already.
# The two long ones that day (101.9 s, 153.5 s) are NOT in this list on
# purpose — they are the collision, not the cost of writing, and the model
# adds that cost where it belongs.
SAVE_SECONDS = 2.0
# What a save has to redo when it lost the slot: put the prefix back by
# prefilling it. The two contended saves of 28.08. took 101.9 s and 153.5 s
# against a ~2 s write.
REPREFILL_SECONDS = 100.0
# One saved prefix, in bytes. The store on 28.08.: 0.69, 0.69, 1.13, 1.14 GB.
BYTES_PER_SAVE = 0.95e9


def from_journal(days):
    """Requests (with their END) and restarts, out of the two units that know.

    The END matters more than the start here: a save can only happen in the
    GAP after an answer, so a model built on start times alone can never see a
    gap at all — which is exactly the bug the first version of this file had.
    """
    def run(cmd):
        return subprocess.run(cmd, capture_output=True, text=True).stdout

    events = []
    gw = run(["journalctl", "--user", "-u", "cc-gateway",
              "--since", "-%dd" % days, "--no-pager", "-o", "short-iso"])
    # Both spellings: the gateway logged German field names until 25.08.
    start = re.compile(r"^(\S+) .*\] START\s+\S+\s+\S+\s+(?:who|wer)=(\S+)\s+"
                       r"(?:prefix|praefix)=(\S+)\s+(COLD|KALT|warm)")
    done = re.compile(r"^(\S+) .*\] DONE\s+\S+\s+\S+\s+(?:who|wer)=(\S+)\s+"
                      r"(?:prefix|praefix)=(\S+)\s+(?:took|dauerte)=([0-9.]+)s")
    pending = None
    for line in gw.splitlines():
        m = start.match(line)
        if m:
            t, who, pid, cold = m.groups()
            pending = (_iso(t), pid, who, cold in ("COLD", "KALT"))
            continue
        m = done.match(line)
        if m and pending:
            t, who, pid, took = m.groups()
            if pid == pending[1]:
                events.append((pending[0], "REQ", pid, who, pending[3],
                               _iso(t)))
            pending = None
    ll = run(["journalctl", "--user", "-u", "llama-user@qwen38",
              "--since", "-%dd" % days, "--no-pager", "-o", "short-iso"])
    for line in ll.splitlines():
        if "Started llama" in line:
            events.append((_iso(line.split()[0]), "RESTART", None, None,
                           False, _iso(line.split()[0])))
    events.sort(key=lambda e: e[0])
    return events


def _iso(s):
    return datetime.datetime.fromisoformat(s).timestamp()


def write_trace(events, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for e in events:
            t, kind, pid, who, cold = e[0], e[1], e[2], e[3], e[4]
            f.write("%.3f\t%s\t%s\t%s\t%d\t%.3f\n"
                    % (t, kind, pid or "-", who or "-", 1 if cold else 0, e[5]))
    return path


def read_trace(path):
    out = []
    for line in open(path):
        p = line.rstrip("\n").split("\t")
        if len(p) != 6:
            continue
        out.append((float(p[0]), p[1], None if p[2] == "-" else p[2],
                    None if p[3] == "-" else p[3], p[4] == "1", float(p[5])))
    return out


class Result:
    def __init__(self, name):
        self.name = name
        self.writes = 0
        self.collisions = 0          # a request met a save in the one slot
        self.evicted = 0             # ... and lost its prefix for it
        self.poisoned = 0            # ... or the file got the wrong state
        self.waited_s = 0.0          # what the new rule costs a turn
        self.max_wait_s = 0.0
        self.warm_from_disk = 0      # restarts survived
        self.cold_after_restart = 0  # restarts NOT survived
        self.forced = 0

    def row(self):
        return ("%-26s writes %3d (%4.1f GB)  collisions %3d (evicted %d, "
                "poisoned %d)  waited %5.1f s (max %4.1f)  after a restart: "
                "%3d warm / %3d cold"
                % (self.name, self.writes, self.writes * BYTES_PER_SAVE / 1e9,
                   self.collisions, self.evicted, self.poisoned,
                   self.waited_s, self.max_wait_s,
                   self.warm_from_disk, self.cold_after_restart))


def _timeline(events):
    """(requests, restarts) — requests as (start, end, prefix)."""
    reqs = [(e[0], e[5], e[2]) for e in events if e[1] == "REQ"]
    restarts = [e[0] for e in events if e[1] == "RESTART"]
    return reqs, restarts


def _serve(r, pid, t, on_disk, in_ram, restarts, last_restart_seen):
    """Book what this request cost, and remember the prefix is now in RAM."""
    fresh_restart = any(last_restart_seen[0] < rt <= t for rt in restarts)
    if fresh_restart:
        in_ram.clear()
        last_restart_seen[0] = max(rt for rt in restarts if rt <= t)
    if pid in in_ram:
        pass                                   # the slot or -cram still has it
    elif pid in on_disk:
        r.warm_from_disk += 1
        in_ram.add(pid)
    else:
        if last_restart_seen[0] > 0:
            r.cold_after_restart += 1
        in_ram.add(pid)


def simulate_today(events):
    """The rule as it stands: write on the FIRST cold sighting, in the
    background, the moment the answer is out — which is exactly when the next
    turn is most likely to arrive (median 1.0 s, measured)."""
    r = Result("today (first sighting)")
    reqs, restarts = _timeline(events)
    on_disk, in_ram, seen = set(), set(), set()
    last_restart = [0.0]
    save = None                                   # (pid, ends_at, disturbed)
    for t0, t1, pid in reqs:
        if save and save[1] <= t0:                # finished in the gap
            r.writes += 1
            if save[2]:
                r.poisoned += 1                   # it got the other prefix
            else:
                on_disk.add(save[0])
            save = None
        if save and save[1] > t0 and save[0] != pid:
            # THE COLLISION. The save holds the one slot; this request takes
            # it away. Both sides lose: the file gets the wrong state, and
            # what was in the slot has to be prefilled again.
            r.collisions += 1
            if pid in in_ram:
                r.evicted += 1
                in_ram.discard(pid)
            save = (save[0], t1 + REPREFILL_SECONDS, True)
        _serve(r, pid, t0, on_disk, in_ram, restarts, last_restart)
        first_sighting = pid not in seen
        seen.add(pid)
        if save is None and first_sighting and pid not in on_disk:
            save = (pid, t1 + SAVE_SECONDS, False)
    if save:
        r.writes += 1
        if save[2]:
            r.poisoned += 1
    return r


def simulate_policy(events, min_sightings, debounce_s, max_defers):
    """The proposed rule: write what has proven itself, in a gap, exclusively.

    The write happens BETWEEN two requests. If it runs into the next one, that
    request waits — it does not take the slot away, which is what makes the
    file trustworthy and costs the only latency this rule has.
    """
    p = savepolicy.Policy(min_sightings=min_sightings, debounce_s=debounce_s,
                          max_defers=max_defers)
    r = Result("min=%d debounce=%4gs defers=%d"
               % (min_sightings, debounce_s, max_defers))
    reqs, restarts = _timeline(events)
    on_disk, in_ram = set(), set()
    last_restart = [0.0]
    prev_end = None
    for t0, t1, pid in reqs:
        if prev_end is not None:
            gap = t0 - prev_end
            # SEVERAL fit in one gap. The first version wrote at most one per
            # quiet stretch, which quietly built a backlog and then blamed the
            # rule for the prefixes that never made it to disk — a modelling
            # artefact, and one that flattered the rule it was compared to.
            cursor = prev_end
            while True:
                due = p.due(cursor + max(gap, 0.0))
                if not due:
                    break
                first = due[0]
                forced = p.forced(first)
                if not (gap >= debounce_s or forced):
                    for i in due:
                        p.note_deferred(i)
                    break
                begin = max(cursor, prev_end + (0.0 if forced else debounce_s))
                finish = begin + SAVE_SECONDS
                if finish > t0:
                    wait = finish - t0
                    r.waited_s += wait
                    r.max_wait_s = max(r.max_wait_s, wait)
                r.writes += 1
                if forced:
                    r.forced += 1
                on_disk.add(first)
                p.note_saved(first)
                cursor = finish
                if finish >= t0:
                    break
        _serve(r, pid, t0, on_disk, in_ram, restarts, last_restart)
        p.saw(t0, pid)
        p.idle_since(t1)
        prev_end = t1
    # The tail: whatever is still owed gets written once the traffic stops.
    for i in p.due(prev_end + 1e6 if prev_end else 0):
        r.writes += 1
        on_disk.add(i)
    return r


_restart_times = None


def _after_restart(t, events):
    """Was this the first sighting of anything since the last restart? Used
    only to count what a restart cost, not to decide anything."""
    global _restart_times
    if _restart_times is None:
        _restart_times = [e[0] for e in events if e[1] == "RESTART"]
    return any(rt <= t for rt in _restart_times)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--trace", help="a trace written by an earlier run")
    ap.add_argument("--out", help="where to write the trace and the result")
    ap.add_argument("--who", default=None,
                    help="keep only this consumer's requests (substring). The "
                         "journal mixes real sessions with this repo's own "
                         "measurement runs, and the two have opposite shapes: "
                         "a suite fires in bursts with no gaps, a session "
                         "pauses. A policy read off the mixture would be tuned "
                         "for the benchmark.")
    ap.add_argument("--restart-every", type=int, default=1,
                    help="keep only every Nth restart. A SENSITIVITY knob, not "
                         "a fudge: this repo's own measurement runs stop and "
                         "start production constantly (86 restarts in 91 hours "
                         "on 28.08.), and a rule tuned on that would be tuned "
                         "for the benchmark rather than for the machine's "
                         "normal life. If the ranking survives thinning the "
                         "restarts, it does not rest on my own noise.")
    ap.add_argument("--sweep", action="store_true",
                    help="every combination, so the defaults are read off "
                         "rather than chosen")
    a = ap.parse_args()

    events = read_trace(a.trace) if a.trace else from_journal(a.days)
    if a.restart_every > 1:
        keep, n = [], 0
        for e in events:
            if e[1] == "RESTART":
                n += 1
                if n % a.restart_every:
                    continue
            keep.append(e)
        events = keep
    if a.who:
        events = [e for e in events
                  if e[1] != "REQ" or (e[3] and a.who in e[3])]
    reqs = [e for e in events if e[1] == "REQ"]
    restarts = [e for e in events if e[1] == "RESTART"]
    prefixes = {e[2] for e in reqs}
    if not reqs:
        raise SystemExit("no requests in the trace — is the journal there?")
    span = (reqs[-1][0] - reqs[0][0]) / 3600.0
    print("trace: %d requests, %d distinct prefixes, %d restarts, %.1f hours"
          % (len(reqs), len(prefixes), len(restarts), span))
    once = sum(1 for p in prefixes
               if sum(1 for e in reqs if e[2] == p) == 1)
    print("       %d of %d prefixes were seen EXACTLY ONCE\n" % (once, len(prefixes)))

    rows = [simulate_today(events)]
    if a.sweep:
        for mins in (1, 2, 3):
            for deb in (0.0, 5.0, 10.0, 30.0):
                for de in (1, 3, 5):
                    rows.append(simulate_policy(events, mins, deb, de))
    else:
        rows.append(simulate_policy(events, 2, 10.0, 3))
    for r in rows:
        print(" ", r.row())

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        write_trace(events, os.path.join(a.out, "trace.tsv"))
        with open(os.path.join(a.out, "sim.json"), "w") as f:
            json.dump({"requests": len(reqs), "prefixes": len(prefixes),
                       "restarts": len(restarts), "hours": round(span, 1),
                       "rows": [vars(r) for r in rows]}, f, indent=1)
        print("\nwritten: %s" % a.out)


if __name__ == "__main__":
    main()
