#!/usr/bin/env python3
"""worklib — the shared core of the modality benches. One copy, not four.

image-, audio- and videobench grew as siblings sharing ~70 % of themselves:
the fence refusal, profile reading, the rep loop with timing and
incremental result.json, hashing, report metadata, the median summary.
The rule-of-three debt was acknowledged in the 01.09.2026 commits and this
is its payment — with the cut the architecture review prescribed, and the
two WRONG cuts it named kept out on purpose:

  * no `mediabench --kind` monolith: the per-kind report fields (realtime
    factor, sequence hashes, frame handling) are the scientific payload
    and stay in the thin per-kind scripts;
  * the CHECKERS are not unified: they judge different physics, their
    heuristic thresholds must stay individually negotiable. videocheck
    composing imagecheck per frame is composition, not unification.

What lives here instead: everything whose second copy would drift — and
build_stamp_of() already HAD diverged at birth (two copies, two search
depths).
"""
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "setup", "lib"))
import systemdfile                                            # noqa: E402
import budget                                                 # noqa: E402


def fence_refusal(workload_arg):
    """True (and says why) when a llama-server is serving — the walk-around
    path beside production says no by itself. Inside the sideserver fence
    production is stopped, so this passes there by construction."""
    if budget.server_pid() is None:
        return False
    print("REFUSING: a llama-server is serving. Run this through\n"
          "  python3 bench/sideserver.py --workload %s --stop "
          "llama-user@<model> -- <this bench>\n"
          "Direct starts beside production are how this machine froze "
          "three times on 26.08.2026." % workload_arg, file=sys.stderr)
    return True


def load_profile(workload_arg):
    """(path, name, argv, prompt) of a workload profile, or SystemExit —
    the same reading every bench and the fence itself use."""
    path = budget._workload_path(workload_arg)
    name = os.path.basename(path).rsplit(".env", 1)[0]
    argv = systemdfile.args_of(path, "WORKLOAD_CMD")
    prompt = systemdfile.variable(path, "WORKLOAD_PROMPT")
    if not argv or not prompt:
        raise SystemExit("the profile lacks WORKLOAD_CMD or WORKLOAD_PROMPT")
    return path, name, argv, prompt


def build_stamp_of(binary):
    """The .build-stamp beside or above the binary, as a dict, or {}.

    THE one copy. imagebench searched two levels, audio-/videobench three
    (qwen-tts-p sits without a bin/), and the two had already disagreed the
    day they were born — the three-parsers disease in miniature, in the
    same branch that avoids it elsewhere (review, 01.09.2026).
    """
    seen = set()
    d = os.path.dirname(os.path.abspath(binary))
    for _ in range(3):
        if d in seen:
            break
        seen.add(d)
        stamp = os.path.join(d, ".build-stamp")
        if os.path.exists(stamp):
            fields = {}
            with open(stamp, encoding="utf-8") as fh:
                for line in fh:
                    k, sep, v = line.partition("=")
                    if sep:
                        fields[k.strip()] = v.strip()
            return fields
        d = os.path.dirname(d)
    return {}


def sha256_of(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def timed_run(job, log_path):
    """(exit code, wall seconds) of one job — the clock around the
    SUBPROCESS alone.

    This is the measurement boundary, and it is a function so it cannot
    drift again: the first worklib cut let run_reps() time the whole
    do_rep, which silently pulled hashing and the checker verdict into
    `seconds` — ~0.3-1 % one-sided inflation on video clips, found
    independently by BOTH re-reviews (01.09.2026) and invisible to the
    hash verification by construction. Every committed report was measured
    with the clock here; benches stamp their own seconds through this, and
    run_reps only backfills a stub that stamped none.
    """
    t0 = time.time()
    with open(log_path, "w") as fh:
        rc = subprocess.run(job, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    return rc, round(time.time() - t0, 2)


class BenchReport:
    """The rep runner: timing, incremental result.json, metadata, summary.

    Owns what bench/run.py's finally-block owns for the cache suites —
    half a measurement run is worth more than a lost one — so result.json
    is rewritten after EVERY rep with partial=True, and finalize() replaces
    it with the full record (no partial key).
    """

    def __init__(self, kind, workload_arg, reps, note, hash_key="sha256",
                 dest=None):
        self.path, self.name, self.argv, self.prompt = \
            load_profile(workload_arg)
        self.kind = kind
        self.note = note
        self.n = reps
        self.hash_key = hash_key
        self.stamp = time.strftime("%Y-%m-%d_%H%M")
        # `dest` overrides the report home for runs that are VERIFICATION,
        # not evidence: tests/live_media.sh re-derives committed hashes and
        # must not spray one report directory per nightly run into the repo.
        self.dest = dest or os.path.join(
            HERE, "reports", "%s_%s_%s" % (self.stamp, kind, self.name))
        os.makedirs(self.dest, exist_ok=True)
        self.reps = []

    def _write(self, payload):
        with open(os.path.join(self.dest, "result.json"), "w") as fh:
            json.dump(payload, fh, indent=2)

    def run_reps(self, do_rep, describe, post=None):
        """for r in 1..n: time do_rep(r), merge, post-process, print, and
        write the partial record. do_rep returns the rep dict WITHOUT
        seconds; `post(rep)` derives fields that need the wall time (the
        realtime factor); `describe(rep)` renders the one printed line."""
        for r in range(1, self.n + 1):
            t0 = time.time()
            rep = do_rep(r)
            rep.setdefault("rep", r)
            # setdefault, NEVER overwrite: the bench stamps its own seconds
            # through timed_run() (clock around the subprocess alone); this
            # fallback exists for stubs and would otherwise re-move the
            # measurement boundary both re-reviews flagged.
            rep.setdefault("seconds", round(time.time() - t0, 2))
            if post:
                post(rep)
            print("  rep %d: %s" % (r, describe(rep)))
            self.reps.append(rep)
            self._write({"workload": self.name, "timestamp": self.stamp,
                         "partial": True, "reps": self.reps})
        return self.reps

    def finalize(self, extra=None, summary_hook=None):
        """The full record: metadata, per-rep data, medians, distinct
        outputs. Returns the exit code (0 only when every rep was sound)."""
        hashes = {p.get(self.hash_key) for p in self.reps
                  if p.get(self.hash_key)}
        result = {
            "workload": self.name,
            "timestamp": self.stamp,
            "machine": budget.machine_identity(),
            "binary": systemdfile.unexpand(self.argv[0]),
            "build_stamp": build_stamp_of(self.argv[0]),
            "argv": [systemdfile.unexpand(t) for t in self.argv],
            "prompt": self.prompt,
            "note": self.note,
            "reps": self.reps,
            # An observation, not a property: 1 means THESE reps were
            # byte-identical, and says nothing beyond them.
            "distinct_outputs": len(hashes),
        }
        if extra:
            result.update(extra)
        ok_reps = [r for r in self.reps if r.get("ok")]
        if ok_reps:
            times = sorted(r["seconds"] for r in ok_reps)
            result["median_seconds"] = times[len(times) // 2]
            result["min_seconds"], result["max_seconds"] = times[0], times[-1]
            if summary_hook:
                summary_hook(result, ok_reps)
            print("  median %.1f s · %d/%d reps sound · %d distinct "
                  "output(s)" % (result["median_seconds"], len(ok_reps),
                                 len(self.reps), len(hashes)))
        else:
            print("  NO sound rep — the numbers above are not a benchmark")
        self._write(result)
        print("  report: %s" % os.path.join(self.dest, "result.json"))
        return 0 if self.reps and len(ok_reps) == len(self.reps) else 1
