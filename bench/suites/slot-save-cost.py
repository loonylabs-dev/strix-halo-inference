#!/usr/bin/env python3
"""slot-save-cost — what writing a session state costs, over the depth.

    D=~/.cache/slot-save-cost && mkdir -p $D
    python3 bench/sideserver.py --env setup/env/flashnext.env --port 8081 \\
        --stop "llama-user@$(bash setup/lib/models.sh serving)" --deadline 90 \\
        --extra "--slot-save-path $D/ -cram 0" -- \\
        python3 bench/suites/slot-save-cost.py --url http://127.0.0.1:8081 \\
            --dir $D --out $D/rows.json

NOT /tmp, AND THIS IS THE WHOLE POINT OF THE SUITE. `/tmp` is tmpfs on this
machine (62.5 GiB, `findmnt -no FSTYPE /tmp` says tmpfs) — a state written
there never reaches a disk at all. It would be measured at RAM speed, the
kernel's sector counter would not move, and the answer would be a confident
number about a write that did not happen. The 02.09. restore-continuation run
wrote its 627 MB into /tmp for exactly this reason; its "4.9 GB/s" is RAM.
Worse than wrong figures: several GiB of state in tmpfs sits in the same RAM
an 88 GiB model is pinned in, on a machine that has frozen three times.
The run refuses a --dir whose filesystem has no block device.

THE TWO NUMBERS THIS EXISTS FOR, and they are not the same number.

`bench/reports/2026-09-02_1815_restore-continuation/` established that a
restored state serves the continuation of its own conversation — so persisting
a live session is mechanically possible. What it could not answer is whether
it is worth doing, and that turns on two quantities its single 15,024-token
point cannot separate:

  HOW LONG THE SLOT IS BLOCKED   llama.cpp runs SERVER_TASK_TYPE_SLOT_SAVE in
      the main task loop (server-context.cpp:2517) and defers it while the
      slot is processing. Nothing generates or prefills during the write, for
      any slot. That is the latency a cooldown has to hide, and the server
      reports it itself as `timings.save_ms`.

  HOW MANY BYTES REACH THE DISK  627 MB in 129 ms is 4.9 GB/s, which this
      drive does not do — that write went into the page cache. The SSD wear a
      cooldown interval costs is a different measurement, taken from the
      kernel rather than from the server, and this suite takes both.

WHY THE INTERMEDIATE POINTS ARE FREE. Every point is a PREFIX of the deepest
one: the run tokenises once, then prefills toks[:2000], toks[:20000], … in
ascending order. Each step reuses everything the step before it computed, so
the whole sweep costs ONE prefill of the deepest point, not the sum. Five
points and two cost the same wall clock; the price is set by `--points`' last
value alone.

WHAT WOULD MAKE THIS LIE

  the kernel counter is noise    A `sync` flushes whatever else the machine is
      doing, and at 600 MB the difference matters. Cell `baseline` measures
      the same window with NO save in it. If the idle drift is within an order
      of magnitude of the smallest point, the disk column is reported as
      unusable rather than quoted.

  the depth is not the depth     `n_saved` from the server is the truth, not
      the requested point. A truncated prompt (`-c` too small for the deepest
      point) silently measures something shorter; the run refuses up front
      against the server's own `n_ctx`.

  the disk fills mid-run         A 180k state is several GiB and there are
      several points. Each file is deleted once measured, except the deepest,
      and the run refuses to start without room for the largest one.

Written 02.09.2026 after the restore-continuation result. Production serves
WITHOUT `--slot-save-path` (flashnext.env, 02.09.), so this only ever runs on
a side server — and the unit to stop is derived from `models.sh serving`,
never written down.
"""
import argparse, json, os, subprocess, sys, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))

SRV = None                                     # set from --url in main()
STATE = "slot-save-cost-%d.bin"


def req(path, payload=None, method=None, t=7200):
    d = json.dumps(payload).encode("utf-8") if payload is not None else None
    r = urllib.request.Request(SRV + path, data=d, method=method,
                               headers={"content-type": "application/json"})
    with urllib.request.urlopen(r, timeout=t) as x:
        return json.loads(x.read().decode("utf-8"))


# ------------------------------------------------------------ the prompt ---
def tokens_for(n, seed):
    """A prompt of EXACTLY n tokens.

    Lifted from bench/suites/cram-state-size.py, including the reason its
    entropy sits in the WORDS and not in a seed printed at the front: seeds
    that differ by a little share leading decimal digits, and llama.cpp
    measures similarity as common prefix over the new prompt's length. Here
    that would matter for the displacement in the restore check, which has to
    land on the LRU path.
    """
    import random
    rnd = random.Random(seed)
    words = ["%s%d" % (rnd.choice("qxzjvkwfy"), rnd.randrange(10 ** 6))
             for _ in range(n * 2 + 40)]
    toks = req("/tokenize", {"content": " ".join(words)})["tokens"]
    if len(toks) < n:
        raise SystemExit("tokenizer returned %d < %d tokens" % (len(toks), n))
    return toks[:n]


def prefill(toks, npredict=1):
    """Prefill `toks` and return what the server did, INCLUDING what it wrote.

    `return_tokens` is not a nicety. After a prefill the slot holds the prompt
    PLUS the generated tokens, and a continuation built from the prompt alone
    is therefore not a superset of the saved state — it diverges at the first
    generated token, and llama.cpp discards such a state whole rather than
    trimming it (bench/reports/2026-08-29_restore-semantics/). The restore
    check would then report NO for a reason that has nothing to do with depth.
    """
    t0 = time.time()
    r = req("/completion", {"prompt": toks, "n_predict": npredict,
                            "cache_prompt": True, "stream": False,
                            "return_tokens": True})
    tm = r.get("timings") or {}
    return {"wall_s": round(time.time() - t0, 2),
            "prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"),
            "truncated": r.get("truncated"),
            "gen": [t for t in (r.get("tokens") or []) if isinstance(t, int)]}


# -------------------------------------------------------- the disk meter ---
def _diskstats(major, minor):
    try:
        for line in open("/proc/diskstats", encoding="utf-8"):
            f = line.split()
            if len(f) > 9 and int(f[0]) == major and int(f[1]) == minor:
                return f[2], int(f[9])
    except OSError:
        pass
    return None, None


def _backing_device(path):
    """The block device behind `path`, from /proc/self/mountinfo.

    `os.stat(path).st_dev` is NOT enough, and the failure is silent. btrfs —
    the filesystem $HOME is on here — reports an ANONYMOUS device (0:34,
    measured 02.09.2026), which appears in no diskstats line. The meter then
    reads None forever and the disk column looks like "nothing was written"
    rather than "this meter is pointed at nothing". Same for any stacked or
    virtual filesystem.

    mountinfo carries the real source after its `-` separator, so the device
    is RESOLVED rather than guessed from a name: the source path is stat'ed
    and its st_rdev gives the numbers diskstats is keyed on. A name taken
    from a mount point would be wrong on LVM and on partitions.
    """
    path = os.path.realpath(path)
    best = None
    try:
        for line in open("/proc/self/mountinfo", encoding="utf-8"):
            parts = line.split(" - ")
            if len(parts) < 2:
                continue
            left, right = parts[0].split(), parts[1].split()
            if len(left) < 5 or len(right) < 2:
                continue
            mountpoint, source = left[4], right[1]
            if path == mountpoint or path.startswith(mountpoint.rstrip("/") + "/"):
                if best is None or len(mountpoint) > len(best[0]):
                    best = (mountpoint, source)
    except OSError:
        return None
    if not best or not best[1].startswith("/dev/"):
        return None
    return best[1]


def disk_row(path):
    """(name, sectors_written) for the block device `path` lives on."""
    src = _backing_device(path)
    if src:
        try:
            rdev = os.stat(src).st_rdev
            name, sect = _diskstats(os.major(rdev), os.minor(rdev))
            if name is not None:
                return name, sect
        except OSError:
            pass
    st = os.stat(path)                       # ext4/xfs answer here directly
    return _diskstats(os.major(st.st_dev), os.minor(st.st_dev))


def sectors_written(path):
    _, s = disk_row(path)
    return s


def flush_and_measure(path, fn):
    """Run fn(), then force the flush, and return (result, bytes to disk).

    The `sync` is what makes the number mean anything: without it the write is
    still in the page cache and the counter has not moved yet. It also flushes
    everything else, which is why `baseline` exists to say how much that is.
    """
    before = sectors_written(path)
    out = fn()
    subprocess.run(["sync"], check=False)
    after = sectors_written(path)
    if before is None or after is None:
        return out, None
    return out, (after - before) * 512


# ---------------------------------------------------------------- cells ----
def save(idx, path):
    name = STATE % idx
    t0 = time.time()
    d = req("/slots/0?action=save", {"filename": name})
    return {"file": os.path.join(path, name),
            "n_saved": d.get("n_saved"), "n_written": d.get("n_written"),
            "save_ms": (d.get("timings") or {}).get("save_ms"),
            "wall_ms": round((time.time() - t0) * 1000)}


def restore(idx):
    t0 = time.time()
    d = req("/slots/0?action=restore", {"filename": STATE % idx})
    return {"n_restored": d.get("n_restored"), "n_read": d.get("n_read"),
            "restore_ms": (d.get("timings") or {}).get("restore_ms"),
            "wall_ms": round((time.time() - t0) * 1000)}


def free_bytes(path):
    s = os.statvfs(path)
    return s.f_bavail * s.f_frsize


def main():
    global SRV
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081",
                    help="the SIDE server, not production")
    ap.add_argument("--dir", default=os.path.expanduser("~/.cache/slot-save-cost"),
                    help="the server's --slot-save-path, so files can be "
                         "measured and deleted here. Must be on a real block "
                         "device — see the module docstring on /tmp")
    ap.add_argument("--allow-no-disk", action="store_true",
                    help="run even though --dir has no block device. Only "
                         "for the save_ms column; the byte columns are then "
                         "not about a disk and the report must say so")
    ap.add_argument("--points", default="2000,20000,80000,140000,180000",
                    help="ascending; the LAST one sets the wall clock")
    ap.add_argument("--seed", type=int, default=0, help="0 = from the clock")
    ap.add_argument("--no-restore-check", action="store_true",
                    help="skip the reuse check at the deepest point")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    SRV = a.url

    if "8080" in a.url:
        raise SystemExit("refusing: 8080 is production. Start a side server "
                         "with bench/sideserver.py and pass its port.")
    points = sorted(int(x) for x in a.points.split(",") if x.strip())
    if not points:
        raise SystemExit("no points")
    if not os.path.isdir(a.dir):
        raise SystemExit("--dir %s does not exist — it must be the same "
                         "directory the server got as --slot-save-path" % a.dir)
    dev, _ = disk_row(a.dir)
    if dev is None and not a.allow_no_disk:
        raise SystemExit(
            "refusing: %s has no block device in /proc/diskstats — on this "
            "machine that means tmpfs, i.e. RAM.\n"
            "  Nothing would reach a disk, the byte columns would measure "
            "memory, and several GiB of state would sit in the RAM the model "
            "is pinned in.\n"
            "  Use a directory on a real filesystem (~/.cache/... is btrfs "
            "here), or pass --allow-no-disk if only save_ms is wanted."
            % a.dir)

    # The context has to hold the deepest point, or the prompt is silently
    # truncated and every number after it describes a shorter state.
    def get(path):
        with urllib.request.urlopen(SRV + path, timeout=60) as x:
            return json.loads(x.read().decode("utf-8"))
    try:
        props = get("/props")
    except Exception as e:
        raise SystemExit("cannot read /props — is the side server up? %s" % e)
    try:
        n_ctx = (get("/slots") or [{}])[0].get("n_ctx")
    except Exception:
        n_ctx = None
    if n_ctx and points[-1] > n_ctx * 0.95:
        raise SystemExit(
            "refusing: deepest point %d against n_ctx %d — the prompt would "
            "be truncated and every figure would describe a shorter state."
            % (points[-1], n_ctx))

    # 41,700 bytes/token is the ONE point measured 02.09. (restore-continuation,
    # 15,024 tokens). It is used here only to refuse a run that cannot fit;
    # measuring that constant properly is what this suite is for.
    want = int(points[-1] * 41700 * 1.3)
    have = free_bytes(a.dir)
    if have < want:
        raise SystemExit("refusing: %.1f GB free in %s, the deepest state is "
                         "expected to need about %.1f GB (extrapolated from a "
                         "single point, so the margin is 30 %%)."
                         % (have / 1e9, a.dir, want / 1e9))

    seed = a.seed or int(time.time())
    print("points: %s   seed: %d   dir: %s" % (points, seed, a.dir))
    print("device: %s   n_ctx: %s\n" % (disk_row(a.dir)[0], n_ctx))

    result = {"url": a.url, "points": points, "seed": seed,
              "build": (props or {}).get("build_info"),
              "n_ctx": n_ctx, "rows": [], "baseline": None,
              "restore_check": None}

    print("=== baseline: the same sync window with no save in it")
    _, idle = flush_and_measure(a.dir, lambda: time.sleep(2.0))
    result["baseline"] = idle
    print("  idle drift over one sync: %s bytes\n"
          % ("unreadable" if idle is None else "%d" % idle))

    print("=== tokenise once; every point is a prefix of the next")
    t0 = time.time()
    toks = tokens_for(points[-1], seed)
    print("  %d tokens in %.1f s\n" % (len(toks), time.time() - t0))

    kept = []
    for i, k in enumerate(points):
        print("=== point %d" % k)
        pf = prefill(toks[:k])
        print("  prefill                       %8.1f s   prompt_n=%-7s cache_n=%s"
              % (pf["wall_s"], pf["prompt_n"], pf["cache_n"]))
        if pf.get("truncated"):
            print("  TRUNCATED — this point and everything after it is void")
            result["rows"].append({"point": k, "truncated": True})
            continue
        try:
            s, disk = flush_and_measure(a.dir, lambda: save(i, a.dir))
        except Exception as e:
            detail = e.read().decode("utf-8")[:300] if hasattr(e, "read") else str(e)
            print("  save FAILED: %s" % detail)
            result["rows"].append({"point": k, "failed": detail})
            continue
        row = dict(s, point=k, disk_bytes=disk,
                   prefill_s=pf["wall_s"], cache_n=pf["cache_n"])
        n = s["n_saved"] or 0
        print("  save                          %8.0f ms  n_saved=%-7s "
              "%.0f MB reported" % (s["save_ms"] or 0, s["n_saved"],
                                    (s["n_written"] or 0) / 1e6))
        print("  on disk                       %8s     %s"
              % ("%.0f MB" % (disk / 1e6) if disk is not None else "n/a",
                 "%.0f bytes/token reported" % ((s["n_written"] or 0) / n)
                 if n else ""))
        result["rows"].append(row)
        # Every file but the deepest goes at once: five states at this size
        # fill a disk, and only the last one is needed for the restore check.
        if k != points[-1]:
            try:
                os.remove(s["file"])
            except OSError as e:
                print("  could not delete %s: %s" % (s["file"], e))
        else:
            kept.append((i, s, pf["gen"]))
        print()

    if not a.no_restore_check and kept:
        i, s, gen = kept[-1]
        print("=== restore check at the deepest point")
        # A short unrelated prompt takes the slot down the LRU path; with
        # -cram 0 there is no RAM cache to hand the state back, so what the
        # continuation reuses can only have come from the file.
        prefill(tokens_for(24, seed + 1))
        print("  displaced")
        r = restore(i)
        print("  restore                       %8.0f ms  n_restored=%-7s "
              "%.0f MB" % (r["restore_ms"] or 0, r["n_restored"],
                           (r["n_read"] or 0) / 1e6))
        # prompt + what the model wrote + something new: a true superset of
        # the saved state, which is the only shape a restore is reused for.
        cont = prefill(toks[:points[-1]] + gen + toks[:8], npredict=1)
        print("  continuation                  %8.1f s   cache_n=%-7s prompt_n=%s"
              % (cont["wall_s"], cont["cache_n"], cont["prompt_n"]))
        carried = (cont["cache_n"] or 0) > (r["n_restored"] or 1) * 0.5
        print("  the file carried the depth  : %s%s"
              % ("YES" if carried else "NO",
                 "" if gen else "   (no generated tokens returned — the "
                                "continuation may not be a superset)"))
        result["restore_check"] = dict(r, cache_n=cont["cache_n"],
                                       carried=carried, n_gen=len(gen))

    print("\n=== the line")
    rows = [r for r in result["rows"] if r.get("n_saved")]
    print("  %9s %9s %12s %12s %10s %10s"
          % ("point", "n_saved", "reported MB", "on disk MB", "B/token", "save ms"))
    for r in rows:
        n = r["n_saved"]
        print("  %9d %9d %12.0f %12s %10.0f %10.0f"
              % (r["point"], n, (r["n_written"] or 0) / 1e6,
                 "%.0f" % (r["disk_bytes"] / 1e6) if r.get("disk_bytes") is not None
                 else "n/a", (r["n_written"] or 0) / n, r["save_ms"] or 0))
    if len(rows) >= 2:
        lo, hi = rows[0], rows[-1]
        dn = hi["n_saved"] - lo["n_saved"]
        if dn:
            slope = ((hi["n_written"] or 0) - (lo["n_written"] or 0)) / dn
            base = (lo["n_written"] or 0) - slope * lo["n_saved"]
            print("\n  two-point fit over the reported bytes: "
                  "%.0f B + %.0f B/token" % (base, slope))
            print("  (two points, not a fit over all of them — read the "
                  "B/token column for whether the line is straight)")
        ms_lo, ms_hi = lo.get("save_ms") or 0, hi.get("save_ms") or 0
        print("  slot blocked: %.0f ms at %d tokens, %.0f ms at %d"
              % (ms_lo, lo["n_saved"], ms_hi, hi["n_saved"]))
    measured = [r["disk_bytes"] for r in rows if r.get("disk_bytes")]
    if result["baseline"] is not None and measured:
        smallest = min(measured)
        if result["baseline"] > smallest * 0.1:
            print("\n  DISK COLUMN UNUSABLE: idle drift %d bytes against a "
                  "smallest measured %d — the sync is carrying somebody "
                  "else's writes, so these bytes are not this suite's."
                  % (result["baseline"], smallest))

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
        print("\n  rows -> %s" % a.out)
    if kept:
        print("  kept for inspection: %s" % kept[-1][1]["file"])


if __name__ == "__main__":
    main()
