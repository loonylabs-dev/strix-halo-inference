#!/usr/bin/env python3
"""The other half of the guard: does it still restore when the server IS cold?

`restore-guard-live.py` measures that RESTORE_ONLY_WHEN_SERVER_COLD stops the
restore while llama-server is warm — the case that was costing 187 s. This
measures the branch that protects against the opposite mistake: after a real
llama-server restart the RAM cache is empty, the file on disk is the only thing
that can help, and the guard has to let it through.

If that branch were broken, every first request after a restart would prefill
the whole prefix instead of loading it — 91.7 s for the production prefix. That
regression would be invisible in a warm-server test, and it is now running on
the operator's machine, which is why this exists rather than trusting the unit
tests for it.

    1  send a body with a big system prompt   -> a cold prefix, saved to disk
    2  RESTART llama-server                   -> its RAM cache is gone; the
                                                 task counter falls to ~0
    3  send the same body again               -> the guard must let the restore
                                                 through, and the answer must
                                                 report the prefix as reused

    python3 bench/suites/restore-guard-cold.py

IT RESTARTS llama-server. That is roughly 90 seconds of downtime and it empties
the prompt cache for everybody. Do not run it while somebody is working. The
prefix file it creates is printed at the end so it can be removed.
"""
import argparse, json, subprocess, sys, time, urllib.request

GATEWAY = "http://127.0.0.1:8090/v1/messages"
LLAMA = "http://127.0.0.1:8080"
SERVER_UNIT = "llama-user@qwen38"
GATEWAY_UNIT = "llm-gateway.service"


def ask(messages, system, label, timeout=1800):
    body = {"model": "qwen38", "max_tokens": 8, "system": system,
            "messages": messages}
    req = urllib.request.Request(
        GATEWAY, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    u = d.get("usage", {})
    took = time.time() - t0
    print("    %-30s %7.1f s   cached=%-7s computed=%-7s"
          % (label, took, u.get("cache_read_input_tokens"), u.get("input_tokens")))
    return {"took_s": round(took, 1), "cached": u.get("cache_read_input_tokens"),
            "computed": u.get("input_tokens")}


def task_counter():
    try:
        with urllib.request.urlopen(LLAMA + "/slots", timeout=20) as r:
            slots = json.loads(r.read().decode())
        return max(x.get("id_task", -1) for x in slots)
    except Exception as e:
        return "unreadable (%r)" % (e,)


def wait_for_server(limit=600):
    """/slots, never /health — the same rule bench/sideserver.py follows."""
    t0 = time.time()
    while time.time() - t0 < limit:
        try:
            with urllib.request.urlopen(LLAMA + "/slots", timeout=10) as r:
                json.loads(r.read().decode())
            return time.time() - t0
        except Exception:
            time.sleep(2)
    raise SystemExit("llama-server did not come back within %d s" % limit)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sys-words", type=int, default=1400)
    ap.add_argument("--salt", default="C")
    a = ap.parse_args()

    system = " ".join("SYS%s%d" % (a.salt, i) for i in range(a.sys_words))
    short = [{"role": "user", "content": "Sag OK."}]

    print(__doc__.split("\n")[0])
    print("  task counter before: %s" % task_counter())
    print("  1/3 warming and saving the prefix")
    first = ask(short, system, "cold, saves the prefix")

    print("  2/3 restarting llama-server — the cache goes with it")
    t0 = time.time()
    subprocess.run(["systemctl", "--user", "restart", SERVER_UNIT], check=True)
    up = wait_for_server()
    print("      back after %.0f s, task counter now: %s"
          % (time.time() - t0, task_counter()))
    # The gateway keeps its own high-water mark in memory; it does not need a
    # restart, and NOT restarting it is the point — this is the case where the
    # counter FELL under a running gateway.
    print("  3/3 the request that decides it")
    got = ask(short, system, "after the llama-server restart")

    print("\n  cached=%s computed=%s took=%s s"
          % (got["cached"], got["computed"], got["took_s"]))
    if (got["cached"] or 0) >= (first["cached"] or 0) * 0.9 and (first["cached"] or 0):
        print("  PASS — the guard let the restore through on a cold server.")
        rc = 0
    else:
        print("  FAIL — the prefix was NOT reused. On a cold server the guard "
              "must not block the restore; this is the regression the warm "
              "test cannot see.")
        rc = 1
    print("  Remove the prefix file this created once you have read the "
          "result: it is named after the id in the trace's `save` record.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
