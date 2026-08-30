#!/usr/bin/env python3
"""HOIST_SYSTEM on and off, through cc-gateway, on the turn that hurts.

`hoist-cost.py` measures the two prompt SHAPES against llama-server directly:
hoisting buys nothing on the counter turn and costs everything on the turn a
new system block appears (203.2 s against 20.4 s). This runs the same three
turns through cc-gateway with the switch set each way, so what is under the
clock is the gateway's own correction, id and restore logic together.

    turn 1   cold — the baseline, not a result
    turn 2   the counter in the system block changes
    turn 3   a NEW system block appears, which is the case that hurt

The body is Claude-Code-shaped: a top-level `system`, a conversation, and a
system message INSIDE the conversation carrying a stable reminder plus a
counter — which is what the gateway's VOLATILE pattern is written for.

    python3 bench/suites/hoist-live.py

IT RESTARTS cc-gateway twice (a systemd drop-in, reverted at the end) and
takes the one slot for a few minutes. It does not touch llama-server.
"""
import argparse, json, subprocess, sys, time, urllib.request

GATEWAY = "http://127.0.0.1:8090/v1/messages"
UNIT = "cc-gateway.service"


def words(tag, n):
    return " ".join("%s%d" % (tag, i) for i in range(n))


def body(salt, base_words, conv_words, counter, blocks):
    msgs = [{"role": "user", "content": "USER " + words(salt + "C", conv_words)},
            {"role": "assistant", "content": "OK"}]
    for k in range(blocks):
        msgs.append({"role": "system",
                     "content": "REMINDER %d: %s\n<total_tokens>%d tokens left"
                                "</total_tokens>"
                                % (k, words("%sR%d" % (salt, k), 40), counter)})
    msgs.append({"role": "user", "content": "Weiter."})
    return {"model": "qwen38", "max_tokens": 8,
            "system": "SYSTEM " + words(salt + "B", base_words),
            "messages": msgs}


def ask(payload, label, timeout=1800):
    req = urllib.request.Request(
        GATEWAY, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    u = d.get("usage", {})
    took = time.time() - t0
    cached, computed = u.get("cache_read_input_tokens"), u.get("input_tokens")
    print("      %-22s %7.1f s   cached=%-7s computed=%-7s"
          % (label, took, cached, computed))
    return {"cached": cached or 0, "computed": computed or 0,
            "took_s": round(took, 1)}


def restart(value):
    if value is None:
        subprocess.run(["systemctl", "--user", "revert", UNIT], capture_output=True)
    else:
        subprocess.run(["systemctl", "--user", "edit", "--stdin", UNIT],
                       input="[Service]\nEnvironment=HOIST_SYSTEM=%s\n" % value,
                       text=True, check=True)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "restart", UNIT], check=True)
    for _ in range(60):
        if subprocess.run(["systemctl", "--user", "is-active", UNIT],
                          capture_output=True, text=True).stdout.strip() == "active":
            time.sleep(1.5)
            return
        time.sleep(0.5)
    raise SystemExit("cc-gateway did not come back")


def run(setting, salt, base_words, conv_words):
    restart(setting)
    print("    HOIST_SYSTEM=%s" % setting)
    out = [ask(body(salt, base_words, conv_words, 1000, 2), "turn 1 (cold)"),
           ask(body(salt, base_words, conv_words, 900, 2), "turn 2 counter--"),
           ask(body(salt, base_words, conv_words, 800, 3), "turn 3 NEW block")]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-words", type=int, default=200)
    ap.add_argument("--conv-words", type=int, default=800)
    ap.add_argument("--salt", default="L")
    a = ap.parse_args()

    print(__doc__.split("\n")[0])
    res = {}
    try:
        for setting in ("1", "0"):
            res[setting] = run(setting, a.salt + setting, a.base_words,
                               a.conv_words)
    finally:
        print("  putting cc-gateway back")
        restart(None)

    print("\n  %-16s %14s %14s %14s" % ("", "turn 1", "turn 2", "turn 3"))
    for setting, name in (("1", "hoist on "), ("0", "hoist off")):
        print("  %-16s %14s %14s %14s"
              % (name, *["%d cached" % r["cached"] for r in res[setting]]))
    print("  %-16s %14s %14s %14s"
          % ("", *["", "", ""]))
    for setting, name in (("1", "hoist on "), ("0", "hoist off")):
        print("  %-16s %14s %14s %14s"
              % (name, *["%.1f s" % r["took_s"] for r in res[setting]]))

    on3, off3 = res["1"][2], res["0"][2]
    print("\n  On the turn a NEW system block appears:")
    print("    hoist on   cached %d, computed %d, %.1f s"
          % (on3["cached"], on3["computed"], on3["took_s"]))
    print("    hoist off  cached %d, computed %d, %.1f s"
          % (off3["cached"], off3["computed"], off3["took_s"]))
    if off3["cached"] > on3["cached"]:
        print("    -> leaving them alone kept %d more tokens."
              % (off3["cached"] - on3["cached"]))
    elif on3["cached"] > off3["cached"]:
        print("    -> hoisting kept %d MORE. The prompt-level result does not "
              "survive the gateway; read turn 2 before concluding."
              % (on3["cached"] - off3["cached"]))
    else:
        print("    -> no difference. At these sizes the experiment cannot "
              "separate them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
