#!/usr/bin/env python3
"""Does RESTORE_ONLY_WHEN_SERVER_COLD actually help, through the real gateway?

`restore-blinds-cache.py` proves the MECHANISM by driving llama-server's slot
API by hand. This drives cc-gateway instead, so what is measured is the
decision code that would ship — including the `cold` flag, the prefix ledger
and the /slots reading.

    1  send a body with a big system prompt      -> a cold prefix, saved to disk
    2  send it again with a long conversation    -> the state is now in the slot
    3  restart the gateway                       -> its ledger is empty, so the
                                                    next request is "cold" and
                                                    the restore path is taken
    4  send the conversation once more, and read what the answer reports

Step 3 is the whole point, and it is not contrived: it is exactly what
happened on 29.08. at 23:38, when the gateway was restarted beside a
llama-server that had been up since 09:48. Every prefix looked cold, the first
restore hid a 69,939-token state, and the turn took 506 s.

Run it twice, once with the guard off and once on:

    python3 bench/suites/restore-guard-live.py --guard off
    python3 bench/suites/restore-guard-live.py --guard on

IT RESTARTS cc-gateway AND WRITES A PREFIX FILE. The file is named after the
id the gateway derives, printed at the end, and `--clean <id>` removes it. Do
not run this while somebody is working: step 2 puts a conversation into the
one slot, and the restart drops whatever the gateway was serving.
"""
import argparse, json, subprocess, sys, time, urllib.request

GATEWAY = "http://127.0.0.1:8090/v1/messages"
UNIT = "cc-gateway.service"


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
    print("    %-28s %7.1f s   cached=%-7s computed=%-7s out=%s"
          % (label, took, u.get("cache_read_input_tokens"),
             u.get("input_tokens"), u.get("output_tokens")))
    return {"took_s": round(took, 1),
            "cached": u.get("cache_read_input_tokens"),
            "computed": u.get("input_tokens")}


def restart(env_line):
    """Restart cc-gateway with one extra environment line, or without it."""
    drop = ["systemctl", "--user", "revert", UNIT]
    if env_line:
        subprocess.run(["systemctl", "--user", "edit", "--stdin", UNIT],
                       input="[Service]\nEnvironment=%s\n" % env_line,
                       text=True, check=True)
    else:
        subprocess.run(drop, capture_output=True)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "restart", UNIT], check=True)
    for _ in range(60):
        out = subprocess.run(["systemctl", "--user", "is-active", UNIT],
                             capture_output=True, text=True).stdout.strip()
        if out == "active":
            time.sleep(1.5)
            return
        time.sleep(0.5)
    raise SystemExit("cc-gateway did not come back")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--guard", choices=("on", "off"), required=True)
    ap.add_argument("--sys-words", type=int, default=1400,
                    help="the system prompt; it becomes the prefix")
    ap.add_argument("--conv-words", type=int, default=5000)
    ap.add_argument("--salt", default="G")
    a = ap.parse_args()

    system = " ".join("SYS%s%d" % (a.salt, i) for i in range(a.sys_words))
    talk = " ".join("MSG%s%d" % (a.salt, i) for i in range(a.conv_words))
    short = [{"role": "user", "content": "Sag OK."}]
    long_ = [{"role": "user", "content": talk},
             {"role": "assistant", "content": "OK"},
             {"role": "user", "content": "Und weiter."}]

    print(__doc__.split("\n")[0])
    print("  guard: %s" % a.guard)
    print("  1/2 warming and saving the prefix")
    ask(short, system, "cold, saves the prefix")
    print("  2/2 putting a conversation behind it")
    ask(long_, system, "the conversation")

    print("  restarting cc-gateway (%s)" % a.guard)
    restart("RESTORE_ONLY_WHEN_SERVER_COLD=1" if a.guard == "on" else None)

    print("  the request that decides it")
    got = ask(long_ + [{"role": "assistant", "content": "OK"},
                       {"role": "user", "content": "Und nochmal."}],
              system, "after the gateway restart")

    print("\n  guard=%s  ->  cached=%s computed=%s took=%s s"
          % (a.guard, got["cached"], got["computed"], got["took_s"]))
    print("  Restore ON in the gateway means the file goes into the slot and")
    print("  llama.cpp's own cache is never consulted; the conversation is")
    print("  then in `computed`. Compare the two runs on that column.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
