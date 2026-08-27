#!/usr/bin/env bash
# slot-scaling — does decode keep scaling beyond two sessions?
#
# Two sessions gain 47 % in aggregate (bench/suites/prefill-decode.py). The open
# question is whether four gain more, because that is what -np 4 would buy —
# at the price of half the context per project (32k instead of 65k).
#
# This stops the model service, starts llama-server by hand with -np 4,
# measures 1, 2 and 4 concurrent decodes, and puts the service back. The trap
# restores the service even if the script dies.
#
#   bash bench/suites/slot-scaling.sh
#
# Costs one model load (~20 s) twice, plus about three minutes of measuring.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$HOME/llama.cpp/build-vulkan/bin/llama-server"
LOG="$(mktemp /tmp/slot-scaling-XXXX.log)"
PID=""

restore() {
  [ -n "$PID" ] && kill "$PID" 2>/dev/null
  sleep 3
  echo "putting llama-user@laguna back …"
  systemctl --user start llama-user@laguna
  echo "log of the hand-started server: $LOG"
}
trap restore EXIT

echo "stopping the service"
systemctl --user stop llama-user@laguna
sleep 3

# Same arguments as the unit, but -np 4. -c stays, so each slot gets a quarter.
# The value spans several lines with backslash continuations — sed alone would
# only take the first. Join them properly.
# The value spans several lines with backslash continuations, and the profile
# carries comment lines AFTER it. The regex that used to stand here had a
# lookahead of (?=^[A-Za-z_]+=|\Z) and ran straight through those comments,
# appending "# Backend for this profile. Measured 24.08. ..." to the server's
# command line; shlex.split then also ate the quotes out of any JSON value.
# setup/lib/systemdfile.py is the one reader that agrees with systemd.
ARGS=$(python3 "$REPO/setup/lib/systemdfile.py" args "$REPO/setup/env/laguna.env")
ARGS=${ARGS/-np 2/-np 4}
echo "starting by hand with -np 4"
# shellcheck disable=SC2086
$BIN $ARGS > "$LOG" 2>&1 &
PID=$!

for _ in $(seq 1 120); do
  curl -sf -m2 http://127.0.0.1:8080/slots >/dev/null 2>&1 && break
  sleep 2
done
SLOTS=$(curl -s http://127.0.0.1:8080/slots | grep -o '"id"' | wc -l)
echo "server up, $SLOTS slots"
[ "$SLOTS" = "4" ] || { echo "expected 4 slots, got $SLOTS — aborting"; exit 1; }

python3 - "$REPO" <<'PY'
import json, sys, threading, time, urllib.request
sys.path.insert(0, sys.argv[1] + "/tools")
from synthetic import body
SRV = "http://127.0.0.1:8080"
Q = "Count slowly from one to sixty, one number per line."

def ask(project, max_tokens=200):
    p = body(project=project, n_tools=4, question=Q)
    p["model"], p["max_tokens"], p["stream"] = "laguna", max_tokens, False
    r = urllib.request.Request(SRV + "/v1/messages", data=json.dumps(p).encode(),
        headers={"content-type": "application/json", "anthropic-version": "2023-06-01"})
    t0 = time.time()
    with urllib.request.urlopen(r, timeout=900) as x:
        a = json.loads(x.read().decode())
    u = a.get("usage", {})
    return {"out": u.get("output_tokens", 0), "s": time.time() - t0,
            "cached": u.get("cache_read_input_tokens", 0), "new": u.get("input_tokens", 0)}

projects = ["/tmp/ss-%d" % i for i in range(4)]
print("\nwarming four prefixes with the same question (then 100 %% cache)")
for p in projects:
    m = ask(p)
    print("   %-12s %5.1f s  cache %5.1f %%" % (p, m["s"],
          100.0*m["cached"]/(m["cached"]+m["new"]) if m["cached"]+m["new"] else 0))

for n in (1, 2, 4):
    res = {}
    def run(k):
        res[k] = ask(projects[k])
    ts = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    t0 = time.time()
    for t in ts: t.start()
    for t in ts: t.join()
    wall = time.time() - t0
    total = sum(res[i]["out"] for i in range(n))
    each = sum(res[i]["out"]/res[i]["s"] for i in range(n)) / n
    print("\n%d session(s) at once" % n)
    print("   wall clock %5.1f s   %5.1f t/s each   %5.1f t/s together"
          % (wall, each, total/wall))
PY
