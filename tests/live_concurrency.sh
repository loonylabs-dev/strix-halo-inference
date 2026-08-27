#!/usr/bin/env bash
# End to end against the real stack: does the admission control hold up under
# load? Complements tests/live_prefix.sh, which covers a single request.
#
# What is asserted here was measured first with
# bench/suites/gateway-concurrency.py — only the parts that came out stable are
# assertions; the rest stays a measurement.
#
#   bash tests/live_concurrency.sh
#
# Duration: about a minute. Needs a running llama-server and cc-gateway, a warm
# prefix (it warms one itself) and an entry in ~/.config/cc-gateway-tokens.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
GATEWAY="${GATEWAY:-http://127.0.0.1:8090}"
DIRECT="${DIRECT:-http://127.0.0.1:8080}"
LAN="${LAN:-$(ip -4 -o addr show scope global 2>/dev/null | grep -v docker \
      | awk '{print $4}' | cut -d/ -f1 | head -1)}"
TOKEN_FILE="${TOKEN_FILE:-$HOME/.config/cc-gateway-tokens}"
TOKEN="$(awk '!/^#/ && NF>=2 {sub(/^[ \t]*[^ \t]+[ \t]+/, ""); print; exit}' \
         "$TOKEN_FILE" 2>/dev/null)"
echo "Admission control under load"
curl -sf -m5 "$GATEWAY/gateway/status" >/dev/null || { echo "cc-gateway unreachable"; exit 2; }
[ -n "$LAN" ]   || { echo "no LAN address found"; exit 2; }
[ -n "$TOKEN" ] || { echo "no access in $TOKEN_FILE"; exit 2; }

python3 - "$REPO" "$GATEWAY" "$DIRECT" "$LAN" "$TOKEN" <<'PY'
import json, statistics, sys, threading, time, urllib.request, urllib.error
REPO, GATEWAY, DIRECT, LAN, TOKEN = sys.argv[1:6]
sys.path.insert(0, REPO + "/tools")
from synthetic import body

WARM = "/tmp/live-conc"
# The served model name, asked from the server. It used to say "laguna"
# here; after a model switch that name no longer exists and every call in
# this file would measure an error instead of the stack.
try:
    with urllib.request.urlopen(DIRECT + "/v1/models", timeout=10) as _x:
        _m = json.load(_x)
    _entries = _m.get("models") or _m.get("data") or []
    MODEL = (_entries[0].get("name") or _entries[0].get("id")) if _entries else "local"
except Exception:
    MODEL = "local"
LOAD = "Count slowly from one to sixty, one number per line."
ERRORS = 0

def ok(m):   print("  \033[32m✓\033[0m %s" % m)
def nope(m):
    global ERRORS
    print("  \033[31m✗\033[0m %s" % m); ERRORS += 1

def call(base, max_tokens, token=None, question="Say alpha.", shape=False):
    p = body(project=WARM, n_tools=4, question=question)
    if shape:
        p = shaped(p)
    p["model"], p["max_tokens"], p["stream"] = MODEL, max_tokens, False
    h = {"content-type": "application/json", "anthropic-version": "2023-06-01"}
    if token:
        h["Authorization"] = "Bearer " + token
    r = urllib.request.Request(base + "/v1/messages",
                               data=json.dumps(p).encode(), headers=h)
    t0 = time.time()
    try:
        with urllib.request.urlopen(r, timeout=600) as x:
            x.read()
        return 200, time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, time.time() - t0

# The DIRECT baseline bypasses the gateway and therefore its body
# corrections. With a template that rejects mid-conversation system
# messages (Qwen 3.8) the raw Claude Code body is answered with an
# instant HTTP 500 — and timing that against a real request reported a
# "gateway overhead" of 631 ms on 25.08. that did not exist (measured
# again with equal bodies: 6 ms). So: shape the direct body the way the
# gateway would, and refuse to report an overhead unless both sides
# really answered 200.
sys.path.insert(0, REPO + "/setup/claude")
import dialects as DIA
import re as _re
_VOLATILE = [_re.compile(r"<total_tokens>\s*\d+\s*tokens left\s*</total_tokens>")]

def shaped(p):
    """What the gateway would forward — so the baseline is comparable."""
    p, _ = DIA.hoist_system_messages(p, DIA.ANTHROPIC, _VOLATILE)
    p, _ = DIA.mid_system_to_user(p, DIA.ANTHROPIC)
    return p

# warm the prefix, otherwise the first call pays a cold start
call(GATEWAY, 8); call(GATEWAY, 8)

# 1 · the gateway must not cost anything worth mentioning.
#     Two conditions, because one alone can pass vacuously: a negative overhead
#     only means the baseline was noisy, and the check would still say yes
#     without having measured anything. So both medians have to look like warm
#     requests in the first place.
dr = [call(DIRECT, 8, shape=True) for _ in range(5)]
gr = [call(GATEWAY, 8) for _ in range(5)]
bad = sorted({c for c, _ in dr + gr} - {200})
d = sorted(t for _, t in dr)[1:]
g = sorted(t for _, t in gr)[1:]
md, mg = statistics.median(d), statistics.median(g)
over = (mg - md) * 1000
if bad:
    # A failing call answers in milliseconds. Timing it against a working
    # one invents an overhead that does not exist — that is how this check
    # reported 631 ms on 25.08. when the truth was 6 ms.
    nope("baseline did not answer 200 (saw %s) — nothing was measured" % bad)
elif md > 1.0 or mg > 1.0:
    nope("baseline unusable: direct %.2f s, gateway %.2f s — was the prefix warm?"
         % (md, mg))
elif over < 100:
    ok("gateway overhead %.0f ms (direct %.3f s, gateway %.3f s)" % (over, md, mg))
else:
    nope("gateway overhead %.0f ms (over 100)" % over)

# 2 · per-access throttle: exactly one of three is turned away
res = {}
def fire(k):
    res[k] = call("http://%s:8090" % LAN, 8, TOKEN)[0]
ts = [threading.Thread(target=fire, args=(i,)) for i in range(3)]
for t in ts: t.start()
for t in ts: t.join()
codes = sorted(res.values())
# Against the CONFIGURED limit, not a hard-wired 2: raising PER_TOKEN_MAX
# to 3 made this report a deviation that was only the new setting.
try:
    with urllib.request.urlopen(GATEWAY + "/gateway/status", timeout=10) as _x:
        LIMIT = json.load(_x).get("per_token_max", 2)
except Exception:
    LIMIT = 2
want = [200] * min(3, LIMIT) + [429] * max(0, 3 - LIMIT)
(ok if codes == want else nope)(
    "three at once from one access -> %s (limit %d, expected %s)"
    % (codes, LIMIT, want))

# 3 · a fourth is turned away too, and the reason is named
res.clear()
ts = [threading.Thread(target=fire, args=(i,)) for i in range(4)]
for t in ts: t.start()
for t in ts: t.join()
n429 = sorted(res.values()).count(429)
want429 = max(0, 4 - LIMIT)
(ok if n429 == want429 else nope)(
    "four at once -> %d turned away (limit %d, expected %d)"
    % (n429, LIMIT, want429))

# 4 · priority: a local request goes ahead of a LAN one. The load has to come
#     from the local zone — PER_TOKEN_MAX does not apply there, so two long
#     requests really can hold both slots.
out = {}
def load(k):
    out[k] = call(GATEWAY, 400, question=LOAD)
def probe(k, base, tok):
    out[k] = call(base, 8, tok)
loads = [threading.Thread(target=load, args=("l%d" % i,)) for i in range(2)]
for t in loads: t.start()
time.sleep(4)
pair = [threading.Thread(target=probe, args=("local", GATEWAY, None)),
        threading.Thread(target=probe, args=("lan", "http://%s:8090" % LAN, TOKEN))]
for t in pair: t.start()
for t in pair: t.join()
for t in loads: t.join()
lo, la = out["local"][1], out["lan"][1]
if out["lan"][0] != 200:
    nope("LAN probe answered %d instead of 200" % out["lan"][0])
elif lo <= la:
    ok("local %.1f s before LAN %.1f s, as intended" % (lo, la))
else:
    nope("LAN %.1f s came before local %.1f s — priority did not bite" % (la, lo))

# 5 · nobody is starved. Four local streams are two Claude Code sessions, and
#     that used to lock a LAN caller out entirely: still waiting after 200 s,
#     while Cloudflare drops a remote one at 125 s. The queue ages now, so the
#     wait has to stay near QUEUE_AGE_AFTER.
LIMIT = 90
stop = threading.Event()
waiter = {}
def busy():
    while not stop.is_set():
        call(GATEWAY, 400, question=LOAD)
def lan_wait():
    waiter["m"] = call("http://%s:8090" % LAN, 8, TOKEN)
loadn = [threading.Thread(target=busy) for _ in range(4)]
for t in loadn: t.start()
time.sleep(2)
w = threading.Thread(target=lan_wait)
w.start()
w.join(timeout=LIMIT)
starved = w.is_alive()
stop.set()
w.join(timeout=120)
for t in loadn: t.join(timeout=60)
if starved:
    nope("LAN caller still waiting after %d s under four local streams" % LIMIT)
elif waiter.get("m", (0, 0))[0] != 200:
    nope("LAN caller answered %d" % waiter["m"][0])
else:
    ok("LAN caller served after %.1f s under four local streams (under %d)"
       % (waiter["m"][1], LIMIT))

sys.exit(ERRORS)
PY
RC=$?
echo
if [ "$RC" = "0" ]; then
  printf "\033[32mAdmission control behaves as designed.\033[0m\n"
else
  printf "\033[31m%d deviation(s).\033[0m\n" "$RC"
fi
exit "$RC"
