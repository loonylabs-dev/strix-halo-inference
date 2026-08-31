#!/usr/bin/env bash
# End to end against the real GPU: does the gateway save a cold prefix in a way
# that lets it find that prefix again afterwards?
#
# This is the test that did not exist before — and the only one that would have
# caught the bug: saving ran, reported success, and the prefix was still never
# restored, because the sidecar file sat under a different key. Everything
# about it looked right in the logs.
#
#   bash tests/live_prefix.sh
#
# Duration: one cold start (100-180 s) plus about a minute. Requires a running
# llama-server and llm-gateway.
#
# SIDE EFFECT: clears every slot in between (action=erase). Warm sessions of
# other projects reload from disk once afterwards.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
GATEWAY="${GATEWAY:-http://127.0.0.1:8090}"
LLAMA="${LLAMA:-http://127.0.0.1:8080}"
SLOT_PATH="${SLOT_PATH:-$HOME/.cache/llama-slots}"
LIMIT="${LIMIT:-30}"
ERRORS=0

ok()   { printf "  \033[32m✓\033[0m %s\n" "$1"; }
nope() { printf "  \033[31m✗\033[0m %s\n" "$1"; ERRORS=$((ERRORS+1)); }

BODY="$(mktemp /tmp/live-body-XXXXXX.json)"
ID=""
# Invoked by `trap cleanup EXIT` below, which shellcheck cannot see.
# shellcheck disable=SC2329
cleanup() {
  rm -f "$BODY"
  [ -n "$ID" ] && rm -f "$SLOT_PATH/$ID.bin" "$SLOT_PATH/$ID.json"
}
trap cleanup EXIT

echo "Prefix saving, end to end"

curl -sf -m5 "$GATEWAY/gateway/status" >/dev/null || { echo "llm-gateway unreachable"; exit 2; }
curl -sf -m5 "$LLAMA/slots"            >/dev/null || { echo "llama-server unreachable"; exit 2; }

# 1 · Build a body and compute the id with the gateway's own code — do not
#     reimplement it, or the test would only check its own copy.
ID="$(python3 - "$REPO" "$BODY" <<'PY'
import importlib.util, json, pathlib, sys, time
repo = pathlib.Path(sys.argv[1])
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, repo / path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m
syn = load("tools/synthetic.py", "synthetic")
gw  = load("setup/gateway/gateway.py", "gateway")
p = syn.body(project="/tmp/live-%d" % time.time(), n_tools=24,
             question="Say only the word test.")
p["model"], p["max_tokens"], p["stream"] = "laguna", 8, False
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(p, f, ensure_ascii=False)
print(gw.prefix_id(json.loads(json.dumps(p)))[0])
PY
)"
[ -n "$ID" ] || { echo "could not determine the id"; exit 2; }
echo "  prefix id: $ID"
rm -f "$SLOT_PATH/$ID.bin" "$SLOT_PATH/$ID.json"

# 2 · Cold request
t0=$(date +%s)
CODE=$(curl -s -o /dev/null -w '%{http_code}' -m 400 "$GATEWAY/v1/messages" \
       -H 'content-type: application/json' --data-binary @"$BODY")
COLD=$(( $(date +%s) - t0 ))
[ "$CODE" = "200" ] && ok "cold request answered ($COLD s)" \
                    || nope "cold request: HTTP $CODE"

# 3 · Wait for the automatic save (it runs in the background)
for _ in $(seq 1 90); do
  [ -f "$SLOT_PATH/$ID.json" ] && break
  sleep 2
done
if [ -f "$SLOT_PATH/$ID.json" ]; then
  ok "saved automatically"
else
  nope "not saved — the rest of the test says nothing"
  exit "$ERRORS"
fi

# 4 · The actual point: does it sit under the id the next request arrives with?
#     That is exactly where the bug was.
GK=$(python3 -c 'import json,sys
with open(sys.argv[1], encoding="utf-8") as f: print(json.load(f).get("gateway_id"))' \
     "$SLOT_PATH/$ID.json" 2>/dev/null)
[ "$GK" = "$ID" ] && ok "store key is the gateway id" \
  || nope "key $GK instead of $ID — this prefix will never be restored"

# 5 · Clear the slots. The gateway notices and throws its bookkeeping away, so
#     the next request counts as cold again — and has to come from disk.
for id in $(curl -s "$LLAMA/slots" | python3 -c 'import json,sys
[print(s["id"]) for s in json.load(sys.stdin)]'); do
  curl -s -X POST "$LLAMA/slots/$id?action=erase" >/dev/null
done
N=1
for _ in $(seq 1 30); do
  N=$(curl -s "$GATEWAY/gateway/status" | python3 -c 'import json,sys
print(len(json.load(sys.stdin)["prefixes"]))' 2>/dev/null || echo 1)
  [ "$N" = "0" ] && break
  sleep 2
done
[ "$N" = "0" ] && ok "gateway reset its bookkeeping" \
               || nope "bookkeeping not reset"

# 6 · Second request — it has to come from the file, not be recomputed
t2=$(date +%s)
CODE=$(curl -s -o /dev/null -w '%{http_code}' -m 400 "$GATEWAY/v1/messages" \
       -H 'content-type: application/json' --data-binary @"$BODY")
WARM=$(( $(date +%s) - t2 ))
[ "$CODE" = "200" ] && ok "second request answered ($WARM s)" \
                    || nope "second request: HTTP $CODE"

if journalctl --user -u llm-gateway --since "-10min" --no-pager 2>/dev/null \
   | grep -q "RESTORED    prefix $ID"; then
  ok "restored from the file"
else
  nope "no RESTORED in the log — the prefix was recomputed"
fi
if [ "$WARM" -le "$LIMIT" ]; then
  ok "$WARM s instead of $COLD s"
else
  nope "$WARM s — that is a cold start, not a reload"
fi

echo
if [ "$ERRORS" = "0" ]; then
  printf "\033[32mSaving and restoring both bite.\033[0m\n"
else
  printf "\033[31m%d deviation(s).\033[0m\n" "$ERRORS"
fi
exit "$ERRORS"
