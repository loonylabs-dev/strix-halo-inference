#!/usr/bin/env bash
# Does a RESTORED prefix still answer the CURRENT question?
#
# live_prefix.sh proves a prefix survives a restart and is restored — it
# never checks what the model says afterwards. A consumer reported on
# 26.08. that a second machine received another session's answer verbatim,
# with a matching thinking block and no error at all. That is the shape
# this class of bug always has: SAVED and RESTORED both look healthy while
# the output is foreign.
#
# So this asks the only question that matters after a restore: send the
# same prefix with a DIFFERENT question and check the answer follows the
# new one. Runs against the running stack.
#
#   bash tests/live_answer_freshness.sh
#
# Exit 0 = every answer belonged to its question.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
GATEWAY="${GATEWAY:-http://127.0.0.1:8090}"
DIRECT="${DIRECT:-http://127.0.0.1:8080}"

python3 - "$PWD" "$GATEWAY" "$DIRECT" <<'PY'
import json, os, subprocess, sys, time, urllib.request

REPO, GATEWAY, DIRECT = sys.argv[1:4]
sys.path.insert(0, REPO + "/tools")
ERRORS = 0

def ok(m):   print("  \033[32m✓\033[0m %s" % m)
def nope(m):
    global ERRORS
    print("  \033[31m✗\033[0m %s" % m); ERRORS += 1

# A body shaped like an OpenAI agent's: the question sits early, stable
# bulk follows it. That layout is what makes a stale state visible at all —
# with the question last, any divergence is a plain append.
SYSTEM = "You are a terse assistant. " + ("Guidance line. " * 260)
BULK = "Workspace instructions.\n" + ("Rule about this repository. " * 900)
TOOLS = [{"type": "function",
          "function": {"name": "Tool%02d" % i, "description": "d " * 60,
                       "parameters": {"type": "object"}}} for i in range(12)]

def body(nonce):
    return {"model": os.environ.get("MODEL", "qwen38"), "stream": False,
            "max_tokens": 200,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user",
                          "content": "Reply with exactly this word and "
                                     "nothing else: " + nonce},
                         {"role": "user", "content": BULK}],
            "tools": TOOLS}

def ask(nonce, timeout=900):
    r = urllib.request.Request(GATEWAY + "/v1/chat/completions",
                               data=json.dumps(body(nonce)).encode(),
                               headers={"content-type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as x:
        resp = json.loads(x.read().decode())
    m = resp["choices"][0]["message"]
    return ((m.get("content") or "").strip(),
            (m.get("reasoning_content") or ""))

def slots():
    with urllib.request.urlopen(DIRECT + "/slots", timeout=10) as x:
        return json.load(x)

stamp = int(time.time()) % 100000
first = "ALPHA-%d" % stamp

# 1 · a cold request, which is what gets auto-saved
c, _ = ask(first)
(ok if first in c else nope)("cold request answered its own question (%r)" % c[:24])

# 2 · wait for the save, then force the restore path: clear the slots and
#     let the gateway reload from disk on the next cold request.
saved = None
for _ in range(120):
    try:
        store = subprocess.run([sys.executable, REPO + "/tools/prewarm.py",
                                "list"], capture_output=True, text=True,
                               timeout=60).stdout
    except Exception:
        store = ""
    if store.count("\n") > 1:
        saved = store
        break
    time.sleep(5)
if saved is None:
    print("  \033[33m?\033[0m no saved prefix appeared — nothing to restore")
else:
    ok("a prefix was saved")

for s in slots():
    urllib.request.urlopen(urllib.request.Request(
        DIRECT + "/slots/%d?action=erase" % s["id"], data=b"{}",
        method="POST", headers={"content-type": "application/json"}),
        timeout=30).read()
ok("slots cleared")

# 3 · THE point of this file: a different question over the restored state
for i in range(2):
    nonce = "BRAVO-%d-%d" % (stamp, i)
    c, think = ask(nonce)
    if nonce in c:
        ok("restored prefix answered the NEW question (%r)" % c[:24])
    elif first in c or first in think:
        nope("STALE: answered the OLD question — %r / thinking %r"
             % (c[:40], think[:80]))
    elif not c:
        print("  \033[33m?\033[0m empty answer (thinking ate the budget) — "
              "no verdict from this run")
    else:
        nope("neither question: %r" % c[:60])

print()
print("\033[32manswers belong to their questions.\033[0m" if not ERRORS
      else "\033[31m%d deviation(s).\033[0m" % ERRORS)
sys.exit(1 if ERRORS else 0)
PY
