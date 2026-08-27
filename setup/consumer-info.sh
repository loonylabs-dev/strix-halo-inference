#!/usr/bin/env bash
# consumer-info — the four facts a consumer of this stack needs, read from the
# RUNNING system rather than from prose.
#
#   bash setup/consumer-info.sh              for a remote consumer
#   bash setup/consumer-info.sh --local      for yourself, on this machine
#   bash setup/consumer-info.sh --markdown   paste-ready, to send to somebody
#
# Why this exists
# ---------------
# docs/CONSUMERS.md is 459 lines and 95 % of it is the same for everybody. The
# other 5 % are VALUES — endpoint, model names, window — and values written
# into prose go stale. That document carried two dead statements on 27.08.:
# a sentence about `laguna` still being available, and an entry that claimed
# "no vision" until 25.08. although the served model had a projector. Both are
# things the running stack can be asked.
#
# So the document is generic and this prints the values. Same idea as
# `budget.py --observe` and `models.sh table`: a small command that reads the
# truth instead of restating it.
set -uo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=lib/models.sh
. "setup/lib/models.sh"

LOCAL=0; MD=0
for a in "$@"; do
  case "$a" in
    --local)    LOCAL=1 ;;
    --markdown) MD=1 ;;
    -h|--help)  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $a" >&2; exit 2 ;;
  esac
done

GW_PORT="$(sed -n 's/^PORT=//p' "$HOME/.config/cc-gateway.env" 2>/dev/null | head -1)"
GW_PORT="${GW_PORT:-8090}"
GW="http://127.0.0.1:$GW_PORT"

if [ "$LOCAL" = 1 ]; then
  ENDPOINT="$GW"
  TOKEN_NOTE="none — the local zone on 127.0.0.1 needs no token"
else
  HOST="$(local_var GATEWAY_HOST)"
  if [ -z "$HOST" ]; then
    echo "No GATEWAY_HOST in $(llm_local_env)." >&2
    echo "  This machine serves no tunnel, or nobody has written it down." >&2
    echo "  For your own use on this machine:  bash setup/consumer-info.sh --local" >&2
    exit 1
  fi
  ENDPOINT="https://$HOST"
  TOKEN_NOTE="a personal token, one line 'name <secret>' in ~/.config/cc-gateway-tokens"
fi

# The model NAMES the gateway serves. Asked, not assumed: they come from
# KWARGS_BY_MODEL and change when the served model changes.
MODELS="$(curl -s -m 5 "$GW/v1/models" 2>/dev/null | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    raise SystemExit(0)
for m in d.get("models", d.get("data", [])):
    n = m.get("name") or m.get("id")
    caps = ",".join(m.get("capabilities", []) or [])
    print("%s%s" % (n, " [" + caps + "]" if caps else ""))
' 2>/dev/null)"

# The real slot size, from the server. The number a client must stay under.
NCTX="$(curl -s -m 5 "http://127.0.0.1:8080/props" 2>/dev/null | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    raise SystemExit(0)
print((d.get("default_generation_settings") or {}).get("n_ctx") or "")
' 2>/dev/null)"
if [ -n "$NCTX" ]; then
  # A client should sit a little below the slot, so a long turn cannot overrun
  # it. Rounded DOWN to the nearest 10k, which is what the doc recommended by
  # hand as "200,000" for a 204,800 slot.
  CLIENT_CTX=$(( (NCTX / 10000) * 10000 ))
else
  NCTX="unknown — is llama-server running?"; CLIENT_CTX=""
fi

if [ "$MD" = 1 ]; then
  printf '**Endpoint:** `%s`\n' "$ENDPOINT"
  printf '**Access:** %s\n' "$TOKEN_NOTE"
  printf '**Context:** %s per slot — set `CLAUDE_CODE_MAX_CONTEXT_TOKENS=%s`\n\n' "$NCTX" "$CLIENT_CTX"
  printf '| model name | capabilities |\n|---|---|\n'
  printf '%s\n' "$MODELS" | sed 's/^\([^ ]*\) \[\(.*\)\]$/| `\1` | \2 |/; t; s/^\(.*\)$/| `\1` | |/'
  printf '\nThe rest is in docs/CONSUMERS.md, which applies unchanged.\n'
  exit 0
fi

printf '  Endpoint    %s\n' "$ENDPOINT"
printf '  Access      %s\n' "$TOKEN_NOTE"
printf '  Window      %s per slot' "$NCTX"
[ -n "$CLIENT_CTX" ] && printf ' — set CLAUDE_CODE_MAX_CONTEXT_TOKENS=%s' "$CLIENT_CTX"
printf '\n  Models      '
if [ -z "$MODELS" ]; then
  printf 'not reachable at %s — is cc-gateway running?\n' "$GW"
else
  printf '%s\n' "$MODELS" | sed '1!s/^/              /'
fi
printf '\n  Everything else — the four Claude Code variants, the cache rules, the\n'
printf '  settings table — is in docs/CONSUMERS.md and does not depend on this.\n'
printf '  Paste-ready:  bash setup/consumer-info.sh --markdown\n'
