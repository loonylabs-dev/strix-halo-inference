#!/usr/bin/env bash
# Functional check of the whole stack: all three zones, access control, allow
# list. Complements check.sh — that one checks configuration and state, this
# one checks whether the protection actually holds.
#
#   bash setup/smoketest.sh                quick checks only (~20 s)
#   bash setup/smoketest.sh --full         plus streaming and cold start
#   bash setup/smoketest.sh --local-only   deliberately skip the zones that
#                                          need a token or a LAN address
#
# Return value 0 = everything as expected, otherwise the number of deviations.
#
# A skipped section counts as a deviation. It did not before: if the token was
# missing, the entire remote zone fell away silently and the script still
# reported "all checks passed" with return value 0 — a green result that had
# checked nothing. Whoever wants to leave a zone out says so with --local-only.
#
# Assumes the slots are warm. After a server restart the first call takes
# 100-180 s; the script allows for that with a generous timeout.
set -uo pipefail

LAN="${LAN:-$(ip -4 -o addr show scope global 2>/dev/null | grep -v docker \
      | awk '{print $4}' | cut -d/ -f1 | head -1)}"
# NO DEFAULT, and this line is why the whole local-config mechanism exists.
# It used to name a private hostname, so `git clone && bash setup/smoketest.sh`
# sent requests to somebody else's tunnel. A hostname cannot be derived and
# must not be guessed: GATEWAY_HOST comes from ~/.config/llm-stack.env, which
# setup/install.sh creates and .gitignore keeps out of the repo.
# shellcheck source=lib/models.sh
. "$(dirname "$0")/lib/models.sh"
HOST="${HOST:-$(local_var GATEWAY_HOST)}"
TOKEN_FILE="${TOKEN_FILE:-${TOKENDATEI:-$HOME/.config/llm-gateway-tokens}}"
ERRORS=0
FULL=0
LOCAL_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --full|--voll)             FULL=1 ;;
    --local-only|--nur-lokal)  LOCAL_ONLY=1 ;;
    *) echo "unknown argument: $arg"; exit 2 ;;
  esac
done

# Read exactly like the gateway: name, whitespace, and the WHOLE rest is the
# secret (split(None, 1) there). With awk '{print $2}' a secret containing a
# space would have been cut short — every token check would have reported 401
# and looked like broken protection.
TOKEN="$(awk '!/^#/ && NF>=2 {sub(/^[ \t]*[^ \t]+[ \t]+/, ""); print; exit}' \
         "$TOKEN_FILE" 2>/dev/null)"
if [ -z "$TOKEN" ]; then
  echo "No access found in $TOKEN_FILE."
fi

BODY='{"model":"laguna","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}'

check() {   # $1 = description, $2 = expected code, rest = curl arguments
  local what="$1" want="$2"; shift 2
  local got
  got=$(curl -s -o /dev/null -w "%{http_code}" -m 240 "$@" 2>/dev/null)
  if [ "$got" = "$want" ]; then
    printf "  \033[32m✓\033[0m %-48s %s\n" "$what" "$got"
  else
    printf "  \033[31m✗\033[0m %-48s %s (expected %s)\n" "$what" "$got" "$want"
    ERRORS=$((ERRORS+1))
  fi
}

unreachable() {   # $1 = description, $2 = URL
  local got
  got=$(curl -s -o /dev/null -w "%{http_code}" -m 6 "$2" 2>/dev/null)
  if [ "$got" = "000" ]; then
    printf "  \033[32m✓\033[0m %-48s unreachable\n" "$1"
  else
    printf "  \033[31m✗\033[0m %-48s REACHABLE (HTTP %s)\n" "$1" "$got"
    ERRORS=$((ERRORS+1))
  fi
}

skipped() {   # $1 = section, $2 = reason
  if [ "$LOCAL_ONLY" = "1" ]; then
    printf "  \033[33m-\033[0m %-48s left out (--local-only)\n" "$1"
  else
    printf "  \033[31m✗\033[0m %-48s NOT CHECKED: %s\n" "$1" "$2"
    ERRORS=$((ERRORS+1))
  fi
}

echo "Stack functional check"
echo "  LAN=$LAN  host=$HOST"

echo
echo "1 · zone local — allowed without a token"
check "POST /v1/messages"          200 -X POST "http://127.0.0.1:8090/v1/messages" \
      -H 'content-type: application/json' -d "$BODY"
check "GET  /gateway/status"       200 "http://127.0.0.1:8090/gateway/status"

echo
echo "2 · zone lan — token required"
if [ -n "$LAN" ]; then
  check "without a token"          401 -X POST "http://$LAN:8090/v1/messages" \
        -H 'content-type: application/json' -d "$BODY"
  check "wrong token"              401 -X POST "http://$LAN:8090/v1/messages" \
        -H "Authorization: Bearer wrong" -H 'content-type: application/json' -d "$BODY"
  check "wrong x-api-key"          401 -X POST "http://$LAN:8090/v1/messages" \
        -H "x-api-key: wrong" -H 'content-type: application/json' -d "$BODY"
  [ -n "$TOKEN" ] && check "valid token"  200 -X POST "http://$LAN:8090/v1/messages" \
        -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' -d "$BODY"
  check "blocked path /slots"      404 "http://$LAN:8090/slots"
  check "/gateway/status from outside" 403 "http://$LAN:8090/gateway/status"
else
  skipped "zone lan" "no LAN address found"
fi

echo
echo "3 · zone remote — through the tunnel"
# An empty HOST is a SKIP, not a request to "https:///v1/messages". A skip
# counts as a deviation here — see the header: a zone that silently falls away
# turned this script green while it had checked nothing.
if [ -z "$HOST" ]; then
  skipped "zone remote" "no GATEWAY_HOST. Set it in $(llm_local_env), or run
    with --local-only if this machine serves no tunnel. It has NO default:
    guessing a hostname means aiming a test at a machine that is not yours"
elif [ -n "$TOKEN" ]; then
  check "without a token"          401 -X POST "https://$HOST/v1/messages" \
        -H 'content-type: application/json' -d "$BODY"
  check "valid token"              200 -X POST "https://$HOST/v1/messages" \
        -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' -d "$BODY"
  check "/slots without a token"   404 "https://$HOST/slots"
  check "/slots WITH a token"      404 "https://$HOST/slots" -H "Authorization: Bearer $TOKEN"
  check "/completion WITH a token" 404 -X POST "https://$HOST/completion" \
        -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
        -d '{"prompt":"hi","n_predict":2}'
  check "/v1/models WITH a token"  200 "https://$HOST/v1/models" -H "Authorization: Bearer $TOKEN"
  # The third allowed path — it was on the allow list but never checked.
  check "/v1/messages/count_tokens WITH a token" 200 -X POST \
        "https://$HOST/v1/messages/count_tokens" \
        -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
        -H 'anthropic-version: 2023-06-01' \
        -d '{"model":"laguna","messages":[{"role":"user","content":"hi"}]}'
else
  skipped "zone remote" "no access in $TOKEN_FILE"
fi

echo
echo "4 · what must NOT be reachable from outside"
if [ -n "$LAN" ]; then
  unreachable "llama-server on the LAN address"  "http://$LAN:8080/health"
  unreachable "tunnel port on the LAN address"   "http://$LAN:8091/v1/models"
else
  skipped "reachability from outside" "no LAN address found"
fi

echo
echo "5 · prompt exposure"
P=$(curl -s -m 10 "http://127.0.0.1:8080/slots" 2>/dev/null \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print("yes" if any("prompt" in s for s in d) else "no")' 2>/dev/null)
if [ "$P" = "no" ]; then
  printf "  \033[32m✓\033[0m %-48s no\n" "/slots contains a prompt field"
else
  printf "  \033[33m!\033[0m %-48s YES — LLAMA_SERVER_SLOTS_DEBUG is set\n" "/slots contains a prompt field"
  ERRORS=$((ERRORS+1))
fi

if [ "$FULL" = "1" ] && [ -n "$TOKEN" ]; then
  echo
  echo "6 · streaming through the tunnel (Cloudflare time limit)"
  OUT=$(curl -sN -m 600 "https://$HOST/v1/messages" \
        -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
        -H 'anthropic-version: 2023-06-01' \
        -d '{"model":"laguna","max_tokens":40,"stream":true,"messages":[{"role":"user","content":"Say only the word smoke."}]}' 2>/dev/null)
  if printf '%s' "$OUT" | grep -q "content_block_delta\|message_stop"; then
    printf "  \033[32m✓\033[0m %-48s stream complete\n" "SSE to the end"
  else
    printf "  \033[31m✗\033[0m %-48s aborted or empty\n" "SSE to the end"
    ERRORS=$((ERRORS+1))
  fi
fi

echo
if [ "$ERRORS" = "0" ]; then
  printf "\033[32mAll checks passed.\033[0m\n"
else
  printf "\033[31m%d deviation(s).\033[0m\n" "$ERRORS"
fi
exit "$ERRORS"
