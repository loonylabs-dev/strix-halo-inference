#!/usr/bin/env bash
# Switches the running tunnel to the prepared configuration and then checks
# function and protection.
#
#   bash setup/tunnel/switch.sh <your-tunnel-host>
#
# The example used to name a real hostname. It is the operator's own, it
# lives in ~/.config/llm-stack.env as GATEWAY_HOST, and a repo other people
# clone is the wrong place for it.
#
# Precondition: the DNS record for the hostname must exist and point as a CNAME
# to <TUNNEL-ID>.cfargotunnel.com (proxied). Without it the tunnel runs but
# receives no traffic.
set -uo pipefail
H="${1:?give a hostname}"
SRC="$(cd "$(dirname "$0")" && pwd)"
ERRORS=0
ANSWER="$(mktemp)"                      # no fixed name in /tmp
trap 'rm -f "$ANSWER"' EXIT

# The token comes from the file with the named access — the same source the
# gateway uses. This used to say '. ~/.config/cc-gateway.env' and take a $TOKEN
# from it. Since named access replaced the shared token, that variable is no
# longer set there: under `set -u` the script died at this point with "TOKEN:
# unbound variable" — and it did so AFTER it had already swapped the
# configuration and restarted the container.
TOKEN_FILE="${TOKEN_FILE:-$HOME/.config/llm-gateway-tokens}"
TOKEN="$(awk '!/^#/ && NF>=2 {sub(/^[ \t]*[^ \t]+[ \t]+/, ""); print; exit}' \
         "$TOKEN_FILE" 2>/dev/null)"
if [ -z "$TOKEN" ]; then
  echo "No access in $TOKEN_FILE — without it the check afterwards is pointless."
  echo "Aborting BEFORE switching; the tunnel stays as it is."
  exit 2
fi

echo "== precheck: does the name resolve? =="
if ! dig +short "$H" A | grep -q .; then
  echo "  $H does not resolve — the DNS record is missing. Aborting."
  echo "  Needed: CNAME $H -> <TUNNEL-ID>.cfargotunnel.com (proxied)"
  exit 1
fi
echo "  ok: $(dig +short "$H" A | tr '\n' ' ')"

echo "== activate the configuration =="
[ -f "$HOME/.cloudflared/config.new.yml" ] || {
  echo "  config.new.yml is missing"; exit 1; }
cp "$HOME/.cloudflared/config.yml" "$HOME/.cloudflared/config.old.yml" 2>/dev/null || true
mv "$HOME/.cloudflared/config.new.yml" "$HOME/.cloudflared/config.yml"
chmod 600 "$HOME/.cloudflared/config.yml"
grep -E "^tunnel:|hostname:" "$HOME/.cloudflared/config.yml" | sed 's/^/  /'

echo "== restart the container =="
if ! (cd "$SRC" && docker compose up -d --force-recreate) >"$ANSWER" 2>&1; then
  echo "  docker compose failed:"
  sed 's/^/    /' "$ANSWER" | tail -10
  echo "  Undo:  mv ~/.cloudflared/config.old.yml ~/.cloudflared/config.yml"
  exit 1
fi
sleep 8
docker ps --filter name=cloudflared --format '  {{.Names}}  {{.Status}}'
docker logs --tail 3 cloudflared 2>&1 | cut -c1-120 | sed 's/^/  /'

echo "== check the protection again =="
# $1 = description, $2 = expected code, rest = curl arguments. A deviation now
# counts — before, the script only printed numbers and always ended with 0, no
# matter how the protection reacted.
p() {
  local what="$1" want="$2"; shift 2
  local got; got=$(curl -s -o "$ANSWER" -w "%{http_code}" -m 25 "$@")
  if [ "$got" = "$want" ]; then
    printf "  \033[32m✓\033[0m %-38s %s\n" "$what" "$got"
  else
    printf "  \033[31m✗\033[0m %-38s %s (expected %s)  %s\n" \
           "$what" "$got" "$want" "$(head -c 44 "$ANSWER" | tr -d '\n')"
    ERRORS=$((ERRORS+1))
  fi
}
p "/slots without a token"   404 "https://$H/slots"
p "/completion without one"  404 -X POST "https://$H/completion" -H 'content-type: application/json' -d '{"prompt":"hi","n_predict":2}'
p "/slots WITH a token"      404 "https://$H/slots" -H "Authorization: Bearer $TOKEN"
p "/v1/messages without one" 401 -X POST "https://$H/v1/messages" -H 'content-type: application/json' -d '{"model":"laguna","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}'
p "/v1/messages WITH a token" 200 -X POST "https://$H/v1/messages" -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' -d '{"model":"laguna","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}'
echo
echo "== what the clients make of the TLS =="
printf "  %-10s " "curl:";   curl -s -o /dev/null -w "verify=%{ssl_verify_result}\n" -m 20 "https://$H/v1/models"
printf "  %-10s " "python:"; python3 -c "
import urllib.request,sys
try: urllib.request.urlopen('https://$H/v1/models', timeout=20); print('accepted')
except urllib.error.HTTPError as e: print('accepted (HTTP %s)' % e.code)
except Exception as e: print('REJECTED:', str(e)[:60])"

echo
if [ "$ERRORS" = "0" ]; then
  printf "\033[32mSwitched, the protection holds.\033[0m\n"
else
  printf "\033[31m%d deviation(s) — the tunnel is up, but not as expected.\033[0m\n" "$ERRORS"
  echo "Undo:  mv ~/.cloudflared/config.old.yml ~/.cloudflared/config.yml"
  echo "       (cd $SRC && docker compose up -d --force-recreate)"
fi
exit "$ERRORS"
