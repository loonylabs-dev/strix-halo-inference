#!/usr/bin/env bash
# Prompt cache verification per runbook section 20:
# 20 calls with an identical prefix and a changing suffix. What is measured
# is how much llama-server reuses from the cache instead of recomputing.
set -uo pipefail
# shellcheck source=../lib/models.sh
# models_dir(): $LLAMA_MODELS, then ~/.config/llm-stack.env, then the
# conventions. No path written down here. The directive above must stand
# ALONE on its line: prose after it makes shellcheck discard it (SC1125).
. "$(dirname "$0")/../lib/models.sh"
OUT=~/llm-setup/cache-result.txt
: > $OUT

# The model comes from the REGISTRY, not from a filename written here. A model
# name in two places is the failure lib/models.sh exists against: the profile
# is the only list, and asking it means this script follows a switch instead of
# quietly measuring whatever used to be called that.
M="$(model_gguf gemma26)"
[ -n "$M" ] || { echo "gemma26 is not in the registry" >&2; exit 1; }
~/llama.cpp/build-vulkan/bin/llama-server -m $M \
  -ngl 999 -fa on -ub 512 -b 2048 -c 32768 -ctk q8_0 -ctv q8_0 \
  --kv-unified -cram 32768 -ctxcp 64 -cms 4096 \
  --jinja --host 127.0.0.1 --port 8080 > ~/llm-setup/server.log 2>&1 &
SRV=$!
echo "Server-PID $SRV" >> $OUT

for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 && break
  sleep 2
done
curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 || { echo "server not ready" >> $OUT; kill $SRV; exit 1; }
echo "Server bereit nach $((i*2)) s" >> $OUT; echo >> $OUT

# Stable prefix: about 1500 tokens, byte-identical across all calls.
PREFIX=$(python3 -c "
import json
block = ('You are a technical assistant for Linux system administration. '
 'You answer briefly, precisely and without embellishment. You know your way '
 'around AMD hardware, ROCm, Vulkan and llama.cpp. Always observe the following rules: '
 'Answer in one sentence. Give no examples. Invent no numbers. ')
print(json.dumps(block*40))")

printf '%-6s %10s %10s %12s %14s\n' "no" "prompt_n" "cache_n" "prompt_ms" "cache share" >> $OUT
printf '%s\n' "------------------------------------------------------------" >> $OUT

for n in $(seq 1 20); do
  BODY=$(python3 -c "
import json,sys
p=json.loads(sys.argv[1])
print(json.dumps({'prompt': p + ' Question number ' + sys.argv[2] + ': answer with a number.',
                  'n_predict': 8, 'temperature': 0, 'cache_prompt': True}))" "$PREFIX" "$n")
  R=$(curl -sf -X POST http://127.0.0.1:8080/completion \
        -H 'Content-Type: application/json' -d "$BODY")
  PN=$(echo "$R" | jq -r '.timings.prompt_n // "?"')
  CN=$(echo "$R" | jq -r '.timings.cache_n // "?"')
  PMS=$(echo "$R" | jq -r '.timings.prompt_ms // "?"')
  PCT=$(python3 -c "
try:
    p,c=float('$PN'),float('$CN'); print(f'{100*c/(p+c):.1f} %' if p+c>0 else '-')
except: print('-')")
  printf '%-6s %10s %10s %12s %14s\n' "$n" "$PN" "$CN" "$PMS" "$PCT" >> $OUT
done

echo >> $OUT
echo "### Log-Auszug" >> $OUT
grep -iE "cache|slot|re-process" ~/llm-setup/server.log | tail -8 >> $OUT
kill $SRV 2>/dev/null
echo "=== FERTIG ===" >> $OUT
