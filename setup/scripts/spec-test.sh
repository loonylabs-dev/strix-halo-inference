#!/usr/bin/env bash
# Speculative decoding: gpt-oss-120b with and without the eagle3 draft.
set -uo pipefail
# shellcheck source=../lib/models.sh
# models_dir(): $LLAMA_MODELS, then ~/.config/llm-stack.env, then the
# conventions. No path written down here. The directive above must stand
# ALONE on its line: prose after it makes shellcheck discard it (SC1125).
. "$(dirname "$0")/../lib/models.sh"
OUT=~/llm-setup/spec-result.txt
: > $OUT
# The model comes from the REGISTRY, not from a filename written here. A model
# name in two places is the failure lib/models.sh exists against: the profile
# is the only list, and asking it means this script follows a switch instead of
# quietly measuring whatever used to be called that.
M="$(model_gguf gptoss)"
[ -n "$M" ] || { echo "gptoss is not in the registry" >&2; exit 1; }
# The drafter has no profile of its own — it is not a model this stack serves,
# only one it would draft with. So it keeps a filename, next to the models.
DRAFT="$(dirname "$M")/eagle3-gpt-oss-120b-Q8_0.gguf"
BIN=~/llama.cpp/build-vulkan/bin/llama-server

run() {  # $1 = Label, Rest = Zusatzflags
  local label="$1"; shift
  $BIN -m $M -ngl 999 -fa on -ub 512 -b 2048 -c 8192 -ctk q8_0 -ctv q8_0 \
      --kv-unified --jinja --host 127.0.0.1 --port 8080 "$@" \
      > ~/llm-setup/spec-$label.log 2>&1 &
  local SRV=$!
  for _ in $(seq 1 150); do curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 2; done
  if ! curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1; then
    echo "$label: server not ready" >> $OUT; kill $SRV 2>/dev/null; sleep 3; return
  fi
  echo "### $label" >> $OUT
  for n in 1 2 3; do
    R=$(curl -sf -X POST http://127.0.0.1:8080/completion -H 'Content-Type: application/json' \
      -d '{"prompt":"Explain in exactly five sentences how mixture-of-experts models work.","n_predict":200,"temperature":0,"cache_prompt":false}')
    echo "$R" | jq -r '"  run \(input.n // "'"$n"'")  predicted_n=\(.timings.predicted_n)  \(.timings.predicted_per_second|tostring[0:6]) t/s  draft_accepted=\(.timings.draft_n_accepted // "-")/\(.timings.draft_n // "-")"' 2>/dev/null \
      || echo "$R" | python3 -c "
import json,sys
t=json.load(sys.stdin).get('timings',{})
print(f\"  predicted_n={t.get('predicted_n')}  {t.get('predicted_per_second',0):.2f} t/s  \"
      f\"draft={t.get('draft_n_accepted','-')}/{t.get('draft_n','-')}\")"
  done >> $OUT
  echo >> $OUT
  kill $SRV 2>/dev/null; sleep 5
}

run baseline
run eagle3 -md $DRAFT --spec-type draft-eagle3 --spec-draft-n-max 5
echo "=== FERTIG ===" >> $OUT
