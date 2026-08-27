#!/usr/bin/env bash
# Checks: (1) sequential cache hit, (2) parallel slots, (3) persistence across a restart
set -uo pipefail
# shellcheck source=../lib/models.sh
# models_dir(): $LLAMA_MODELS, then ~/.config/llm-stack.env, then the
# conventions. No path written down here. The directive above must stand
# ALONE on its line: prose after it makes shellcheck discard it (SC1125).
. "$(dirname "$0")/../lib/models.sh"
OUT=~/llm-setup/slot-result.txt; : > $OUT
BIN=~/llama.cpp/build-vulkan/bin/llama-server
# The model comes from the REGISTRY, not from a filename written here. A model
# name in two places is the failure lib/models.sh exists against: the profile
# is the only list, and asking it means this script follows a switch instead of
# quietly measuring whatever used to be called that.
M="$(model_gguf gemma26)"
[ -n "$M" ] || { echo "gemma26 is not in the registry" >&2; exit 1; }
SLOTDIR=/tmp/llama-slots; mkdir -p $SLOTDIR

start() {
  $BIN -m $M --alias gemma26 -ngl 999 -fa on -ub 512 -b 2048 -c 65536 \
    -ctk q8_0 -ctv q8_0 --kv-unified -np 4 -cram 32768 -ctxcp 64 -cms 4096 \
    --slot-save-path $SLOTDIR --cache-idle-slots \
    --jinja --host 127.0.0.1 --port 8081 > ~/llm-setup/slot-server.log 2>&1 &
  SRV=$!
  for _ in $(seq 1 90); do curl -sf http://127.0.0.1:8081/health >/dev/null 2>&1 && break; sleep 2; done
}
stop(){ kill $SRV 2>/dev/null; sleep 4; }

# ~6000 tokens of stable prefix, the way a system prompt looks
PREFIX=$(python3 -c "
import json
b=('Du bist ein Coding-Agent. Werkzeuge: read_file(path), write_file(path,content), '
   'grep(pattern,path), bash(cmd). Rules: answer briefly. Invent no paths. '
   'Pruefe Annahmen am Dateisystem. Nutze grep vor read_file bei unbekannter Struktur. ')
print(json.dumps(b*160))")

ask() { # $1=suffix  $2=id_slot(optional)
  local slot=${2:--1}
  python3 -c "
import json,sys
print(json.dumps({'prompt': json.loads(sys.argv[1]) + ' Aufgabe: ' + sys.argv[2],
 'n_predict':4,'temperature':0,'cache_prompt':True,'id_slot':int(sys.argv[3])}))" "$PREFIX" "$1" "$slot" \
  | curl -sf -X POST http://127.0.0.1:8081/completion -H 'Content-Type: application/json' -d @- \
  | python3 -c "
import json,sys
t=json.load(sys.stdin).get('timings',{})
p,c=t.get('prompt_n',0),t.get('cache_n',0)
print(f\"{p:>7} {c:>8} {100*c/(p+c) if p+c else 0:>7.1f}%  {t.get('prompt_ms',0):>9.0f} ms\")"
}

start
echo "=== 1. Sequenziell, gleicher Slot ===" >> $OUT
printf '%-10s %7s %8s %8s %12s\n' "run" "new" "cached" "share" "prompt_ms" >> $OUT
for n in 1 2 3; do printf '%-10s %s\n' "seq-$n" "$(ask "A$n")" >> $OUT; done

echo >> $OUT
echo "=== 2. four PARALLEL requests, same prefix ===" >> $OUT
printf '%-10s %7s %8s %8s %12s\n' "run" "new" "cached" "share" "prompt_ms" >> $OUT
rm -f /tmp/par.*.out
for n in 1 2 3 4; do ( printf '%-10s %s\n' "par-$n" "$(ask "P$n")" > /tmp/par.$n.out ) & done
wait
cat /tmp/par.*.out >> $OUT
echo "   (Reihenfolge zufaellig, siehe unten)" >> $OUT
for n in 1 2 3 4; do printf '%-10s %s\n' "par2-$n" "$(ask "Q$n")" >> $OUT; done

echo >> $OUT
echo "=== 3. Slot 0 auf Platte sichern ===" >> $OUT
curl -sf -X POST "http://127.0.0.1:8081/slots/0?action=save" -H 'Content-Type: application/json' \
  -d '{"filename":"agent-prefix.bin"}' \
  | python3 -c "import json,sys;d=json.load(sys.stdin);print('   gespeichert:',d.get('n_saved'),'Token,',round(d.get('n_written',0)/2**20),'MiB,',round(d.get('timings',{}).get('save_ms',0)),'ms')" >> $OUT 2>&1
ls -l $SLOTDIR >> $OUT 2>&1
stop

echo >> $OUT
echo "=== 4. Server NEU gestartet, Slot wiederherstellen ===" >> $OUT
start
printf '%-10s %s\n' "kalt" "$(ask "R0")" >> $OUT
curl -sf -X POST "http://127.0.0.1:8081/slots/1?action=restore" -H 'Content-Type: application/json' \
  -d '{"filename":"agent-prefix.bin"}' \
  | python3 -c "import json,sys;d=json.load(sys.stdin);print('   wiederhergestellt:',d.get('n_restored'),'Token,',round(d.get('timings',{}).get('restore_ms',0)),'ms')" >> $OUT 2>&1
printf '%-10s %s\n' "nach-restore" "$(ask "R1" 1)" >> $OUT
stop
echo "=== FERTIG ===" >> $OUT
