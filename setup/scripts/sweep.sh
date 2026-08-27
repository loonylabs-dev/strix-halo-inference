#!/usr/bin/env bash
# Power profile sweep: balanced and performance, all five models.
# Telemetry line by line with sync -> survives an EC power cut.
set -uo pipefail
# printf and the locale. bash's printf parses its ARGUMENTS according to
# LC_NUMERIC, so `printf '%.1f' 8.9` fails with "invalid number" in de_DE,
# fr_FR and every other comma-decimal locale — while awk, which produced the
# 8.9, always writes a dot. The repo has hit this before (setup/scripts/gtt.sh
# carries the same line) and tests/test_gtt.py pins it.
export LC_ALL=C
# shellcheck source=../lib/models.sh
# models_dir(): $LLAMA_MODELS, then ~/.config/llm-stack.env, then the
# conventions. No path written down here. The directive above must stand
# ALONE on its line: prose after it makes shellcheck discard it (SC1125).
. "$(dirname "$0")/../lib/models.sh"
# The sweep never uses the path. It cannot run without a model directory
# either, and failing here beats failing inside the first cell.
models_dir >/dev/null || exit 1
cd ~/llama.cpp
RES=~/llm-setup/bench-profiles.txt
CSV=~/llm-setup/telemetry-sweep.csv
D=/sys/class/drm/card1/device
H=$(ls -d $D/hwmon/hwmon* | head -1)
B=/sys/class/power_supply/BAT0

setprof(){ busctl set-property net.hadess.PowerProfiles /net/hadess/PowerProfiles \
   net.hadess.PowerProfiles ActiveProfile s "$1" >/dev/null 2>&1; sleep 5; }
trap 'setprof power-saver; kill $TPID 2>/dev/null' EXIT

echo "ts,profil,modell,gpu_w,gpu_c,gpu_busy,akku_pct,akku_status,cpu_mhz" > $CSV
telemetry(){
  while :; do
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$(date +%s)" "$(cat /tmp/.sweep_prof 2>/dev/null)" \
      "$(cat /tmp/.sweep_mod 2>/dev/null)" \
      "$(awk '{printf "%.1f",$1/1000000}' $H/power1_average)" \
      "$(awk '{printf "%.0f",$1/1000}' $H/temp1_input)" \
      "$(cat $D/gpu_busy_percent)" "$(cat $B/capacity)" "$(cat $B/status)" \
      "$(( $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq) / 1000 ))" >> $CSV
    sync -d $CSV
    sleep 3
  done
}
telemetry & TPID=$!

: > $RES
echo "Leistungsprofil-Sweep 2026-08-22, Vulkan/RADV, b10577, -p 512 -n 128 -r 3" >> $RES
echo >> $RES

for PROF in balanced performance; do
  echo "$PROF" > /tmp/.sweep_prof
  setprof "$PROF"
  echo "===== PROFIL: $PROF (platform_profile=$(cat /sys/firmware/acpi/platform_profile)) =====" >> $RES
  # Every model the REGISTRY knows, not five filenames written down here. The
  # old list named exactly the models of 22.08.; a sixth profile would have
  # been swept silently past. models_all reads setup/env/*.env, which is the
  # only list there is.
  for NAME in $(models_all); do
    GGUF="$(model_gguf "$NAME")"
    [ -r "$GGUF" ] || { echo "skip $NAME — not on this disk" >&2; continue; }
    echo "$NAME" > /tmp/.sweep_mod
    ./build-vulkan/bin/llama-bench -m "$GGUF" -ngl 999 -fa 1 -p 512 -n 128 -r 3 2>&1 \
      | grep -E "^\| (gemma|qwen|laguna|gpt)" >> $RES
    sleep 8
  done
  echo >> $RES
done
echo "idle" > /tmp/.sweep_mod
setprof power-saver
echo "=== FERTIG ===" >> $RES
