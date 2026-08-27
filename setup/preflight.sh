#!/usr/bin/env bash
# preflight — is this repo for this machine, and what of it can run here?
#
#   bash setup/preflight.sh
#
# Run it BEFORE anything else. It reads only; it changes nothing and needs no
# root. Return value 0 = this is the machine this repo was measured on.
#
# What it is for
# --------------
# This repo is measured on ONE configuration: Strix Halo (gfx1151) with 128 GB
# of shared memory. Every number in it comes from that machine — the window,
# the KV cost per token, the RAM prompt cache, the GTT cap, the memory
# budget's own constants.
#
# The same silicon also ships with 32 and 64 GB. Those are not the target, and
# this script does not pretend otherwise: it does not scale anything, and it
# does not invent numbers for a machine nobody measured. What it does is tell
# you where you stand before you spend an afternoon finding out — and show
# which profiles fit AS WRITTEN, which is a filter over measured values rather
# than a guess about unmeasured ones.
set -uo pipefail
# printf and the locale. bash's printf parses its ARGUMENTS according to
# LC_NUMERIC, so `printf '%.1f' 8.9` fails with "invalid number" in de_DE,
# fr_FR and every other comma-decimal locale — while awk, which produced the
# 8.9, always writes a dot. The repo has hit this before (setup/scripts/gtt.sh
# carries the same line) and tests/test_gtt.py pins it.
export LC_ALL=C
cd "$(dirname "$0")/.."
# shellcheck source=lib/models.sh
. "setup/lib/models.sh"

ok()   { printf "  \033[32m✓\033[0m %s\n" "$1"; }
no()   { printf "  \033[31m✗\033[0m %s\n" "$1"; }
hm()   { printf "  \033[33m?\033[0m %s\n" "$1"; }
head_() { printf "\n%s\n" "$1"; }

STATUS=0
JSON="$(python3 setup/lib/hardware.py --json 2>/dev/null)"
if [ -z "$JSON" ]; then
  echo "could not read this machine — is python3 present?" >&2
  exit 2
fi
val() { printf '%s' "$JSON" | python3 -c "import json,sys;d=json.load(sys.stdin);print(eval('d'+sys.argv[1]) if eval('d'+sys.argv[1]) is not None else '')" "$1" 2>/dev/null; }

head_ "This machine"
python3 setup/lib/hardware.py

head_ "Against what this repo was measured on"
GFX="$(val "['gpu']['gfx']")"
TARGET_GPU="$(val "['is_target_gpu']")"
TARGET_RAM="$(val "['is_target_ram']")"

if [ "$TARGET_GPU" = "True" ]; then
  ok "gfx1151 — the GPU every measurement here was taken on"
else
  STATUS=1
  no "this repo is measured on gfx1151 (Strix Halo); this machine reports '${GFX:-nothing}'"
  printf "    Nothing here is expected to be right on other hardware: the flags,\n"
  printf "    the defect registry and the memory arithmetic are all about that GPU.\n"
fi

UMA_BIG="$(val "['uma_is_large']")"
FITTED="$(val "['machine_ram_gib']")"
VRAM="$(val "['vram_reserved_gib']")"

if [ "$UMA_BIG" = "True" ]; then
  hm "the BIOS has parked $(printf '%.0f' "${VRAM:-0}") GiB as UMA — give it back"
  printf "    This stack does NOT use UMA. llama.cpp reaches the GPU through GTT,\n"
  printf "    which comes out of ordinary system RAM, so memory reserved as UMA is\n"
  printf "    lost to both sides. Set the BIOS split to its minimum; the amount the\n"
  printf "    GPU may then take is bash setup/scripts/gtt.sh, not a firmware menu.\n"
  printf "    ->  docs/setup/01-before-you-start.md\n"
fi

if [ "$TARGET_RAM" = "True" ]; then
  ok "$(printf '%.0f' "${FITTED:-0}") GiB fitted — the 128 GB configuration this repo is written for"
else
  STATUS=1
  no "$(printf '%.0f' "${FITTED:-0}") GiB fitted — this repo is written for the 128 GB configuration"
  printf "    Strix Halo also ships with 32 and 64 GB. Those are NOT the target, and\n"
  printf "    this repo will not scale its numbers to them: they were measured on\n"
  printf "    one machine and a scaled value is a guess. What you get instead is\n"
  printf "    the list below — profiles that fit as written — and bench/, which\n"
  printf "    measures YOUR machine if you want your own numbers.\n"
fi

head_ "Which profiles fit this machine, as written"
python3 - <<'PYEOF'
import glob, os, sys
sys.path.insert(0, "setup/lib")
import systemdfile as S, budget
m = budget.read_machine()
if m.mem_total is None:
    print("    /proc/meminfo unreadable — cannot judge"); raise SystemExit(0)
rows, fit = [], 0
for env in sorted(glob.glob("setup/env/*.env")):
    name = os.path.basename(env)[:-4]
    argv = S.llama_args(env)
    w = budget.weights_gib(argv)
    title = (S.variable(env, "MODEL_TITLE") or "")[:46]
    if w is None:
        rows.append("  \033[33m?\033[0m %-11s  weights not on this disk — not judged" % name)
        continue
    p = budget.plan(argv, w, budget.declared_kv(env), name,
                    gtt_base=budget.declared_gtt(env),
                    host_anon=budget.declared_anon(env))
    okk = budget.fits_the_machine(p, m)
    need = p.host_gib + budget.host_reserve_gib()
    mark = "\033[32m✓\033[0m" if okk else "\033[31m✗\033[0m"
    rows.append("  %s %-11s %6.1f of %.1f GiB   %s" % (mark, name, need, m.mem_total, title))
    fit += 1 if okk else 0
print("\n".join(rows))
judged = len([r for r in rows if "not judged" not in r])
print("\n    %d of %d fit." % (fit, judged), end=" ")
if fit == judged:
    print("This is what the 128 GB configuration looks like.")
else:
    print("The rest are refused BEFORE they start — GTT is")
    print("    pinned, so a model that does not fit does not fail, it freezes the")
    print("    machine. Nothing here is scaled down to make them fit: those numbers")
    print("    were measured on one machine, and a scaled one would be a guess.")
PYEOF

head_ "Prerequisites outside this repo"
if command -v rocminfo >/dev/null 2>&1; then ok "ROCm present"
else hm "ROCm not installed  ->  docs/setup/03-gpu-and-memory.md"; fi
if [ -x "$HOME/llama.cpp/build-rocm-patched/bin/llama-server" ]; then
  ok "patched llama.cpp build present"
else
  hm "no patched llama.cpp build  ->  docs/setup/04-build-llama.md"
  printf "    The patch is not optional on this GPU: without it a second slot\n"
  printf "    corrupts every answer to '////'. setup/defects.json carries the evidence.\n"
fi
CAP="$(val "['gtt_cap_gib']")"
if [ -n "$CAP" ]; then ok "GTT cap set on the kernel command line ($(printf '%.0f' "$CAP") GiB)"
else hm "no GTT cap on the kernel command line  ->  docs/setup/03-gpu-and-memory.md"; fi

head_ "Known defects of this hardware"
python3 setup/lib/defects.py 2>/dev/null | sed -n '1,4p' || true
printf "    full report:  python3 setup/lib/defects.py\n"

printf "\n"
if [ "$STATUS" = 0 ]; then
  printf "This is the configuration this repo was measured on.\n"
  printf "  Setting up a machine from scratch:  docs/setup/\n"
  printf "  Already there:                      bash setup/install.sh\n"
else
  printf "This is NOT the configuration this repo was measured on. Read the two\n"
  printf "sections above before going further; nothing here is stopping you, but\n"
  printf "no number in this repo was taken on a machine like yours.\n"
  printf "  What it IS measured on:  docs/setup/README.md\n"
fi
exit $STATUS
