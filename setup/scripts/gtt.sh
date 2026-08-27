#!/usr/bin/env bash
# How much system memory the GPU may take — read it, raise it, undo it.
#
#   bash setup/scripts/gtt.sh                    what is set, what is in use
#   bash setup/scripts/gtt.sh --set 108          set the cap to 108 GiB
#   bash setup/scripts/gtt.sh --set 108 --dry-run   show the diff, change nothing
#   bash setup/scripts/gtt.sh --set 96           back to the conservative start
#   bash setup/scripts/gtt.sh --reset            remove the parameters entirely
#   bash setup/scripts/gtt.sh --verify           after the reboot: did it take?
#
# What this actually changes
# --------------------------
# On Strix Halo there is no separate VRAM. The BIOS reserves a minimum (0.5 GiB
# here, deliberately), and everything else the GPU uses comes out of system RAM
# through GTT. `ttm.pages_limit` is the ceiling on that, in 4 KiB pages.
#
# **It is a CAP, not a reservation.** Raising it allocates nothing and takes
# nothing from the host at boot; GTT allocations are dynamic. The risk is at
# RUNTIME and only then: a model that really claims 110 of 124.9 GiB leaves the
# desktop 15. That is why this script refuses values that leave the host too
# little, and why the runbook says to climb the ladder rather than jump to the
# top.
#
# Under Windows the same machine caps out around 96 GB (UMA + VGM). Being able
# to go past that is the actual argument for Linux here — not the 10-20 %
# throughput. Measured 17.08.2026 on this machine.
#
#     GiB   ttm.pages_limit   host      what the runbook says
#      96          25165824   32 GiB    start here; everything up to ~120B-MoE
#     108          28311552   20 GiB    plus room for containers alongside
#     116          30408704   12 GiB    "Arbeitseinstellung" — was set on 26.08.
#                                       and LOWERED again the same night, see below
#     120          31457280    8 GiB    edge cases; nothing else runs alongside
#
# The rest-for-host column is the runbook's, computed from a nominal 128 GB.
# This machine reports 124.9 GiB, so the real remainder is ~3 GiB less.
#
# THIS MACHINE RUNS 108, and how it got there is the argument for reading the
# column on the right as seriously as the one on the left. It was raised to
# 116 on 26.08. for a Flash-Next footprint the plan had computed as 110 GiB.
# Measured, that footprint is 87.4 — the plan was 30 GiB pessimistic about
# weights and 56 % low on KV, two errors that happened to cancel. So the raise
# bought room nobody needed, and it cost the thing this file's own header
# calls a cap rather than a reservation: the CEILING. Twice that day a
# measurement started a model that did not fit, and above the cap such a start
# does not fail, it takes the machine — no OOM kill, no log, a power cycle.
# At 108 the largest model here still has 20 GiB of headroom and a second one
# cannot allocate at all.
#
# Do NOT set amdgpu.gttsize
# -------------------------
# It is deprecated and ignored; the kernel says so at boot. Set both and let
# them disagree and the driver reports "this is unusual" — after which ROCm
# sees LESS memory than with no parameter at all. Documented as a ROCm issue
# (62.2 GB instead of 120). docs/archive/README.md is where the two documents
# that recommended it were retired to.
set -uo pipefail
# The numbers here pass through awk and back into printf. Under a German
# locale bash's printf refuses awk's "8.9" ("Ungültige Zahl") and silently
# substitutes 0 — a memory figure that is wrong rather than absent. C locale
# for the whole script, and floats are formatted by awk and printed as %s.
export LC_ALL=C
cd "$(dirname "$0")/../.." || exit 1
REPO="$PWD"
LIB="$REPO/setup/lib/kernelcmdline.py"

KCMD=/etc/kernel/cmdline
GRUBDEF=/etc/default/grub
PARAMS=(ttm.pages_limit ttm.page_pool_size)

MODE=show; WANT=""; DRY=0; FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --set)     MODE="set"; WANT="${2:?--set needs a size in GiB}"; shift 2 ;;
    --reset)   MODE="reset"; shift ;;
    --verify)  MODE="verify"; shift ;;
    --show)    MODE="show"; shift ;;
    --dry-run|-n) DRY=1; shift ;;
    --force)   FORCE=1; shift ;;
    -h|--help) sed -n '2,45p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

say()  { printf '%s\n' "$*"; }
head_(){ printf '\n%s\n' "$*"; }
ok()   { printf '  \033[32m=\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31m!\033[0m %s\n' "$*"; }
die()  { printf '\n\033[31mABORT\033[0m %s\n' "$*" >&2; exit 2; }
run()  { if [ "$DRY" = 1 ]; then printf '  would: %s\n' "$*"; else "$@"; fi; }

kib()  { awk '/^MemTotal:/{print $2}' /proc/meminfo; }
gib_total() { awk '/^MemTotal:/{printf "%.1f", $2/1048576}' /proc/meminfo; }
pages_now() { cat /sys/module/ttm/parameters/pages_limit 2>/dev/null; }
gtt_total_gib() {
  cat /sys/class/drm/card*/device/mem_info_gtt_total 2>/dev/null | head -1 \
    | awk '{printf "%.1f", $1/1073741824}'
}
gtt_used_gib() {
  cat /sys/class/drm/card*/device/mem_info_gtt_used 2>/dev/null | head -1 \
    | awk '{printf "%.1f", $1/1073741824}'
}
vram_gib() {
  cat /sys/class/drm/card*/device/mem_info_vram_total 2>/dev/null | head -1 \
    | awk '{printf "%.1f", $1/1073741824}'
}
pages_to_gib() { awk -v p="$1" 'BEGIN{printf "%.0f", p/262144}'; }

# --------------------------------------------------------------------------
show_state() {
  head_ "Memory"
  say   "    system RAM        $(gib_total) GiB"
  say   "    VRAM (BIOS UMA)   $(vram_gib) GiB   — the minimum, on purpose"
  say   "    GTT cap           $(gtt_total_gib) GiB"
  say   "    GTT in use now    $(gtt_used_gib) GiB"

  head_ "Where the setting lives"
  local booted etc grubv p
  booted=$(python3 "$LIB" get /proc/cmdline ttm.pages_limit 2>/dev/null)
  [ -n "$booted" ] && ok "booted with        ttm.pages_limit=$booted  ($(pages_to_gib "$booted") GiB)" \
                   || warn "booted WITHOUT ttm.pages_limit — the driver default applies"
  p=$(pages_now)
  if [ -n "$p" ] && [ -n "$booted" ] && [ "$p" != "$booted" ]; then
    bad "the module reports $p, the command line says $booted"
  fi
  if [ -r "$KCMD" ]; then
    etc=$(python3 "$LIB" get "$KCMD" ttm.pages_limit 2>/dev/null)
    if [ "$etc" = "$booted" ]; then ok "$KCMD  in step"
    else bad "$KCMD says ${etc:-nothing} — a NEW kernel would boot with that"; fi
  else
    warn "$KCMD does not exist — a new kernel inherits /proc/cmdline instead"
  fi
  if [ -r "$GRUBDEF" ]; then
    grubv=$(sed -n 's/^GRUB_CMDLINE_LINUX="\(.*\)"$/\1/p' "$GRUBDEF" \
            | tr ' ' '\n' | sed -n 's/^ttm.pages_limit=//p' | head -1)
    if [ "$grubv" = "$booted" ]; then ok "$GRUBDEF  in step"
    else warn "$GRUBDEF says ${grubv:-nothing} — inert on BLS, but it is what the next reader believes"; fi
  fi
  say
  say "  /boot/loader/entries/*.conf is what actually boots and needs root to read:"
  say "      sudo grubby --info=ALL | grep args"

  head_ "The ladder (docs/setup/03-gpu-and-memory.md)"
  local g pg cur
  cur=$(pages_to_gib "${booted:-0}")
  for g in 96 108 116 120; do
    pg=$(( g * 262144 ))
    printf '    %s %3d GiB   %-9d  host keeps %s GiB\n' \
      "$([ "$g" = "$cur" ] && printf '\033[32m->\033[0m' || printf '  ')" \
      "$g" "$pg" "$(awk -v t="$(gib_total)" -v g="$g" 'BEGIN{printf "%.1f", t-g}')"
  done
  say
  say "  It is a CAP, not a reservation: raising it takes nothing from the host"
  say "  until a model actually claims it."
}

# --------------------------------------------------------------------------
apply() {          # $1 = "set" or "reset", $2 = pages (for set)
  local what="$1" pages="${2:-}" args=() a
  local stamp; stamp=$(date +%Y%m%d-%H%M%S)

  if [ "$what" = set ]; then
    for a in "${PARAMS[@]}"; do args+=("$a=$pages"); done
  fi

  head_ "1/4  /boot/loader/entries/*.conf — what actually boots"
  # grubby does its own replace-or-add, so it gets the parameters directly.
  if [ "$what" = set ]; then
    run sudo grubby --update-kernel=ALL --args="${args[*]}"
  else
    run sudo grubby --update-kernel=ALL --remove-args="${PARAMS[*]}"
  fi

  head_ "2/4  $KCMD — what a NEW kernel inherits"
  # kernel-install prefers this file over /proc/cmdline. Without it, the next
  # kernel update quietly boots with the old value. grubby --update-kernel=ALL
  # only ever touches entries that already exist.
  if [ -r "$KCMD" ]; then
    local new_k
    if [ "$what" = set ]; then
      new_k=$(python3 "$LIB" set "$KCMD" "${args[@]}") || die "refused to edit $KCMD (see above)"
    else
      new_k=$(python3 "$LIB" remove "$KCMD" "${PARAMS[@]}") || die "refused to edit $KCMD"
    fi
    say "    - $(cat "$KCMD")"
    say "    + $new_k"
    run sudo cp -a "$KCMD" "$KCMD.bak-$stamp"
    if [ "$DRY" = 1 ]; then say "  would: write the + line to $KCMD"
    else printf '%s\n' "$new_k" | sudo tee "$KCMD" >/dev/null && ok "written, backup at $KCMD.bak-$stamp"; fi
  else
    warn "$KCMD does not exist — skipping"
  fi

  head_ "3/4  $GRUBDEF — inert on BLS, kept honest anyway"
  if [ -r "$GRUBDEF" ]; then
    local new_g
    if [ "$what" = set ]; then
      new_g=$(python3 "$LIB" grub "$GRUBDEF" "${args[@]}") || die "refused to edit $GRUBDEF"
    else
      new_g=$(python3 "$LIB" grub-remove "$GRUBDEF" "${PARAMS[@]}") || die "refused to edit $GRUBDEF"
    fi
    say "    $(printf '%s' "$new_g" | sed -n 's/^GRUB_CMDLINE_LINUX=/    + GRUB_CMDLINE_LINUX=/p')"
    run sudo cp -a "$GRUBDEF" "$GRUBDEF.bak-$stamp"
    if [ "$DRY" = 1 ]; then say "  would: write the whole file back with only that line changed"
    else printf '%s' "$new_g" | sudo tee "$GRUBDEF" >/dev/null && ok "written, backup at $GRUBDEF.bak-$stamp"; fi
  fi

  head_ "4/4  read it back"
  if [ "$DRY" = 1 ]; then
    say "  would: sudo grubby --info=ALL | grep args"
    say
    say "DRY RUN — nothing was changed."
    return 0
  fi
  local entries hits
  entries=$(sudo grubby --info=ALL 2>/dev/null | grep -c '^args=')
  if [ "$what" = set ]; then
    hits=$(sudo grubby --info=ALL 2>/dev/null | grep -c "ttm.pages_limit=$pages")
  else
    hits=$(( entries - $(sudo grubby --info=ALL 2>/dev/null | grep -c 'ttm.pages_limit') ))
  fi
  if [ "$entries" -gt 0 ] && [ "$hits" = "$entries" ]; then
    ok "all $entries boot entries carry the new value"
  else
    bad "$hits of $entries boot entries carry it — check: sudo grubby --info=ALL | grep args"
  fi

  cat <<END

Nothing has changed for the RUNNING system — ttm.pages_limit is read when the
amdgpu driver initialises, which is at boot. Reboot when it suits you:

    sudo reboot
    bash setup/scripts/gtt.sh --verify

If the desktop comes up unusable
--------------------------------
At the GRUB menu press 'e', delete the two ttm. parameters from the 'linux'
line, and boot that once with Ctrl-X. Append systemd.unit=multi-user.target as
well if the graphical session is what hangs. Then, from that session:

    bash setup/scripts/gtt.sh --set 96

The backups are at
    $KCMD.bak-$stamp
    $GRUBDEF.bak-$stamp
END
}

# --------------------------------------------------------------------------
verify() {
  head_ "After the reboot"
  local booted p gtt want_gib
  booted=$(python3 "$LIB" get /proc/cmdline ttm.pages_limit 2>/dev/null)
  p=$(pages_now)
  gtt=$(gtt_total_gib)
  if [ -z "$booted" ]; then
    warn "no ttm.pages_limit on the booted command line"
  else
    want_gib=$(pages_to_gib "$booted")
    ok "booted with ttm.pages_limit=$booted ($want_gib GiB)"
    if [ "$p" = "$booted" ]; then ok "the ttm module agrees"
    else bad "the module says $p, the command line says $booted"; fi
    # The test that really counts: what the amdgpu driver made of it.
    if awk -v a="$gtt" -v b="$want_gib" 'BEGIN{exit !(a>b-1 && a<b+1)}'; then
      ok "amdgpu reports a GTT cap of $gtt GiB — it took"
    else
      bad "amdgpu reports $gtt GiB, not $want_gib. Check for a conflicting"
      bad "amdgpu.gttsize:  sudo dmesg | grep -i 'this is unusual'"
    fi
  fi
  # What ROCm ITSELF believes it may allocate — the number the loader acts on.
  # The parser is a file, because the one-liner that used to stand here read
  # the CPU agent's pool (also flagged COARSE GRAINED, also sized at the whole
  # of system RAM) and confidently reported 124.9 GiB on a machine whose GPU
  # limit was 96. See setup/lib/rocm-gpu-pool.awk.
  if command -v rocminfo >/dev/null 2>&1; then
    local pool pool_gib
    pool=$(rocminfo 2>/dev/null | awk -f "$REPO/setup/lib/rocm-gpu-pool.awk")
    if [ -n "$pool" ]; then
      pool_gib=$(awk -v k="$pool" 'BEGIN{printf "%.1f", k/1048576}')
      if awk -v a="$pool_gib" -v b="$gtt" 'BEGIN{exit !(a>b-1 && a<b+1)}'; then
        ok "ROCm sees $pool_gib GiB on the GPU agent — agrees with amdgpu"
      else
        bad "ROCm sees $pool_gib GiB on the GPU agent but amdgpu reports $gtt GiB."
        bad "That gap is the amdgpu.gttsize trap:  sudo dmesg | grep -i 'this is unusual'"
      fi
    else
      warn "rocminfo names no GPU agent — is the render group membership still there?"
    fi
  fi
  say
  say "  in use right now: $(gtt_used_gib) of $gtt GiB"
}

# --------------------------------------------------------------------------
case "$MODE" in
  show)   show_state ;;
  verify) verify ;;
  reset)  apply reset ;;
  set)
    case "$WANT" in
      ''|*[!0-9]*) die "--set takes whole GiB, got '$WANT'" ;;
    esac
    TOTAL_GIB=$(gib_total)
    REST=$(awk -v t="$TOTAL_GIB" -v w="$WANT" 'BEGIN{printf "%.1f", t-w}')
    if awk -v t="$TOTAL_GIB" -v w="$WANT" 'BEGIN{exit !(w>=t)}'; then
      die "$WANT GiB is more RAM than this machine has ($TOTAL_GIB GiB of RAM)."
    fi
    if awk -v r="$REST" 'BEGIN{exit !(r<6)}' && [ "$FORCE" = 0 ]; then
      die "$WANT GiB would leave the host $REST GiB.

    That is not enough for a desktop session — the runbook's own table stops
    recommending anywhere near it. If you mean it (a benchmark with nothing
    else running), say so:  --force"
    fi
    awk -v r="$REST" 'BEGIN{exit !(r<8)}' && warn "the host is left $REST GiB — close the browser and any VM before loading a model that big"
    USED=$(gtt_used_gib)
    if awk -v u="$USED" -v w="$WANT" 'BEGIN{exit !(w<u)}'; then
      die "$WANT GiB is below what the GPU is using right now ($USED GiB)."
    fi
    PAGES=$(( WANT * 262144 ))
    head_ "GTT $(gtt_total_gib) GiB  ->  $WANT GiB   (ttm.pages_limit=$PAGES)"
    say   "  host keeps $REST GiB of $TOTAL_GIB — and only loses it when a model claims the cap"
    apply set "$PAGES" ;;
esac
