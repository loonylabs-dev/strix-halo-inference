#!/usr/bin/env bash
# Build a patched llama.cpp — versioned, verified, and reversible.
#
#   bash setup/scripts/build-llama.sh                    build from origin/master
#   bash setup/scripts/build-llama.sh --ref b10631       build a tag
#   bash setup/scripts/build-llama.sh --ref pr/27700     build a pull request
#   bash setup/scripts/build-llama.sh --ref master --no-patch
#                                                        WITHOUT the patch, to
#                                                        measure upstream itself
#   bash setup/scripts/build-llama.sh --unroll           patched PLUS the ROCm
#                                                        unroll workaround, to
#                                                        measure the flag
#   bash setup/scripts/build-llama.sh --with-bench       build llama-bench too
#   bash setup/scripts/build-llama.sh --rocm-path DIR    build against a
#                                                        ROCm that is not
#                                                        the system one
#   bash setup/scripts/build-llama.sh --backend vulkan   the other backend
#   bash setup/scripts/build-llama.sh --activate         build, then point the
#                                                        profiles at the result
#   bash setup/scripts/build-llama.sh --list             which builds exist
#   bash setup/scripts/build-llama.sh --use <id>         switch to one (rollback)
#   bash setup/scripts/build-llama.sh --prune            what could be deleted
#   bash setup/scripts/build-llama.sh --prune --yes      delete it (keeps the
#                                                        active one and --keep N)
#   bash setup/scripts/build-llama.sh --dry-run          say what would happen
#
# Why this exists at all
# ----------------------
# setup/env/qwen38.env ends with a sentence that used to be a manual chore:
# "Rebuild the patched binary after every llama.cpp update." The build itself
# is four lines from setup/patches/README.md. What is NOT four lines is doing
# it without breaking the machine, and there are three separate reasons:
#
# 1. THE PATCH. `git pull` removes it and nothing says so. The server still
#    starts; the answers just turn to '////' once a second slot is used. It is
#    a COMMIT on a branch now (default gfx1151-patched) instead of a loose
#    working-tree diff, so an update is a rebase — which either replays it or
#    tells you it cannot, rather than silently losing it.
#
# 2. THE RUNNING SERVER. llama-server is a 12 KB executable that maps
#    libggml-hip.so and friends out of the SAME bin/ directory. Rebuilding in
#    place overwrites files the running process has mapped, and overwriting a
#    mapped file is a SIGBUS in that process, not an error in the build. So
#    every build gets its OWN directory and the stable path becomes a symlink.
#
# 3. THE WAY BACK. gfx1151 collects ROCm and llama.cpp regressions at a
#    remarkable rate — setup/defects.json now holds the evidence for that
#    claim rather than an unverified survey. A new build that is
#    slower or wrong has to be undone in one command, with the old binary
#    still on disk — not rebuilt from a tag under time pressure.
#
# Layout it creates:
#
#     ~/llama.cpp/build-rocm-patched-b10631/     a build, with .build-stamp
#     ~/llama.cpp/build-rocm-patched  ->  build-rocm-patched-b10631
#
# The symlink is the name setup/env/*.env points LLAMA_BIN at, so profiles
# never change when the build does.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO="$PWD"

SRC="${LLAMA_SRC:-$HOME/llama.cpp}"
# WHY NOT `gfx1151-patched` ANY MORE. That branch still exists and still
# carries 22 commits over master: the gfx1151 patch, the EOG fix, and twenty
# more that were the local qwen4exp/Flash-Next implementation. Upstream merged
# its own on 27.08.2026 (#27742), one day after this stack's previous build,
# so those twenty are dead weight — and MAX_REPLAY refuses a 22-commit replay,
# correctly. `master-2patches` is the same two patches on current master and
# nothing else. Keep the old branch until Flash-Next has served from a master
# build in anger; it is the way back if upstream's version turns out to differ.
PATCH_BRANCH="${PATCH_BRANCH:-master-2patches}"
# One marker per patch the branch has to carry, as `file:text`. Both are
# checked before a build, because a patch that goes missing does not fail —
# it degrades, silently and in a different way each time:
#   hip-integrated-off      wrong ANSWERS once a second slot is used
#   speculation-stops-at-eog  the previous answer re-prefilled every turn
# setup/patches/README.md carries both stories.
PATCH_MARKERS=(
  "ggml/src/ggml-cuda/ggml-cuda.cu:gfx1151/ROCm: trusting prop.integrated"
  "tools/server/server-context.cpp:accepted token(s) past EOG"
)
BACKEND=rocm
REF=""
JOBS="${JOBS:-$(( $(nproc) > 8 ? $(nproc) - 8 : 2 ))}"
DRY=0; ACTIVATE=0; USE=""; LIST=0; PRUNE=0; YES=0; NOPATCH=0; KEEP="${KEEP:-1}"
# An extra compiler flag, and a second build target. Both are for MEASURING
# rather than for serving, which is why each gets a family or a control of its
# own rather than being folded into the ordinary build. See step 3.
UNROLL=0; WITHBENCH=0; ROCMPATH=""
# How many commits the patch branch may carry over the ref being built.
# TWO patches since 30.08.2026 — hip-integrated-off and
# speculation-stops-at-eog — so three is now one spare rather than two. See
# step 2 for why this exists at all.
MAX_REPLAY="${MAX_REPLAY:-3}"

while [ $# -gt 0 ]; do
  case "$1" in
    --ref)      REF="${2:?--ref needs a value}"; shift 2 ;;
    --no-patch) NOPATCH=1; shift ;;
    --unroll)   UNROLL=1; shift ;;
    --with-bench) WITHBENCH=1; shift ;;
    --rocm-path) ROCMPATH="${2:?--rocm-path needs a directory}"; shift 2 ;;
    --backend)  BACKEND="${2:?--backend needs rocm or vulkan}"; shift 2 ;;
    --jobs|-j)  JOBS="${2:?-j needs a number}"; shift 2 ;;
    --activate) ACTIVATE=1; shift ;;
    --use)      USE="${2:?--use needs a build id}"; shift 2 ;;
    --list)     LIST=1; shift ;;
    --prune)    PRUNE=1; shift ;;
    --keep)     KEEP="${2:?--keep needs a number}"; shift 2 ;;
    --yes)      YES=1; shift ;;
    --dry-run|-n) DRY=1; shift ;;
    # Bounded by the heading that follows it, not by a line NUMBER: `2,20p`
    # silently started cutting the last option off the moment two were added
    # above it, and a truncated help says nothing about being truncated.
    -h|--help)  sed -n '2,/^# Why this exists/p' "$0" | sed '$d' \
                  | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)          echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$BACKEND" in rocm|vulkan) ;; *) echo "backend must be rocm or vulkan" >&2; exit 2 ;; esac
# Four FAMILIES of build directory, and the difference is not cosmetic.
#
#   build-<backend>-patched-<id>     carries setup/patches/hip-integrated-off.patch
#   build-<backend>-unpatched-<id>   deliberately does not
#   build-<backend>-unroll-<id>      patched, plus an extra compiler flag
#   build-<backend>-altsdk-<id>      patched, built against another ROCm
#
# An unpatched build is a legitimate SUBJECT — "is the corruption still in
# upstream master, and does PR #27311 fix it" cannot be answered on a binary
# that already suppresses the symptom, which is exactly how 27.08. came to
# measure a PR on top of its own competitor. What must never exist is an
# unpatched build that CLAIMS to be patched: on gfx1151 that shows up as wrong
# answers rather than as an error, and this whole script exists so that a
# build cannot lie about what is in it.
#
# So the families have different names and different stamps, only the patched
# one can be activated, and activate() refuses anything else by name.
#
# THE UNROLL FAMILY is named `<backend>-unroll` and NOT `<backend>-patched-unroll`,
# and that is load-bearing rather than taste. builds_of_backend() globs
# `build-$fam-*`, so any name starting with `<backend>-patched-` is swept up by
# every list and every prune of the patched family: the unroll build would show
# up there as a build whose id begins `unroll-`, sort into the ranking by
# built_at, and eventually be offered for deletion as a stale patched build.
# tests/test_buildllama.py asserts the absence of that collision, because it is
# a rename away and nothing else would notice.
FAMILY="$BACKEND-patched"
[ "$NOPATCH" = 1 ] && FAMILY="$BACKEND-unpatched"
[ "$UNROLL" = 1 ] && FAMILY="$BACKEND-unroll"
[ -n "$ROCMPATH" ] && FAMILY="$BACKEND-altsdk"
# Every family there is. Listing and pruning walk THIS, so a family added above
# and forgotten here becomes ~950 MB per build that nothing mentions and
# therefore nobody prunes — the failure the old hard-wired pair of names was
# one family away from.
ALL_FAMILIES="$BACKEND-patched $BACKEND-unpatched $BACKEND-unroll $BACKEND-altsdk"
STABLE="$SRC/build-$BACKEND-patched"

say()  { printf '%s\n' "$*"; }
step() { printf '\n== %s\n' "$*"; }
ok()   { printf '  \033[32m=\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '\n\033[31mABORT\033[0m %s\n' "$*" >&2; exit 2; }
run()  { if [ "$DRY" = 1 ]; then printf '  would: %s\n' "$*"; else "$@"; fi; }

git_() { git -C "$SRC" "$@"; }

# The registry, so --prune can ask which builds a PROFILE names. Not a fourth
# parser: setup/lib/models.sh is the one reader for setup/env/*.env, and the
# alternative here was a sed for LLAMA_BIN — which is exactly how this repo
# ended up with three disagreeing parsers for LLAMA_ARGS.
# shellcheck source=../lib/models.sh
. "$REPO/setup/lib/models.sh"

# Build directories that a profile PINS by name.
#
# in_use_by() only sees RUNNING processes, so a build that nothing is running
# out of looked deletable — and setup/env/flashnext.env pins
# build-rocm-patched-b10636-20-g035e22731 deliberately, because that PR moved
# 20 commits on its first day and a profile following the symlink would have
# changed backend under a rebuild. `--prune --yes` offered to delete it.
#
# Found 27.08. while making --prune family-aware. It is the same shape as
# everything else this repo keeps finding: the check ran, it was correct about
# what it checked, and the thing it needed to know was somewhere else.
pinned_builds() {
  local m bin
  for m in $(models_all); do
    bin="$(model_bin "$m")"
    case "$bin" in
      */build-*/bin/llama-server)
        basename "$(dirname "$(dirname "$bin")")" ;;
    esac
  done
}

# The build a directory holds, from its stamp — or from the binary itself for
# a directory built before stamps existed.
build_id_of() {   # $1 = build directory
  local d="$1"
  if [ -r "$d/.build-stamp" ]; then
    sed -n 's/^build_id=//p' "$d/.build-stamp" | head -1
  elif [ -x "$d/bin/llama-server" ]; then
    "$d/bin/llama-server" --version 2>&1 \
      | sed -n 's/.*commit \([0-9a-f]\{7,\}\).*/\1/p' | head -1
  fi
}

# Is any process running out of this directory? Overwriting a mapped .so is a
# SIGBUS in that process, so this is a hard stop, not a warning.
in_use_by() {     # $1 = build directory -> prints pids
  local d p
  d="$(readlink -f "$1")"
  for p in $(pgrep -x llama-server 2>/dev/null); do
    case "$(readlink -f "/proc/$p/exe" 2>/dev/null)" in
      "$d"/*) printf '%s ' "$p" ;;
    esac
  done
}

# `head -n "$KEEP"` looks harmless and is not. With KEEP=0 it closes the pipe
# at once, `sort` upstream takes SIGPIPE, `pipefail` turns that into a failed
# command substitution, and `set -e` acts on it — SILENTLY, because it happens
# inside a subshell. The effect was that --keep 0, the option that sounds the
# most dangerous, printed nothing and did nothing at all. Fourth bug in this
# one function; the last two were found by tests/test_prune.py rather than by
# reading output, which is the difference between a delete you can trust and
# one that has merely not gone wrong yet.
head_or_none() {
  if [ "$KEEP" -le 0 ]; then cat >/dev/null; else head -n "$KEEP"; fi
}

# $1 = family, default the one this invocation is about. A build directory
# that nothing lists is a build directory nobody prunes, and these are ~950 MB
# each — which is the reason --prune exists at all.
builds_of_backend() {
  local fam="${1:-$FAMILY}" d
  for d in "$SRC/build-$fam-"*; do
    [ -d "$d" ] || continue
    printf '%s\n' "${d##*/build-$fam-}"
  done
}

# --------------------------------------------------------------------------
# --prune: old builds are ~950 MB each and nothing was ever deleting them
# --------------------------------------------------------------------------
# Every build keeps its predecessor so a rollback is one --use away. Nothing
# ever removed the one before that, and by 26.08. there were three patched
# directories of ~950 MB, two of them the SAME build under two names.
#
# It deletes only with --yes, which is a deliberate departure from --dry-run
# everywhere else in this repo: a model switch is reversible in seconds, a
# deleted build is a fifteen-minute rebuild. So the default is to say what it
# would remove and remove nothing.
prune_builds() {
  local active="" id d keep_ids="" pids removed=0
  if [ -L "$STABLE" ]; then active="$(basename "$(readlink "$STABLE")")"; fi
  # Spaces both sides, so the `case` below cannot match a prefix of an id.
  # Declared and assigned separately: `local X=$(…)` makes local the command
  # whose status is reported, so a failing substitution reads as success.
  local PINNED
  PINNED=" $(pinned_builds | tr '\n' ' ')"

  # Newest first by built_at, ACTIVE EXCLUDED — it is kept unconditionally and
  # must not eat a rollback slot. KEEP is therefore "how many fallbacks", which
  # is the number a reader actually wants to reason about. A directory without
  # a stamp sorts last and is a candidate, which is right: it predates stamps.
  #
  # tr '\n' ' ' is not cosmetic. The membership test below is a `case` on
  # " $keep_ids ", and with newlines still in the list an id preceded by a
  # newline never matches the pattern " $id " — the first version of this
  # offered to delete a build it had just decided to keep.
  # `if`, not `[ … ] && continue`. Under `set -e` a failing test as the last
  # command of a loop body takes the whole command substitution down with it,
  # and the substitution is a subshell, so the failure is SILENT: keep_ids ends
  # up empty and prune prints nothing at all. Third bug in this function, and
  # the first one a test found rather than a person reading output.
  keep_ids="$(for id in $(builds_of_backend); do
                if [ "build-$FAMILY-$id" = "$active" ]; then continue; fi
                local at
                at="$(sed -n 's/^built_at=//p' "$SRC/build-$FAMILY-$id/.build-stamp" 2>/dev/null | head -1)"
                # An explicit sentinel, not an empty field. With an empty one
                # the line begins with a tab, and `sort -r` under a normal
                # locale ignores leading punctuation — the stampless build came
                # out FIRST, was kept as "newest", and the actual second-newest
                # was offered for deletion. Wrong by 944 MB. LC_ALL=C as well,
                # so the collation cannot change the answer again.
                printf '%s\t%s\n' "${at:-0000-00-00}" "$id"
              done | LC_ALL=C sort -r | cut -f2 | head_or_none | tr '\n' ' ')"


  say "builds in $SRC (family $FAMILY) — keeping the active one and $KEEP fallback(s)"
  say
  for id in $(builds_of_backend); do
    d="$SRC/build-$FAMILY-$id"
    if [ "build-$FAMILY-$id" = "$active" ]; then
      ok "keep  $id  (active)"; continue
    fi
    case " $keep_ids " in *" $id "*) ok "keep  $id  (recent)"; continue ;; esac
    pids="$(in_use_by "$d")"
    if [ -n "$pids" ]; then
      warn "keep  $id  — IN USE by pid(s) $pids; unmapping a live .so is a SIGBUS"
      continue
    fi
    case "$PINNED" in
      *" build-$FAMILY-$id "*)
        warn "keep  $id  — PINNED by a profile's LLAMA_BIN, not running now"
        continue ;;
    esac
    if [ "$YES" = 1 ]; then
      step "remove $id  ($(du -sh "$d" 2>/dev/null | cut -f1))"
      rm -rf "$d" && removed=$((removed + 1))
    else
      say "  would remove $id  ($(du -sh "$d" 2>/dev/null | cut -f1))"
      removed=$((removed + 1))
    fi
  done
  say
  # The OTHER family is not silently skipped. A directory nothing mentions is
  # a directory nobody prunes, and these are ~950 MB each.
  # EVERY other family, not just the one. While this was a hard-wired pair, a
  # third family would have been silently omitted here — and a family nothing
  # mentions is ~950 MB per build that nobody prunes.
  local other n
  for other in $ALL_FAMILIES; do
    # `if`, not `[ … ] && continue` — the shape this file carries two other
    # comments about, and not worth being clever with a third time.
    if [ "$other" = "$FAMILY" ]; then continue; fi
    n="$(builds_of_backend "$other" | grep -c . || true)"
    # An `if`, not `[ … ] && A && B`. Under set -e a false test as the last
    # command of a chain ends the function — the same shape this file already
    # carries two comments about.
    if [ "$n" != "0" ]; then
      say "  $n build(s) also exist in family $other. Prune those with:"
      case "$other" in
        *-unpatched) say "      bash setup/scripts/build-llama.sh --prune --no-patch" ;;
        *-unroll)    say "      bash setup/scripts/build-llama.sh --prune --unroll" ;;
        *)           say "      bash setup/scripts/build-llama.sh --prune" ;;
      esac
    fi
  done
  say
  if [ "$YES" = 1 ]; then
    say "Removed $removed build(s)."
  elif [ "$removed" = 0 ]; then
    say "Nothing to remove."
  else
    say "$removed build(s) would go. Add --yes to actually delete them."
  fi
}

# --------------------------------------------------------------------------
# --list / --use: no build, just look or switch
# --------------------------------------------------------------------------
show_list() {
  local active="" d id
  [ -L "$STABLE" ] && active="$(basename "$(readlink "$STABLE")")"
  say "builds in $SRC (backend $BACKEND)"
  say
  printf '  %-3s %-16s %-26s %s\n' "" ID BUILT UPSTREAM
  # Both families, always. --list is read-only, and a build that is not listed
  # is a build nobody knows they have — 950 MB at a time.
  local fam
  for fam in $ALL_FAMILIES; do
   [ -n "$(builds_of_backend "$fam")" ] || continue
   say "  [$fam]"
   for id in $(builds_of_backend "$fam"); do
    d="$SRC/build-$fam-$id"
    printf '  %-3s %-16s %-20s %s\n' \
      "$([ "build-$fam-$id" = "$active" ] && echo '->' || echo '')" \
      "$id" \
      "$(sed -n 's/^built_at=//p' "$d/.build-stamp" 2>/dev/null | head -1 || true)" \
      "$(sed -n 's/^upstream_commit=//p' "$d/.build-stamp" 2>/dev/null | head -1 || true)"
   done
  done
  say
  if [ -L "$STABLE" ]; then
    say "  $STABLE -> $(readlink "$STABLE")"
  elif [ -d "$STABLE" ]; then
    say "  $STABLE is still a real directory (id $(build_id_of "$STABLE" || echo '?'))"
    say "  — the first --activate turns it into a symlink and keeps it as a build."
  else
    say "  $STABLE does not exist yet"
  fi
  local pids; pids="$(in_use_by "$STABLE")"
  [ -n "$pids" ] && say "  in use right now by pid(s): $pids"
  return 0
}

# Point the stable name at a build. Atomic where it can be: the very first
# time, the real directory has to be renamed out of the way first, and that
# leaves a sub-millisecond window in which the path does not exist. A service
# restart landing exactly there would fail and retry — every later switch is a
# single atomic rename of the symlink.
activate() {      # $1 = build id
  local id="$1" target="$SRC/build-$BACKEND-patched-$1"
  [ -d "$target" ] || die "no such build: $target
    bash setup/scripts/build-llama.sh --list"
  [ -x "$target/bin/llama-server" ] || die "$target has no bin/llama-server — the build did not finish"
  # Belt and braces beside the name. activate() only ever looks in the patched
  # family, so an unpatched id cannot be found here — but a directory can be
  # renamed, and the stamp is the thing that says what is actually in it.
  if [ "$(sed -n 's/^patched=//p' "$target/.build-stamp" 2>/dev/null | head -1)" = "no" ]; then
    die "$target says patched=no in its .build-stamp.

    That binary has no setup/patches/hip-integrated-off.patch. Serving it does
    not fail — it returns degenerate output once a second slot is used. It was
    built to be MEASURED, not to be served."
  fi
  if [ -L "$STABLE" ]; then
    :
  elif [ -d "$STABLE" ]; then
    # Renaming a build directory invalidates an ABSOLUTE RUNPATH. Builds made
    # before CMAKE_BUILD_RPATH_USE_ORIGIN have one, and it points at $STABLE —
    # which keeps working, but only for as long as they ARE $STABLE. Say so
    # rather than let a rollback fail at 3 a.m.
    if [ -x "$STABLE/bin/llama-server" ] \
       && [ "$(readelf -d "$STABLE/bin/llama-server" 2>/dev/null | grep -c 'ORIGIN')" = 0 ]; then
      warn "the build being moved aside has an absolute RUNPATH (built before"
      warn "CMAKE_BUILD_RPATH_USE_ORIGIN). It will only run again while it is"
      warn "the active build — which is what --use makes it. Rebuild it to be safe."
    fi
    local old; old="$(build_id_of "$STABLE")"
    [ -n "$old" ] || old="previous"
    # A COUNTER, not $(date +%s). If date is not on PATH the substitution is
    # empty and the directory ends in a bare '-' — which is exactly what
    # happened here on 26.08.: build-rocm-patched-b10631- sat next to
    # build-rocm-patched-b10631, same stamp, same build id, 944 MB twice, and
    # `--use b10631` had two candidates. A name that depends on an external
    # command can be empty; a counter cannot.
    if [ -e "$SRC/build-$BACKEND-patched-$old" ]; then
      local n=2
      while [ -e "$SRC/build-$BACKEND-patched-$old.$n" ]; do n=$((n + 1)); done
      old="$old.$n"
    fi
    step "keeping the existing real directory as build-$BACKEND-patched-$old"
    run mv "$STABLE" "$SRC/build-$BACKEND-patched-$old"
    if [ "$DRY" = 0 ] && [ ! -r "$SRC/build-$BACKEND-patched-$old/.build-stamp" ]; then
      printf 'build_id=%s\nbackend=%s\nnote=renamed by build-llama.sh; predates stamps\n' \
        "$old" "$BACKEND" > "$SRC/build-$BACKEND-patched-$old/.build-stamp"
    fi
  fi
  # ln -sfn on an existing SYMLINK replaces it atomically via rename(2).
  run ln -sfn "build-$BACKEND-patched-$id" "$STABLE"
  ok "$STABLE -> build-$BACKEND-patched-$id"
  # Run it THROUGH THE SYMLINK — the exact path the unit will exec. Verifying
  # the binary in its build directory is not the same statement, and on 26.08.
  # the difference was the whole incident: the build verified fine where it
  # was built, the directory was renamed afterwards, its absolute RUNPATH went
  # stale, and the service died on the next restart with a linker error.
  if [ "$DRY" = 0 ] && ! "$STABLE/bin/llama-server" --version >/dev/null 2>&1; then
    ERR="$("$STABLE/bin/llama-server" --version 2>&1 | head -2)"
    die "the binary does not run through $STABLE:

$(printf '%s' "$ERR" | sed 's/^/      /')

    Nothing has been switched back automatically — decide with:
      bash setup/scripts/build-llama.sh --list
      bash setup/scripts/build-llama.sh --use <a build that works>"
  fi
  [ "$DRY" = 0 ] && ok "runs through the symlink: $("$STABLE/bin/llama-server" --version 2>&1 | head -1)"
  say
  say "VERIFY IT, do not trust it. Since 28.08. the gfx1151 corruption has a"
  say "deterministic reproducer, so a build is checkable in about a minute"
  say "instead of being hoped about — 10 of 10 corrupt without the patch, 0 of"
  say "10 with it, on the same upstream commit:"
  say
  # $id, not $BUILD_ID. This function is reached from BOTH paths and only the
  # build path sets BUILD_ID, so `--use` died here under `set -u` AFTER moving
  # the symlink correctly: the rollback worked and exited 2 saying it had not.
  # Found 30.08.2026 by using it; the test that would have caught it existed
  # only for the refusal case.
  say "    python3 bench/suites/slot-corruption.py par-two-prefixes \\"
  say "        --binary $id --starts 3"
  say
  say "  It runs on a side server (port 8081) and leaves production alone."
  say
  say "The RUNNING server keeps the libraries it already mapped — it does not"
  say "change until it is restarted:"
  say
  say "    systemctl --user restart llama-user@\$(bash setup/lib/models.sh active)"
  say "    bash setup/check.sh"
  say
  say "Back:  bash setup/scripts/build-llama.sh --use <id>   (--list shows them)"
}

# --list and --prune work on BUILD DIRECTORIES and never touch the source, so
# they must not require a checkout. That coupling had one visible cost and one
# invisible one: it made --prune untestable, and --prune is the only thing in
# this repo that deletes. Two bugs had already been found in it by reading its
# output — a keep-list that never matched, and a sort that called the oldest
# build the newest. tests/test_prune.py runs it against fake builds now.
if [ "$LIST" = 1 ]; then show_list; exit 0; fi
if [ "$PRUNE" = 1 ]; then prune_builds; exit 0; fi
[ -d "$SRC/.git" ] || die "no llama.cpp checkout at $SRC (set LLAMA_SRC)"
if [ -n "$USE" ]; then activate "$USE"; exit 0; fi

# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------
step "0/6 preflight"
if [ -n "$ROCMPATH" ]; then
  # PROVEN PRESENT, not assumed. Without this check a wrong --rocm-path falls
  # back to the system toolchain, the build succeeds, and the stamp claims the
  # other SDK — a lie every measurement downstream would inherit.
  [ -d "$ROCMPATH" ] || die "--rocm-path: no such directory: $ROCMPATH"
  ALT_CLANG="$ROCMPATH/llvm/bin/clang"
  [ -x "$ALT_CLANG" ] || die "--rocm-path: no clang at $ALT_CLANG
    A ROCm tarball unpacks with its compiler under llvm/bin. Without it this
    build would silently use the system toolchain and be stamped as this SDK's."
  ALT_VERSION="$(cat "$ROCMPATH/.info/version" 2>/dev/null | head -1)"
  [ "$ACTIVATE" = 0 ] || die "--rocm-path and --activate cannot be combined.

    The serving binary must run against the ROCm the system actually has.
    This build's libraries are found only with LD_LIBRARY_PATH pointing into
    $ROCMPATH — and its libamdhip64 carries the SAME soname as the system one,
    so without that path it loads the system's and nothing says so."
  ok "building against ROCm ${ALT_VERSION:-unknown} at $ROCMPATH — not activatable"
fi
if [ "$UNROLL" = 1 ]; then
  # Same rule as --no-patch, for the same reason: a build that differs from
  # the serving one is a SUBJECT. It is reachable for a measurement through
  # `bench/sideserver.py --bin`, which puts production back afterwards and
  # arms a dead man's switch while it is away; the symlink offers neither.
  [ "$ACTIVATE" = 0 ] || die "--unroll and --activate cannot be combined.

    The symlink $STABLE is what the model unit execs, and an unroll build is
    there to be COMPARED against it, not to replace it unmeasured. Run it
    beside production instead:
      python3 bench/sideserver.py --bin \$HOME/llama.cpp/build-$FAMILY-<id>/bin/llama-server \\
          --env setup/env/qwen38.env --stop llama-user@qwen38 -- <command>"
  # Two variables at once is the mistake llama.cpp#19984 already contains: it
  # compares self-built-with-flag against an official binary, so where a build
  # without the flag lands is exactly the cell nobody has. Refusing this
  # combination is what keeps our own answer to one variable.
  [ "$NOPATCH" = 0 ] || die "--unroll and --no-patch cannot be combined.

    That build would differ from the serving binary in TWO ways, and the
    question the flag is being measured for could not be attributed to it."
  ok "building WITH -mllvm --amdgpu-unroll-threshold-local=600, into the $FAMILY family — not activatable"
fi
if [ "$NOPATCH" = 1 ]; then
  # --no-patch and --activate are mutually exclusive, and this is the hard
  # stop of the whole feature. The stable symlink is what the production unit
  # execs; pointing it at a binary without the gfx1151 fix does not fail, it
  # returns wrong answers once a second slot is used. An unpatched build is a
  # subject to measure, never a thing to serve.
  [ "$ACTIVATE" = 0 ] || die "--no-patch and --activate cannot be combined.

    The symlink $STABLE is what the model unit execs. A binary without
    setup/patches/hip-integrated-off.patch does not fail there — it returns
    degenerate output once a second slot is used, silently. Build it, measure
    it with
      python3 bench/suites/restore-safety.py --binary <build id>
    and leave the serving binary alone."
  ok "building WITHOUT the patch, into the $FAMILY family — not activatable"
else
git_ rev-parse --verify -q "$PATCH_BRANCH" >/dev/null \
  || die "branch '$PATCH_BRANCH' does not exist in $SRC.

    The gfx1151 patch has to be a COMMIT, not a working-tree diff, or an
    update loses it silently. Create it once:

      cd $SRC
      git checkout -b $PATCH_BRANCH
      git apply $REPO/setup/patches/hip-integrated-off.patch
      git commit -am 'gfx1151: do not trust prop.integrated on HIP'"

# grep -c, NOT grep -q. This script runs with 'set -o pipefail'; grep -q stops
# at the first match, git gets SIGPIPE, and pipefail reports the pipeline as
# failed — a hit would look like a miss. setup/check.sh carries the same note
# about journalctl, and this is the second time it has bitten.
for entry in "${PATCH_MARKERS[@]}"; do
  marker_file="${entry%%:*}"
  marker_text="${entry#*:}"
  if [ "$(git_ show "$PATCH_BRANCH:$marker_file" | grep -c "$marker_text")" = 0 ]; then
    die "branch '$PATCH_BRANCH' does not carry a patch marker.
    Expected in $marker_file:  $marker_text
    See setup/patches/README.md."
  fi
done
ok "patch branch $PATCH_BRANCH carries all ${#PATCH_MARKERS[@]} markers"
fi

[ -z "$(git_ status --porcelain)" ] \
  || die "$SRC has uncommitted changes. A rebase would take them along or
    refuse. Commit them onto $PATCH_BRANCH or stash them:
$(git_ status --short | sed 's/^/      /')"
ok "working tree clean"

# An if, not 'A || B && C': that parses as '(A || B) && C', and under set -e
# the failing list would end the script when NO rebase is in progress — the
# exact opposite of what it is guarding.
if [ -d "$SRC/.git/rebase-merge" ] || [ -d "$SRC/.git/rebase-apply" ]; then
  die "a rebase is already in progress in $SRC — finish or 'git rebase --abort' it first"
fi

case "$BACKEND" in
  rocm)   [ -x /usr/lib64/rocm/llvm/bin/clang ] \
            || warn "the ROCm clang is not where the recipe expects it (/usr/lib64/rocm/llvm/bin/clang)" ;;
esac
command -v cmake >/dev/null || die "cmake is not installed"
ok "jobs: $JOBS of $(nproc) — the rest stays for the running model"

# --------------------------------------------------------------------------
step "1/6 fetch"
if [ -z "$REF" ]; then REF=origin/master; fi
case "$REF" in
  pr/*)
    PRNUM="${REF#pr/}"
    run git -C "$SRC" fetch origin "pull/$PRNUM/head:pr-$PRNUM" --force
    REF="pr-$PRNUM" ;;
  *)
    run git -C "$SRC" fetch origin --tags ;;
esac
if [ "$DRY" = 0 ]; then
  git_ rev-parse --verify -q "$REF^{commit}" >/dev/null \
    || die "'$REF' is not a commit in $SRC"
  TARGET="$(git_ rev-parse "$REF^{commit}")"
  # tr -c turns the trailing NEWLINE into a '-' too, which produced ids like
  # "b10631-" on the first run. Delete the newline first, then sanitise, then
  # trim any dashes the sanitising left at the ends.
  BUILD_ID="$(git_ describe --tags --always "$TARGET" 2>/dev/null \
              | tr -d '\n' | tr -c 'A-Za-z0-9._-' '-' | sed 's/^-*//; s/-*$//')"
  [ -n "$BUILD_ID" ] || die "could not derive a build id for $TARGET"
  ok "$REF -> $(git_ log --oneline -1 "$TARGET")"
  ok "build id: $BUILD_ID"
else
  TARGET="$REF"; BUILD_ID="dry-run"
fi
BUILD_DIR="$SRC/build-$FAMILY-$BUILD_ID"

# The one thing that must never happen: building into the directory the
# running server has mapped.
PIDS="$(in_use_by "$BUILD_DIR")"
[ -z "$PIDS" ] || die "pid(s) $PIDS are running out of $BUILD_DIR.
    Overwriting a mapped .so is a SIGBUS in that process, not a build error.
    Stop the service first, or build a different ref."

# --------------------------------------------------------------------------
if [ "$NOPATCH" = 1 ]; then
  step "2/6 check out $BUILD_ID WITHOUT the patch"
  if [ "$DRY" = 0 ]; then
    # Remembered so the tree can be put back. An unpatched checkout LEFT
    # BEHIND is a trap: setup/check.sh reads the source, not the running
    # binary, and says "THE PATCH IS GONE" — which is true of the tree and
    # false of the server. Found 28.08. after a night of --no-patch builds,
    # by check.sh, which was right.
    BEFORE="$(git_ rev-parse --abbrev-ref HEAD)"
    [ "$BEFORE" = "HEAD" ] && BEFORE="$(git_ rev-parse HEAD)"
    git_ checkout -q --detach "$TARGET"
    # The MIRROR of the patched check, and it has to be here rather than
    # assumed. "I passed --no-patch" is an intention; "the marker is not in
    # the source I am about to compile" is the fact. If upstream ever adopts
    # the same change, this fires and says so — which is good news and must
    # not be reported as an unpatched build.
    for entry in "${PATCH_MARKERS[@]}"; do
      marker_file="${entry%%:*}"
      marker_text="${entry#*:}"
      [ -f "$SRC/$marker_file" ] || continue
      if [ "$(grep -c "$marker_text" "$SRC/$marker_file")" != "0" ]; then
        die "a marker is PRESENT in $BUILD_ID and --no-patch was asked for:
      $marker_file:  $marker_text
    Either the checkout did not take, or upstream now does this itself. The
    second would be good news — setup/patches/README.md, 'When can this be
    dropped?' — but it is not an unpatched build either way."
      fi
    done
    ok "markers absent — this is upstream $BUILD_ID as it stands"
  else
    say "  would: git checkout --detach $TARGET, then verify the marker is ABSENT"
  fi
else
step "2/6 replay the patch onto $BUILD_ID"
# WHAT IS ABOUT TO BE REPLAYED, looked at before the rebase rather than after
# the report. `git rebase <target>` replays EVERY commit in target..HEAD, not
# "the patch" — and those are the same thing only while the patch branch is
# what its name says.
#
# On 27.08.2026 it was not, and nothing said so. gfx1151-patched had been
# rebased onto PR #27742 on the 26th to build Flash-Next, so it carried 26
# commits over origin/master; and PR #27311's base sat 65 commits behind that
# master. The documented `--ref pr/27311` would therefore have replayed 91
# commits — and it does not fail. Tried in a throwaway worktree first: the
# rebase SUCCEEDS, exit 0, no conflict, and the build comes out stamped
# `upstream_ref=pr-27311` while containing an entire unmerged 180B-model PR
# and 65 extra master commits. The measurement would have been attributed to
# the wrong change, and every check downstream would have agreed with it.
#
# A wrong answer that exits 0 is the defect this repository keeps finding.
# The count is cheap and it is the whole guard.
REPLAY=""
if git_ rev-parse --verify -q "$TARGET^{commit}" >/dev/null 2>&1; then
  REPLAY="$(git_ rev-list --count "$TARGET..$PATCH_BRANCH")"
fi
if [ -n "$REPLAY" ] && [ "$REPLAY" -gt "$MAX_REPLAY" ]; then
  die "the patch branch '$PATCH_BRANCH' carries $REPLAY commits over $BUILD_ID,
    and a rebase would replay ALL of them into this build. At most
    $MAX_REPLAY is expected — the gfx1151 patch is one commit.

$(git_ log --oneline -8 "$TARGET..$PATCH_BRANCH" | sed 's/^/      /')
$([ "$REPLAY" -gt 8 ] && printf '      … and %s more\n' "$((REPLAY - 8))")

    This is not a failure to work around; it is the answer to a question
    nobody asked. Two ways on, and the first is almost always the right one:

      * Build the ref plus the patch and NOTHING else, on a branch of its
        own — which is also what makes it a one-variable comparison:

            git -C $SRC branch -f mybranch <the ref you want>
            git -C $SRC checkout mybranch
            git -C $SRC cherry-pick \$(git -C $SRC rev-parse $PATCH_BRANCH)
            PATCH_BRANCH=mybranch bash setup/scripts/build-llama.sh --ref mybranch

      * Or say you meant it:  MAX_REPLAY=$REPLAY bash setup/scripts/build-llama.sh …"
fi
[ -n "$REPLAY" ] && ok "$REPLAY commit(s) to replay onto $BUILD_ID"
if [ "$DRY" = 0 ]; then
  BEFORE="$(git_ rev-parse --abbrev-ref HEAD)"
  git_ checkout -q "$PATCH_BRANCH"
  if ! git_ rebase -q "$TARGET"; then
    git_ rebase --abort || true
    git_ checkout -q "$BEFORE" || true
    die "the patch does not apply to $BUILD_ID any more.

    That is a RESULT, not just a failure: upstream has touched the same code.
    Look at what changed — llama.cpp #27572 / #27579 may have been fixed, in
    which case the patch and '-np 1' can both go, after remeasuring with
      python3 bench/suites/np2-candidates.py rocm+cram+mmproj"
  fi
  grep -q "${PATCH_MARKERS[0]#*:}" "$SRC/${PATCH_MARKERS[0]%%:*}" \
    || die "the rebase succeeded but the marker is GONE from the source.
    Most likely the commit became empty because upstream now does the same
    thing. Check by hand — and if so, this is good news: setup/patches/README.md,
    'When can this be dropped?'"
  ok "patch replayed, marker present"
else
  say "  would: git checkout $PATCH_BRANCH && git rebase $TARGET"
fi
fi

# --------------------------------------------------------------------------
step "3/6 configure $BUILD_DIR"
# The flags come from setup/patches/README.md verbatim. Do NOT add
# GGML_HIP_ROCWMMA_FATTN: measured as a 41 % prefill regression on gfx1151,
# and the option was removed from llama.cpp anyway — in fa72aeccb "HIP: remove
# rocWMMA FlashAttention" (#26046), where it had lived in ggml/CMakeLists.txt,
# ggml/src/ggml-hip/CMakeLists.txt and ggml-cuda/vendors/hip.h. The commit and
# the file are here so that setup/defects.json can PROBE for it rather than
# take this sentence's word for it.
#
# CMAKE_BUILD_RPATH_USE_ORIGIN is not cosmetic and was learned the hard way on
# 26.08. Without it CMake bakes the ABSOLUTE build path into the binary as its
# RUNPATH — llama-server is a 12 KB executable that finds libllama-server-impl.so
# and seven more that way. A build directory that is later renamed is then a
# binary that cannot start:
#
#     llama-server: error while loading shared libraries:
#     libllama-server-impl.so: cannot open shared object file
#
# and because Restart=on-failure gives up after three tries in fifteen seconds,
# the model server simply stays down. With $ORIGIN the libraries are found
# relative to the binary and a build directory can be moved, renamed or reached
# through the symlink.
# ONE place decides the compiler. Appending an override later left
# -DCMAKE_HIP_COMPILER in the line TWICE — cmake takes the last, so it worked,
# and the stamp then showed both and settled nothing about which was used.
HIPCC="/usr/lib64/rocm/llvm/bin/clang"
[ -n "$ROCMPATH" ] && HIPCC="$ROCMPATH/llvm/bin/clang"
case "$BACKEND" in
  rocm)
    CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release
                -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
                -DGGML_HIP=ON -DGPU_TARGETS=gfx1151
                -DGGML_HIP_GRAPHS=ON -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON
                -DBUILD_SHARED_LIBS=ON
                "-DCMAKE_HIP_COMPILER=$HIPCC"
                -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF) ;;
  vulkan)
    CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release
                -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
                -DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=ON
                -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF) ;;
esac
# THE UNROLL FLAG. llama.cpp#19984 measures, on this exact hardware, a prefill
# collapse it attributes to a loop-unrolling regression in ROCm 7+, and works
# around it with this. Whether it changes anything HERE is the open question —
# the issue's own comparison is self-built-with-flag against an official
# binary, two variables, so the cell for a self-built build WITHOUT it does not
# exist anywhere yet.
#
# One array element, not two: the value contains a space, and split across
# elements cmake receives `-mllvm` as a flag and the threshold as a source file.
# HIP compile flags are COLLECTED, then passed once. Two separate appends of
# -DCMAKE_HIP_FLAGS would leave the second overwriting the first, silently, so
# --unroll and --rocm-path together would have lost the unroll flag.
HIP_FLAGS=""
[ "$UNROLL" = 1 ] && HIP_FLAGS="$HIP_FLAGS -mllvm --amdgpu-unroll-threshold-local=600"
# clang's OWN --rocm-path, and it is not the same thing as CMAKE_PREFIX_PATH.
# Without it the compiler keeps finding /usr/include/hip — the SYSTEM headers —
# and a ROCm 10.1 clang against ROCm 7.1 headers dies on
# `use of undeclared identifier '__ocml_log2_f32'`. Measured 31.08.2026; the
# first attempt at this build failed exactly there.
# BOTH are needed, and they do different jobs. --rocm-path points clang at the
# SDK's device libraries; -isystem is what actually decides which hip/*.h gets
# included. Without the second, /usr/include wins because it is a default
# search path and the hip cmake target's include directories never reach the
# HIP compile line — flags.make carried exactly one -I, and it was ggml's own.
# The symptom is not subtle but it is misleading: ROCm 10.1's clang against
# ROCm 7.1's headers dies on `use of undeclared identifier '__ocml_log2_f32'`,
# a name the 7.1 header uses and the 10.1 header does not. Measured 31.08.2026.
[ -n "$ROCMPATH" ] && HIP_FLAGS="$HIP_FLAGS --rocm-path=$ROCMPATH -isystem $ROCMPATH/include"
[ -n "$HIP_FLAGS" ] && CMAKE_ARGS+=("-DCMAKE_HIP_FLAGS=${HIP_FLAGS# }")
# ANOTHER ROCm. The compiler comes from the SDK, and CMAKE_PREFIX_PATH is what
# lets find_package(hip) resolve there instead of against /usr. ROCM_PATH is
# exported as well because parts of the HIP cmake package still read it.
if [ -n "$ROCMPATH" ]; then
  CMAKE_ARGS+=("-DCMAKE_PREFIX_PATH=$ROCMPATH"
               "-DROCM_PATH=$ROCMPATH")
  export ROCM_PATH="$ROCMPATH"
  export PATH="$ROCMPATH/bin:$PATH"
fi
run cmake -S "$SRC" -B "$BUILD_DIR" "${CMAKE_ARGS[@]}"

step "4/6 build (nice, so the running model keeps its CPU)"
run nice -n 10 cmake --build "$BUILD_DIR" --target llama-server -j "$JOBS"
# llama-bench is a SECOND target, never a replacement: the server is what this
# stack serves and what every other suite measures. It is built only on
# request because it is what llama.cpp#19984 measured with, and reproducing
# somebody else's number needs their instrument rather than an equivalent one.
[ "$WITHBENCH" = 1 ] && \
  run nice -n 10 cmake --build "$BUILD_DIR" --target llama-bench -j "$JOBS"

# --------------------------------------------------------------------------
step "5/6 verify"
if [ "$DRY" = 0 ]; then
  BIN="$BUILD_DIR/bin/llama-server"
  [ -x "$BIN" ] || die "the build produced no $BIN"
  VER="$("$BIN" --version 2>&1 | head -1)"
  ok "$VER"
  # llama.cpp stamps the commit it was built from, and that is checked against
  # the commit that SHOULD be in it. For a patched build that is the tip of the
  # patch branch — if it is the upstream commit instead, the build picked up a
  # source tree without the patch, the failure this pipeline exists to make
  # impossible. For an unpatched build the expected commit is the target
  # itself, and the same check catches a stale tree just as well.
  if [ "$NOPATCH" = 1 ]; then
    WANT="$(git_ rev-parse --short=9 "$TARGET")"; WHAT="unpatched upstream commit"
  else
    WANT="$(git_ rev-parse --short=9 "$PATCH_BRANCH")"; WHAT="patched commit"
  fi
  case "$VER" in
    *"$WANT"*) ok "the binary was built from the $WHAT $WANT" ;;
    *) die "the binary reports $VER but the expected $WHAT is $WANT.
    The build did not come from the source it claims. Do not use it." ;;
  esac
  # The stamp says which family this is, in a field of its own. A reader — and
  # bench/suites/restore-safety.py, which records provenance from here — must
  # not have to infer "patched" from a directory name that somebody could
  # rename.
  if [ "$NOPATCH" = 1 ]; then
    PATCH_COMMIT=none; PATCH_BRANCH_FIELD=none; PATCHED=no
  else
    PATCH_COMMIT="$(git_ rev-parse "$PATCH_BRANCH")"; PATCH_BRANCH_FIELD="$PATCH_BRANCH"; PATCHED=yes
  fi
  printf 'build_id=%s\nbackend=%s\nfamily=%s\npatched=%s\nupstream_ref=%s\nupstream_commit=%s\npatch_commit=%s\npatch_branch=%s\nrocm_path=%s\nrocm_version=%s\nbuilt_at=%s\nversion=%s\ncmake=%s\n' \
    "$BUILD_ID" "$BACKEND" "$FAMILY" "$PATCHED" "$REF" "$(git_ rev-parse "$TARGET")" \
    "$PATCH_COMMIT" "$PATCH_BRANCH_FIELD" \
    "${ROCMPATH:-none}" "${ALT_VERSION:-none}" \
    "$(date -Is)" "$VER" "${CMAKE_ARGS[*]}" > "$BUILD_DIR/.build-stamp"
  ok "stamp written: $BUILD_DIR/.build-stamp"
fi

# --------------------------------------------------------------------------
step "6/6 activate"
if [ "$NOPATCH" = 1 ] && [ "$DRY" = 0 ] && [ -n "${BEFORE:-}" ]; then
  # Put the tree back where it was found. The build directory keeps the
  # unpatched build; the SOURCE goes back to carrying the patch, so the next
  # reader — human or check.sh — is not told the patch was lost.
  if git_ checkout -q "$BEFORE" 2>/dev/null; then
    ok "source tree back on $BEFORE (the build keeps its own copy)"
  else
    warn "could not put the source tree back on $BEFORE — it is left on"
    warn "$BUILD_ID, which has NO patch. check.sh will say so."
  fi
fi

if [ "$NOPATCH" = 1 ]; then
  say "  NOT activatable, by design. This build has no gfx1151 patch; it is a"
  say "  subject to measure, not a binary to serve:"
  say
  say "      python3 bench/suites/restore-safety.py --binary $BUILD_ID"
  say
  say "  The serving binary is untouched: $STABLE"
elif [ "$ACTIVATE" = 1 ]; then
  activate "$BUILD_ID"
else
  say "  not activated (pass --activate, or later:)"
  say "      bash setup/scripts/build-llama.sh --use $BUILD_ID"
  say
  say "  The profiles keep pointing at $STABLE until then, so nothing"
  say "  about the running stack has changed."
fi
