#!/usr/bin/env bash
# build-qwentts.sh — build qwentts.cpp (Qwen3-TTS, GGML) into a PINNED dir.
#
#   bash setup/scripts/build-qwentts.sh          build the checkout (Vulkan)
#   bash setup/scripts/build-qwentts.sh --list   what exists, with stamps
#
# Same shape as build-sd.sh, same two decisions: NO stable symlink (workload
# profiles pin the full build directory — a moved symlink turns measured
# figures into claims), and Vulkan FIRST as the hypothesis to measure, not a
# preference. The upstream buildvulkan.sh wipes and reuses one `build/`
# directory, which is exactly the unpinned build this repo does not do.
set -euo pipefail

SRC="${QWENTTS_SRC:-$HOME/qwentts.cpp}"

say() { printf '%s\n' "$*"; }

if [ "${1:-}" = "--list" ]; then
  for d in "$SRC"/build-*; do
    [ -d "$d" ] || continue
    # `|| true`: under set -euo pipefail, sed on a MISSING stamp kills the
    # substitution and with it the whole listing — and the stamp is written
    # last, so an aborted build creates exactly the state that then broke
    # the tool for inspecting builds (measured, review 01.09.2026).
    id="$(sed -n 's/^build_id=//p' "$d/.build-stamp" 2>/dev/null | head -1 || true)"
    at="$(sed -n 's/^built_at=//p' "$d/.build-stamp" 2>/dev/null | head -1 || true)"
    say "$(basename "$d")  ${id:-no-stamp}  ${at:-}"
  done
  exit 0
fi

[ -d "$SRC/.git" ] || {
  say "no qwentts.cpp checkout at $SRC" >&2
  say "  git clone --recurse-submodules https://github.com/ServeurpersoCom/qwentts.cpp.git $SRC" >&2
  exit 1
}

# --ref <commit>: build the EXACT commit a workload profile pins — see the
# twin comment in build-sd.sh (architecture review, 01.09.2026).
if [ "${1:-}" = "--ref" ]; then
  REF="${2:?--ref needs a commit, tag or branch}"
  # checkout FETCH_HEAD after a successful fetch, NOT the ref name: a
  # LOCAL branch of the same name may lag origin, and `--ref master` would
  # silently build its stale tip (re-review, 01.09.2026). A ref the remote
  # does not serve (a bare commit on some servers) falls back to a plain
  # local checkout, which is exactly right for SHA pins already on disk.
  if git -C "$SRC" fetch --quiet origin "$REF" 2>/dev/null; then
    git -C "$SRC" checkout --quiet FETCH_HEAD
  else
    # The fallback is for SHA pins already on disk, and ONLY for them: a
    # BRANCH name whose fetch failed (network down, origin unreachable)
    # would otherwise resolve to the stale local tip — the exact hole the
    # FETCH_HEAD line closes, reopened on the offline path (review,
    # 01.09.2026; the twin comment sits in build-sd.sh).
    if git -C "$SRC" show-ref --verify --quiet "refs/heads/$REF"; then
      say "REFUSING: the fetch of '$REF' failed and a LOCAL branch of that" >&2
      say "name exists — checking it out could silently build a stale tip." >&2
      say "Check the network, or pass the exact commit id instead." >&2
      exit 1
    fi
    git -C "$SRC" checkout --quiet "$REF" || {
      say "cannot check out $REF — a shallow clone may not carry it; try:" >&2
      say "  git -C $SRC fetch --unshallow origin" >&2
      exit 1
    }
  fi
  # A failed submodule sync must FAIL the build — see build-sd.sh's twin
  # comment: `|| true` stamped hybrid binaries as the new commit (review,
  # 01.09.2026).
  git -C "$SRC" submodule --quiet update --init --recursive || {
    say "REFUSING: submodule update failed after the checkout — building" >&2
    say "now would pair the new tree with the OLD submodule state and" >&2
    say "stamp the hybrid as $REF." >&2
    exit 1
  }
fi

COMMIT="$(git -C "$SRC" rev-parse --short=9 HEAD)"
BUILD="$SRC/build-vulkan-$COMMIT"

# qwen-tts reads its text from STDIN and nothing else; the workload contract
# (bench/sideserver.py smoke, audiobench) appends `-p TEXT -o OUT`. The
# adapter lives BESIDE the binary — a machine artifact versioned with the
# build, which a profile can name through @HOME@ — and translates one flag.
write_adapter() {
  cat > "$1/qwen-tts-p" <<'ADAPTER'
#!/usr/bin/env bash
# -p TEXT -> stdin adapter for qwen-tts; every other argument passes through.
set -euo pipefail
text=""
args=()
while [ $# -gt 0 ]; do
  case "$1" in
    -p) text="$2"; shift 2 ;;
    *)  args+=("$1"); shift ;;
  esac
done
printf '%s\n' "$text" | "$(dirname "$0")/qwen-tts" "${args[@]}"
ADAPTER
  chmod +x "$1/qwen-tts-p"
}

# The adapter goes BESIDE the binary in BOTH branches. The first version
# wrote it into the build root on a rerun while the build branch wrote it
# next to the binary — a bin/-layout build then got a root-level DECOY whose
# relative ./qwen-tts does not exist, and the profile pinning it would die
# with ENOENT at run time (review, 01.09.2026).
EXISTING=""
[ -x "$BUILD/qwen-tts" ] && EXISTING="$BUILD/qwen-tts"
[ -x "$BUILD/bin/qwen-tts" ] && EXISTING="$BUILD/bin/qwen-tts"
if [ -n "$EXISTING" ]; then
  [ -x "$(dirname "$EXISTING")/qwen-tts-p" ] || write_adapter "$(dirname "$EXISTING")"
  say "already built: $BUILD"
  exit 0
fi

# Same missing-header story as build-sd.sh (01.09.2026): ggml-vulkan needs
# <spirv/unified1/spirv.hpp>, Fedora ships it in spirv-headers-devel, and a
# checkout of KhronosGroup/SPIRV-Headers in $HOME covers it without root.
EXTRA_CXX=""
if [ ! -e /usr/include/spirv/unified1/spirv.hpp ]; then
  if [ -e "$HOME/SPIRV-Headers/include/spirv/unified1/spirv.hpp" ]; then
    EXTRA_CXX="-isystem $HOME/SPIRV-Headers/include"
    say "spirv-headers-devel is not installed — using ~/SPIRV-Headers instead"
  else
    say "missing <spirv/unified1/spirv.hpp>: install spirv-headers-devel, or" >&2
    say "  git clone --depth 1 https://github.com/KhronosGroup/SPIRV-Headers $HOME/SPIRV-Headers" >&2
    exit 1
  fi
fi

say "building qwentts.cpp $COMMIT (Vulkan) into $BUILD"
cmake -S "$SRC" -B "$BUILD" -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release \
      ${EXTRA_CXX:+-DCMAKE_CXX_FLAGS="$EXTRA_CXX"}
cmake --build "$BUILD" --config Release -j"$(nproc)"

BIN="$BUILD/qwen-tts"
[ -x "$BIN" ] || BIN="$BUILD/bin/qwen-tts"
[ -x "$BIN" ] || { say "build produced no qwen-tts binary" >&2; exit 1; }
write_adapter "$(dirname "$BIN")"

{
  printf 'build_id=vulkan-%s\n' "$COMMIT"
  printf 'commit=%s\n' "$(git -C "$SRC" rev-parse HEAD)"
  printf 'backend=vulkan\n'
  printf 'built_at=%s\n' "$(date +%Y-%m-%d_%H%M)"
  printf 'cmake_flags=-DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release\n'
} > "$BUILD/.build-stamp"

say "done: $BIN"
say "stamp: $(tr '\n' ' ' < "$BUILD/.build-stamp")"
