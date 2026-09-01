#!/usr/bin/env bash
# build-sd.sh — build stable-diffusion.cpp into a PINNED directory.
#
#   bash setup/scripts/build-sd.sh              build the checkout (Vulkan)
#   bash setup/scripts/build-sd.sh --list       what exists, with stamps
#
# Layout it creates:
#
#     ~/stable-diffusion.cpp/build-vulkan-<commit>/     a build, with .build-stamp
#
# Two deliberate differences from build-llama.sh:
#
#   * NO stable symlink. Workload profiles (setup/workloads/*.env) pin their
#     WORKLOAD_CMD to the full build directory — the same discipline
#     flashnext.env applies to LLAMA_BIN, and for the same reason: a symlink
#     that moves turns every figure measured on the old build back into a
#     claim about the new one.
#   * NO patch machinery. Nothing is patched here yet; the day a gfx1151
#     defect needs one, this script grows the stamp fields build-llama.sh has
#     (patched=, patch_commit=) rather than a second convention.
#
# Vulkan first, and that is a decision to be MEASURED, not a preference:
# docs/MODELS.md's backend rule ("the backend choice belongs to the model")
# was derived from llama.cpp speculation workloads and says nothing about
# diffusion. Whether Vulkan is right for sd.cpp on gfx1151 is exactly the
# first measurement this build exists for.
set -euo pipefail

SRC="${SD_SRC:-$HOME/stable-diffusion.cpp}"

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
  say "no stable-diffusion.cpp checkout at $SRC" >&2
  say "  git clone --recursive https://github.com/leejet/stable-diffusion.cpp $SRC" >&2
  exit 1
}

# --ref <commit>: build the EXACT commit a workload profile pins, instead of
# whatever HEAD happens to be. Without it, reproducing build-vulkan-<id> on
# a new machine was archaeology — clone, hand-checkout, then build
# (architecture review, 01.09.2026). Moves the checkout; build-llama.sh set
# the precedent.
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
    # 01.09.2026).
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
  # A failed submodule sync must FAIL the build: `|| true` here paired the
  # new tree with the old ggml and stamped the result as the new commit —
  # a hybrid binary whose figures are attributed to a tree upstream never
  # shipped (review, 01.09.2026). The stamp records no submodule SHAs, so
  # the checkout is the only place this can be caught.
  git -C "$SRC" submodule --quiet update --init --recursive || {
    say "REFUSING: submodule update failed after the checkout — building" >&2
    say "now would pair the new tree with the OLD submodule state and" >&2
    say "stamp the hybrid as $REF." >&2
    exit 1
  }
fi

COMMIT="$(git -C "$SRC" rev-parse --short=9 HEAD)"
BUILD="$SRC/build-vulkan-$COMMIT"

if [ -x "$BUILD/bin/sd-cli" ]; then
  say "already built: $BUILD"
  exit 0
fi

say "building stable-diffusion.cpp $COMMIT (Vulkan) into $BUILD"
mkdir -p "$BUILD"

# ggml-vulkan includes <spirv/unified1/spirv.hpp>. Fedora ships it in
# spirv-headers-devel, which is not part of the Vulkan SDK metapackage — on
# 01.09.2026 this machine had glslc and vulkan-headers and still failed at
# 89 %. A checkout of KhronosGroup/SPIRV-Headers in $HOME covers it without
# root; the package is the cleaner fix and this branch says so.
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

cmake -S "$SRC" -B "$BUILD" -DSD_VULKAN=ON -DSD_BUILD_EXAMPLES=ON \
      -DCMAKE_BUILD_TYPE=Release \
      ${EXTRA_CXX:+-DCMAKE_CXX_FLAGS="$EXTRA_CXX"}
cmake --build "$BUILD" --config Release -j"$(nproc)"

[ -x "$BUILD/bin/sd-cli" ] || { say "build produced no bin/sd-cli" >&2; exit 1; }

{
  printf 'build_id=vulkan-%s\n' "$COMMIT"
  printf 'commit=%s\n' "$(git -C "$SRC" rev-parse HEAD)"
  printf 'backend=vulkan\n'
  printf 'built_at=%s\n' "$(date +%Y-%m-%d_%H%M)"
  printf 'cmake_flags=-DSD_VULKAN=ON -DSD_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release\n'
} > "$BUILD/.build-stamp"

say "done: $BUILD/bin/sd-cli"
say "stamp: $(tr '\n' ' ' < "$BUILD/.build-stamp")"
