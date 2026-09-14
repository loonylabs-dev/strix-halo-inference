#!/usr/bin/env bash
# Fetch Halogen Flash model bundle from Hugging Face — resumable, verified,
# size-checked, and stored in local model storage.
#
# Usage:
#   bash setup/scripts/fetch-halogen.sh [--dry-run] [--with-vision]
#
set -uo pipefail
export LC_ALL=C
cd "$(dirname "$0")/../.."
TOP="$PWD"
# shellcheck source=../lib/models.sh
. "$TOP/setup/lib/models.sh"

# ASKED, not derived. This used to be `$(models_dir)/halogen-models` — inside
# the .gguf directory — while halogenexec looked beside it, so a fresh fetch
# put 124 GiB where nothing serves from. setup/lib/models.sh answers for both.
DEST="${DEST:-$(halogen_models_dir)}"
REPO="peonist-ai/halogen-qwen3.8-flash-next"
DRY=0
WITH_VISION=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --with-vision) WITH_VISION=1; shift ;;
    --dest) DEST="${2:?--dest needs a directory}"; shift 2 ;;
    -h|--help)
      echo "usage: fetch-halogen.sh [--dry-run] [--with-vision] [--dest <dir>]"
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

say()  { printf '%s\n' "$*"; }
ok()   { printf '  \033[32m=\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '\n\033[31mABORT\033[0m %s\n' "$*" >&2; exit 2; }

verify() {
  local f="$1" want="$2" got
  if [ "$want" = "-" ]; then
    warn "$(basename "$f") — no checksum published, size checked only"
    return 0
  fi
  got=$(sha256sum "$f" 2>/dev/null | cut -d' ' -f1)
  [ "$got" = "$want" ]
}

# Fetch metadata from Hugging Face
say "Fetching file manifest from Hugging Face..."
TREE_JSON=$(curl -s -m 60 "https://huggingface.co/api/models/$REPO/tree/main?recursive=true")
[ -n "$TREE_JSON" ] || die "Failed to fetch tree listing from Hugging Face"

FILES=$(echo "$TREE_JSON" | WITH_VISION="$WITH_VISION" python3 -c '
import json, os, sys
with_vision = os.environ.get("WITH_VISION", "0") == "1"
try:
    tree = json.load(sys.stdin)
except Exception as e:
    sys.exit(1)

for e in sorted(tree, key=lambda x: x.get("path", "")):
    p = e.get("path", "")
    sz = e.get("size", 0)
    lfs = e.get("lfs") or {}
    oid = lfs.get("oid", "-")
    if p in [".gitattributes", "README.md", "halogen.jpg", "tokenizer"]:
        continue
    if p == "qwen38-flash-next-vision.hgn" and not with_vision:
        continue
    print(f"{sz}\t{oid}\t{p}")
')

[ -n "$FILES" ] || die "No files selected to download"

TOTAL=$(printf '%s\n' "$FILES" | awk -F'\t' '{s+=$1} END {printf "%.1f", s/1073741824}')
COUNT=$(printf '%s\n' "$FILES" | grep -c .)
say ""
say "$REPO"
printf '%s\n' "$FILES" | awk -F'\t' '{printf "  %8.2f GiB  %s\n", $1/1073741824, $3}'
say "  ---------------"
say "  $TOTAL GiB in $COUNT file(s)"

# Disk space check
# The destination may not exist yet — ask the nearest parent that does.
FREE_AT="$DEST"; while [ ! -d "$FREE_AT" ] && [ "$FREE_AT" != "/" ]; do FREE_AT="$(dirname "$FREE_AT")"; done
FREE=$(df -B1 --output=avail "$FREE_AT" 2>/dev/null | tail -1)
FREE_G=$(python3 -c "print('%.1f' % ($FREE/1073741824))" 2>/dev/null || echo 0)
say "  destination $DEST has $FREE_G GiB free"
python3 -c "
import sys
sys.exit(0 if $FREE/1073741824 > $TOTAL * 1.05 + 20 else 1)" \
  || die "not enough room: $TOTAL GiB wanted, $FREE_G GiB free, and 20 GiB must stay."

if [ "$DRY" = 1 ]; then
  say ""
  say "  destination would be $DEST"
  say "DRY RUN — nothing fetched."
  exit 0
fi

# Only now: a --dry-run used to create the destination AND the convenience
# symlink before printing "nothing fetched", which is a dry run that changes
# the machine.
mkdir -p "$DEST"
mkdir -p "$DEST/tokenizer"
if [ ! -e "$HOME/halogen-models" ]; then
  ln -s "$DEST" "$HOME/halogen-models"
  say "created symlink $HOME/halogen-models -> $DEST"
fi

# Download files
FAIL=0
while IFS=$'\t' read -r size sha path; do
  [ -n "$path" ] || continue
  dir="$(dirname "$path")"
  if [ "$dir" != "." ]; then
    mkdir -p "$DEST/$dir"
  fi
  out="$DEST/$path"
  part="$DEST/.$path.part"
  lock="$DEST/.$path.lock"

  mkdir -p "$(dirname "$part")"
  exec 9>"$lock" 2>/dev/null
  if ! flock -n 9; then
    warn "$path — another fetch is already working on it, skipping"
    exec 9>&-
    FAIL=1
    continue
  fi

  have=$(stat -c%s "$out" 2>/dev/null || echo 0)
  if [ "$have" != "0" ] && [ "$have" != "$size" ]; then
    warn "$path is already here at $have bytes, expected $size."
    warn "  To restart this file: rm $out"
    FAIL=1
    exec 9>&-
    continue
  fi

  if [ "$have" = "0" ]; then
    held=$(stat -c%s "$part" 2>/dev/null || echo 0)
    [ "$held" -gt 0 ] && say "  resuming $path at $(python3 -c "print('%.1f' % ($held/1073741824))") GiB"
    say "  fetching $path  ($(python3 -c "print('%.2f' % ($size/1073741824))") GiB)"
    
    curl -L -C - --retry 20 --retry-delay 10 --retry-all-errors \
         --connect-timeout 30 --speed-limit 1048576 --speed-time 60 -# \
         "https://huggingface.co/$REPO/resolve/main/$path" -o "$part"

    held=$(stat -c%s "$part" 2>/dev/null || echo 0)
    if [ "$held" != "$size" ]; then
      warn "$path is $held bytes, expected $size — run again to resume"
      FAIL=1
      exec 9>&-
      continue
    fi

    say "  checking $path  (sha256 of $(python3 -c "print('%.2f' % ($held/1073741824))") GiB)"
    if verify "$part" "$sha"; then
      mv -- "$part" "$out"
      ok "$path complete and verified"
    else
      warn "$path downloaded with WRONG CHECKSUM. Deleting partial."
      rm -f "$part"
      FAIL=1
    fi
    exec 9>&-
    continue
  fi

  say "  verifying existing $path  ($(python3 -c "print('%.2f' % ($have/1073741824))") GiB)"
  if verify "$out" "$sha"; then
    ok "$path complete and verified"
  else
    warn "$path existing file checksum MISMATCH! Deleting corrupted file."
    rm -f "$out"
    FAIL=1
  fi
  exec 9>&-
done <<< "$FILES"

say ""
if [ "$FAIL" = 0 ]; then
  say "All Halogen files complete and verified in $DEST."
else
  say "Incomplete or errors encountered. Run again to resume."
  exit 1
fi
