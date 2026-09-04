#!/usr/bin/env bash
# Fetch a GGUF from Hugging Face — resumable, size-checked, and able to WAIT
# for a quant that has not been uploaded yet.
#
#   bash setup/scripts/fetch-model.sh <repo> <pattern>
#   bash setup/scripts/fetch-model.sh <repo> <pattern> --wait 600
#   bash setup/scripts/fetch-model.sh <repo> <pattern> --dry-run
#
#   pattern   a substring of the file path, e.g. UD-Q4_K_XL. Every matching
#             .gguf is fetched, so a sharded quant comes down whole.
#   --wait N  poll every N seconds until something matches, then fetch.
#             A release trickles in over hours — IQ1_S was up 20 minutes
#             before anything else — and this is the difference between
#             watching a page and having the file.
#
# Why a script rather than `huggingface-cli download`:
#
#   * it must be RESUMABLE without a cache directory. 100 GiB at 10 MiB/s is
#     three hours; a dropped connection must cost seconds, not the download.
#     The files land directly where the profiles point (see models_dir in
#     setup/lib/models.sh — $LLAMA_MODELS, then ~/.config/llm-stack.env),
#     with their original names, so a sharded model's part one finds the rest.
#   * it must REFUSE before it starts rather than fill the partition. The
#     model partition also holds every other model here.
#   * it must REFUSE TO COLLIDE. Two of these running against the same file
#     both resume from the same offset and both write — 26.08., and it was
#     caught by looking at `lsof`, not by anything the script did. curl
#     resumes in append mode, so the second writer does not overwrite the
#     first, it interleaves with it, and the damage is invisible to a size
#     check. One flock per output file, and the second one says so and stops.
#   * it must CHECK what it got, and SIZE IS NOT A CHECK. A truncated GGUF
#     fails at load time with a confusing error hours later; a corrupted one
#     fails as wrong answers, which is worse. Hugging Face publishes a
#     sha256 per file in the tree listing (`lfs.oid`) and this script had
#     been ignoring it. It does not any more — ~1 minute per 46 GiB, paid
#     once, against a fault that otherwise surfaces as bad output.
set -uo pipefail
export LC_ALL=C
cd "$(dirname "$0")/../.."

# shellcheck source=../lib/models.sh
. "$(dirname "$0")/../lib/models.sh"
DEST="${DEST:-$(models_dir)}" || exit 1
REPO=""; PATTERN=""; WAIT=0; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --wait)    WAIT="${2:?--wait needs seconds}"; shift 2 ;;
    --dest)    DEST="${2:?--dest needs a directory}"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*)        echo "unknown option: $1" >&2; exit 2 ;;
    *)         if [ -z "$REPO" ]; then REPO="$1"; else PATTERN="$1"; fi; shift ;;
  esac
done
[ -n "$REPO" ] && [ -n "$PATTERN" ] || { echo "usage: fetch-model.sh <repo> <pattern> [--wait N]" >&2; exit 2; }

say()  { printf '%s\n' "$*"; }
ok()   { printf '  \033[32m=\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '\n\033[31mABORT\033[0m %s\n' "$*" >&2; exit 2; }

listing() {   # -> "size<TAB>sha256<TAB>path" per matching .gguf
  # A test seam, and a narrow one: FETCH_MODEL_LISTING names a file holding
  # exactly the lines this function would otherwise build. It replaces the
  # NETWORK call and nothing else — every decision below still runs against
  # it. The collision refusal added 04.09.2026 cannot be exercised any other
  # way without a network or a fake HTTP server, and a guard that is never
  # made to go red is the failure class this repo names most often
  # (bench/README.md, "A check that cannot fail is not a check").
  if [ -n "${FETCH_MODEL_LISTING:-}" ]; then
    PATTERN="$PATTERN" python3 -c '
import os, sys
pat = os.environ["PATTERN"]
for line in open(os.environ["FETCH_MODEL_LISTING"], encoding="utf-8"):
    line = line.rstrip("\n")
    if line and pat in line.split("\t")[-1]:
        print(line)'
    return 0
  fi
  curl -s -m 60 "https://huggingface.co/api/models/$REPO/tree/main?recursive=true" 2>/dev/null \
    | PATTERN="$PATTERN" python3 -c '
import json, os, sys
pat = os.environ["PATTERN"]
try:
    tree = json.load(sys.stdin)
except Exception:
    sys.exit(0)
for e in sorted(tree, key=lambda e: e.get("path", "")):
    p = e.get("path", "")
    if p.endswith(".gguf") and pat in p:
        lfs = e.get("lfs") or {}
        print("%d\t%s\t%s" % (e.get("size") or 0, lfs.get("oid") or "-", p))'
}


# A file is right when its bytes hash to what Hugging Face says, and at no
# other time. "-" means the repo published no checksum for it, which happens
# for small non-LFS files; then the size is all there is and it says so.
verify() {
  local f="$1" want="$2" got
  if [ "$want" = "-" ]; then
    warn "$(basename "$f") — no checksum published, size checked only"
    return 0
  fi
  got=$(sha256sum "$f" 2>/dev/null | cut -d' ' -f1)
  [ "$got" = "$want" ]
}

# --- wait for it to exist --------------------------------------------------
FILES="$(listing)"
if [ -z "$FILES" ] && [ "$WAIT" -gt 0 ]; then
  say "nothing matching '$PATTERN' in $REPO yet — polling every ${WAIT}s"
  while [ -z "$FILES" ]; do
    sleep "$WAIT"
    FILES="$(listing)"
    [ -z "$FILES" ] && printf '%s  still nothing\n' "$(date '+%H:%M:%S')"
  done
  say ""
  say "$(date '+%H:%M:%S')  it is there."
fi
[ -n "$FILES" ] || die "nothing matching '$PATTERN' in $REPO (and no --wait given)"

TOTAL=$(printf '%s\n' "$FILES" | awk -F'\t' '{s+=$1} END {printf "%.1f", s/1073741824}')
COUNT=$(printf '%s\n' "$FILES" | grep -c .)
say ""
say "$REPO · $PATTERN"
printf '%s\n' "$FILES" | awk -F'\t' '{printf "  %8.1f GiB  %s\n", $1/1073741824, $3}'
say "  ---------------"
say "  $TOTAL GiB in $COUNT file(s)"

# --- will it fit on the partition? ----------------------------------------
FREE=$(df -B1 --output=avail "$DEST" 2>/dev/null | tail -1)
FREE_G=$(python3 -c "print('%.1f' % ($FREE/1073741824))" 2>/dev/null || echo 0)
say "  destination $DEST has $FREE_G GiB free"
python3 -c "
import sys
sys.exit(0 if $FREE/1073741824 > $TOTAL * 1.05 + 20 else 1)" \
  || die "not enough room: $TOTAL GiB wanted, $FREE_G GiB free, and 20 GiB must
    stay for everything else on that partition."

if [ "$DRY" = 1 ]; then say ""; say "DRY RUN — nothing fetched."; exit 0; fi

# --- fetch ----------------------------------------------------------------
mkdir -p "$DEST"
say ""
FAIL=0
while IFS=$'\t' read -r size sha path; do
  [ -n "$path" ] || continue
  name="$(basename "$path")"
  out="$DEST/$name"

  # One writer per file. Held for the whole download AND the verification, so
  # a second run cannot start hashing a file that is still growing either.
  exec 9>"$DEST/.$name.lock" 2>/dev/null
  if ! flock -n 9; then
    warn "$name — another fetch is already working on it, leaving it alone"
    exec 9>&-
    FAIL=1
    continue
  fi

  # A DOWNLOAD IN PROGRESS LIVES IN ITS OWN FILE, and that is not tidiness.
  #
  # Until 04.09.2026 this resumed straight into "$out" whenever its size did
  # not match, which silently assumed that a file of the right NAME is a
  # partial copy of the file we want. This model directory is one flat
  # namespace and quantisers name the vision encoder after its PRECISION
  # rather than after its model: `mmproj-F16.gguf` is what unsloth calls
  # Qwen3.6-35B-A3B's (899,283,680 B) AND what it calls Qwen3.8-27B's
  # (927,607,488 B, the file qwen38 has been serving from since 17.08.).
  # Two different files, one path, sizes that differ — so `curl -C -` would
  # have appended to a complete, working, ANOTHER MODEL'S file. Measured
  # 04.09.2026 while integrating qwen36; nothing in the script could have
  # told the two apart afterwards, and the sha256 check fires only after the
  # damage.
  #
  # So: bytes accumulate in "$DEST/.$name.part" and reach "$out" only after
  # they hash to what Hugging Face published. The .part file is also the only
  # thing this script will ever resume into — its existence is the claim
  # "these bytes are mine". A stranger sitting at "$out" is refused by name.
  have=$(stat -c%s "$out" 2>/dev/null || echo 0)
  part="$DEST/.$name.part"

  if [ "$have" != "0" ] && [ "$have" != "$size" ]; then
    warn "$name is already here at $have bytes, and this repo publishes it at $size."
    warn "  NOT resuming into it. A file of the right name is not evidence of"
    warn "  the right content, and appending to the wrong one destroys it."
    warn "  Either it belongs to another model — check which profile points at"
    warn "    $out"
    warn "  — or it is a partial download from before 04.09.2026, in which case"
    warn "  it can be adopted deliberately:"
    warn "    mv $out $part"
    FAIL=1
    exec 9>&-
    continue
  fi

  if [ "$have" = "0" ]; then
    held=$(stat -c%s "$part" 2>/dev/null || echo 0)
    [ "$held" -gt 0 ] && say "  resuming $name at $(python3 -c "print('%.1f' % ($held/1073741824))") GiB"
    say "  fetching $name  ($(python3 -c "print('%.1f' % ($size/1073741824))") GiB)"
    # -C - resumes, --retry survives a dropped connection without losing the file
    curl -L -C - --retry 20 --retry-delay 10 --retry-all-errors \
         --connect-timeout 30 -# \
         "https://huggingface.co/$REPO/resolve/main/$path" -o "$part"
    held=$(stat -c%s "$part" 2>/dev/null || echo 0)
    if [ "$held" != "$size" ]; then
      warn "$name is $held bytes, expected $size — run again to resume"
      FAIL=1
      exec 9>&-
      continue
    fi
    say "  checking $name  (sha256 of $(python3 -c "print('%.1f' % ($held/1073741824))") GiB)"
    if verify "$part" "$sha"; then
      mv -- "$part" "$out"
      ok "$name complete and verified"
    else
      warn "$name downloaded to the right SIZE with the WRONG CONTENT."
      warn "  It stays out of $DEST until it is right. Delete the partial and"
      warn "  fetch again — resuming cannot repair it:"
      warn "  rm $part"
      FAIL=1
    fi
    exec 9>&-
    continue
  fi

  # $out is here at exactly the published size. Verify it rather than assume
  # it: size is not a check, which is the lesson this script already carries.
  say "  checking $name  (sha256 of $(python3 -c "print('%.1f' % ($have/1073741824))") GiB)"
  if verify "$out" "$sha"; then
    ok "$name complete and verified"
  else
    warn "$name has the right SIZE and the WRONG CONTENT."
    warn "  Delete it and fetch again — resuming cannot repair it:"
    warn "  rm $out"
    FAIL=1
  fi
  exec 9>&-
done <<< "$FILES"

say ""
if [ "$FAIL" = 0 ]; then
  say "All files complete in $DEST."
  say ""
  say "Next:"
  say "  python3 ~/llama.cpp/gguf-py/gguf/scripts/gguf_dump.py --no-tensors \\"
  say "      $DEST/$(printf '%s\n' "$FILES" | head -1 | cut -f3 | xargs basename)"
  say "    -> read general.architecture, then"
  say "  python3 setup/scripts/scout.py --arch <that name>"
  say "  and fill in the two REPLACE lines in setup/env/flashnext.env"
else
  say "Incomplete. Run the same command again — it resumes."
  exit 1
fi
