#!/usr/bin/env bash
# Fetch the model a profile needs — this stack's `ollama pull`.
#
#   bash setup/get-model.sh --list          what exists, what is already here
#   bash setup/get-model.sh qwen38          fetch what qwen38 needs
#   bash setup/get-model.sh qwen38 --dry-run
#
# The profile IS the registry, and since 26.08. it also knows where its model
# comes from (MODEL_SOURCE). So the name you fetch is the name you then serve:
#
#     bash setup/get-model.sh qwen38
#     bash setup/switch-model.sh qwen38
#
# Why not just a list of links in the README. Links rot, they verify nothing,
# and they cannot say whether the thing fits YOUR machine. This resolves a
# repo plus a pattern to whatever is current, checks the size against the
# partition, downloads resumably and verifies every file against Hugging
# Face's sha256 — all of that is setup/scripts/fetch-model.sh, which this only
# points at the right thing.
#
# And the list it offers is the real difference from a generic catalog:
# `bash setup/lib/models.sh table` shows what has been MEASURED on this
# hardware, with the verdict in the title — "Coding agent + vision + judge ·
# 16.7 GiB · production since 25.08." is a different kind of statement from
# "qwen3.8:27b".
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
MODELS_REPO="$REPO"
# shellcheck source=lib/models.sh
. "$REPO/setup/lib/models.sh"

# Where the weights land. models_dir() in lib/models.sh: $LLAMA_MODELS, then
# ~/.config/llm-stack.env, then the conventional locations — and no absolute
# path written down here, because the one that used to be here existed on a
# single computer.
DEST="${DEST:-$(models_dir)}" || exit 1
NAME=""; DRY=""; LIST=0
for a in "$@"; do
  case "$a" in
    --list) LIST=1 ;;
    --dry-run|-n) DRY="--dry-run" ;;
    -h|--help) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "unknown argument: $a" >&2; exit 2 ;;
    *) NAME="$a" ;;
  esac
done

say()  { printf '%s\n' "$*"; }
ok()   { printf '  \033[32m=\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m?\033[0m %s\n' "$*"; }
die()  { printf '\n\033[31mABORT\033[0m %s\n' "$*" >&2; exit 2; }

if [ "$LIST" = 1 ] || [ -z "$NAME" ]; then
  say "models this repo knows (setup/env/*.env), and whether the file is here"
  say
  printf '  %-12s %-6s %s\n' NAME FILE TITLE
  for m in $(models_all); do
    g="$(model_gguf "$m" 2>/dev/null || true)"
    if [ -n "$g" ] && [ -r "$g" ]; then here="here"
    elif [ -n "$(model_source "$m")" ]; then here="fetch"
    else here="—"; fi
    printf '  %-12s %-6s %s\n' "$m" "$here" "$(model_title "$m")"
  done
  say
  say "  bash setup/get-model.sh <name>       fetch one"
  say "  bash setup/switch-model.sh <name>    serve it"
  [ -z "$NAME" ] && exit 0
  exit 0
fi

models_known "$NAME" || die "unknown model '$NAME'. Known: $(models_all | tr '\n' ' ')"

SOURCE="$(model_source "$NAME")"
[ -n "$SOURCE" ] || die "$(model_repo_env "$NAME") has no MODEL_SOURCE.

    Nobody has written down where this model comes from, and guessing a
    Hugging Face repo is how the wrong weights end up on disk. Add a line:

      MODEL_SOURCE=<owner>/<repo> <pattern> [<pattern>…]

    Verify it by matching the exact FILENAME the profile's -m points at:
      python3 setup/scripts/scout.py <owner>/<repo>"

REPO_ID="${SOURCE%% *}"
PATTERNS="${SOURCE#* }"
[ "$REPO_ID" != "$PATTERNS" ] || die "MODEL_SOURCE in $(model_repo_env "$NAME") has a repo but no pattern"

say
say "$NAME — $(model_title "$NAME")"
say "  from $REPO_ID"
GGUF="$(model_gguf "$NAME" 2>/dev/null || true)"
if [ -n "$GGUF" ] && [ -r "$GGUF" ]; then
  ok "$(basename "$GGUF") is already here"
  say "  Fetching again is harmless — every file is size- and sha256-checked,"
  say "  and a complete one is verified rather than re-downloaded."
fi
say

for pat in $PATTERNS; do
  say "== $pat"
  DEST="$DEST" bash "$REPO/setup/scripts/fetch-model.sh" "$REPO_ID" "$pat" ${DRY:+$DRY} \
    || die "fetching '$pat' from $REPO_ID failed. Run the same command again — it resumes."
done

[ -n "$DRY" ] && exit 0

# The profile's own -m is the only test that counts: a fetch can succeed and
# still leave the profile pointing somewhere else, which is exactly what a
# renamed quant does.
if [ -n "$GGUF" ] && [ -r "$GGUF" ]; then
  say
  ok "$(model_repo_env "$NAME") points at a file that now exists"
  say
  say "Next:  bash setup/switch-model.sh $NAME --dry-run"
else
  say
  warn "the files are here, but $(model_repo_env "$NAME") points at"
  warn "  $GGUF"
  warn "which still does not exist. The pattern fetched something else, or the"
  warn "profile's -m needs updating. ls $DEST to see what arrived."
  exit 1
fi
