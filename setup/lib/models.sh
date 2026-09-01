#!/usr/bin/env bash
# The model registry — ONE place that knows which models exist.
#
#   source setup/lib/models.sh          as a library
#   bash   setup/lib/models.sh list     as a command (for tests and humans)
#
# Why this file exists
# --------------------
# Until 26.08. the set of models was written down in SIX places: the pair in
# switch-model.sh, the Conflicts= line of llama-user@.service, the same line
# in llama@.service, the service check in check.sh, the SWA list in check.sh,
# and the SWA table in setup/README.md. Adding a seventh model meant finding
# all six. Missing one of them fails SILENTLY in the worst way there is:
# a Conflicts= line that does not name the new model lets TWO llama-servers
# start, and the second one loses the race for port 8080 — the service says
# "active", the gateway talks to the wrong model.
#
# That is the same class of bug tests/README.md is about: nothing breaks, an
# effect simply fails to appear. So: one source of truth (setup/env/*.env —
# the profile IS the model), everything else derived, and the places that
# CANNOT be derived (systemd has no wildcard in Conflicts=) pinned by a test
# in tests/test_models.py.
#
# The profile carries its own metadata, as ordinary systemd variables:
#
#   MODEL_TITLE=…    one line, shown by 'switch-model.sh --list'
#   MODEL_SWA=yes|no does this model have sliding window attention?
#                    yes -> the profile MUST carry --swa-full, or every
#                    Claude Code turn runs cold (docs section 15)
#
# They are passed to llama-server as environment variables and ignored there.
# That is deliberate: a magic comment is invisible to systemd and to anyone
# reading the file, a variable is neither.

# Resolve the repo root from this file, not from $PWD — sourced from
# anywhere, called from anywhere.
_models_lib_dir() {
  local src="${BASH_SOURCE[0]}"
  while [ -L "$src" ]; do src="$(readlink -f "$src")"; done
  cd "$(dirname "$src")" && pwd
}
MODELS_LIB_DIR="${MODELS_LIB_DIR:-$(_models_lib_dir)}"
MODELS_REPO="${MODELS_REPO:-$(dirname "$(dirname "$MODELS_LIB_DIR")")}"
MODELS_ENV_DIR="$MODELS_REPO/setup/env"
# Where install.sh symlinks the profiles to. This is what the USER service
# really reads (EnvironmentFile=%h/.config/llm-profile/%i.env) — a model whose
# profile is only in the repo cannot start.
MODELS_USER_ENV_DIR="${MODELS_USER_ENV_DIR:-$HOME/.config/llm-profile}"

# --- what exists ----------------------------------------------------------

models_all() {          # every model the repo knows, one per line, sorted
  local f
  for f in "$MODELS_ENV_DIR"/*.env; do
    [ -e "$f" ] || continue
    basename "$f" .env
  done | sort
}

models_known() {        # rc 0 if $1 is a model in the repo
  local want="$1" m
  [ -n "$want" ] || return 1
  while IFS= read -r m; do [ "$m" = "$want" ] && return 0; done < <(models_all)
  return 1
}

model_repo_env()  { printf '%s/%s.env\n' "$MODELS_ENV_DIR" "$1"; }
model_user_env()  { printf '%s/%s.env\n' "$MODELS_USER_ENV_DIR" "$1"; }
model_unit()      { printf 'llama-user@%s.service\n' "$1"; }

# --- foreign workloads ----------------------------------------------------
#
# setup/workloads/*.env — jobs that are not llama-server and still pin GTT
# (image, later audio/video). Same file discipline as the model profiles,
# different lifecycle: a workload is started by bench/sideserver.py as a
# transient unit, never by llama-user@, so it appears in NO Conflicts= line
# and switch-model.sh does not offer it. Enumerated here so tests and humans
# have ONE list — the six-places failure this file exists to prevent.
MODELS_WORKLOADS_DIR="$MODELS_REPO/setup/workloads"

workloads_all() {       # every workload the repo knows, one per line, sorted
  local f
  for f in "$MODELS_WORKLOADS_DIR"/*.env; do
    [ -e "$f" ] || continue
    basename "$f" .env
  done | sort
}

workload_repo_env() { printf '%s/%s.env\n' "$MODELS_WORKLOADS_DIR" "$1"; }

workload_meta() {       # $1 = workload, $2 = variable, $3 = default
  local f v
  f="$(workload_repo_env "$1")"
  [ -f "$f" ] || { printf '%s\n' "${3-}"; return 0; }
  v="$(sed -n "s/^$2=//p" "$f" | head -1)"
  v="${v%\"}"; v="${v#\"}"
  printf '%s\n' "${v:-${3-}}"
}

# --- what is running ------------------------------------------------------

models_active() {       # instances of llama-user@ that are ACTIVE right now
  systemctl --user list-units --plain --no-legend --state=active \
      'llama-user@*.service' 2>/dev/null \
    | awk '{print $1}' \
    | sed -n 's/^llama-user@\(.*\)\.service$/\1/p' \
    | sort
}

models_enabled() {      # instances that would come back after a reboot
  local m
  while IFS= read -r m; do
    [ "$(systemctl --user is-enabled "$(model_unit "$m")" 2>/dev/null)" = "enabled" ] \
      && printf '%s\n' "$m"
  done < <(models_all)
  return 0
}

# The one statement no unit file can argue away: which model does the process
# that actually holds port 8080 serve? Reads --alias out of its command line.
models_serving() {
  local pid
  for pid in $(pgrep -x llama-server 2>/dev/null); do
    tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null \
      | awk '$0=="--alias"{getline; print; exit}'
  done | sort -u
}

# --- reading a profile ----------------------------------------------------

# LLAMA_ARGS as systemd would hand it to llama-server. The parsing itself
# lives in systemdfile.py next door — ONE implementation, shared with bench/, and
# tests/test_models.py pins that no second one grows back. Three used to
# exist and they disagreed; the worst of them appended the words of the
# COMMENT lines after LLAMA_ARGS to the server command line.
model_args() {          # $1 = model name (or a path to a .env file)
  local f="$1"
  [ -f "$f" ] || f="$(model_repo_env "$1")"
  [ -f "$f" ] || return 1
  python3 "$MODELS_LIB_DIR/systemdfile.py" args "$f"
}

# A plain scalar variable out of a profile (MODEL_TITLE, MODEL_SWA,
# LLAMA_BIN). Not sourceable: systemd's EnvironmentFile syntax is NOT bash —
# '. file.env' reads 'VAR=value command args' and fails silently.
# --- this machine's answers, for the shell side ----------------------------
#
# The Python counterpart is setup/lib/systemdfile.py (local_var, models_dir),
# and the two must agree — same file, same order of precedence. Kept short
# here on purpose: one grep, no parsing beyond what model_meta already does.
#
# NOT `. "$LOCAL_ENV"`. It is systemd EnvironmentFile syntax and bash reads
# `VAR=value command args` — the trap this repo documents in three places.
llm_local_env() { printf '%s\n' "${LLM_STACK_ENV:-$HOME/.config/llm-stack.env}"; }

local_var() {           # $1 = variable, $2 = default
  local f v
  f="$(llm_local_env)"
  [ -f "$f" ] || { printf '%s\n' "${2-}"; return 0; }
  v="$(sed -n "s/^$1=//p" "$f" | head -1)"
  v="${v%\"}"; v="${v#\"}"
  printf '%s\n' "${v:-${2-}}"
}

# Where the .gguf live. $LLAMA_MODELS, then the local config, then the
# conventional locations — and then it gives up rather than guessing. No
# absolute path is written down here: a directory that exists on one machine
# is not a fallback, it is the same hard-coding one level down.
# The conventions are NOT listed here. They live in setup/lib/systemdfile.py
# and are read from it — this function had its own copy for a few hours on
# 27.08., install.sh had a third, and the three had already diverged before
# anyone used them. One list, or it is not a rule.
llm_conventions() {
  python3 "${MODELS_LIB:-$(dirname "${BASH_SOURCE[0]}")}/systemdfile.py" conventions 2>/dev/null
}

models_dir() {
  local d hit
  if [ -n "${LLAMA_MODELS:-}" ]; then printf '%s\n' "$LLAMA_MODELS"; return 0; fi
  d="$(local_var LLAMA_MODELS)"
  if [ -n "$d" ]; then printf '%s\n' "$d"; return 0; fi
  while IFS= read -r d; do
    [ -z "$d" ] && continue
    # An entry may be a glob (/mnt/*/LLM); sorted, so two matching volumes
    # give the same answer on every run.
    for hit in $(eval "printf '%s\n' ${d/#\~/$HOME}" 2>/dev/null | sort); do
      if [ -d "$hit" ] && [ -n "$(find "$hit" -maxdepth 1 -name '*.gguf' -print -quit 2>/dev/null)" ]; then
        printf '%s\n' "$hit"; return 0
      fi
    done
  done <<EOF_CONV
$(llm_conventions)
EOF_CONV
  printf 'no model directory: set LLAMA_MODELS, or run bash setup/install.sh\n' >&2
  return 1
}

model_meta() {          # $1 = model, $2 = variable, $3 = default
  local f v
  f="$(model_repo_env "$1")"
  [ -f "$f" ] || { printf '%s\n' "${3-}"; return 0; }
  v="$(sed -n "s/^$2=//p" "$f" | head -1)"
  # strip one layer of surrounding quotes, the way systemd does
  v="${v%\"}"; v="${v#\"}"
  printf '%s\n' "${v:-${3-}}"
}

model_title() { model_meta "$1" MODEL_TITLE "(no MODEL_TITLE in $(model_repo_env "$1"))"; }
model_swa()   { model_meta "$1" MODEL_SWA   "unknown"; }
# "<repo> <pattern>…" or empty. Empty means nobody has written down where
# this model comes from — get-model.sh says so rather than guessing a repo.
model_source() { model_meta "$1" MODEL_SOURCE ""; }
# What one token of context costs in KV, and how much of the file the GPU
# really pins. Both MEASURED and both empty when nobody has measured them —
# setup/lib/budget.py then charges a pessimistic estimate and says so, which
# is the honest state. See the note in setup/env/gemma31.env for why an empty
# value must not be filled in from the profile next door.
model_kv()      { model_meta "$1" MODEL_KV_KIB_PER_TOKEN ""; }
model_gtt_gib() { model_meta "$1" MODEL_WEIGHTS_GTT_GIB ""; }

# The binary a profile starts. Same fallback the unit has, so a profile
# without LLAMA_BIN reads the same here as it behaves there.
model_bin() {
  printf '%s/%s\n' "$HOME" \
    "$(model_meta "$1" LLAMA_BIN "llama.cpp/build-vulkan/bin/llama-server")"
}

# The model file, taken out of LLAMA_ARGS the way setup/waitformodel does.
model_gguf() {
  model_args "$1" | awk '{for(i=1;i<NF;i++) if($i=="-m"||$i=="--model"){print $(i+1); exit}}'
}

# --- CLI ------------------------------------------------------------------

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  set -euo pipefail
  case "${1:-list}" in
    list)     models_all ;;
    workloads) workloads_all ;;
    active)   models_active ;;
    enabled)  models_enabled ;;
    serving)  models_serving ;;
    known)    models_known "${2:-}" ;;
    args)     model_args "${2:?model name}" ;;
    meta)     model_meta "${2:?model name}" "${3:?variable}" "${4-}" ;;
    bin)      model_bin  "${2:?model name}" ;;
    gguf)     model_gguf "${2:?model name}" ;;
    table)
      printf '%-12s %-7s %-6s %-8s %s\n' MODEL SWA BIN STATE TITLE
      while IFS= read -r m; do
        state="$(systemctl --user is-active "$(model_unit "$m")" 2>/dev/null || true)"
        printf '%-12s %-7s %-6s %-8s %s\n' \
          "$m" "$(model_swa "$m")" \
          "$(case "$(model_meta "$m" LLAMA_BIN)" in *rocm-patched*) echo rocm+;; *rocm*) echo rocm;; *) echo vulk;; esac)" \
          "${state:-inactive}" "$(model_title "$m")"
      done < <(models_all)
      ;;
    *) echo "usage: bash setup/lib/models.sh {list|workloads|active|enabled|serving|table|known N|args N|meta N VAR|bin N|gguf N}" >&2
       exit 2 ;;
  esac
fi
