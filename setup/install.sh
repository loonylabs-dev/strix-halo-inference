#!/usr/bin/env bash
# Connects this repo to the running system.
#
#   bash setup/install.sh              everything
#   bash setup/install.sh --user-only  only the user-side part, no sudo
#
# Principle: ONE source of truth, and that is this repo.
#
#   user side    -> SYMLINK into the repo. Editing then means changing the
#                   running thing, and drift is impossible.
#   system wide  -> COPY. /etc and /etc/systemd/system need root, and systemd
#                   reads EnvironmentFile at start; a copy also survives an
#                   unmounted partition.
#
# The copies can therefore drift apart — that is exactly what check.sh is for.
set -euo pipefail

# Do NOT call with sudo. The user-side part creates symlinks under $HOME — with
# sudo that is /root, and your own files are never touched. The script asks for
# root itself for the /etc part.
if [ "$(id -u)" = "0" ]; then
  echo "Do not call this script with sudo."
  echo
  if [ -n "${SUDO_USER:-}" ]; then
    echo "  Under sudo HOME=/root — the symlinks would land there instead of in"
    echo "  /home/$SUDO_USER. The right way is:"
  else
    echo "  As root the user-side files land in /root. The right way is to call"
    echo "  it as a normal user:"
  fi
  echo
  echo "      bash $0"
  echo
  echo "  The script then asks for the sudo password once, but only for /etc"
  echo "  and /usr/local/bin."
  exit 1
fi

SRC="$(cd "$(dirname "$0")" && pwd)"      # .../inference-stack/setup
REPO="$(dirname "$SRC")"                  # .../inference-stack
USER_ONLY=0
SYSTEM_UNIT=0
for a in "$@"; do
  case "$a" in
    --user-only|--nur-user) USER_ONLY=1 ;;   # --nur-user: pre-rename name
    # A system service for a host with no user session to linger on — a Strix
    # Halo in a cupboard rather than a laptop. OPT-IN, and everything it needs
    # is installed only with it: the predecessor was installed unconditionally,
    # never started, and rotted where nobody could see it.
    --system-unit) SYSTEM_UNIT=1 ;;
    *) echo "unknown argument: $a"; exit 2 ;;
  esac
done

echo "repo: $REPO"
echo

link_() {   # $1 = source in the repo, $2 = target in the system
  local q="$1" z="$2"
  [ -e "$q" ] || { echo "  ! missing in the repo: $q"; return 0; }
  mkdir -p "$(dirname "$z")"
  if [ -L "$z" ] && [ "$(readlink -f "$z")" = "$(readlink -f "$q")" ]; then
    echo "  = $z"
  else
    [ -e "$z" ] && [ ! -L "$z" ] && mv "$z" "$z.before-repo"
    ln -sfn "$q" "$z"
    echo "  -> $z"
  fi
}

# --- user side: symlinks, no root ----------------------------------------
echo "== Claude Code profiles and gateway (symlinks) =="
link_ "$SRC/claude/local.json"        "$HOME/.claude/profiles/local.json"
link_ "$SRC/claude/hybrid.json"       "$HOME/.claude/profiles/hybrid.json"
link_ "$SRC/claude/PROFILE.md"        "$HOME/.claude/profiles/README.md"
# dialects.py must sit NEXT TO both cc-gateway.py and prewarm.py: they import
# it by directory, and it is the shared truth about how a request body is read.
# modes.py is the second such module and was missed when it was added on
# 28.08.2026 — the gateway kept starting, because Python 3.11+ puts the
# RESOLVED script directory on sys.path[0] and that is the repo, so the import
# found it there instead. It worked for a reason the code does not state, which
# is not the same as working. tests/test_install.py now walks these links.
link_ "$SRC/claude/dialects.py"       "$HOME/.claude/bin/dialects.py"
link_ "$SRC/claude/modes.py"          "$HOME/.claude/bin/modes.py"
link_ "$SRC/claude/cc-gateway.py"     "$HOME/.claude/bin/cc-gateway.py"
link_ "$SRC/claude/cc-router.py"      "$HOME/.claude/bin/cc-router.py"
# cc-cachefix.py and cc-cachefix2.py are NOT installed. Both were superseded by
# cc-gateway.py, which does the same job and more, and setup/README.md has said
# so since 26.08. — while install.sh kept linking the newer one into
# ~/.claude/bin anyway. They stay in the tree because docs/measurements/
# measures against them and the comparison is the argument for the gateway;
# they do not belong on a running system.

echo
echo "== systemd units (symlinks) =="
link_ "$SRC/systemd/cc-gateway.service" \
      "$HOME/.config/systemd/user/cc-gateway.service"
link_ "$SRC/systemd/llama-user@.service" \
      "$HOME/.config/systemd/user/llama-user@.service"
link_ "$SRC/systemd/prefix-cleanup.service" \
      "$HOME/.config/systemd/user/prefix-cleanup.service"
link_ "$SRC/systemd/llama-probe.service" \
      "$HOME/.config/systemd/user/llama-probe.service"
link_ "$SRC/systemd/llama-probe.timer" \
      "$HOME/.config/systemd/user/llama-probe.timer"
link_ "$SRC/systemd/prefix-cleanup.timer" \
      "$HOME/.config/systemd/user/prefix-cleanup.timer"
link_ "$REPO/tools/prewarm.py"        "$HOME/.claude/bin/prewarm.py"

# Model profiles for the USER service. A symlink, not a copy: a copy
# drifts, and the root-owned copies under /etc/llm-profile made every
# profile change need sudo — which locks out a remote operator entirely.
# The system unit llama@.service keeps using /etc.
mkdir -p "$HOME/.claude/env"
for f in "$SRC"/env/*.env; do
  link_ "$f" "$HOME/.claude/env/$(basename "$f")"
done
link_ "$SRC/waitformodel"             "$HOME/.claude/bin/waitformodel"
link_ "$SRC/llamaexec"                "$HOME/.claude/bin/llamaexec"
# The memory guard, and the two modules it is made of. budget.py imports
# systemdfile.py from its OWN directory, so the pair travels together — the
# same rule that keeps dialects.py next to cc-gateway.py, and for the same
# reason: a module that is found by directory has to BE in the directory.
link_ "$SRC/checkroom"                "$HOME/.claude/bin/checkroom"
link_ "$SRC/lib/budget.py"            "$HOME/.claude/bin/budget.py"
link_ "$SRC/lib/systemdfile.py"       "$HOME/.claude/bin/systemdfile.py"
link_ "$REPO/setup/scripts/probe.py"  "$HOME/.claude/bin/probe.py"
# --- this machine's answers, derived once ---------------------------------
#
# The one file that says what is specific to this computer. Written before
# anything else needs it, because the units below load it and the tools read
# it. Derived rather than asked wherever that is possible; a hostname cannot
# be derived and is left empty on purpose.
LOCAL_ENV="${LLM_STACK_ENV:-$HOME/.config/llm-stack.env}"
if [ ! -e "$LOCAL_ENV" ]; then
  mkdir -p "$(dirname "$LOCAL_ENV")"
  cp "$SRC/local.env.template" "$LOCAL_ENV"
  # Find the directory that actually holds .gguf files, using THE convention
  # list — setup/lib/systemdfile.py, not a copy. This loop had its own list for
  # a few hours on 27.08. and it was already WIDER than the two readers': a
  # directory install.sh found could be one models_dir() would then not. Three
  # copies of one rule is the failure this repo keeps writing about, and it
  # took three hours to reproduce it.
  #
  # A directory that exists on one machine is deliberately not in the list. It
  # is found because the .gguf files are in it, not because it is named.
  FOUND=""
  while IFS= read -r d; do
    [ -z "$d" ] && continue
    for hit in $(eval "printf '%s\n' ${d/#\~/$HOME}" 2>/dev/null | sort); do
      if [ -d "$hit" ] && [ -n "$(find "$hit" -maxdepth 1 -name '*.gguf' -print -quit 2>/dev/null)" ]; then
        FOUND="$hit"; break 2
      fi
    done
  done <<EOF_CONV
$(python3 "$SRC/lib/systemdfile.py" conventions)
EOF_CONV
  [ -n "$FOUND" ] && sed -i "s|^LLAMA_MODELS=.*|LLAMA_MODELS=$FOUND|" "$LOCAL_ENV"
  [ -d "$HOME/llama.cpp" ] && sed -i "s|^LLAMA_CPP=.*|LLAMA_CPP=$HOME/llama.cpp|" "$LOCAL_ENV"
  echo "  -> $LOCAL_ENV (from the template; stays local)"
  [ -n "$FOUND" ] && echo "     models found at $FOUND"                   || echo "     no .gguf found anywhere — set LLAMA_MODELS in that file by hand"
  echo "     GATEWAY_HOST is empty: a hostname cannot be derived, and guessing"
  echo "     one would point the smoke test at a machine that is not yours."
else
  echo "  = $LOCAL_ENV (left untouched)"
fi

if [ ! -e "$HOME/.config/cc-gateway.env" ]; then
  cp "$SRC/cc-gateway.env.template" "$HOME/.config/cc-gateway.env"
  echo "  -> ~/.config/cc-gateway.env (from the template; stays local)"
else
  echo "  = ~/.config/cc-gateway.env (left untouched)"
fi
systemctl --user daemon-reload 2>/dev/null || true

if [ "$USER_ONLY" = "1" ]; then
  echo
  echo "User-side part done. For /etc, call again without --user-only."
  exit 0
fi

# --- system wide: copies, needs sudo -------------------------------------
echo
echo "== system wide (sudo, copies) =="
sudo install -m755 "$SRC/llmprofile" /usr/local/bin/llm-profile
# 4 GiB instead of Fedora's 8: zram is swap that lives in RAM, and on a box
# whose pressure comes from pinned GTT it can only take memory away and delay
# the failure. The file itself carries the full reasoning.
sudo install -m644 "$SRC/zram-generator.conf" /etc/systemd/zram-generator.conf
# Read by llm-profile, which is installed either way.
sudo mkdir -p /etc/llm-profile /var/log/llm-profile
sudo install -m644 "$SRC"/env/*.env /etc/llm-profile/
sudo install -m644 "$SRC/systemd/llm-watch.service" /etc/systemd/system/
sudo systemctl daemon-reload
echo "  -> /usr/local/bin/llm-profile, /etc/llm-profile/, /etc/systemd/"

# --- the system unit, on request only --------------------------------------
#
# Everything below exists ONLY for llama@.service, which is why it is behind a
# flag. Its predecessor was a hand-written file installed unconditionally; it
# never ran on this machine, and by 27.08. it had silently drifted to the wrong
# binary (Vulkan, unpatched — the '////' corruption) and to half the memory
# ceilings. Generated now, from the unit that actually runs.
STALE=/etc/systemd/system/llama@.service
if [ "$SYSTEM_UNIT" = "1" ]; then
  echo
  echo "== system unit llama@.service (generated from llama-user@.service) =="
  # A system unit may not execute out of a home directory (SELinux, 203/EXEC),
  # so the three scripts it calls get a system location of their own.
  sudo install -m755 "$SRC/waitformodel" /usr/local/bin/llm-wait-for-model
  sudo install -m755 "$SRC/checkroom"    /usr/local/bin/llm-check-room
  sudo install -m755 "$SRC/llamaexec"    /usr/local/bin/llm-exec
  # budget.py imports systemdfile.py from its own directory, so the pair
  # travels together. /usr/local/lib, not bin: these are imported, not called.
  sudo mkdir -p /usr/local/lib/llm-profile
  sudo install -m644 "$SRC/lib/budget.py" "$SRC/lib/systemdfile.py" \
    /usr/local/lib/llm-profile/
  sudo install -m644 "$LOCAL_ENV" /etc/llm-stack.env
  python3 "$SRC/lib/systemunit.py" > "$SRC/.llama@.service.generated" || exit 1
  sudo install -m644 "$SRC/.llama@.service.generated" "$STALE"
  rm -f "$SRC/.llama@.service.generated"
  sudo systemctl daemon-reload
  echo "  -> $STALE, /usr/local/bin/llm-{exec,check-room,wait-for-model}"
  echo "  Verify:  python3 setup/lib/systemunit.py --check"
  echo
  echo "  NOTE: on Fedora with SELinux this unit cannot start — it execs a"
  echo "  binary under \$HOME (AVC denied, 203/EXEC). setup/README.md has the"
  echo "  relabel that fixes it, and the user unit that does not need one."
elif [ -e "$STALE" ]; then
  echo
  echo "  ! $STALE exists but was NOT generated by this run."
  echo "    It is the hand-written predecessor, removed from the repo on 27.08."
  echo "    because it had drifted to the wrong binary and half the ceilings."
  echo "    Regenerate it:   bash setup/install.sh --system-unit"
  echo "    Or remove it:    sudo rm $STALE && sudo systemctl daemon-reload"
fi

cat <<'END'

Done. Cross-check:

  bash setup/check.sh          differences between repo and system
  bash tests/run.sh            logic, without GPU and without a service
  llm-profile probe

Start:

  systemctl --user --now enable llama-user@qwen38   # model server
  systemctl --user --now enable cc-gateway          # gateway

  The USER service, not llama@laguna: on Fedora with SELinux a system service
  may not execute a binary from the home directory (AVC denied, 203/EXEC).
  Details in setup/README.md.

  So that both come up at boot, once:
      sudo loginctl enable-linger $USER

  And the weekly cleanup of the saved prefixes — without it the store fills up
  to AUTO_MAX_GB, after which the gateway saves nothing more and only says so
  in the log:
      systemctl --user --now enable prefix-cleanup.timer

Check that --swa-full bites (only for models WITH a sliding window —
laguna, gemma26, gemma31, gptoss; qwen38 has none and needs no switch):

  journalctl --user -u llama-user@laguna -n 300 | grep "full-size SWA cache"
  -> llama_kv_cache_iswa: using full-size SWA cache

Note: the .env files are systemd syntax and are NOT bash-sourceable.
A '. file.env' fails silently, because bash reads 'VAR=value command args'.
END
