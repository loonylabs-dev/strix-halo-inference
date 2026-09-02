#!/usr/bin/env bash
# Reports differences between this repo and the running system.
#
#   bash setup/check.sh
#
# Why it is needed: the user-side files are symlinks, drift is impossible
# there. /etc and /etc/systemd/system are copies — and that is exactly where it
# has already happened: /etc/llm-profile/laguna.env was a day older than the
# version in the repo, and nobody had noticed.
#
# Return value 0 = identical, 1 = differences found.
set -uo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$SRC")"
DIFF=0
# The model registry. Nothing in this script may name a model — which models
# exist comes from setup/env/*.env, and which one is meant comes from what is
# running. Hardcoded "laguna" survived here for a day after the switch to
# qwen38 and reported the production model as missing.
MODELS_REPO="$REPO"
# shellcheck source=lib/models.sh
. "$SRC/lib/models.sh"

head_() { printf "\n%s\n" "$1"; }
ok()    { printf "  \033[32m=\033[0m %s\n" "$1"; }
gone()  { printf "  \033[33m?\033[0m %s  (not installed)\n" "$1"; DIFF=1; }
old()   { printf "  \033[31m!\033[0m %s\n" "$1"; DIFF=1; }

# --- copy expected, but nothing running reads it -------------------------
info_copy() {   # like check_copy, but a difference is not a defect
  local q="$1" z="$2"
  if [ ! -e "$z" ]; then
    printf "  \033[33m?\033[0m %s  (not installed — only llama@.service reads it)\n" "$z"
  elif diff -q "$q" "$z" >/dev/null 2>&1; then
    ok "$z"
  else
    printf "  \033[33m?\033[0m %s differs — informational: the USER service reads \$HOME, not /etc\n" "$z"
  fi
}

# --- symlink expected ----------------------------------------------------
check_link() {   # $1 = repo source, $2 = system target
  local q="$1" z="$2" short="${2/#$HOME/~}"
  if [ ! -e "$z" ] && [ ! -L "$z" ]; then gone "$short"; return; fi
  if [ -L "$z" ]; then
    if [ "$(readlink -f "$z")" = "$(readlink -f "$q")" ]; then ok "$short -> repo"
    else old "$short points at $(readlink "$z") instead of into the repo"; fi
  else
    if cmp -s "$q" "$z"; then old "$short is a COPY (identical content) — run install.sh again"
    else old "$short is a COPY and differs:"; diff -u "$z" "$q" | head -12 | sed 's/^/      /'; fi
  fi
}

# --- copy expected -------------------------------------------------------
check_copy() {  # $1 = repo source, $2 = system target
  local q="$1" z="$2"
  if [ ! -e "$z" ]; then gone "$z"; return; fi
  if cmp -s "$q" "$z"; then ok "$z"
  else
    old "$z differs:"
    diff -u "$z" "$q" | sed -n '4,16p' | sed 's/^/      /'
  fi
}

# Is the gateway installed here at all?
#
# This repo holds two things: an inference layer for this hardware, and the
# gateway plus consumer tooling on top of it. The dependency points ONE way —
# the gateway needs the inference layer, the inference layer must not need
# the gateway. Until 26.08. this script did not know that: a machine running
# llama-server for something else got `gone()` on six links, DIFF=1, and
# "Differences found" with exit 1. Nothing was wrong with it.
#
# So the gateway is checked WHERE IT IS, and its absence is a fact, not a
# defect. Either signal is enough — the unit, or the gateway's config. The
# pre-rename names count as present too: on such a machine the gateway IS
# installed, and the checks below are what says it needs migrating.
gateway_present() {
  [ -e "$HOME/.config/systemd/user/llm-gateway.service" ] || \
  [ -r "${GATEWAY_ENV:-$HOME/.config/llm-gateway.env}" ] || \
  [ -e "$HOME/.config/systemd/user/cc-gateway.service" ] || \
  [ -r "$HOME/.config/cc-gateway.env" ]
}

# …and the Claude Code consumer on top of the gateway — router and profiles.
# A machine can serve DeepSeek Harness alone; their absence is a fact too.
claude_present() {
  [ -e "$HOME/.claude/bin/cc-router.py" ] || \
  [ -e "$HOME/.claude/profiles/local.json" ]
}

head_ "This machine's own answers"
# The one file that says what is specific to this computer. Not a copy of
# anything in the repo, so there is nothing to diff — what is checked is
# whether it exists and whether its answers still resolve. A model directory
# that has been unmounted or renamed is invisible until a start fails.
LOCAL_ENV="$(llm_local_env)"
if [ ! -e "$LOCAL_ENV" ]; then
  old "$LOCAL_ENV is missing — 'bash setup/install.sh' writes it from setup/local.env.template"
else
  ok "$LOCAL_ENV"
  MDIR="$(models_dir 2>/dev/null)"
  if [ -z "$MDIR" ]; then
    old "no model directory: LLAMA_MODELS is unset here and nothing in the config resolves"
  elif [ ! -d "$MDIR" ]; then
    old "LLAMA_MODELS points at $MDIR, which does not exist (unmounted?)"
  elif [ -z "$(find "$MDIR" -maxdepth 1 -name '*.gguf' -print -quit 2>/dev/null)" ]; then
    old "$MDIR exists but holds no .gguf — mounted empty?"
  else
    ok "models at $MDIR ($(find "$MDIR" -maxdepth 1 -name '*.gguf' | wc -l) files)"
  fi
  GH="$(local_var GATEWAY_HOST)"
  if [ -n "$GH" ]; then ok "gateway host $GH"
  else
    printf "  \033[33m?\033[0m no GATEWAY_HOST — 'smoketest.sh' skips the remote zone.\n"
    printf "    That is a choice, not a difference: it has no default on purpose.\n"
  fi
fi

head_ "User side — symlinks into the repo expected"
# The inference layer. waitformodel is ExecStartPre of llama-user@.service and
# the profiles ARE the model registry; both belong to the server, not to any
# consumer of it.
LIB="$HOME/.local/lib/llm-stack"
check_link "$SRC/waitformodel"                "$LIB/waitformodel"
check_link "$SRC/llamaexec"                   "$LIB/llamaexec"
# The memory guard. It matters MORE than the others that this is checked:
# checkroom deliberately fails OPEN when it cannot find budget.py — a guard
# that is missing has no opinion about whether the model fits, and refusing on
# that basis would turn a packaging mistake into an outage. The price of that
# choice is that a missing link is silent at start time. This is where it
# stops being silent.
check_link "$SRC/checkroom"                   "$LIB/checkroom"
check_link "$SRC/lib/budget.py"               "$LIB/budget.py"
check_link "$SRC/lib/systemdfile.py"          "$LIB/systemdfile.py"
# llama-probe.service execs it; unchecked before the 09/2026 move — a gap,
# not a choice.
check_link "$REPO/setup/scripts/probe.py"     "$LIB/probe.py"
for f in "$SRC"/env/*.env; do
  check_link "$f" "$HOME/.config/llm-profile/$(basename "$f")"
done

if gateway_present; then
  # The gateway and its sibling modules (they import each other by
  # directory), and prewarm, the gateway's saving arm.
  check_link "$SRC/gateway/gateway.py"          "$LIB/gateway.py"
  check_link "$SRC/gateway/dialects.py"         "$LIB/dialects.py"
  check_link "$SRC/gateway/modes.py"            "$LIB/modes.py"
  check_link "$SRC/gateway/tracelog.py"         "$LIB/tracelog.py"
  check_link "$REPO/tools/prewarm.py"           "$LIB/prewarm.py"
  check_link "$SRC/systemd/llm-gateway.service" "$HOME/.config/systemd/user/llm-gateway.service"
else
  printf "  \033[33m?\033[0m no gateway installed — its links are skipped.\n"
  printf "    That is a choice, not a difference.\n"
fi

if claude_present; then
  check_link "$SRC/claude/local.json"           "$HOME/.claude/profiles/local.json"
  check_link "$SRC/claude/hybrid.json"          "$HOME/.claude/profiles/hybrid.json"
  check_link "$SRC/claude/cc-router.py"         "$HOME/.claude/bin/cc-router.py"
else
  printf "  \033[33m?\033[0m no Claude Code consumer installed — router and profiles\n"
  printf "    are skipped. That is a choice, not a difference.\n"
fi

# Links install.sh USED to make. A symlink it stops creating is not a symlink
# it removes, and nothing else would notice: cc-cachefix2.py was linked into
# ~/.claude/bin until 27.08. while setup/README.md had called it superseded
# since 26.08. Reported, not deleted — removing files from a home directory is
# the operator's call.
for f in cc-cachefix.py cc-cachefix2.py; do
  if [ -e "$HOME/.claude/bin/$f" ]; then
    printf "  \033[33m?\033[0m ~/.claude/bin/%s is left over — the gateway replaced it.\n" "$f"
    printf "      rm ~/.claude/bin/%s\n" "$f"
  fi
done
# The 09/2026 move: everything consumer-agnostic left ~/.claude/bin for
# ~/.local/lib/llm-stack, and the model profiles left ~/.claude/env for
# ~/.config/llm-profile. The old links go DANGLING the moment the repo files
# move, so -L matters in every test below: -e alone is false for a dangling
# symlink and would pass over exactly the thing being reported.
for f in cc-gateway.py dialects.py modes.py tracelog.py prewarm.py \
         waitformodel llamaexec checkroom budget.py systemdfile.py probe.py; do
  p="$HOME/.claude/bin/$f"
  if [ -e "$p" ] || [ -L "$p" ]; then
    printf "  \033[33m?\033[0m ~/.claude/bin/%s is left over — moved to ~/.local/lib/llm-stack (09/2026).\n" "$f"
    printf "      rm ~/.claude/bin/%s\n" "$f"
  fi
done
if [ -d "$HOME/.claude/env" ] || [ -L "$HOME/.claude/env" ]; then
  printf "  \033[33m?\033[0m ~/.claude/env is left over — the profiles moved to ~/.config/llm-profile (09/2026).\n"
  printf "      rm -r ~/.claude/env\n"
fi
p="$HOME/.config/systemd/user/cc-gateway.service"
if [ -e "$p" ] || [ -L "$p" ]; then
  printf "  \033[33m?\033[0m ~/.config/systemd/user/cc-gateway.service is left over — the unit is llm-gateway since 09/2026.\n"
  printf "      systemctl --user disable cc-gateway 2>/dev/null; rm %s\n" "$p"
fi
# The gateway's own local files, reported only once install.sh has copied
# them to the new name — reporting them earlier would invite an rm BEFORE the
# copy, and these are the two files that cannot be regenerated from the repo.
for pair in "cc-gateway.env llm-gateway.env" "cc-gateway-tokens llm-gateway-tokens"; do
  set -- $pair
  if [ -e "$HOME/.config/$1" ] && [ -e "$HOME/.config/$2" ]; then
    printf "  \033[33m?\033[0m ~/.config/%s is left over — copied to %s (09/2026).\n" "$1" "$2"
    printf "      rm ~/.config/%s\n" "$1"
  fi
done
# Same rule for the gateway's cache: the seen file is copied by install.sh,
# the trace directory is diagnosis history and is moved rather than copied.
if [ -e "$HOME/.cache/cc-gateway-seen.json" ] && [ -e "$HOME/.cache/llm-gateway-seen.json" ]; then
  printf "  \033[33m?\033[0m ~/.cache/cc-gateway-seen.json is left over — copied to llm-gateway-seen.json (09/2026).\n"
  printf "      rm ~/.cache/cc-gateway-seen.json\n"
fi
if [ -d "$HOME/.cache/cc-gateway-trace" ]; then
  printf "  \033[33m?\033[0m ~/.cache/cc-gateway-trace is left over — the trace dir is llm-gateway-trace since 09/2026.\n"
  printf "      mv ~/.cache/cc-gateway-trace ~/.cache/llm-gateway-trace\n"
fi
# A viewer that was started before the move keeps the OLD directory: the path
# is read once at import, so the process goes on serving an empty table from a
# directory that no longer exists while the gateway writes to the new one —
# and the page says "no rows", not "wrong directory" (01.09.2026, an hour).
# Ask the running viewer instead of guessing from its start time; it reports
# the directory it is reading in /days.
for vpid in $(pgrep -f "tracelog\.py serve" 2>/dev/null); do
  vport=$(tr '\0' ' ' < "/proc/$vpid/cmdline" 2>/dev/null \
          | sed -n 's/.*--port  *\([0-9][0-9]*\).*/\1/p')
  [ -n "$vport" ] || vport=8092          # tools/tracelog.py's own default
  vdir=$(curl -s -m3 "http://127.0.0.1:$vport/days" 2>/dev/null \
         | sed -n 's/.*"dir": *"\([^"]*\)".*/\1/p')
  if [ -n "$vdir" ] && [ ! -d "$vdir" ]; then
    old "the trace viewer on :$vport (pid $vpid) reads $vdir, which does not exist — started before the directory moved, or given a TRACE_DIR that is gone; either way the table stays empty"
    printf "      kill %s && python3 %s/tools/tracelog.py serve --port %s\n" \
           "$vpid" "$REPO" "$vport"
  fi
done
if systemctl --user is-active cc-gateway.service >/dev/null 2>&1; then
  old "the pre-rename unit cc-gateway is RUNNING; llm-gateway is the name since 09/2026"
  printf "      systemctl --user disable --now cc-gateway\n"
  printf "      systemctl --user enable --now llm-gateway\n"
fi

head_ "System wide — copies"
# Copies drift; symlinks cannot. That asymmetry is the whole reason this
# script exists, and it is the reason as little as possible lives down here.
check_copy "$SRC/llmprofile"    /usr/local/bin/llm-profile
check_copy "$SRC/systemd/llm-watch.service" /etc/systemd/system/llm-watch.service
# Read by llm-profile, which is installed either way. The USER service takes
# its profile from $HOME since 25.08., so a difference here does NOT affect
# the running server — informational, not a defect.
for f in "$SRC"/env/*.env; do
  info_copy "$f" "/etc/llm-profile/$(basename "$f")"
done

head_ "System unit llama@.service — opt-in, for a host with no user session"
# Everything here exists only for that unit, and it is installed only with
# 'bash setup/install.sh --system-unit'. Absent is a CHOICE, not a difference:
# a desktop uses the user unit plus 'loginctl enable-linger'.
#
# The unit itself is not compared against a file in the repo, because there is
# no such file any more. It is DERIVED from llama-user@.service, and the
# comparison is with what that derivation produces right now. Its hand-written
# predecessor is why: never started, never noticed, and by 27.08. it pinned the
# Vulkan binary while production had long moved to the patched ROCm build — so
# enabling it would have served the unpatched build and the '////' corruption.
if [ ! -e /etc/systemd/system/llama@.service ]; then
  printf "  \033[33m?\033[0m not installed — that is a choice, not a difference.\n"
  printf "    bash setup/install.sh --system-unit   if this host has no user session.\n"
  # Files that exist ONLY for that unit. Until 27.08. they were installed
  # unconditionally, so a machine that never wanted the unit still carries
  # them — and with the unit absent nothing above watches them any more. An
  # unwatched copy of budget.py is exactly the thing this script exists to
  # notice, so it is reported rather than quietly left behind. Reported, not
  # deleted: removing files from /usr/local is the operator's call.
  ORPHANS=""
  for f in /usr/local/bin/llm-exec /usr/local/bin/llm-check-room \
           /usr/local/bin/llm-wait-for-model /usr/local/lib/llm-profile \
           /etc/llm-stack.env; do
    [ -e "$f" ] && ORPHANS="$ORPHANS $f"
  done
  if [ -n "$ORPHANS" ]; then
    printf "  \033[33m?\033[0m left over from before the unit became opt-in, and now unwatched:\n"
    for f in $ORPHANS; do printf "      %s\n" "$f"; done
    printf "    While they are there, setup/checkroom can still FIND a stale\n"
    printf "    budget.py in /usr/local/lib/llm-profile — it is the last entry in\n"
    printf "    its search path, and nothing watches it once the unit is gone.\n"
    printf "    Keep them (install --system-unit) or remove them:\n"
    printf "      sudo rm -r%s\n" "$ORPHANS"
    # Three paths, one word. /usr/local/lib/llm-profile holds two Python
    # modules and is disposable; /usr/local/bin/llm-profile is the power-profile
    # script and /etc/llm-profile is the model registry — both stay. The line
    # above names only the first, and that is worth saying out loud.
    printf "    (that is /usr/local/LIB/llm-profile. /usr/local/bin/llm-profile and\n"
    printf "     /etc/llm-profile are different things and both stay.)\n"
  fi
else
  if python3 "$SRC/lib/systemunit.py" --check >/dev/null 2>&1; then
    ok "/etc/systemd/system/llama@.service matches llama-user@.service"
  else
    old "/etc/systemd/system/llama@.service is not what llama-user@.service would produce
      python3 setup/lib/systemunit.py --check     says how
      bash setup/install.sh --system-unit         regenerates it"
  fi
  # The three scripts it execs, and the two modules the guard imports. A stale
  # budget.py here would not fail — it would guard the system unit with OLD
  # arithmetic and say nothing, which is the exact shape of the bug the guard
  # exists for.
  check_copy "$SRC/waitformodel"       /usr/local/bin/llm-wait-for-model
  check_copy "$SRC/checkroom"          /usr/local/bin/llm-check-room
  check_copy "$SRC/llamaexec"          /usr/local/bin/llm-exec
  check_copy "$SRC/lib/budget.py"      /usr/local/lib/llm-profile/budget.py
  check_copy "$SRC/lib/systemdfile.py" /usr/local/lib/llm-profile/systemdfile.py
  # Not from the repo — from this machine's own config, which the unit loads
  # as its first EnvironmentFile.
  if [ ! -e /etc/llm-stack.env ]; then
    old "/etc/llm-stack.env is missing, and the unit loads it"
  elif cmp -s "$LOCAL_ENV" /etc/llm-stack.env; then
    ok "/etc/llm-stack.env"
  else
    old "/etc/llm-stack.env differs from $LOCAL_ENV"
  fi
fi

head_ "Effective service environment — what the service REALLY reads"
# Comparing the env file in the repo with the one in /etc is not enough. A unit
# may carry several EnvironmentFile lines, and for the same variable the last
# one wins. That is how --slot-save-path came to hang on an untracked file in
# ~/.config here, while /etc/llm-profile/laguna.env had long stopped carrying
# it: the prefix saving only worked by accident. check.sh did report the copy
# as differing, but nobody could see WHY it still worked.
# Which model to check? The one that is actually serving, not a name written
# down here. Falls back through active -> enabled -> first in the registry, so
# the section still says something useful with nothing running.
INSTANCE="${INSTANCE:-$(models_serving | head -1)}"
[ -z "$INSTANCE" ] && INSTANCE="$(models_active  | head -1)"
[ -z "$INSTANCE" ] && INSTANCE="$(models_enabled | head -1)"
[ -z "$INSTANCE" ] && INSTANCE="$(models_all     | head -1)"
printf "  checking instance: %s\n" "$INSTANCE"
# Since 25.08. the user service reads the profile from the user's own
# directory — see the unit for why. /etc stays for llama@.service.
EXPECTED="$HOME/.config/llm-profile/$INSTANCE.env"
# 'show' rather than 'cat': cat gives the unit raw, with %i and %h. show names
# the resolved paths — exactly the ones the service really reads.
FILES=$(systemctl --user show "llama-user@$INSTANCE" -p EnvironmentFiles 2>/dev/null \
        | sed -n 's/^EnvironmentFiles=//p' | sed 's/ (ignore_errors=.*)$//')
if [ -z "$FILES" ]; then
  printf "  \033[33m?\033[0m llama-user@%s names no EnvironmentFile\n" "$INSTANCE"
else
  LAST=""
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    if grep -qs '^LLAMA_ARGS=' "$f"; then
      LAST="$f"
      [ "$f" != "$EXPECTED" ] && old "$f sets LLAMA_ARGS and is not in the repo"
    elif [ "$f" = "$(llm_local_env)" ]; then
      # Expected since 27.08.: the unit's first EnvironmentFile is this
      # machine's own answers, which is where $LLAMA_MODELS comes from. It
      # sets no LLAMA_ARGS, so it cannot decide which model runs.
      ok "$f (this machine's answers, loaded first)"
    elif [ "$f" != "$EXPECTED" ]; then
      printf "  \033[33m?\033[0m %s is loaded in addition — nothing in the repo puts it there\n" "$f"
    fi
  done <<< "$FILES"
  if [ "$LAST" = "$EXPECTED" ]; then
    ok "effective LLAMA_ARGS comes from $EXPECTED"
  elif [ -n "$LAST" ]; then
    old "effective LLAMA_ARGS comes from $LAST instead of $EXPECTED"
  fi
fi

# And what ends up in the process line? That is the one statement no file can
# argue away any more.
PID=$(pgrep -f "llama-server.*--alias $INSTANCE" | head -1)
if [ -n "$PID" ] && [ -r "$SRC/env/$INSTANCE.env" ]; then
  # The reader for LLAMA_ARGS lives in setup/lib/models.sh and NOWHERE else.
  # It used to be a copy right here, and a copy that is allowed to drift,
  # drifts: two bugs lived in it — a regex that ran on to the next VAR= and
  # reported the words of the COMMENT lines in between as missing arguments,
  # and shlex.split eating the quotes out of --chat-template-kwargs
  # {"a":false}. switch-model.sh needs the same reader; that made three
  # copies. tests/test_models.py pins that there is one.
  MISSING=$(tr '\0' '\n' < "/proc/$PID/cmdline" \
    | python3 -c '
import sys
have = [z.rstrip("\n") for z in sys.stdin]
want = sys.argv[1].split()
print(" ".join([x for x in want if x not in have]))' "$(model_args "$INSTANCE")")
  if [ -z "$MISSING" ]; then
    ok "running llama-server has every argument from the repo"
  else
    old "the running llama-server lacks these arguments from the repo: $MISSING"
  fi
fi

head_ "Documentation"
# One operator's convenience: a symlink somewhere handy that points at the
# repo's docs, so they can be reached without knowing where the repo is. The
# path used to be hard-wired to /mnt/shared/docs, which is one machine's
# answer — on anybody else's it reported a missing link forever and set the
# exit code. It is now opt-in: set DOCS_LINK in ~/.config/llm-stack.env, or
# leave it empty and this section says nothing.
DOCS_LINK="$(local_var DOCS_LINK)"
if [ -z "$DOCS_LINK" ]; then
  printf "  \033[33m?\033[0m no DOCS_LINK configured — nothing to check. Set it in\n"
  printf "    %s to have a shortcut to docs/ watched here.\n" "$(llm_local_env)"
elif [ -L "$DOCS_LINK" ] && [ "$(readlink -f "$DOCS_LINK")" = "$REPO/docs" ]; then
  ok "$DOCS_LINK -> repo"
elif [ -d "$DOCS_LINK" ]; then
  old "$DOCS_LINK is a real directory — a second copy of the docs"
else
  gone "$DOCS_LINK"
fi

head_ "Services"
# No model name is written down here. Which llama-user@ instances exist comes
# from the registry, which of them run comes from systemd — the old version
# asked after "llama-user@laguna" and kept reporting the production model as
# missing for a day after the switch to qwen38.
ACTIVE="$(models_active)"
case "$(printf '%s' "$ACTIVE" | grep -c .)" in
  0) printf "  \033[33m?\033[0m no llama-user@ instance is active\n" ;;
  1) E=$(systemctl --user is-enabled "$(model_unit "$ACTIVE")" 2>/dev/null)
     if [ "$E" = "enabled" ]; then ok "$(model_unit "$ACTIVE") (active, enabled) — $(model_title "$ACTIVE")"
     else old "$(model_unit "$ACTIVE") is active but ${E:-not enabled} — a reboot brings back a different model"; fi ;;
  *) old "MORE THAN ONE model is active: $(printf '%s' "$ACTIVE" | tr '\n' ' ')
      They all want port 8080. That is the Conflicts= failure — see
      setup/systemd/llama-user@.service." ;;
esac
# And the same question asked of the processes, which cannot be argued with.
SERVING="$(models_serving)"
case "$(printf '%s' "$SERVING" | grep -c .)" in
  0) : ;;
  1) [ "$SERVING" = "$ACTIVE" ] || old "systemd says $ACTIVE, the running process serves $SERVING" ;;
  *) old "$(printf '%s' "$SERVING" | grep -c .) llama-server processes are running: $(printf '%s' "$SERVING" | tr '\n' ' ')" ;;
esac
D=llm-gateway
A=$(systemctl --user is-active "$D" 2>/dev/null)
E=$(systemctl --user is-enabled "$D" 2>/dev/null)
if [ "$A" = "active" ] && [ "$E" = "enabled" ]; then ok "$D ($A, $E)"
elif systemctl --user is-active cc-gateway.service >/dev/null 2>&1; then
  printf "  \033[33m?\033[0m %s (%s, %s) — cc-gateway still serves; the switch-over lines are above\n" "$D" "${A:-?}" "${E:-?}"
else printf "  \033[33m?\033[0m %s (%s, %s)\n" "$D" "${A:-?}" "${E:-?}"; fi
L=$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)
[ "$L" = "yes" ] && ok "linger active — services start at boot" \
  || printf "  \033[33m?\033[0m Linger=%s — services start only on login\n" "${L:-?}"
if docker ps --filter name=cloudflared --format '{{.Names}}' 2>/dev/null | grep -q cloudflared; then
  ok "cloudflared is running (docker)"
else printf "  \033[33m?\033[0m cloudflared is not running\n"; fi

head_ "Timer"
# Without this timer the prefix store fills up to AUTO_MAX_GB; after that the
# gateway saves nothing more and only says so in the log.
T=$(systemctl --user is-enabled prefix-cleanup.timer 2>/dev/null)
if [ "$T" = "enabled" ]; then
  # The question is whether the CLEANUP works, not whether the timer has
  # happened to fire yet. A weekly timer enabled on the 25th cannot have
  # fired before the 31st, and reporting that as an open point every day
  # until then trains the reader to skip the line.
  #
  # BUT LastTriggerUSec IS NOT PROOF OF A FIRING, and this said it was until
  # 27.08.2026 — a green line reading "last fired Mon 2026-08-24 20:01:17"
  # for a timer that had never run anything. With Persistent=true systemd
  # writes ~/.local/share/systemd/timers/stamp-<unit> when the timer is first
  # activated, precisely so a newly enabled timer does not fire once for every
  # window it missed in the past. LastTriggerUSec reports that stamp. The
  # journal for that minute is empty, and the branch below could therefore
  # never be reached: the check could not tell "it fired" from "it was
  # switched on".
  #
  # So the stamp is not consulted for the verdict. Two things are asked
  # instead, and both are answerable:
  #
  #   did the SERVICE ever complete cleanly, however it was started
  #   did a run actually happen AT the stamp — asked of the journal
  #
  # The second one is what will eventually settle whether the timer path
  # works, without anybody remembering to look.
  STAMP=$(systemctl --user show prefix-cleanup.timer -p LastTriggerUSec --value 2>/dev/null)
  NEXT=$(systemctl --user show prefix-cleanup.timer -p NextElapseUSecRealtime --value 2>/dev/null)
  SVC_RUN=$(systemctl --user show prefix-cleanup.service -p ExecMainExitTimestamp --value 2>/dev/null)
  SVC_RC=$(systemctl --user show prefix-cleanup.service -p ExecMainStatus --value 2>/dev/null)

  TRIGGERED=no
  if [ -n "$STAMP" ] && [ "$STAMP" != "0" ] && [ "$STAMP" != "n/a" ]; then
    E=$(date -d "$STAMP" +%s 2>/dev/null || echo "")
    if [ -n "$E" ] && journalctl --user -u prefix-cleanup.service \
         --since "@$((E-60))" --until "@$((E+180))" -q -o cat 2>/dev/null \
         | grep -q .; then
      TRIGGERED=yes
    fi
  fi

  if [ "$TRIGGERED" = "yes" ]; then
    ok "prefix-cleanup.timer has fired and the service ran ($STAMP)"
  elif [ -n "$SVC_RUN" ] && [ "$SVC_RC" = "0" ]; then
    ok "prefix-cleanup: the service ran cleanly at $SVC_RUN. The TIMER has not triggered it yet — next ${NEXT:-unknown}"
  else
    printf "  \033[33m?\033[0m prefix-cleanup.timer active, and the cleanup has never run — prove it works once:  systemctl --user start prefix-cleanup.service\n"
  fi

  # Overdue is a different question from "not yet", and it is the one nobody
  # would notice: the due date passes, nothing runs, and the line above still
  # reads as an ordinary "not yet".
  if [ "$TRIGGERED" != "yes" ] && [ -n "$NEXT" ] && [ "$NEXT" != "n/a" ]; then
    NE=$(date -d "$NEXT" +%s 2>/dev/null || echo "")
    if [ -n "$NE" ] && [ "$NE" -lt "$(date +%s)" ]; then
      # old() prints the red line AND sets DIFF, which is what exit reads.
      # The first version of this branch printed the line by hand and set an
      # RC nothing consumes — so it would have gone red on screen and exited
      # 0. shellcheck caught it as "RC appears unused", in the same hour the
      # job was made blocking, in the code written against exactly this.
      old "prefix-cleanup.timer was due at $NEXT and still has not triggered the service"
      printf "      That is no longer 'not yet'. journalctl --user -u prefix-cleanup.timer\n"
    fi
  fi
else
  gone "prefix-cleanup.timer (${T:-not enabled})"
fi

head_ "Patched llama.cpp build (the corruption fix lives OUTSIDE this repo)"
# Load-bearing and invisible: the profile points LLAMA_BIN at a build that
# carries setup/patches/hip-integrated-off.patch. A llama.cpp update wipes
# the patch and the corruption comes back SILENTLY — the server starts, the
# answers just turn to garbage once a second slot is used. So check three
# things: the binary exists, the source still carries the patch, and the
# binary is not older than the source.
LLAMA_SRC="${LLAMA_SRC:-$HOME/llama.cpp}"
PATCH_SRC="$LLAMA_SRC/ggml/src/ggml-cuda/ggml-cuda.cu"
STABLE="$LLAMA_SRC/build-rocm-patched"
PATCH_BIN="$STABLE/bin/llama-server"
if [ ! -e "$PATCH_BIN" ]; then
  old "patched build missing: $PATCH_BIN — bash setup/scripts/build-llama.sh"
elif [ ! -e "$PATCH_SRC" ]; then
  gone "llama.cpp source not found at $PATCH_SRC"
else
  # grep -c, not grep -q: with 'set -o pipefail' a -q would SIGPIPE the
  # producer and a hit would read as a miss. Same note as for journalctl below.
  if [ "$(grep -c "gfx1151/ROCm: trusting prop.integrated" "$PATCH_SRC")" != 0 ]; then
    ok "llama.cpp source still carries the patch"
  else
    old "THE PATCH IS GONE from $PATCH_SRC — bash setup/scripts/build-llama.sh, or two slots corrupt again"
  fi

  # WHICH build is in use, and does it match the source? Comparing file dates
  # only ever said "older" or "newer", which is a half-answer once builds are
  # versioned: the stamp names the commit the binary was actually built from.
  if [ -L "$STABLE" ]; then
    ok "active build: $(readlink "$STABLE")"
  else
    printf "  \033[33m?\033[0m %s is a real directory, not a symlink — 'bash setup/scripts/build-llama.sh --list' explains\n" "$STABLE"
  fi
  STAMP="$STABLE/.build-stamp"
  SRC_HEAD=$(git -C "$LLAMA_SRC" rev-parse HEAD 2>/dev/null)
  if [ -r "$STAMP" ]; then
    BUILT_FROM=$(sed -n 's/^patch_commit=//p' "$STAMP" | head -1)
    if [ -n "$SRC_HEAD" ] && [ "$BUILT_FROM" = "$SRC_HEAD" ]; then
      ok "the binary was built from exactly the source that is checked out"
    elif [ -n "$BUILT_FROM" ]; then
      printf "  \033[33m?\033[0m the source has moved on since this build (%s vs %s) — that is fine until you want the new code:  bash setup/scripts/build-llama.sh --activate\n" \
        "${BUILT_FROM:0:9}" "${SRC_HEAD:0:9}"
    fi
  elif [ "$PATCH_BIN" -nt "$PATCH_SRC" ]; then
    ok "patched binary is newer than the source (no build stamp — built before build-llama.sh)"
  else
    printf "  \033[33m?\033[0m patched binary is older than the source and has no stamp — bash setup/scripts/build-llama.sh --list\n"
  fi
  # Does the RUNNING process actually come out of the active build?
  RPID=$(pgrep -x llama-server 2>/dev/null | head -1)
  if [ -n "$RPID" ]; then
    REXE=$(readlink -f "/proc/$RPID/exe" 2>/dev/null)
    if [ "$REXE" = "$(readlink -f "$PATCH_BIN")" ]; then
      ok "the running server comes out of the active build"
    else
      printf "  \033[33m?\033[0m the running server runs from %s, the active build is %s — restart to pick it up\n" \
        "${REXE/#$HOME/\~}" "$(readlink -f "$PATCH_BIN" | sed "s#^$HOME#~#")"
    fi
  fi
fi

head_ "Saved prefixes"
# A slot state restored into the wrong model is garbage. switch-model.sh
# writes the owner into the store; a store without one is a store nobody can
# safely hand to the next model.
SLOTS="${SLOTS:-$HOME/.cache/llama-slots}"
if [ ! -d "$SLOTS" ]; then
  printf "  \033[33m?\033[0m %s does not exist yet\n" "${SLOTS/#$HOME/\~}"
elif [ ! -r "$SLOTS/.owner" ]; then
  printf "  \033[33m?\033[0m %s has no .owner marker — the next switch has to guess who wrote these prefixes\n" "${SLOTS/#$HOME/\~}"
else
  OWNER=$(tr -d '[:space:]' < "$SLOTS/.owner")
  RUNNING=$(models_serving | head -1)
  if [ -z "$RUNNING" ] || [ "$OWNER" = "$RUNNING" ]; then
    ok "prefixes belong to $OWNER ($(du -sh "$SLOTS" 2>/dev/null | cut -f1))"
  else
    old "the prefix store belongs to $OWNER but $RUNNING is serving — a restore would feed one model's KV state to another"
  fi
fi

head_ "Running state"
if pgrep -x llama-server >/dev/null; then
  SLOTS=$(curl -s -m3 http://127.0.0.1:8080/slots 2>/dev/null | grep -o '"id"' | wc -l)
  ok "llama-server is running, $SLOTS slots"
  # Check both the system and the user service. On Fedora with SELinux the
  # server runs as a user service (llama-user@), see README.
  #
  # CAREFUL: do NOT use "| grep -q" here. The script runs with 'set -o
  # pipefail'; grep -q stops at the first match, journalctl gets SIGPIPE, and
  # pipefail reports the pipeline as failed — a hit would look like a miss.
  # So count instead of stopping.
  N_SWA=$(( $(journalctl --user -u 'llama-user@*' -n 400 --no-pager 2>/dev/null | grep -c "full-size SWA cache")
          + $(journalctl        -u 'llama@*'      -n 400 --no-pager 2>/dev/null | grep -c "full-size SWA cache")
          + $(grep -rs "full-size SWA cache" "$HOME/llm-setup"/*.log 2>/dev/null | wc -l) ))
  # Only models WITH sliding window attention need --swa-full. Which ones
  # those are is a property of the MODEL and is declared in its profile as
  # MODEL_SWA — it used to be a list of four names right here, which is a
  # second place to forget when a model is added.
  SWA_OF_RUNNING="$(model_swa "$INSTANCE")"
  if [ "$N_SWA" -gt 0 ]; then
    ok "--swa-full is active"
  elif [ "$SWA_OF_RUNNING" = "no" ]; then
    ok "--swa-full not needed ($INSTANCE has no sliding window)"
  elif [ "$SWA_OF_RUNNING" = "unknown" ]; then
    printf "  \033[33m?\033[0m MODEL_SWA is 'unknown' for %s — read the model config and set it in %s\n" \
      "$INSTANCE" "setup/env/$INSTANCE.env"
  else
    printf "  \033[33m?\033[0m --swa-full not found in the log (started by hand? other log target?)\n"
  fi
else
  printf "  \033[33m?\033[0m llama-server is not running\n"
fi
if curl -s -m3 http://127.0.0.1:8090/gateway/status >/dev/null 2>&1; then
  COLL=$(curl -s -m3 http://127.0.0.1:8090/gateway/status \
         | python3 -c 'import json,sys; print(len(json.load(sys.stdin).get("collisions",[])))' 2>/dev/null)
  ok "gateway is running, $COLL prefix collisions"
  [ "${COLL:-0}" != "0" ] && old "collisions present — see docs/CONSUMERS.md"
  # MAX_INFLIGHT is asked from the server at start. If the gateway starts
  # before llama-server at boot, that fails and it stays at the default of 2 —
  # with -np 4 the gateway would leave half the slots unused.
  MI=$(curl -s -m3 http://127.0.0.1:8090/gateway/status \
       | python3 -c 'import json,sys; print(json.load(sys.stdin).get("max_concurrent"))' 2>/dev/null)
  if [ -n "$SLOTS" ] && [ -n "$MI" ] && [ "$SLOTS" != "0" ]; then
    if [ "$MI" = "$SLOTS" ]; then
      ok "MAX_INFLIGHT $MI matches the slot count"
    else
      old "MAX_INFLIGHT is $MI, the server has $SLOTS slots — restart the gateway"
    fi
  fi
else
  printf "  \033[33m?\033[0m gateway is not running\n"
fi
# The RAM prompt cache is a MiB budget (-cram), and llama-server says when the
# budget is too small for the load it is actually asked to hold: one WARN line
# per entry it throws out to make room. Counting those asks the machine,
# instead of computing an entry size from a declared KiB/token — the derived
# route is the one this repo keeps getting wrong, and an entry turns out to
# carry the draft head and the checkpoints on top of the KV (measured
# 02.09.2026, bench/suites/cram-state-size.py).
#
# An eviction is not a fault by itself: a cache that never evicts is merely a
# cache nobody filled. What it means is that whatever was evicted has to be
# prefilled again if its conversation comes back, and for a Claude-Code-sized
# prefix that is minutes, not seconds. See
# `cram-holds-only-one-claude-code-state` in setup/defects.json.
#
# grep -c, not grep -q — same pipefail trap the SWA count above documents.
N_EVICT=$(( $(journalctl --user -u 'llama-user@*' --since "-24h" --no-pager 2>/dev/null | grep -c "making room for prompt cache entry")
          + $(journalctl        -u 'llama@*'      --since "-24h" --no-pager 2>/dev/null | grep -c "making room for prompt cache entry") ))
if [ "$N_EVICT" = "0" ]; then
  ok "no prompt-cache evictions in the last 24 h"
else
  printf "  \033[33m?\033[0m %s prompt-cache evictions in the last 24 h — -cram is smaller\n" "$N_EVICT"
  printf "    than the load this machine actually serves. Each one is a conversation\n"
  printf "    that pays a full prefill if it comes back:\n"
  printf "    journalctl --user -u 'llama-user@*' | grep 'making room'\n"
fi

# AUTO_SAVE against a profile that cannot be saved. The serving profile may
# deliberately omit --slot-save-path — flashnext does, because the QSA indexer
# has no state save/load and a poisoned restore degrades output rather than
# failing. The gateway does not know that: it auto-saves every warmed prefix,
# the save cannot succeed, the id never enters SAVED, and it is retried on the
# NEXT cold prefix. Each retry is a prewarm subprocess that holds the one slot
# for its full 600 s timeout while nothing in Claude Code shows a request at
# all — measured 02.09.2026, found only because the GPU was at full load with
# an empty request log after a restart.
#
# Cold prefixes are exactly what a restart produces, so this fires when it
# costs most. The profile names AUTO_SAVE=0 in its checklist; a checklist is
# what gets missed at 12:20, which is why it is checked here.
if pgrep -x llama-server >/dev/null 2>&1; then
  SAVEPATH=$(tr '\0' '\n' < "/proc/$(pgrep -x llama-server | head -1)/cmdline" 2>/dev/null \
             | grep -c -- "--slot-save-path")
  AS=$(grep -sE "^[[:space:]]*AUTO_SAVE=" "$HOME/.config/llm-gateway.env" | tail -1 | cut -d= -f2)
  if [ "${SAVEPATH:-0}" = "0" ] && [ "${AS:-1}" != "0" ]; then
    old "the served profile has no --slot-save-path, but AUTO_SAVE is ${AS:-unset}"
    printf "    Every automatic save will fail and be retried on the next cold\n"
    printf "    prefix, each retry holding the slot for 600 s. Set AUTO_SAVE=0 in\n"
    printf "    ~/.config/llm-gateway.env and restart llm-gateway.\n"
  elif [ "${SAVEPATH:-0}" = "0" ]; then
    ok "AUTO_SAVE=0, matching a profile without --slot-save-path"
  fi
fi

head_ "The watchdog for the silent failure modes"
# The registry above says WHAT can go wrong. This says whether anything would
# notice if it did. Both two-slot defects here end as degraded output with no
# error anywhere, so an unattended machine needs something that asks.
PT=$(systemctl --user is-enabled llama-probe.timer 2>/dev/null)
if [ "$PT" = "enabled" ]; then
  ok "llama-probe.timer enabled"
  LAST=$(systemctl --user show llama-probe.service -p ExecMainStatus --value 2>/dev/null)
  # The EXIT CODE cannot say whether anything was looked at. Since 29.08. a
  # probe that could not get past production's single slot exits 0 with the
  # verdict BUSY — correct, because a queue is not a fault, and misleading
  # here, because `last probe passed` for a round that checked nothing is the
  # silent-failure detector failing silently. The verdict is in the journal,
  # in the fixed format probe.py prints: "<date> <time>  VERDICT  detail".
  LASTV=$(journalctl --user -u llama-probe.service -n 200 --no-pager -o cat 2>/dev/null \
          | sed -nE 's/^[0-9-]{10} [0-9:]{8}  ([A-Za-z]+).*/\1/p' | tail -1)
  case "$LAST" in
    0) case "$LASTV" in
         BUSY|UNKNOWN)
           printf "  \033[33m?\033[0m the last probe got no turn ($LASTV) — the slot was busy,\n"
           printf "    so nothing was checked. Not a fault; not a pass either\n" ;;
         *) ok "last probe passed${LASTV:+ ($LASTV)}" ;;
       esac ;;
    "") printf "  \033[33m?\033[0m llama-probe.service has not run yet\n" ;;
    *) old "the last probe FAILED (status $LAST${LASTV:+, $LASTV}) — journalctl --user -u llama-probe" ;;
  esac
else
  printf "  \033[33m?\033[0m llama-probe.timer is %s — nothing would notice a\n" "${PT:-not installed}"
  printf "    poisoned server; it answers, it just answers wrongly\n"
fi

head_ "Memory budget — what the running profile claimed, and what it took"
# The half that keeps the declaration honest. setup/lib/budget.py refuses a
# start that would not fit; this asks the opposite question afterwards — was
# the number it refused or allowed on actually RIGHT?
#
# Why that matters more than it sounds: every figure in this repo that was
# DERIVED turned out wrong (the architecture arithmetic for KV by 4x, the
# per-prefix cache estimate by 3-4x, the Flash-Next footprint by 30 GiB), and
# every figure that was MEASURED held. A declared number with nothing checking
# it drifts back into an assertion — which is exactly what `-cram 32768` was
# when it got copied into five profiles.
#
# It reports, it does not edit. A profile is a file people are invited to
# change, and a tool that rewrites it behind them turns a measured number back
# into a copied one.
if pgrep -x llama-server >/dev/null 2>&1; then
  python3 - "$REPO" <<'PYBUDGET'
import os, sys
repo = sys.argv[1]
sys.path.insert(0, os.path.join(repo, "setup", "lib"))
import budget                                     # noqa: E402

argv = budget.running_argv()
if not argv:
    print("  \033[33m?\033[0m llama-server is running but its command line is unreadable")
    raise SystemExit(0)

alias = budget.flag(argv, "--alias") or ""
env = os.path.join(repo, "setup", "env", alias + ".env")
declared = budget.declared_kv(env) if alias else None
in_gtt = budget.declared_gtt(env) if alias else None
anon = budget.declared_anon(env) if alias else None
plan = budget.plan(argv, budget.weights_gib(argv), declared,
                   alias or "the running server", gtt_base=in_gtt, host_anon=anon)

for line in budget.render(plan, budget.read_machine(),
                          budget.Verdict(True, [], [])).split("\n"):
    print("  " + line)

cmp = budget.compare(plan, budget.observe(argv=argv))
if cmp is None:
    print("  \033[33m?\033[0m nothing observable yet — GTT or the weights could not be read")
    raise SystemExit(0)

if cmp.ok:
    print("  \033[32m=\033[0m GTT predicted %.1f GiB, observed %.1f (%+.0f %%) — not under-predicting"
          % (cmp.predicted, cmp.observed, cmp.margin * 100))
else:
    print("  \033[31m!\033[0m GTT predicted %.1f GiB, observed %.1f (%+.0f %%) — the guard UNDER-predicts"
          % (cmp.predicted, cmp.observed, cmp.margin * 100))
    print("    That is the dangerous direction: a profile that passes the guard and")
    print("    then takes more than it promised is how this machine froze. Re-measure")
    print("    the KV with the two-point method — load at two windows, take the")
    print("    difference — and correct MODEL_KV_KIB_PER_TOKEN in the profile.")
print("    %s" % cmp.note)

if declared is None and cmp.kv_upper is not None:
    print("  \033[33m?\033[0m %s declares no MODEL_KV_KIB_PER_TOKEN. Observed here: at MOST"
          % (alias or "this profile"))
    print("    %.1f KiB/token — an upper bound, because a single reading cannot" % cmp.kv_upper)
    print("    separate the KV from the compute buffers. Conservative is the right")
    print("    direction to declare in; write it into setup/env/%s.env with its date." % alias)
PYBUDGET
else
  printf "  \033[33m?\033[0m no llama-server running — nothing to weigh against the declaration\n"
fi

head_ "Known defects of this hardware and build (setup/defects.json)"
# A separate axis from everything above. The checks so far ask "is the system
# what the repo says"; this one asks "is what the repo says still SAFE on this
# hardware". install.sh cannot fix a defect, so the two must not share a
# conclusion line — but an exposed one must still fail the exit code, because
# that is the only thing a script notices.
DEFECTS=0
if python3 "$REPO/setup/lib/defects.py"; then :; else DEFECTS=1; fi

printf "\n"
if [ "$DIFF" = "0" ]; then
  printf "Repo and system are identical.\n"
else
  printf "Differences found. 'bash setup/install.sh' puts the repo back.\n"
fi
if [ "$DEFECTS" != "0" ]; then
  printf "A known defect is EXPOSED — see the section above. install.sh does not fix that.\n"
fi
exit $(( DIFF | DEFECTS ))
