#!/usr/bin/env bash
# Switch the serving model.
#
#   bash setup/switch-model.sh --list             what exists, what runs
#   bash setup/switch-model.sh qwen38             switch to qwen38
#   bash setup/switch-model.sh flashnext --dry-run  print the plan, change nothing
#   bash setup/switch-model.sh qwen38 --sync-etc  also refresh the /etc copies
#
# The models come from setup/lib/models.sh, which reads setup/env/*.env. There
# is no list of model names in this script — there used to be, as a PAIR
#
#     case "$NEW" in qwen38) OLD=laguna ;; laguna) OLD=qwen38 ;; esac
#
# and a third model did not fit into it at all. The model to stop is now the
# one that is actually running, which is also the only answer that stays right
# when something was started by hand.
#
# Two things this script is built around
# --------------------------------------
# 1. EVERYTHING IS CHECKED BEFORE ANYTHING IS TOUCHED. setup/tunnel/switch.sh
#    cost that lesson: it died on an unset $TOKEN, but only AFTER it had
#    swapped the tunnel configuration and restarted the container. Preflight
#    here exits 2 and leaves the running model exactly as it was;
#    tests/test_models.py pins that property.
#
# 2. /etc IS NOT LOAD-BEARING ANY MORE, so it must not be able to block a
#    switch. Since 25.08. the user service reads its profile from
#    %h/.config/llm-profile/%i.env (a symlink into this repo). /etc/llm-profile is
#    read by /usr/local/bin/llm-profile and by the opt-in system unit, which
#    is generated on request and which SELinux keeps from running here anyway.
#    The old script began with an
#    unconditional `sudo install` — and a REMOTE operator cannot authenticate
#    to sudo at all: the prompt appears on the machine's own screen. That made
#    the stack unswitchable from outside for exactly the reason the profile
#    had been moved out of /etc in the first place. Now: synced when passwordless
#    sudo happens to be there, on request with --sync-etc, and otherwise skipped
#    with a note.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
# The registry resolves its own location through symlinks, which is right when
# it is sourced from an installed copy — but here the caller KNOWS its repo, and
# saying so is what makes this script testable against a scratch repo.
MODELS_REPO="$REPO"
# shellcheck source=lib/models.sh
. "$REPO/setup/lib/models.sh"

SLOTS="${SLOTS:-$HOME/.cache/llama-slots}"
GATEWAY="${GATEWAY:-http://127.0.0.1:8090}"
# SERVER is NOT a constant — it comes from the profile, below. It used to be
# hard-wired to 8080, and profiles in the repo do not all serve there
# (gemma31 8082, batch 8083; gemma26 was 8081 until 04.09.2026, when it was
# measured and moved to 8080 so it could be switched to at all). Switching to
# one of those would
# have started the model, then waited fifteen minutes on port 8080, then
# failed with "never served /slots" — with the old model already stopped and
# disabled. The model was fine. The script was looking in the wrong place.
GATEWAY_ENV="${GATEWAY_ENV:-$HOME/.config/llm-gateway.env}"
# Half-migrated machine (pulled, install.sh not yet run): fall back to the
# pre-09/2026 file rather than reading no port at all. check.sh is what
# reports that state; this script just has to survive it.
[ -r "$GATEWAY_ENV" ] || ! [ -r "$HOME/.config/cc-gateway.env" ] || \
  GATEWAY_ENV="$HOME/.config/cc-gateway.env"
UNIT_FILE="$REPO/setup/systemd/llama-user@.service"
LOCAL_JSON="$REPO/setup/claude/local.json"

# Is there a gateway on this machine at all?
#
# The dependency between the two halves of this repo points ONE way. The
# harness — the gateway, the Claude Code profiles, prewarm — may REQUIRE the
# inference layer. The inference layer may NOTICE the harness; it must not
# NEED it. Until 26.08. this script needed it: it read the gateway's port and
# aborted when the profile disagreed, restarted the gateway unconditionally,
# and smoked only through it. On a machine that serves llama-server to
# anything else, the model could not be switched at all — for reasons that
# had nothing to do with the model.
#
# Same failure class as the two the units carry a note about: ExecStartPost
# without a leading '-', which let a convenience take the server down with
# it, and EnvironmentFile= without one, which stopped the service starting
# when the file went away. A companion made compulsory.
#
# Either signal is enough: the config file (then we know its port) or the
# unit (then there is something to restart).
gateway_present() {
  [ -r "$GATEWAY_ENV" ] || \
  [ -e "$HOME/.config/systemd/user/llm-gateway.service" ] || \
  [ -e "$HOME/.config/systemd/user/cc-gateway.service" ]
}
if gateway_present; then GW_PRESENT=1; else GW_PRESENT=0; fi
# Restart what is actually INSTALLED. On a half-migrated machine only the
# pre-rename unit file cc-gateway.service exists; restarting llm-gateway
# there would start a second gateway that loses the race for the port.
# Decided by unit FILES, not by `systemctl is-active`: the tests drive this
# script in a sandbox $HOME, and an is-active probe would answer from the
# test machine's real user bus — a result that changes with the weather.
GW_UNIT="llm-gateway"
[ -e "$HOME/.config/systemd/user/llm-gateway.service" ] || \
  ! [ -e "$HOME/.config/systemd/user/cc-gateway.service" ] || GW_UNIT="cc-gateway"

NEW=""; DRY=0; SYNC_ETC=auto
for a in "$@"; do
  case "$a" in
    --list|-l)  bash "$REPO/setup/lib/models.sh" table
                echo
                echo "serving right now (from the process command line): $(models_serving | tr '\n' ' ')"
                exit 0 ;;
    --dry-run|-n) DRY=1 ;;
    --sync-etc)   SYNC_ETC=yes ;;
    --no-sync-etc) SYNC_ETC=no ;;
    -h|--help)  sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*)         echo "unknown option: $a" >&2; exit 2 ;;
    *)          [ -n "$NEW" ] && { echo "one model at a time, got '$NEW' and '$a'" >&2; exit 2; }
                NEW="$a" ;;
  esac
done

say()  { printf '%s\n'            "$*"; }
step() { printf '\n== %s\n'       "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
ok()   { printf '  \033[32m=\033[0m %s\n' "$*"; }
die()  { printf '\n\033[31mABORT\033[0m %s\n' "$*" >&2; exit 2; }
run()  { if [ "$DRY" = 1 ]; then printf '  would: %s\n' "$*"; else "$@"; fi; }
run_write() {   # $1 = content, $2 = file. Separate from run(): a redirect is
                # not an argv, and --dry-run must not create the file.
  if [ "$DRY" = 1 ]; then printf '  would: write "%s" to %s\n' "$1" "$2"
  else printf '%s\n' "$1" > "$2"; fi
}

# --------------------------------------------------------------------------
# Preflight — nothing below this block may run if anything here fails.
# --------------------------------------------------------------------------
[ -n "$NEW" ] || die "no model given.

    bash setup/switch-model.sh --list      what exists
    bash setup/switch-model.sh <model>     switch"

models_known "$NEW" || die "unknown model '$NEW'. The repo knows:
$(models_all | sed 's/^/      /')

    A model IS its profile: create setup/env/$NEW.env, then
    bash setup/install.sh --user-only"

step "0/7 preflight for '$NEW'  ($(model_title "$NEW"))"

# The profile the SERVICE reads. Not the one in the repo — the symlink.
USER_ENV="$(model_user_env "$NEW")"
if [ ! -e "$USER_ENV" ]; then
  die "the service would read $USER_ENV, and it does not exist.
    The unit names it without a leading '-', so it fails loudly rather than
    starting llama-server with an empty \$LLAMA_ARGS and serving a wrong
    model. Fix:  bash setup/install.sh --user-only"
fi
if [ -L "$USER_ENV" ] && [ "$(readlink -f "$USER_ENV")" = "$(readlink -f "$(model_repo_env "$NEW")")" ]; then
  ok "profile -> repo ($USER_ENV)"
else
  warn "$USER_ENV is not a symlink into the repo — it can drift. bash setup/install.sh --user-only"
fi

ARGS="$(model_args "$NEW")"
[ -n "$ARGS" ] || die "$(model_repo_env "$NEW") carries no LLAMA_ARGS"

# Which port does THIS profile serve on, and is it the one the gateway asks?
PORT="$(printf '%s' "$ARGS" | awk '{for(i=1;i<NF;i++) if($i=="--port") print $(i+1)}' | head -1)"
PORT="${PORT:-8080}"
SERVER="${SERVER:-http://127.0.0.1:$PORT}"
# An if, not 'sed … 2>/dev/null | head -1' in a command substitution: with
# 'set -o pipefail' a sed on a missing file fails the whole pipeline, and a
# failing command substitution under 'set -e' ends the script — silently, with
# an empty stderr, BEFORE any of the checks below have run. Found by
# tests/test_models.py, which runs against a home that has no gateway config.
GW_URL=""
if [ -r "$GATEWAY_ENV" ]; then
  GW_URL="$(sed -n 's/^LLAMA_URL=//p' "$GATEWAY_ENV" | head -1)"
fi
GW_URL="${GW_URL:-http://127.0.0.1:8080}"
GW_PORT="${GW_URL##*:}"; GW_PORT="${GW_PORT%%/*}"
if [ "$GW_PRESENT" = 0 ]; then
  # Nothing to disagree with. The abort below protects consumers from being
  # left on a dead port; with no gateway installed there is no such consumer,
  # and the wait in step 5 targets $SERVER from this profile anyway.
  ok "serves on port $PORT — no gateway here, so no port to agree with"
elif [ "$PORT" = "$GW_PORT" ]; then
  ok "serves on port $PORT — the port the gateway asks ($GW_URL)"
else
  die "$(model_repo_env "$NEW") serves on port $PORT, but the gateway talks to
    $GW_URL.

    Switching would start the model and leave every consumer talking to a
    dead port. Either change --port in the profile to $GW_PORT, or point the
    gateway at this one (LLAMA_URL in $GATEWAY_ENV) and restart it.

    The profiles on other ports (8081-8083) were written to run ALONGSIDE the
    main model. The Conflicts= line in llama-user@.service makes that
    impossible today — only one instance can be active. That is a real
    inconsistency in the repo, not something this script should paper over."
fi

# Conflicts=. systemd has no wildcard here, so the unit must NAME every
# instance. A model missing from that line is the worst failure this stack
# has: both servers start, the second loses the race for the port, the unit
# says "active" — and the gateway talks to whichever won. Silent and total.
CONFLICTS="$(python3 "$MODELS_LIB_DIR/systemdfile.py" directive "$UNIT_FILE" Conflicts)"
if printf '%s' "$CONFLICTS" | grep -qw -- "$(model_unit "$NEW")"; then
  ok "llama-user@.service conflicts with $NEW — no second server can start"
else
  die "$(model_unit "$NEW") is MISSING from the Conflicts= line of
    setup/systemd/llama-user@.service.

    Without it, starting $NEW does not stop the running model: two
    llama-servers come up, one loses the race for the port, and the unit
    still reports 'active'. Add it there — the system unit is DERIVED from
    this one since 27.08., so there is no second list to keep in step — then
    'bash tests/run.sh' will go green again."
fi

BIN="$(model_bin "$NEW")"
[ -x "$BIN" ] || die "LLAMA_BIN points at $BIN, which is not executable.
    Build it:  bash setup/scripts/build-llama.sh --help"
ok "binary $(printf '%s' "$BIN" | sed "s#^$HOME#~#")"
case "$BIN" in
  *rocm-patched*)
    if grep -q "gfx1151/ROCm: trusting prop.integrated" \
         "$HOME/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu" 2>/dev/null; then
      ok "llama.cpp source still carries the gfx1151 patch"
    else
      warn "the gfx1151 patch is NOT in the llama.cpp source — the binary may predate an update (setup/patches/README.md)"
    fi ;;
esac

# The model file, checked with the same script the unit uses as ExecStartPre.
# One reader, unit-tested (tests/test_scripts.py::TestWaitForModel).
if [ -n "$(model_gguf "$NEW")" ]; then
  if LLAMA_ARGS="$ARGS" WAIT_MAX=1 bash "$REPO/setup/waitformodel" >/dev/null 2>&1; then
    ok "model file readable ($(model_gguf "$NEW"))"
  else
    # Two very different causes, and the old message only named one of them.
    # "the mount is gone" and "you never downloaded it" want opposite actions,
    # and telling a newcomer to look at fstab when they simply have no weights
    # sends them into the machine's plumbing for nothing.
    if [ -d "$(dirname "$(model_gguf "$NEW")")" ]; then
      die "the model file is not here:  $(model_gguf "$NEW")

    The directory exists, so this is most likely a model that has not been
    fetched yet rather than a missing mount:

      bash setup/get-model.sh $NEW

    If you expected it to be there: a model volume in fstab with nofail passes
    silently when it is not mounted — check with  findmnt -T '$(model_gguf "$NEW")'
    that the directory is not empty."
    else
      die "the model file is not readable:  $(model_gguf "$NEW")
    Its directory does not exist at all. A model volume in fstab with nofail
    passes silently when it is not mounted, which looks exactly like this:
    findmnt -T '$(dirname "$(model_gguf "$NEW")")'"
    fi
  fi
fi

# Does it FIT? Same arithmetic the unit runs as ExecStartPre, called here so
# the answer arrives before anything is stopped rather than after — a switch
# that takes production down and then refuses to bring the replacement up is a
# worse outcome than a switch that never started.
#
# The check is against the machine as it is RIGHT NOW, which during a switch
# still includes the outgoing model's GTT. That is why it only warns here and
# refuses in ExecStartPre: at this point in the script the memory the new model
# needs is legitimately still held by the old one. The unit asks again, after
# the old server is gone and after waiting for GTT to actually fall.
#
# --static is the question that CAN be answered now: would this profile fit an
# idle machine of this size at all? A `no` there is a property of the profile,
# not of the moment, and it will not improve by stopping anything.
if [ -r "$REPO/setup/lib/budget.py" ]; then
  if BUDGET_OUT="$(LLAMA_ARGS="$ARGS" \
        MODEL_KV_KIB_PER_TOKEN="$(model_kv "$NEW")" \
        MODEL_GTT_BASE_GIB="$(model_gtt_gib "$NEW")" \
        MODEL_HOST_ANON_GIB="$(model_anon_gib "$NEW")" \
        python3 "$REPO/setup/lib/budget.py" --from-env --static --check 2>&1)"; then
    ok "memory budget fits this machine"
    printf '%s\n' "$BUDGET_OUT" | sed 's/^/  /'
  else
    die "this profile does not fit this machine, and stopping the running model
    will not change that — the numbers below are about the profile, not about
    what happens to be running.

$BUDGET_OUT"
  fi
else
  warn "setup/lib/budget.py is missing — the memory budget was NOT checked"
fi

# Sliding window. Only a warning: gemma26/gemma31/gptoss are KNOWN to be
# missing --swa-full (docs/DOCUMENTS.md, open points) because the memory cost
# there is unmeasured. tests/test_models.py carries that list explicitly, so a
# NEW model cannot join it unnoticed.
if [ "$(model_swa "$NEW")" = "yes" ] && ! printf '%s' "$ARGS" | grep -q -- "--swa-full"; then
  warn "MODEL_SWA=yes but no --swa-full: every changed question re-processes the whole prompt (measured 100.2 s vs 10.4 s)"
fi

# Does the window this profile offers match what Claude Code will ask for?
# A real session on the second machine sent 78,826 tokens and was rejected
# with "exceeds the available context size". Only meaningful where Claude
# Code is actually the consumer — this reads setup/claude/local.json, which
# is harness configuration and has nothing to say to anyone else.
if [ "$GW_PRESENT" = 1 ]; then
python3 - "$ARGS" "$LOCAL_JSON" "$MODELS_LIB_DIR" <<'PY' || true
import json, sys
sys.path.insert(0, sys.argv[3])
from systemdfile import flag                 # the one place that knows -c == --ctx-size
args = sys.argv[1].split()
try:
    ctx = int(flag(args, "-c", "--ctx-size", default=0))
    np_ = int(flag(args, "-np", "--parallel", default=1))
except ValueError:
    sys.exit(0)
per_slot = ctx // np_ if "--no-kv-unified" in args and np_ > 1 else ctx
try:
    want = int(json.load(open(sys.argv[2]))["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"])
except Exception:
    sys.exit(0)
if per_slot and want > per_slot:
    print("  \033[33m!\033[0m CLAUDE_CODE_MAX_CONTEXT_TOKENS=%d in setup/claude/local.json "
          "exceeds this profile's %d tokens per slot — requests will be rejected "
          "with 'exceeds the available context size'" % (want, per_slot))
elif per_slot:
    print("  \033[32m=\033[0m window %d per slot, local.json asks for %d" % (per_slot, want))
PY
fi

OLD="$(models_active | grep -vx "$NEW" || true)"
SERVING="$(models_serving | tr '\n' ' ')"
if [ -n "$OLD" ]; then
  ok "running now: $(printf '%s' "$OLD" | tr '\n' ' ') — will be stopped"
else
  ok "no other model is active"
fi
[ -n "${SERVING// /}" ] && ok "process command line says: $SERVING"

# Who owns the saved prefixes? A slot state restored into the WRONG model is
# garbage, so this question must be answered BEFORE anything moves — and it
# must not be answered by guessing. The store therefore carries its owner in
# a marker file that every switch writes; the derivation from the running
# model is only the fallback for a store written before this existed.
PARK_AS=""; RESTORE=0
if [ -d "$SLOTS" ] && [ -n "$(ls -A "$SLOTS" 2>/dev/null)" ]; then
  if [ -r "$SLOTS/.owner" ]; then
    SLOT_OWNER="$(tr -d '[:space:]' < "$SLOTS/.owner")"
    # The marker decides the target of an `rm -rf "$SLOTS.$SLOT_OWNER"` a few
    # steps down. It is written by this script and should always be a model
    # name — but a file that steers a recursive delete gets checked, not
    # trusted. A hand-edited or truncated .owner containing "../.." would
    # otherwise point that delete somewhere else entirely.
    if ! models_known "$SLOT_OWNER"; then
      die "$SLOTS/.owner says '$SLOT_OWNER', which is not a model in this repo.

    That file decides where the current prefixes get parked, and parking is a
    move plus a recursive delete of the previous parking spot. Fix it by hand:
      printf '%s\n' <model> > $SLOTS/.owner       if you know whose they are
      rm -rf $SLOTS                                if you do not"
    fi
    ok "saved prefixes belong to $SLOT_OWNER (from .owner)"
  else
    # Deliberate word splitting: each model name has to become its own LINE
    # so head -1 picks the first non-empty one. Quoting makes it one word.
    # shellcheck disable=SC2046,SC2086
    SLOT_OWNER="$(printf '%s\n' $OLD $(models_serving) | sed '/^$/d' | head -1)"
    [ -n "$SLOT_OWNER" ] && warn "no .owner in $SLOTS — deriving '$SLOT_OWNER' from what is running"
  fi
  [ -n "$SLOT_OWNER" ] || die "$SLOTS holds saved prefixes ($(du -sh "$SLOTS" | cut -f1)) but nothing says which model wrote them, and no model is running to derive it from.

    Restoring them into $NEW would feed one model's KV state to another.
    Decide by hand, then switch again:
      mv $SLOTS $SLOTS.<the model it belongs to>     keep them
      rm -rf $SLOTS                                  throw them away"
  [ "$SLOT_OWNER" != "$NEW" ] && PARK_AS="$SLOT_OWNER"
else
  SLOT_OWNER=""
fi
if [ "$SLOT_OWNER" != "$NEW" ] && [ -d "$SLOTS.$NEW" ]; then
  RESTORE=1
  ok "$NEW has parked prefixes ($(du -sh "$SLOTS.$NEW" | cut -f1)) — they come back"
fi

if [ "$(models_active | tr '\n' ' ')" = "$NEW " ] && [ "$DRY" = 0 ]; then
  say
  say "$NEW is already the active model. Nothing to do."
  say "  restart it:  systemctl --user restart $(model_unit "$NEW")"
  exit 0
fi

[ "$DRY" = 1 ] && say "
  (--dry-run: preflight passed, nothing below is executed)"

# --------------------------------------------------------------------------
# From here on the system is changed.
# --------------------------------------------------------------------------
step "1/7 /etc/llm-profile (read by llm-profile and by the opt-in system unit)"
do_sync=0
case "$SYNC_ETC" in
  yes) do_sync=1 ;;
  no)  say "  skipped (--no-sync-etc)" ;;
  auto)
    if [ "$DRY" = 1 ] || sudo -n true 2>/dev/null; then do_sync=1
    else
      say "  skipped: no passwordless sudo, and nothing that runs reads /etc."
      say "  To sync anyway:  bash setup/switch-model.sh $NEW --sync-etc"
    fi ;;
esac
if [ "$do_sync" = 1 ]; then
  if [ "$SYNC_ETC" = yes ]; then
    run sudo install -m 644 -o root -g root "$REPO"/setup/env/*.env /etc/llm-profile/
  else
    run sudo -n install -m 644 -o root -g root "$REPO"/setup/env/*.env /etc/llm-profile/ \
      || warn "the /etc sync failed — not fatal, see above"
  fi
  [ "$DRY" = 0 ] && ok "/etc/llm-profile refreshed"
fi

step "2/7 daemon-reload"
run systemctl --user daemon-reload

step "3/7 stop the old model and re-key the prefix store"
# Stop FIRST: llama-server holds the slot files open and keeps writing them.
for m in $OLD; do run systemctl --user stop "$(model_unit "$m")"; done
# Every model keeps its own store — including models that were never the
# immediate predecessor. The plan was decided in preflight.
if [ -n "$PARK_AS" ]; then
  run rm -rf "$SLOTS.$PARK_AS"
  run mv "$SLOTS" "$SLOTS.$PARK_AS"
  [ "$DRY" = 0 ] && ok "prefixes of $PARK_AS parked in $SLOTS.$PARK_AS"
fi
if [ "$RESTORE" = 1 ]; then
  # rmdir, not rm -rf: it can only succeed on an EMPTY directory, so a store
  # that unexpectedly still has content stops the switch instead of being
  # silently merged into $NEW's.
  if [ -z "$PARK_AS" ] && [ -d "$SLOTS" ]; then run rmdir "$SLOTS"; fi
  run mv "$SLOTS.$NEW" "$SLOTS"
  [ "$DRY" = 0 ] && ok "prefixes of $NEW brought back"
fi
run mkdir -p "$SLOTS"
run_write "$NEW" "$SLOTS/.owner"

step "4/7 swap the services"
for m in $OLD; do run systemctl --user disable "$(model_unit "$m")" || true; done
run systemctl --user enable --now "$(model_unit "$NEW")"

step "5/7 wait for the model"
if [ "$DRY" = 0 ]; then
  for _ in $(seq 1 450); do
    curl -sf --max-time 3 "$SERVER/slots" >/dev/null 2>&1 && break
    sleep 2
  done
  curl -sf --max-time 3 "$SERVER/slots" >/dev/null || die "llama-user@$NEW never served /slots — check:
    journalctl --user -u $(model_unit "$NEW") -n 80"
  ok "$SERVER/slots answers"

  # The check no unit file can argue away: exactly ONE server, serving $NEW.
  S="$(models_serving | tr '\n' ' ')"
  case "$(printf '%s' "$S" | wc -w)" in
    1) [ "${S// /}" = "$NEW" ] && ok "one llama-server, serving $NEW" \
         || die "the running server serves '${S// /}', not $NEW" ;;
    0) warn "no llama-server found by command line (started differently?)" ;;
    *) die "MORE THAN ONE llama-server is running: $S
    That is the Conflicts= failure. Stop them all and switch again:
      systemctl --user stop 'llama-user@*'" ;;
  esac
fi

# ONLY NOW the gateway, and the order is the whole point. The gateway asks the
# server for its slot count at startup (query_slots) and falls back to a
# default of 2 when nobody answers. Restarting it BEFORE the model is up —
# which is what this script did until 26.08. — therefore left MAX_INFLIGHT at
# 2 against a server with one slot, every single time. Nothing breaks
# visibly: the gateway simply admits a second request that llama.cpp then
# queues internally, where the priority ordering between local, LAN and
# remote no longer applies. Found by rehearsing a real switch; check.sh had
# been reporting it all along.
if [ "$GW_PRESENT" = 1 ]; then
  step "6/7 restart the gateway (MID_SYSTEM_TO_USER / KWARGS_BY_MODEL, and the slot count)"
  run systemctl --user restart "$GW_UNIT"
else
  step "6/7 gateway — none installed, skipped"
fi

# The smoke happens either way, and that is the point of asking. Step 5 proved
# the server ANSWERS /slots; only this proves it GENERATES. Through the
# gateway where there is one, because then the whole path is under test;
# straight at the server where there is not — "no gateway" must not come to
# mean "no verification", which would trade a coupling for a blind spot.
if [ "$GW_PRESENT" = 1 ]; then
  step "7/7 smoke through the gateway"
else
  step "7/7 smoke against the server"
fi
if [ "$DRY" = 0 ] && [ "$GW_PRESENT" = 0 ]; then
  curl -sf --max-time 300 --retry 2 --retry-delay 3 \
    "$SERVER/v1/chat/completions" \
    -H 'content-type: application/json' \
    -d "{\"model\":\"$NEW\",\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"Say ok.\"}]}" \
    | python3 -c "import json,sys; r=json.load(sys.stdin); \
c=r.get('choices') or []; assert c and c[0].get('message'), r; \
print('  smoke ok:', (c[0]['message'].get('content') or '')[:40])" || die "llama-server answered /slots but did not generate — check:
    journalctl --user -u $(model_unit "$NEW") -n 80"
elif [ "$DRY" = 0 ]; then
  # The gateway was just restarted and needs a moment to bind — smoking
  # straight away raced it once (25.08.) and crashed on an empty reply.
  for _ in $(seq 1 30); do
    curl -sf --max-time 3 "$GATEWAY/v1/models" >/dev/null 2>&1 && break
    sleep 1
  done
  curl -sf --max-time 300 --retry 2 --retry-delay 3 \
    "$GATEWAY/v1/messages" \
    -H 'content-type: application/json' -H 'anthropic-version: 2023-06-01' \
    -d "{\"model\":\"$NEW\",\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"Say ok.\"}]}" \
    | python3 -c "import json,sys; r=json.load(sys.stdin); \
assert r.get('content'), r; \
print('  smoke ok:', [b.get('text','') for b in r['content'] if b.get('type')=='text'][0][:40])" || die "the smoke through the gateway did not answer — check:
    journalctl --user -u $GW_UNIT -n 80"
fi

say
if [ "$DRY" = 1 ]; then
  say "DRY RUN — nothing was changed."
else
  say "DONE — $NEW is serving.  $(model_title "$NEW")"
  say
  say "Claude Code sessions started from now use the names in"
  say "setup/claude/local.json. Running sessions keep their old name; the"
  say "gateway passes it through and llama-server ignores it."
  if [ -n "$OLD" ]; then
    say
    say "Back:  bash setup/switch-model.sh $(printf '%s' "$OLD" | head -1)"
  fi
fi
