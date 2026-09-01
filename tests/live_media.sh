#!/usr/bin/env bash
# live_media.sh — do the media workloads still produce EXACTLY their pinned
# output? The determinism lane.
#
#   bash tests/live_media.sh              --quick: qwen3-tts + sdxl (~5 min)
#   bash tests/live_media.sh --all        every pinned workload (~40 min)
#   bash tests/live_media.sh --selftest   prove the comparison can go red (no GPU)
#
# Every measured workload is byte-deterministic at its pinned seed (measured
# 01.09.2026, up to six reps across process lifetimes and one refactor), so
# regression testing here is not statistics — it is a HASH COMPARISON. Each
# profile declares WORKLOAD_SMOKE_SHA256; this lane runs a 1-rep bench
# through the sideserver fence and diffs. A flip means build, weights or
# flags changed the output: sometimes deliberately (re-declare, with the
# report), never silently.
#
# It STOPS PRODUCTION once per workload (sideserver's dance, dead man's
# switch armed) — this is the GPU lane the 15-second gate deliberately is
# not. Reports go to ~/.cache/llm-stack/live-media/, NOT bench/reports/:
# verification output is not evidence, and a nightly lane must not spray
# report directories into a public repo.
#
# The selftest was seen RED on 01.09.2026 before the first green counted:
# it tampers a profile copy's pin and demands the comparison fail.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"

QUICK="qwen3-tts sdxl"
DEST_ROOT="$HOME/.cache/llm-stack/live-media/$(date +%Y-%m-%d_%H%M)"

say() { printf '%s\n' "$*"; }

expected_hash() {  # $1 = profile path
  python3 setup/lib/systemdfile.py value "$1" WORKLOAD_SMOKE_SHA256
}

bench_for_kind() {  # $1 = kind
  case "$1" in
    image) echo "bench/imagebench.py" ;;
    audio) echo "bench/audiobench.py" ;;
    video) echo "bench/videobench.py" ;;
    *)     return 1 ;;
  esac
}

got_hash() {  # $1 = result.json, $2 = kind
  python3 - "$1" "$2" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
key = "sequence_sha256" if sys.argv[2] == "video" else "sha256"
print(next((r[key] for r in d.get("reps", []) if r.get(key)), ""))
EOF
}

compare() {  # $1 = name, $2 = profile, $3 = result.json -> 0 ok / 1 flip
  local kind want got
  kind="$(python3 setup/lib/systemdfile.py value "$2" WORKLOAD_KIND)"
  want="$(expected_hash "$2")"
  got="$(got_hash "$3" "$kind")"
  if [ -z "$got" ]; then
    say "FAIL  $1 — the run produced no hash (see $3)"
    return 1
  fi
  if [ "$want" != "$got" ]; then
    say "FAIL  $1 — HASH FLIP"
    say "      pinned $want"
    say "      got    $got"
    say "      Build, weights or flags changed the output. If deliberate:"
    say "      re-declare WORKLOAD_SMOKE_SHA256 with the report behind it."
    return 1
  fi
  say "ok    $1 — output byte-identical to the pin (${want:0:12}...)"
  return 0
}

if [ "${1:-}" = "--selftest" ]; then
  # Red first: a tampered pin against a REAL committed result must fail.
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  src="setup/workloads/qwen3-tts.env"
  # tail -1: the NEWEST committed report (they sort chronologically). The
  # first version took the oldest, so the first legitimate pin
  # re-declaration — new report committed, old ones kept as history —
  # would have compared the new pin against the old report and left this
  # selftest permanently red (review, 01.09.2026).
  result="$(ls bench/reports/*_audio_qwen3-tts/result.json | tail -1)"
  sed 's/^WORKLOAD_SMOKE_SHA256=.*/WORKLOAD_SMOKE_SHA256=0000000000000000000000000000000000000000000000000000000000000000/' \
    "$src" > "$tmp/tampered.env"
  if compare "tampered-pin" "$tmp/tampered.env" "$result" >/dev/null 2>&1; then
    say "SELFTEST FAILED: a tampered pin was accepted — the lane checks nothing"
    exit 1
  fi
  compare "genuine-pin" "$src" "$result" || {
    say "SELFTEST FAILED: the genuine pin did not verify against its own report"
    exit 1
  }
  say "selftest green: the tampered pin went red, the genuine one passed"
  exit 0
fi

WORKLOADS="$QUICK"
[ "${1:-}" = "--all" ] && WORKLOADS="$(bash setup/lib/models.sh workloads | tr '\n' ' ')"

# The unit to fence is DERIVED, never hard-wired. The first version said
# `--stop llama-user@qwen38`: with any OTHER model serving, that stop was a
# no-op, every bench refused beside the still-serving server — and the
# teardown then STARTED qwen38, whose Conflicts= stopped the serving
# model. A production change from a test script, against the repo's own
# hard rule (review, 01.09.2026). models.sh asks the PROCESS on the port,
# not a unit file.
SERVING="$(bash setup/lib/models.sh serving | head -1)"
STOP_ARGS=()
if [ -n "$SERVING" ]; then
  STOP_ARGS=(--stop "llama-user@$SERVING")
  say "fence production: llama-user@$SERVING (derived from models.sh serving)"
else
  say "note: no llama-server is serving — running without a production fence"
fi

mkdir -p "$DEST_ROOT"
failures=0
for name in $WORKLOADS; do
  profile="setup/workloads/$name.env"
  want="$(expected_hash "$profile")"
  if [ -z "$want" ]; then
    say "skip  $name — no WORKLOAD_SMOKE_SHA256 pinned"
    continue
  fi
  kind="$(python3 setup/lib/systemdfile.py value "$profile" WORKLOAD_KIND)"
  bench="$(bench_for_kind "$kind")" || { say "skip  $name — kind $kind has no bench"; continue; }
  dest="$DEST_ROOT/$name"
  say "run   $name (1 rep through the fence -> $dest)"
  # Video needs the wide bounds; harmless for the others. --deadline 70:
  # deadline_covers now charges the settle and release waits (180 s each)
  # on top of job-timeout + slack, so 3000 s of job needs 3660 s of
  # deadline — 60 min stopped covering it when the arithmetic got honest.
  python3 bench/sideserver.py --workload "$profile" ${STOP_ARGS[@]+"${STOP_ARGS[@]}"} \
      --deadline 70 --job-timeout 3000 -- \
      python3 "$REPO/$bench" --workload "$REPO/$profile" --reps 1 \
      --dest "$dest" --note "live_media determinism lane" \
      > "$dest.fence.log" 2>&1
  compare "$name" "$profile" "$dest/result.json" || failures=$((failures + 1))
done

if [ "$failures" -gt 0 ]; then
  say "$failures workload(s) FLIPPED — read the fence logs under $DEST_ROOT"
  exit 1
fi
say "all pinned workloads byte-identical · logs under $DEST_ROOT"
