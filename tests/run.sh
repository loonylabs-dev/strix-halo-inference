#!/usr/bin/env bash
# Every test that runs without a GPU, without llama-server and without a
# running service. Takes under two seconds.
#
#   bash tests/run.sh              everything
#   bash tests/run.sh -v           with names
#   bash tests/run.sh -k router    only matching ones
#
# Return value 0 = all green. Its counterparts are tests/live_prefix.sh (needs
# the GPU) and setup/smoketest.sh (needs the running stack).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

# The media/ border (media/README.md, rule 1): the base stack never imports
# from the Torch world, or this gate loses exactly the property its header
# promises — no GPU, no heavy deps, seconds. A cheap grep, run BEFORE the
# suite so a violation is the first line anyone sees, not a buried import
# error. Its red was seen once on 01.09.2026 (a planted import) before the
# green counted.
if grep -rEn '^[[:space:]]*(from[[:space:]]+media[.[:space:]]|import[[:space:]]+media([.[:space:]]|$))' \
     --include='*.py' setup bench tools tests 2>/dev/null; then
  echo "FAIL: the base stack imports from media/ — that border is what keeps" >&2
  echo "this gate torch-free (media/README.md, rule 1)." >&2
  exit 1
fi

exec python3 -m unittest discover -s tests -t tests "$@"
