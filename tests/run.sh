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
exec python3 -m unittest discover -s tests -t tests "$@"
