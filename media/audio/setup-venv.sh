#!/usr/bin/env bash
# setup-venv.sh — the Torch lane's environment, OUTSIDE the repo tree.
#
#   bash media/audio/setup-venv.sh          create/refresh ~/.venvs/media-audio
#
# The venv lives under $HOME (like ~/llama.cpp and ~/stable-diffusion.cpp:
# machine state, not repo state) so that setup/workloads/*.env can point at
# it with the @HOME@ token — a profile may not carry a repo path, because
# the repo's location is a property of this machine. The wrapper script is
# COPIED into the venv's bin for the same reason; re-run this script after
# editing it (the copy is the deploy step, and this line is the reminder).
#
# PYTHON 3.12 VIA uv, and that is measured, not taste: this machine ships
# python3.14 ONLY, and chatterbox's dependency chain does not compile there
# (spacy-pkuseg C++ build error, 01.09.2026 — full log in that night's
# session). uv fetches a managed interpreter without root. media/README.md
# originally said "plain venv, not uv"; the machine refuted that the same
# night, and the README says so now.
#
# Torch comes from the CPU index ON PURPOSE. ROCm torch on gfx1151 is its
# own measurement project — a backend nobody here has measured is not a
# default, it is an experiment (media/README.md, rule 4). The day someone
# runs it: new venv, new measurements, profile figures re-declared.
set -euo pipefail
cd "$(dirname "$0")"

VENV="${MEDIA_AUDIO_VENV:-$HOME/.venvs/media-audio}"
UV="${UV_BIN:-$HOME/.local/bin/uv}"

[ -x "$UV" ] || {
  echo "no uv at $UV — bootstrap once with any python that has pip:" >&2
  echo "  pip install --user uv    (or: python3 -m venv /tmp/b && /tmp/b/bin/pip install uv" >&2
  echo "   && cp /tmp/b/bin/uv ~/.local/bin/)" >&2
  exit 1
}

if [ ! -x "$VENV/bin/python" ]; then
  echo "creating $VENV (managed CPython 3.12)"
  "$UV" venv --python 3.12 --seed "$VENV"
fi

# The LOCK is the torch lane's LLAMA_BIN (architecture review, 01.09.2026):
# the venv is what the profile's figures were measured on, and a refresh
# that resolves different transitives silently turns them into claims. So:
# with a lock present, install EXACTLY it; regenerating is an explicit act
# (--relock) that says out loud what it costs.
if [ "${1:-}" = "--relock" ] || [ ! -f requirements.lock ]; then
  echo "installing torch (CPU index) ..."
  "$UV" pip install --python "$VENV/bin/python" --quiet torch torchaudio \
    --index-url https://download.pytorch.org/whl/cpu
  echo "installing requirements.txt (fresh resolve) ..."
  "$UV" pip install --python "$VENV/bin/python" --quiet -r requirements.txt
  # Written to a TEMP file and moved into place only on success: the
  # group redirect used to truncate requirements.lock BEFORE freeze ran,
  # so a failed or interrupted freeze left a header-only lock — which the
  # next plain run's `uv pip sync` then faithfully enforced, stripping
  # the measured venv bare (review, 01.09.2026). mv on the same
  # filesystem is atomic; the trap covers the abort paths.
  lock_tmp="requirements.lock.tmp.$$"
  trap 'rm -f "$lock_tmp"' EXIT
  {
    echo "# Frozen $(date +%d.%m.%Y) by setup-venv.sh --relock."
    echo "# THIS is the torch lane's LLAMA_BIN: setup-venv.sh installs exactly"
    echo "# this set when the file exists, and regenerating it turns every"
    echo "# measured figure in setup/workloads/chatterbox.env back into a"
    echo "# claim until re-measured under bench/sideserver.py."
    "$UV" pip freeze --python "$VENV/bin/python"
  } > "$lock_tmp"
  mv "$lock_tmp" requirements.lock
  echo
  echo "requirements.lock REGENERATED — the measured figures in"
  echo "setup/workloads/chatterbox.env were taken on the previous stack and"
  echo "are claims now: re-measure under bench/sideserver.py and re-declare."
  # Not only a request anymore: budget.workload_plan compares this hash
  # against the `lock sha256:` stamp in WORKLOAD_MEASURED_ON and degrades
  # the figures to ESTIMATE until the profile is re-declared with it
  # (ultrareview, 01.09.2026).
  echo "the guard enforces this now — the new lock identity to declare is:"
  echo "  lock sha256:$(sha256sum requirements.lock | cut -c1-12)"
else
  echo "installing requirements.lock (the measured stack, exactly) ..."
  "$UV" pip sync --python "$VENV/bin/python" --quiet requirements.lock
fi

# The shebang is REWRITTEN to the venv's python: a plain copy kept
# `#!/usr/bin/env python3`, so calling the deployed tool directly ran it
# under the system 3.14 — where the import fails and the error message
# recommends re-running this script, which would not have helped (review,
# 01.09.2026).
sed "1s|.*|#!$VENV/bin/python|" chatterbox_tts.py > "$VENV/bin/chatterbox-tts-cli"
chmod +x "$VENV/bin/chatterbox-tts-cli"

echo "done. Sanity: $("$VENV/bin/python" -c 'import torch; print("torch", torch.__version__)')"
echo "wrapper: $VENV/bin/chatterbox-tts-cli (a COPY — re-run this script after edits)"
