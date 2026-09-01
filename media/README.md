# media/ — the Torch world, behind a hard border

This directory is RESERVED for the modalities that need a PyTorch/ROCm
stack — text-to-audio, text-to-video, ComfyUI-class pipelines. As of
01.09.2026 it is a skeleton: text-to-image runs WITHOUT any of this, on
stable-diffusion.cpp (ggml/Vulkan), as ordinary workload profiles in
`setup/workloads/` — see the border rule below for why that split is the
whole point.

## The border rules (from the media work order, 01.09.2026)

These are what made the monorepo decision safe. Each one is enforced or
enforceable; eroding them is what would justify splitting this directory
into its own repository.

1. **The base installation stays torch-free.** Nothing under `setup/`,
   `bench/`, `tests/` or `tools/` imports anything from `media/`.
   Enforced: `tests/run.sh` fails on any such import before running a
   single test.
2. **The contract gate keeps its property.** `bash tests/run.sh` runs
   without GPU, without network, in seconds. Anything here that needs a
   GPU gets a live suite, never a gate test.
3. **Every workload registers with the budget guard before its first
   start.** A `media/` job is a foreign workload like any other: it gets a
   profile in `setup/workloads/` (fields born UNMEASURED, filled with date
   + method + machine), and it starts through `bench/sideserver.py
   --workload` — never by hand next to production. A start that can walk
   around the guard is a design fault, not a missing feature
   (bench/sideserver.py's docstring counts the times that cost this
   machine).
4. **Repo culture applies unchanged.** Measured, not estimated; every
   figure carries date and origin; a new checker is shown red once before
   its green counts.

## Environment convention

One virtualenv per modality, under `~/.venvs/` — machine state like
~/llama.cpp, NOT inside this tree, so a workload profile can name it with
the @HOME@ token (a profile may not carry a repo path). Created and
refreshed by the modality's own setup script:

    ~/.venvs/media-<modality>          # created by media/<modality>/setup-venv.sh
    media/<modality>/requirements.txt  # top-level intent, dated
    media/<modality>/requirements.lock # the venv the figures were measured on —
                                       # the torch lane's LLAMA_BIN. setup-venv.sh
                                       # installs exactly this; --relock regenerates
                                       # and says out loud that the profile's
                                       # measured figures are claims again

Interpreters come from **uv** (`uv venv --python 3.12`). The first draft of
this file said "plain venv, not uv" — the machine refuted that within the
hour (01.09.2026): it ships python3.14 only, and chatterbox's dependency
chain does not compile there (spacy-pkuseg C++ build failure). uv fetches
a managed 3.12 without root; the uv binary itself sits in ~/.local/bin,
bootstrapped once via pip.

## Structure

    media/
      audio/     text-to-audio — empty until its first measured workload
      video/     text-to-video — empty until its first measured workload

A future workload here begins the same way sdxl did: a profile in
`setup/workloads/` born UNMEASURED, one fenced run, declared figures, then
a bench with a machine checker that has been seen red.
