# strix-halo-inference

[![tests](https://github.com/loonylabs-dev/strix-halo-inference/actions/workflows/tests.yml/badge.svg)](https://github.com/loonylabs-dev/strix-halo-inference/actions/workflows/tests.yml)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![GPU](https://img.shields.io/badge/GPU-gfx1151-red)](docs/setup/03-gpu-and-memory.md)
[![RAM](https://img.shields.io/badge/RAM-128%20GB-orange)](#does-this-fit-your-machine)
[![backend](https://img.shields.io/badge/backend-llama.cpp%20%C2%B7%20ROCm-lightgrey)](setup/patches/README.md)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.12%20%7C%203.13-blue)](.github/workflows/tests.yml)

**A local coding agent on a Strix Halo, with the settings measured rather than
guessed.** One configuration — Ryzen AI Max+ 395 (gfx1151), 128 GB of shared
memory — and every number here was taken on it.

```bash
bash setup/preflight.sh              # is this repo for your machine?
bash setup/install.sh                # once — writes ~/.config/llm-stack.env
bash setup/get-model.sh qwen38       # fetch: resumable, sha256-checked
bash setup/switch-model.sh qwen38    # serve it
```

The name you fetch is the name you serve. There is no `pull` command, because
a model **is** its profile in [`setup/env/`](setup/env/) — and the profile
carries where the weights come from, what a token of context costs in KV, and
the measurement behind every flag on its command line.

## Does this fit your machine?

`bash setup/preflight.sh` answers it in five seconds, without root, changing
nothing. It filters over measured values and never guesses about unmeasured
ones:

| RAM | profiles that fit as written |
|---|---|
| **128 GB** | 7 of 7 — the configuration everything here was measured on |
| 64 GB | 3 of 7 — `batch`, `gemma26`, `gemma31` |
| 32 GB | none: 17.6 weights + 6.0 buffer floor + 12 host is already over |

Nothing is scaled down to change that. A scaled number is a guess, and
measured numbers are the whole argument.

## What you get that `ollama` does not

`ollama` has something answering in five minutes, and that is a real
advantage. Three things it will not do on this hardware.

**Tell you what will freeze your machine.** There is no separate VRAM — the
GPU takes system RAM through GTT, and that allocation is **pinned**. A model
that does not fit does not page and does not get OOM-killed: it stops the
machine, takes every process with it, and leaves nothing in any log. That
happened three times in one day before [`setup/lib/budget.py`](setup/lib/budget.py)
existed. It now weighs the profile before every start and refuses:

```
REFUSING TO START qwen38: it needs about 70.1 GiB and it does not fit.
    the host has 44.2 GiB available, 12 must stay free
```

**Know the failures that do not announce themselves.** On gfx1151 the
dangerous defects do not raise. Output degenerates to `////`; a session
answers from another session's context. No error, no crash, no log line.
[`setup/defects.json`](setup/defects.json) is that knowledge as data, and
`python3 setup/lib/defects.py` says whether **your** build and flags are
exposed, guarded, or unaffected.

**Make a coding agent usable rather than impressive.** The founding
measurement: Claude Code against a local model cost about **140 seconds per
turn**, because a sliding-window cache threw the prompt away on every edit.
With `--swa-full` and a saved prefix it is **1.3 seconds**. The whole chain —
how a request body is read, which prefix id it produces, when a state may be
restored — is in [`setup/claude/`](setup/claude/) with the measurements behind
each step.

## Not a fork

The backend is llama.cpp **master plus a short list of patches**. The list,
with the defect and the measurement behind each, lives in
[`setup/patches/`](setup/patches/README.md); `python3 setup/lib/defects.py`
says whether a given build still needs them. A patch here is a debt being
worked off, not a feature — each is tied to an upstream issue, and when the
fix lands on master the local copy is retired. The build script guards the
set: a binary that silently lost a patch refuses instead of serving the
defect again.

Custom-format forks tie your model files and your pace to their releases.
The weights here are standard GGUF from public repositories — no private
quant format only one binary can read — and because the base is master,
every upstream improvement arrives at the next build, not at a fork's next
release.

## What it is not

- **Not a generic stack.** See the table above; `preflight.sh` says where you
  stand before you spend an afternoon.
- **Not a model benchmark.** Others do that with more scale than a home-grown
  battery survives, and such a battery ages the week a new model lands. What is
  measured here is the STACK: what fits, what it costs, and whether the answers
  stay CORRECT as the window fills — a property of the build and the flags,
  which nobody else measures for this hardware.

## Where to start

| If you want to … | go here |
|---|---|
| find out whether this repo is **for your machine** | `bash setup/preflight.sh` — run it first |
| **set the machine up from scratch** — BIOS to first token | [docs/setup/](docs/setup/README.md), six chapters |
| point **Claude Code or an OpenAI agent** at it — yours or somebody else's | [docs/CONSUMERS.md](docs/CONSUMERS.md), and `bash setup/consumer-info.sh` for the values |
| decide **which model to take**, and why Flash-Next is not served | [docs/MODELS.md](docs/MODELS.md) |
| see the **raw measurements** | [docs/measurements/](docs/measurements/README.md) and [`bench/reports/`](bench/reports/) |
| **repeat** them | [bench/](bench/) |
| **install** it | [setup/](setup/) |
| know **what is protected** | [docs/SECURITY.md](docs/SECURITY.md) — the model. [SECURITY.md](SECURITY.md) — where a finding goes |
| **change** something without breaking it | [tests/](tests/), and [CONTRIBUTING.md](CONTRIBUTING.md) |
| report what it did on **your** machine | an issue — the one thing this repo cannot measure for itself |

## Layout

| | |
|---|---|
| [`docs/setup/`](docs/setup/README.md) | getting a machine to the starting line: BIOS, Linux, ROCm, the GTT cap, the build, the first token |
| [`docs/`](docs/) | the model decision, pointing a client at it, the security model, the measurement records |
| [`setup/env/`](setup/env/) | the model registry — one profile per model, and nothing else holds a list |
| [`setup/lib/budget.py`](setup/lib/budget.py) | the one memory budget; refuses a start that would not fit |
| [`setup/defects.json`](setup/defects.json) | what is known to go wrong on this hardware, as data |
| [`setup/claude/`](setup/claude/) | the gateway: three zones, per-consumer tokens, prefix save and restore |
| [`setup/scripts/`](setup/scripts/) | the patched llama.cpp build, the GTT cap, model scouting and fetching |
| [`bench/`](bench/) | repeatable measurements, one report per run |
| [`tools/`](tools/) | synthetic request bodies, prefix save/restore |
| [`tests/`](tests/) | unit tests without a GPU, plus three end-to-end against it |

## Running it

```bash
systemctl --user --now enable llama-user@qwen38
systemctl --user --now enable llm-gateway
systemctl --user --now enable prefix-cleanup.timer
sudo loginctl enable-linger $USER      # so they come up at boot
```

User services, not system ones — [setup/README.md](setup/README.md) explains
why, and what SELinux has to do with it.

> **One warning before you measure anything.** GTT is pinned, so a model that
> does not fit freezes the machine rather than failing. Four ceilings exist to
> stop that; the first refuses the start before anything is allocated. Never
> start a second model by hand — use `python3 bench/sideserver.py`.
> [setup/README.md](setup/README.md) explains all four before you need them.

## Checking it

```bash
bash tests/run.sh                  # contracts between the parts (1024 tests, ~13 s, no GPU)
bash setup/check.sh                # configuration and state
bash setup/smoketest.sh            # function and protection, all three zones
bash tests/live_prefix.sh          # prefix saving end to end against the GPU
bash tests/live_concurrency.sh     # admission control under load
```

All of them return 0 when everything is right. What separates the levels is in
[tests/README.md](tests/README.md): `check.sh` and `smoketest.sh` look at the
running system, `tests/run.sh` at the contracts between the parts — which is
where the bugs live that break nothing and simply let an effect fail to appear.

## Repeating the measurements

```bash
python3 bench/speed.py --label qwen38-production
python3 bench/sweep.py --variants bench/variants/qwen38.json --restore qwen38
python3 bench/compare.py bench/reports/<stamp>_sweep_qwen38
```

Every run writes a report with full context to
`bench/reports/<date>_<model>_<build>/`. No captured requests are needed —
[`tools/synthetic.py`](tools/synthetic.py) produces Claude-Code-shaped bodies.

**And captures do not belong here.** A real one contains an e-mail address, a
`device_id`, an `account_uuid` and Anthropic's system prompt. `.gitignore`
covers them; they do not belong in this repository, not even a private one.
