# strix-halo-inference

[![tests](https://github.com/loonylabs-dev/strix-halo-inference/actions/workflows/tests.yml/badge.svg)](https://github.com/loonylabs-dev/strix-halo-inference/actions/workflows/tests.yml)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![GPU](https://img.shields.io/badge/GPU-gfx1151-red)](docs/setup/03-gpu-and-memory.md)
[![RAM](https://img.shields.io/badge/RAM-128%20GB-orange)](#does-this-fit-your-machine)
[![backend](https://img.shields.io/badge/backends-llama.cpp%20%C2%B7%20sd.cpp%20%C2%B7%20qwentts.cpp-lightgrey)](setup/patches/README.md)
[![python](https://img.shields.io/badge/python-3.10%20%E2%80%93%203.14-blue)](.github/workflows/tests.yml)

**A Strix Halo as a measured inference machine: a local coding agent in
production — and text-to-image, speech and video under the same memory
authority.** One configuration — Ryzen AI Max+ 395 (gfx1151), 128 GB of
shared memory — and every number here was taken on it, with the date and
method beside it.

```bash
bash setup/preflight.sh              # is this repo for your machine?
bash setup/install.sh                # once — writes ~/.config/llm-stack.env
bash setup/get-model.sh qwen38       # fetch: resumable, sha256-checked
bash setup/switch-model.sh qwen38    # serve it
bash tests/run.sh                    # the gate (1305 tests, ~19 s, no GPU)
```

The name you fetch is the name you serve. There is no `pull` command, because
a model **is** its profile in [`setup/env/`](setup/env/) — and the profile
carries where the weights come from, what a token of context costs in KV, and
the measurement behind every flag on its command line.

## Does this fit your machine?

`bash setup/preflight.sh` answers it in five seconds, without root, changing
nothing:

| RAM | profiles that fit as written |
|---|---|
| **128 GB** | 7 of 7 — the configuration everything here was measured on |
| 64 GB | 3 of 7 — `batch`, `gemma26`, `gemma31` |
| 32 GB | none: 17.6 weights + 6.0 buffer floor + 12 host is already over |

Nothing is scaled down to change that — a scaled number is a guess. Not a
generic stack, on purpose; `preflight.sh` says where you stand before you
spend an afternoon.

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

> For the same reason, never start a second model — or a media workload — by
> hand: `python3 bench/sideserver.py` is the only safe way, and
> [setup/README.md](setup/README.md) explains the four ceilings before you
> need them.

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

**Keep a conversation, and know that keeping it works.** llama-server can save
a slot to a file and load it back; that is not the hard part. The hard part is
that on a hybrid model it silently did nothing: the restore reported success,
`/slots` confirmed the tokens were there, and the next request re-processed the
entire prompt anyway. Nobody would see it — there is no error, only a bill.
Found here on 05.09.2026 because this repo measures what others assume, traced
to two lines four hundred apart (`prompt.clear()` empties the context
checkpoints; the next prompt needs one and resets when it finds none), and
fixed in [`setup/patches/`](setup/patches/README.md) by carrying the
checkpoints beside the state:

```
a conversation continued from its file, RAM prompt cache OFF
    before   3.10 s   cached 0     — the whole prompt again
    after    0.40 s   cached 2,299 — and the planted needle still correct
```

Three checks decide that, not one: the reuse, the answer against a **forced
recomputation** on an erased slot, and a six-digit value planted mid-context —
because a fix that buys speed and changes answers would be worse than the
defect. A fourth run, `restore-determinism.py`, is identical on both builds and
is quoted here for what it rules OUT: the patch does not disturb the path that
already worked.

## Not only a language model — and not a toolbox

Since 01.09.2026 the same machine renders images, speaks and films. Measured
on this box, n=3 each, idle machine, every output machine-judged:

| workload | what | cost | licence |
|---|---|---|---|
| `flux-schnell` | text-to-image, 1024² | 56 s / image | Apache 2.0 |
| `sdxl` | text-to-image, 1024² | 112 s / image | OpenRAIL++ |
| `qwen-image` | text-to-image, top quality tier | 409 s / image | Apache 2.0 |
| `qwen3-tts` | text-to-speech, German included, Vulkan | **2.65× realtime** | Apache 2.0 |
| `chatterbox` | text-to-speech, voice cloning, 23 languages | 0.29× realtime (CPU) | MIT |
| `wan21-t2v` | text-to-video, 480p | ~9 min / 2 s clip | Apache 2.0 |
| `wan22-ti2v` | text-to-video, 5B — faster AND flagged | 288 s / clip, [see its profile](setup/workloads/wan22-ti2v.env) | Apache 2.0 |

Toolbox repos collect start commands. Every workload above is a **tenant of
the same memory authority** — the freeze story one section up is just as true
when the bytes pinning GTT come from a diffusion sampler — and lives by the
same rules:

- **One guard, one fence.** Each declares its measured footprint in
  [`setup/workloads/`](setup/workloads/), is weighed by `budget.py` before it
  starts, and runs only through the `sideserver` fence.
- **Determinism as an instrument.** All seven profiles pin a seed and carry
  their exact output hash — `bash tests/live_media.sh` re-derives it. A
  regression is a hash flip, not a statistical argument.
- **Machine-judged output**, each probe's selftest seen red before its green
  counted ([bench/](bench/README.md)).
- **Honest defects.** The 5B video model shows an artifact in dark regions —
  its profile says so, with the A/B that exonerated flash attention.
- **The Torch border.** The base install stays torch-free (the 16-second
  gate proves it); torch tenants live behind [`media/`](media/README.md).

And not a model benchmark: others do that at more scale than a home-grown
battery survives. Measured here is the STACK — what fits, what it costs, and
whether the output stays CORRECT — which nobody else measures for this
hardware.

## Not a fork

The backend is llama.cpp **master plus a short list of patches**. The list,
with the defect and the measurement behind each, lives in
[`setup/patches/`](setup/patches/README.md); a patch here is a debt being
worked off, not a feature, and the build script refuses a binary that
silently lost one. The weights are standard GGUF from public repositories,
and because the base is master, upstream improvements arrive at the next
build — not at a fork's next release.

## Where to start

| If you want to … | go here |
|---|---|
| find out whether this repo is **for your machine** | `bash setup/preflight.sh` — run it first |
| **set the machine up from scratch** — BIOS to first token | [docs/setup/](docs/setup/README.md), six chapters |
| **run it** — services, boot, the four ceilings | [setup/README.md](setup/README.md) |
| point **Claude Code or an OpenAI agent** at it — yours or somebody else's | [docs/CONSUMERS.md](docs/CONSUMERS.md), and `bash setup/consumer-info.sh` for the values |
| decide **which model to take**, and why the current one is production | [docs/MODELS.md](docs/MODELS.md) |
| generate **images, speech or video** on the same box | [`setup/workloads/`](setup/workloads/) — the profiles carry every measured number — and the workload-registry section of [setup/README.md](setup/README.md) |
| **verify** a running box — the gate, the live lanes, the smoke test | [tests/README.md](tests/README.md) |
| see the **raw measurements** | [docs/measurements/](docs/measurements/README.md) and [`bench/reports/`](bench/reports/) |
| **repeat or extend** them | [bench/](bench/README.md) |
| know **what is protected** | [docs/SECURITY.md](docs/SECURITY.md) — the model. [SECURITY.md](SECURITY.md) — where a finding goes |
| **change** something without breaking it | [tests/](tests/README.md), and [CONTRIBUTING.md](CONTRIBUTING.md) |
| report what it did on **your** machine | an issue — the one thing this repo cannot measure for itself |
