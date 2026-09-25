# strix-halo-inference

[![tests](https://github.com/loonylabs-dev/strix-halo-inference/actions/workflows/tests.yml/badge.svg)](https://github.com/loonylabs-dev/strix-halo-inference/actions/workflows/tests.yml)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![GPU](https://img.shields.io/badge/GPU-gfx1151-red)](docs/setup/03-gpu-and-memory.md)
[![RAM](https://img.shields.io/badge/RAM-128%20GB-orange)](#does-this-fit-your-machine)
[![backend](https://img.shields.io/badge/backends-Halogen%20%C2%B7%20llama.cpp%20%C2%B7%20sd.cpp%20%C2%B7%20qwentts.cpp-lightgrey)](setup/README.md)
[![python](https://img.shields.io/badge/python-3.10%20%E2%80%93%203.14-blue)](.github/workflows/tests.yml)

**A Strix Halo as a measured inference machine: a local coding agent in
production — and text-to-image, speech and video under the same memory
authority.** One configuration — Ryzen AI Max+ 395 (gfx1151), 128 GB of
shared memory — and every number here was taken on it, with the date and
method beside it.

```bash
bash setup/preflight.sh              # is this repo for your machine?
bash setup/install.sh                # once — writes ~/.config/llm-stack.env
bash setup/switch-model.sh halogen-qwen38flash # (alias: halogen) serve the high-speed Halogen container
# or: bash setup/switch-model.sh halogen-qwen38  # serve 27B via Halogen
# or: bash setup/switch-model.sh qwen38         # serve GGUF via llama-server
bash tests/run.sh                    # the gate (1529 tests, ~20 s, no GPU)
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

## The engines: Turnkey performance on Strix Halo

Instead of fighting ROCm compilation flags or manual container mounts, the
stack provides turnkey configurations for the fastest inference engines on
this silicon:

* **Halogen Flash Server** ([`setup/halogen/`](setup/halogen/)): A rootless
  container backend running Qwen 3.8 Flash-Next with MTP speculative decoding
  — production here for daily agent work. Measured 24.09.2026 on 0.13.8: a
  cold 186k-token prompt in 192 s (~970 tok/s), a long answer at 31-34 tok/s
  from 2k to 193k of context. Its prompt cache lives in RAM **and on NVMe**:
  a deep conversation that was evicted, or survived a restart, resumes in
  2-3 s instead of ~3 minutes
  ([report](bench/reports/2026-09-24_halogen-disk-cache/README.md)). Its
  engine is closed source, so the container runs with **no route out** and
  its port on loopback only
  ([why and how it was tested](setup/README.md#no-route-out)).
* **llama.cpp** ([`setup/scripts/build-llama.sh`](setup/scripts/build-llama.sh)):
  Upstream master plus a small curated set of hardware patches
  ([`setup/patches/`](setup/patches/README.md)) for the broad GGUF model
  ecosystem.

Switching between them is one command:

```bash
bash setup/switch-model.sh halogen-qwen38flash # switch to Halogen Flash Server (or: halogen)
bash setup/switch-model.sh halogen-qwen38      # switch to Halogen Server (27B)
bash setup/switch-model.sh qwen38              # switch back to llama.cpp
```

The gateway shields your clients completely: changing the underlying engine
requires zero configuration changes in your editor or harness.

## The Gateway: Universal interface and instant turns

The LLM gateway ([`setup/gateway/`](setup/gateway/)) sits between your clients
and the serving engine:

* **Dual-dialect translation:** Speaks Anthropic (`/v1/messages`) and OpenAI
  (`/v1/chat/completions`) in-process. Claude Code, DeepSeek Harness, Cursor,
  and scripts talk to whichever backend is running without separate bridge
  daemons.
* **True prompt and prefix caching:** Agent prompts often carry 20k–40k tokens
  of system instructions and tool definitions. On llama.cpp the gateway tracks
  prefix hashes and saves KV states, turning a ~120-second cold prefill into a
  **1.3-second** follow-up turn (>90% cache reuse). On Halogen the engine
  caches itself, and the gateway passes Claude Code's cache marks through so
  that even its auto-mode safety check resumes where its transcript ends:
  **32.5 s → 18.6 s** per check, verdicts unchanged (24.09.2026,
  [report](bench/reports/2026-09-24_classifier/README.md)).
* **Edge-stable streaming:** Incremental token-by-token parameter streaming for
  tool calls and 10-second SSE keepalive heartbeats prevent Cloudflare Tunnel
  and proxy dropouts (`500` / `524`) on long generations.
* **Stable model aliases:** Use `local-low`, `local`, or `local-medium` in your
  clients. They resolve to the active engine's equivalent mode, so switching
  models never breaks your client configuration.
* **Dedicated CPU Vision Sidecar:** Exposes `qwen3-vl-4b` and `vision` alongside
  the primary GPU model. Runs on 8 dedicated Zen 5 CPU cores (CCD1: cores 8–15)
  via `llama-vision.service` with an independent admission gate (`VISION_GATE`),
  protecting Halogen GPU decoding from memory bus contention. Auto-starts on
  demand and stops after 10 minutes of inactivity (`VISION_IDLE_TIMEOUT=600`) to reclaim RAM for the Linux
  page cache. Fully compatible with Unity Asset Inventory, DeepSeek Harness,
  and OpenAI multimodal clients.

## Memory authority: Never freeze the machine

On unified memory architectures like Strix Halo, there is no discrete VRAM.
The GPU allocates host RAM through GTT, and that allocation is **pinned**.
A workload that exceeds available memory does not page out and does not get
OOM-killed: it hard-freezes the entire machine, taking down all processes
without writing to kernel logs.

[`setup/lib/budget.py`](setup/lib/budget.py) weighs every profile before start
and actively refuses if it does not fit:

```
REFUSING TO START qwen38: it needs about 70.1 GiB and it does not fit.
    the host has 44.2 GiB available, 12 must stay free
```

For the same reason, never start a second model — or a media workload — by
hand: `python3 bench/sideserver.py` is the only safe way, stopping production,
metering memory ceilings, and putting production back.

## Not only a language model: Multimodal tenants

Since 01.09.2026 the same machine renders images, speaks and films under the
same memory authority. Measured on this box, n=3 each, idle machine, every
output machine-judged:

| workload | what | cost | licence |
|---|---|---|---|
| `flux-schnell` | text-to-image, 1024² | 56 s / image | Apache 2.0 |
| `sdxl` | text-to-image, 1024² | 112 s / image | OpenRAIL++ |
| `qwen-image` | text-to-image, top quality tier | 409 s / image | Apache 2.0 |
| `qwen3-tts` | text-to-speech, German included, Vulkan | **2.65× realtime** | Apache 2.0 |
| `chatterbox` | text-to-speech, voice cloning, 23 languages | 0.29× realtime (CPU) | MIT |
| `wan21-t2v` | text-to-video, 480p | ~9 min / 2 s clip | Apache 2.0 |
| `wan22-ti2v` | text-to-video, 5B — faster AND flagged | 288 s / clip, [see its profile](setup/workloads/wan22-ti2v.env) | Apache 2.0 |

Each declares its measured footprint in [`setup/workloads/`](setup/workloads/),
guarded by `budget.py`. The base install remains torch-free (the ~20-second
test gate proves it); torch workloads stay contained behind [`media/`](media/README.md).

## Why not a generic runner (like `ollama`)?

Generic runners get an endpoint running quickly, but they are built for
standard discrete GPUs or generic CPU fallback. On unified-memory APUs like
Strix Halo, that leaves crucial gaps:

| Capability | Generic runner (`ollama` etc.) | This stack |
|---|---|---|
| **Silicon target** | Generic CPU / CUDA | Tailored & measured for Strix Halo gfx1151 (128 GB UMA) |
| **Engine choice** | Single internal runtime | Fastest engine per task: Halogen (MTP) or patched llama.cpp |
| **Multi-dialect gateway** | OpenAI only | In-process OpenAI ↔ Anthropic translation (Claude Code & DSH) |
| **Prefix & state caching** | In-memory only per run | Persistent across restarts, disk reload, instant follow-up turns |
| **Streaming stability** | Buffered tool payloads | Incremental argument streaming & SSE heartbeats against proxy timeouts |
| **Memory safety** | System OOM (hard-freezes Strix Halo) | Strict GTT budgeting (`budget.py`) across LLMs, Diffusion, TTS & Video |

## We measure, not claim

Nothing in this repository is based on estimates or marketing claims:

* **Every flag is measured:** Speeds, context windows, and KV costs carry
  their date and measurement method directly beside them in the profile
  comments.
* **Defects as testable data:** Silent hardware corruptions (e.g. `////`
  degeneration) and upstream bugs are recorded as data in
  [`setup/defects.json`](setup/defects.json). `python3 setup/lib/defects.py`
  verifies whether your running build is affected.
* **Rigorous test gate:** `bash tests/run.sh` runs 1529 tests in ~20 seconds
  without needing a GPU, verifying parser integrity, dialect conversions,
  budget calculations, and guardrails before anything touches production.

## Where to start

| If you want to … | go here |
|---|---|
| find out whether this repo is **for your machine** | `bash setup/preflight.sh` — run it first |
| see **what changed** and which component versions it was measured against | [CHANGELOG.md](CHANGELOG.md) |
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
