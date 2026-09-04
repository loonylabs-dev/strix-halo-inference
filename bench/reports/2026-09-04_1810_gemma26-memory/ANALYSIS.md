# gemma26 memory — three two-point pairs and one prediction that held

**Measured 04.09.2026, 18:11-18:21**, seven loads behind `bench/sideserver.py`,
every figure read off the RUNNING server with `setup/lib/budget.py --observe`.
Raw output: `memory-observations.txt`, verbatim and unedited except that the
checkout path was folded away.

This is the first memory measurement Gemma 4 26B-A4B has ever had in this
repo. `setup/env/gemma26.env` declared `MODEL_KV_KIB_PER_TOKEN=10.4` from a
note rather than a measurement, and `MODEL_KV_SOURCE` said so itself.

## Conditions

`platform_profile = performance` and `tuned-adm active = throughput-performance`,
read at the start of the run and recorded in the raw file's second line.
**`tuned-adm verify` was NOT run** — it needs a password this session could not
supply — so the one check that compares the sysfs attribute against the D-Bus
intent is missing, and the gap is recorded rather than papered over. The
standing guard for that drift (`platform-profile-guard.timer`) was `active`.

Production (qwen36, 27.71 GiB of GTT) was stopped and restored by sideserver
around every point; GTT settled at 0.7 GiB before each start, which is in the
raw file for every point. Nothing else ran on the GPU.

**All seven loads used `build-vulkan`**, the binary the profile's `LLAMA_BIN`
names. Nothing here is a statement about the ROCm build.

## The numbers

| arm | -c 65536 | -c 131072 | -c 262144 | KiB/token | base GiB |
|---|---:|---:|---:|---:|---:|
| no `--swa-full`, no draft | 17.04 | — | 21.16 | **21.97** | 15.67 |
| no `--swa-full`, WITH draft | 17.73 | — | 22.23 | **24.00** | 16.23 |
| `--swa-full`, no draft | 29.37 | 43.37 | — | **224.00** | 15.37 |
| `--swa-full`, WITH draft | 30.18 | — | — | *(check point)* | |

**Every base is computed from BOTH points of its pair independently and they
agree to two decimals** — 17.04-1.37=15.67 and 21.16-5.49=15.67, and so on for
the other two. That agreement is the check that the model is linear in the
window and that neither point was taken in a disturbed state; it is not a
coincidence and it is the reason the pairs are run at all.

RssAnon of the running server: 0.13-0.14 GiB without the drafter, 0.24 with it,
over seven loads. The largest is 0.24.

## 1 · The drafter does NOT double the KV cache, and that is the finding

Isolated from the two pairs that differ only in the drafter:

    the draft model costs  +2.03 KiB/token  and  +0.56 GiB of base

**GLM-4.7-Flash's MTP head measured 55.0 against 110.0 KiB/token on this same
machine seven hours earlier — exactly double**, because llama.cpp's MTP draft
path allocates a second, full-size KV cache. This one does not. The reason is
in the driver: `common/speculative.cpp:1337` documents three modes for
`draft-mtp`, and the gemma4 one is `is_mem_shared` — "shares the target KV,
runs all heads in one graph" — set when the draft context's `ctx_other` is the
target context. So the cost here is a small second base and two extra KiB per
token, not a second cache.

**The check that says the model is right rather than merely consistent.** Point
7 was run against a prediction made before it, from arms 2 and 3 added
together: 226.0 KiB/token, base 15.93, i.e. **30.06 GiB** at -c 65536.
Measured: **30.18 GiB**, +0.4 %. `--swa-full` and the drafter are additive and
neither one changes what the other costs.

## 2 · `--swa-full` costs ten times the KV per token

    21.97 KiB/token without      224.00 KiB/token with       10.2x

That is not a surprise once the config is read, and the config predicted it
almost exactly (see below). Gemma 4 26B has 30 layers in a 5-sliding /
1-global pattern, and the two kinds of layer are inverted from what the old
profile note assumed:

| | layers | KV heads | key+value | window |
|---|---:|---:|---:|---|
| global | 5 | **2** | 512+512 | full context |
| sliding | 25 | **8** | 256+256 | 1024 cells |

The GLOBAL layers are the cheap ones — two KV heads each, 20.00 KiB/token for
all five together. The SLIDING layers carry four times the heads and would cost
200.00 KiB/token if they were allocated at full length, which is exactly what
`--swa-full` does. Without it they hold 1024 cells apiece, a fixed 0.20 GiB
that lands in the base.

**The old profile note said "210 KiB/token on the 5 global layers of 30" and
that is wrong in both halves** — it is 20.00 KiB/token, and 210 is roughly what
the OTHER 25 layers cost. The declared `MODEL_KV_KIB_PER_TOKEN=10.4` followed
from it and is out by 2.1x.

## 3 · The derivation from the config came closest yet

| | derived from the GGUF | measured | error |
|---|---:|---:|---:|
| no `--swa-full` | 20.00 | 21.97 | +9.9 % |
| `--swa-full` | 220.00 | 224.00 | **+1.8 %** |

For scale, the closest a derivation had come on this machine before was 4 %
(glm47flash, 04.09.). The `--swa-full` figure is better than that; the
no-`--swa-full` one is worse, and the ~2 KiB/token gap is not explained here —
the candidate is the context-checkpoint store (`-ctxcp 64 -cms 4096`, which
this profile carries and which has to keep something per checkpoint), and
nothing in this run isolates it. Recorded as observed.

## 4 · What the profile would cost, by window

| `-c` | as the profile serves | + draft | + `--swa-full` | both |
|---:|---:|---:|---:|---:|
| 65536 | 17.04 | 17.73 | 29.37 | 30.06 |
| 131072 | 18.41 | 19.23 | **43.37** | 44.19 |
| 262144 | 21.16 | 22.23 | 71.37 | 72.44 |

Beside production (qwen36, 27.75 GiB) under this machine's 108 GiB GTT cap:
44.19 + 27.75 = 71.9, leaving 36 GiB — at -c 131072 with everything on. At
-c 262144 with `--swa-full` the pair would be 100.2 of 108, which is not a
margin anyone should serve on.

## 5 · Two things this run exposed that are not about gemma26

**A loop of `sideserver.py` invocations trips systemd's start rate limiter and
leaves production down.** `llama-user@qwen36.service` carries
`StartLimitBurst=3` inside `StartLimitIntervalUSec=2min`; this run stopped and
restarted it seven times, and the fourth start inside the window was refused
with `Start request repeated too quickly` / `Result=start-limit-hit`. The unit
then stays down until someone runs `reset-failed`.

Two details make it worth writing down rather than shrugging at:

* **It could not have happened yesterday.** qwen36 loads in **3 seconds** from
  page cache (`model loaded` at 0.02.978 in the journal), and gemma26 measures
  in about 75 s, so seven points fit inside a handful of two-minute windows.
  The bigger models of the previous days were slow enough to stay under the
  limit on their own.
* **The cost is 13 minutes per point, not one failed restart.** sideserver
  reports the refusal loudly — `did not come back within 180 s — starting the
  probe timer anyway, it is better loud than absent` — and then calls
  `wait_for_slots(PRODUCTION_URL, 600)`, so each subsequent point waits 180 s
  and then 600 s for a unit systemd will not start. The REASON is only in the
  journal; sideserver's own output names the symptom.

**`--extra` bypasses the memory guard's arithmetic.** `budget.plan()` reads the
PROFILE, so a point run with `--extra "--swa-full -c 131072"` was budgeted as
"16 GiB the model will pin in GTT" while it pinned 43.37. Harmless here — GTT
had 60 GiB spare and the guard's job is host RAM — but it is the same silent
shape as a declared `MODEL_GTT_BASE_GIB` replacing the file size in `plan()`,
which cost an hour on 04.09.

## What this settles for the profile

* `MODEL_KV_KIB_PER_TOKEN` — **21.97** for the shape without the drafter,
  24.00 with it. Not 10.4.
* `MODEL_GTT_BASE_GIB` — **15.67** without the drafter, 16.23 with it. Both are
  ABOVE the 14.56 GiB of file (main model plus mmproj), by 1.11 and 1.67, which
  is the qwen36 pattern (0.80 above) rather than the glm47flash one (0.02
  below).
* `MODEL_HOST_ANON_GIB` — **0.24**, the largest of seven loads. Note the
  standing caveat: these are SIDE-SERVER loads, without the gateway and without
  a warm `-cram`; a production shape reads more (qwen36 declared 0.29 from side
  servers and the production server showed 0.38 at the switch).
* `--swa-full` is affordable at -c 131072 and not at -c 262144. Whether it is
  WORTH 24.96 GiB is a question about prompt-cache reuse, not about memory, and
  it is not answered here.
