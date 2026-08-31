# Documentation

ASUS ProArt PX13 HN7306EA · Ryzen AI Max+ 395 · 128 GB · Fedora 44 alongside Windows 11

## How this is organised

Three layers, and they are different in kind. Filing a document under the
wrong one is what makes documentation rot: a measurement that gets edited
stops being evidence, and a guide that never gets edited stops being true.

| Layer | What it is | Rule |
|---|---|---|
| **Prerequisites** | getting the MACHINE ready: BIOS, dual boot, Fedora, ROCm, the GTT cap. Not this product — its preconditions | [`setup/`](setup/README.md), six chapters. Command-first, and the chapters are what `preflight.sh` points at |
| **The product** | what runs once the machine is ready: profiles, gateway, benchmarks, the defect registry | lives and is edited |
| **Evidence** | measurements, decision logs, dated machine records | **never edited** — superseded or corrected by a newer document, not rewritten. Translated, though: see below |

### What goes to `archive/`, and what only looks like it should

`archive/` is gitignored: it is not published. So the test for it cannot be
"would a user need this to set the machine up" — by that test almost every
document here goes, including every measurement. It is three-way:

| | Goes where |
|---|---|
| Somebody reads it to **DO** something | published, English |
| It is the **EVIDENCE** for something published | published, in the language it was recorded in, behind a door |
| **Neither** | `archive/`, gitignored |

**The evidence is the product** — and on 27.08. that principle was applied to
the wrong thing, so the correction belongs here rather than the slogan alone.

`MODELLWAHL.md` was defended as "the evidence under `MODELS.md`", on the
argument that archiving it turns "9 of 9, 40.1 s per correct answer" into an
assertion with nothing behind it. **That argument confused the PROSE with the
EVIDENCE.** The evidence is `bench/reports/2026-08-24_2251_sweep_qwen38/` —
every task, pass or fail with its reason, wall time and token count, per
variant, in JSON. It is 1 MB, it is machine-readable, and it does not move.
The 294-line German log was a NARRATIVE around that data.

So the test is not "would deleting this leave a claim unbacked". It is:
**would deleting this leave the DATA unreachable?** For `measurements/` and
`bench/reports/` the answer is yes and they stay. For the log it was no, and
it went — along with a headline margin its own footnote recorded as a checker
artefact, and an instrument (`bench/quality.py`) that no longer exists. What
survived is a table in `MODELS.md`, where the claim is made.

What genuinely went: two unverified surveys that read like guidance and aged in
a fortnight, plus `.orig` copies and withdrawn versions. That is the shape of
an archive candidate — not *unneeded*, but **misleading or superseded**.

### One rule, and it used to be two

Everything published here is **English**, including the records. The rule was
narrower until 27.08.2026 and drew the line in the wrong place:

> *"Does somebody READ this in order to DO something? Yes → English. No, it is
> a RECORD → it stays in the language it was recorded in. A translated
> measurement is no longer the record; it is a claim about the record."*

The first half was right. The second half conflated **not changing the
numbers**, which is the whole point of evidence, with **not translating the
prose around them**, which does not follow — a translated number is the same
number. The repository had already made that distinction once, in the other
direction, when it deleted a 294-line decision log: *"that argument confused
the PROSE with the EVIDENCE."*

What makes translating a record safe is not care, it is a check:

* every data line byte-identical to the original,
* every measured value of three digits or more compared as a multiset,
* the German originals kept OUT of the repository as `*.de.md`, so any figure
  can still be traced to the text it came from.

It earned its keep on the first run: three quoted Claude Code counter messages
had been retyped instead of copied, and the value comparison found them.

Number formatting was converted (`1.637` → `1,637`, `100,2 s` → `100.2 s`).
That is not an exception to "do not change the numbers" — leaving a German
separator in an English document does not preserve a number, it changes what
it reads as.

### The gap this used to leave, and how it was closed

Until 27.08.2026 the setup path — the thing a newcomer needs first — existed
only as two German HTML documents totalling 18,000 words, mixed with a
purchase narrative and a survey of models that ages in a fortnight.

They were not translated. They were **extracted**: roughly 5,000 words of
actual instruction, rewritten as [`setup/`](setup/README.md) in the order
somebody does them, with the corrections that cost afternoons kept prominent —
the `excludepkgs='kernel*'` glob that also blocks `kernel-headers`, the build
flag that no longer exists, the `/etc/kernel/cmdline` step without which the
parameters quietly vanish at the next kernel.

What was dropped: the biography ("Day 0 — before unpacking"), the model
surveys with no commands in them, and 1,900 words about Node, Unity and
Android that have nothing to do with inference.

The German originals are the maintainer's own now — see below.

## What is NOT in this repository, and why

Two documents and one tool are the maintainer's own working files. They are on
the maintainer's disk and excluded by `.gitignore`:

| | |
|---|---|
| `docs/HANDOVER.md` | a session log — what was built on which day, what is open. Over 2,000 lines, and it is the record of DEVELOPING this stack, not of running it |
| `docs/FLASHNEXT-PLAN.md` | the closed plan behind a model that is not served. Its conclusion is public in [MODELS.md](MODELS.md); the plan is the reasoning that got there |
| `setup/scripts/watch-flashnext.sh` | a watcher for two upstream conditions. Useful to one person on one machine |

Nothing in this repository depends on them. That is checked rather than
assumed: `tests/test_docs.py` fails if any published file points at one of
them, because a link into a file nobody has is worse than no link.

What a user setting up their own machine needs is here — the profiles, the
memory budget, the defect registry, the measurements. What is missing is one
person's notebook.

## The documents of this project

| Document | For what |
|---|---|
| [setup/](setup/README.md) | **getting a machine to the starting line** — BIOS to first token, six chapters. English, command-first, and the chapters are what `preflight.sh` points at when a prerequisite is missing |
| [CONSUMERS.md](CONSUMERS.md) | how someone sets up Claude Code — or an OpenAI-speaking agent — against this inference |
| [MODELS.md](MODELS.md) | **which model and which settings** — compact, English, no history |
| [SECURITY.md](SECURITY.md) | zone model, allow list, every security measurement |
| [measurements/](measurements/) | the raw records: backends, power profiles, prefill depth, the prompt-cache series, the sliding-window finding. [measurements/README.md](measurements/README.md) summarises what each file shows |
| [../setup/README.md](../setup/README.md) | operations: services, access, tunnel, prefix cache |
| [../setup/defects.json](../setup/defects.json) | the defect registry: what is known to go wrong on this hardware, as data. `python3 setup/lib/defects.py` says whether this machine is exposed |
| [../tests/README.md](../tests/README.md) | the three test levels and why the lowest one exists |

The documents under `px13/` described the **machine**, not the project. Every
one of them was cited by something — `gtt.sh` for the memory ladder,
`tools/synthetic.py` for the sliding-window chain, `tests/test_gtt.py` for the
numbers `gtt.sh --set` climbs.

That is exactly why making them internal was work rather than a `git rm`. Each
citation was one of three kinds, and only the middle one needed anything real:

| kind | example | what was done |
|---|---|---|
| **provenance note** | `gtt.sh`: "the ladder (runbook § 7)" | reworded — the number was already in the script, so the date replaces the document |
| **evidence pointer** | README: "the full chain of evidence is § 15" | the evidence was translated into `measurements/`, and the pointer moved there |
| **documentation map** | this file, twenty times | rewritten |

Two survey documents had already gone to `archive/` on 26.08. for a different
reason — they read like guidance, aged in a fortnight, and one was cited by
`build-llama.sh` for an OPINION rather than a measurement.

## Where to start

| If you want to … | go here |
|---|---|
| see the **raw measurements** | [measurements/](measurements/README.md) |
| **set a machine up** | [setup/](setup/README.md) — six chapters, BIOS to first token |
| **pick a model** | [MODELS.md](MODELS.md) — the short answer |
| look up **flags** | the profile itself: `../setup/env/<model>.env`, with the measurement above each line |
| **clarify terms** (backend? harness? runtime?) | [../setup/README.md](../setup/README.md), *The one rule* |

> **Where two documents disagree, the newer measurement wins** — and the
> measurement is in `measurements/` or `bench/reports/`, not in the prose
> around it. A document that has been superseded says so; it is not edited to
> look right.

---

## The documents in detail

### The machine documents that are no longer here

Five German documents under `px13/` described this one machine: a dated state
record, a runbook, the setup path, a glossary and a note on the shared NTFS
partition. They were the source for `setup/` and for the translated
measurement records, and they are the maintainer's own files now — on disk,
excluded by `.gitignore`.

Nothing published depends on them, and `tests/test_docs.py` fails if that
stops being true. What they uniquely carried has a public home:

| what it was | where it is now |
|---|---|
| the setup path, ISO to first token | [`setup/`](setup/README.md) |
| the memory ladder and the GTT reasoning | [`setup/03-gpu-and-memory.md`](setup/03-gpu-and-memory.md) and `setup/scripts/gtt.sh` |
| the sliding-window finding, in full | [`measurements/cache-hunt-finding.md`](measurements/cache-hunt-finding.md) |
| the backend and flag measurements | [`measurements/measurements.md`](measurements/measurements.md) |
| build flags and the patch | [`setup/04-build-llama.md`](setup/04-build-llama.md), `setup/patches/README.md` |

## [setup/](../setup/) — the runnable configuration

The source of everything installed on the Fedora side: model configurations, systemd
units, Claude Code profiles, the gateway and the measurement scripts. After a
reinstall, `bash setup/install.sh` from the repo is enough.
Details in [setup/README.md](../setup/README.md).

## Configuration and scripts

| File | Content |
|---|---|
| [llmprofile](../setup/llmprofile) | power profiles and backend start. Rev. 2 of 22 August — the `card0` bug that had devalued the watchdog is fixed. |
| `~/.local/lib/llm-stack/gateway.py` | **The current gateway** (llm-gateway; cc-gateway until 09/2026). Splits stable and volatile system messages, and much more. Source: [setup/gateway/](../setup/gateway/). |
| `~/.claude/bin/cc-cachefix.py` | Predecessor. **Do not use any more** — it pulls a changing counter to the start of the prompt and turns a 1.5-second turn into an 89-second one. |

Installed, these live under `/usr/local/bin/llm-profile` and `/etc/llm-profile/` on the
Fedora side. The working copies including test scripts are in `~/llm-setup/`.

---


---

## [measurements/](measurements/)

Raw data for every number in the state document. In German.

| File | Content |
|---|---|
| [measurements/measurements.md](measurements/measurements.md) | all benchmarks: Vulkan, ROCm, power profiles, prefill scaling, prompt cache, speculative decoding. **Section 8** holds the SWA series of 23 August. |
| [measurements/cache-hunt-finding.md](measurements/cache-hunt-finding.md) | detailed log of the cache investigation: tools, bisection, counter-checks, source evidence |
| [measurements/telemetry-sweep.csv](measurements/telemetry-sweep.csv) | watts, degrees, load, battery on a 3-second grid across the profile sweep |

---

## archive/ — on disk, not in the repo

Withdrawn predecessor versions, `.orig` copies, and since 26.08. the two
surveys. It is in `.gitignore`: nothing in it is needed to set up a Strix Halo
or to get this stack running, and some of it is actively wrong —
`px13-einrichtung.html` carries the `amdgpu.gttsize` recommendation that
runbook rev. 7 revoked.

Since 27.08. the machine documents under `px13/` are excluded too — for a
different reason. `archive/` is withdrawn and partly wrong; `px13/` is the
maintainer's own source material, from which `setup/` and the translated
measurement records were made. Neither is needed to set up a Strix Halo, and
that is the test both had to pass.

Note what the gitignore does not do: files already committed remain in the git
HISTORY. This keeps them out of a clone and out of the file browser, which is
the point. It is not deletion.


---

## The most important numbers at a glance

    memory        VRAM 0.5 GiB (BIOS)  ·  GTT 108 GiB  ·  system RAM 124.9 GiB
                  (GTT raised from 96 on 26.08. and lowered from 116 the same
                  night: a higher cap is more room AND less protection — below
                  it, too large is an error message. setup/scripts/gtt.sh)
    bandwidth     180 GB/s (quiet)  ·  203 GB/s (balanced)  ·  205 GB/s (performance)
    backend       Vulkan wins decode by 4–31 %, ROCm prefill by 16–31 %
    thermals      balanced 64 W / 67 °C  ·  performance 84 W / 77 °C

    decode in quiet mode (tok/s)
      Gemma 4 26B-A4B   57.6      gpt-oss-120b      45.7
      Laguna S 2.1      26.8      Qwen3.8-27B       10.1      Gemma 4 31B    7.8

---

## Open points

- **Prompt cache with Claude Code**: done. The cause was sliding window attention, not
  Claude Code. With `--swa-full` tool turns cost 1.5–2.0 s, a complete `claude -p` run
  with two tool calls 13 s. The full chain is in
  [measurements/cache-hunt-finding.md](measurements/cache-hunt-finding.md).
- **`--swa-full` is still missing** in `gemma26.env`, `gemma31.env` and `gptoss.env` —
  all three models have SWA as well, but the memory requirement there is unmeasured.
- **Four projects warm at once** works with `--swa-full` and `-np 4` (measured: 8 of 8
  queries at 100 % cache). The finding "from three agents on the cache fails" is
  therefore superseded. The price: 32,768 instead of 65,536 tokens of context per
  project.
- **The `-cram` RAM cache carries about 65 project prefixes** (~500 MiB each at 32 GiB).
  Measured with five projects on two slots — a Laguna-era figure; qwen38 runs
  ONE slot since 26.08., see `setup/env/qwen38.env` for the two defects that
  force it. All at 100 % cache, 0.3 s to swap in. `-np`
  therefore only decides how many *compute* at once, not how many stay warm.
- **Speculative decoding** is unusable with eagle3 (factor 2.2–2.7 slower).
- **The internal speakers** need kernel ≥ 7.0 plus firmware from Windows.
