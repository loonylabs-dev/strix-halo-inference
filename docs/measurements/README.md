# Measurements — what was measured, and what came out

**These files are the record**: the raw numbers, the tools, the
counter-checks, dated. This page is the summary — find the result here, then
read the file for the evidence behind it.

They were in German until 27.08.2026, and this page used to argue that they
should stay that way: *"a translated measurement is no longer the record, it
is a claim about the record."* That argument confused two things. Not changing
the NUMBERS is right and it is the whole point. Not translating the PROSE
around them does not follow — translating changes no measurement.

What it does risk is a typo, so the translation was checked rather than
trusted: every data line is byte-identical to the original, every measured
value of three digits or more was compared as a multiset, and the German
originals are kept out of the repository as `*.de.md` so any figure can still
be traced to the text it came from. The check earned its keep immediately — it
caught three quoted counter messages that had been retyped instead of copied.

Number formatting was converted (`1.637` → `1,637`, `100,2 s` → `100.2 s`):
leaving German separators in an English document does not preserve a number,
it changes what it reads as.

Everything below was measured on one machine (ASUS ProArt PX13, Ryzen AI
Max+ 395, 128 GB, Fedora 44) and says nothing about any other.

---

## [measurements.md](measurements.md) · 22–24 August 2026

Seven sections. What each one settled:

| § | Subject | Result |
|---|---|---|
| 1–2 | **Vulkan vs ROCm**, five models, quiet mode | Vulkan wins decode by 4–31 %, ROCm wins prefill by 16–31 %. Neither backend is simply better — it depends on which half of a request dominates |
| 3 | **Power profile sweep** | quiet 180 GB/s · balanced 203 · performance 205. The top profile costs 84 W and 77 °C for ~1 % over balanced |
| 4 | **Prefill over context depth** | how the cost grows as the window fills, dense vs MoE-with-SWA |
| 5 | **Prompt cache** | the series that led to the sliding-window finding; context checkpoints alone do NOT fix it |
| 6 | **Speculative decoding with eagle3** | unusable: 2.2–2.7× SLOWER, on both backends |
| 7 | **Telemetry** | watts, degrees, load and battery on a 3-second grid — [telemetry-sweep.csv](telemetry-sweep.csv) |

## [cache-hunt-finding.md](cache-hunt-finding.md) · 23 August 2026

The investigation behind the founding finding of this repo: Claude Code
against a local model cost ~140 s **per request**, and the prompt cache never
bit.

* The rendering was fine — that was ruled out first, with
  `LLAMA_SERVER_SLOTS_DEBUG=1`, which shows the fully rendered prompt per slot.
  (That switch is also this project's worst security finding; see
  [../SECURITY.md](../SECURITY.md).)
* **The cause: sliding window attention.** llama.cpp can only roll the KV state
  back INSIDE the window, and Claude Code appends a 1,624-token block behind
  the user question — far outside a 512-token window.
* Bisection, the numbers that close the circle, and a synthetic counter-check.
* **The fix: `--swa-full`.** 140 s becomes 1.3 s.
* Why the earlier `cc-cachefix.py` helped for chat and not for tool calls.

The full chain of evidence, with the counter-checks, is section 15 of
[cache-hunt-finding.md](cache-hunt-finding.md).

---

## What has changed since

These are dated records, not living documentation, and some of what they
measured has moved on. Where a newer document contradicts them, the newer one
wins:

* **The model changed.** Laguna S 2.1 (whose SWA caused the finding) was
  replaced by Qwen 3.8 27B on 25.08., which has no sliding window at all. The
  lesson survived the model: what decides between usable and unusable here is
  whether the prefix survives from turn to turn.
* **Slots changed.** These runs predate the gfx1151 two-slot corruption; the
  stack has run `-np 1` since 26.08. See [../../setup/defects.json](../../setup/defects.json).
* **The memory cap changed, twice.** GTT went 96 -> 116 GiB on the morning of
  26.08., and back down to **108** the same night: the raise had been made for
  a predicted footprint that measurement put 22 GiB lower, and a cap is a
  ceiling as well as a budget. Above it, a model that does not fit takes the
  machine instead of failing to allocate.
* **Repeatability.** Speculative decoding makes a run unrepeatable
  (`bench/suites/spec-determinism.py`), so measure correctness with it off and
  speed with it on. Anything in these files measured with speculation on is a
  speed number, not a correctness one.

To repeat any of it against a different model, build or flag set, see
[../../bench/README.md](../../bench/README.md) — every run writes its own
report with the full context it ran under.
