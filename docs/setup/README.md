# Setting up a Strix Halo for local inference

From a machine in a box to a coding agent answering in about a second.

This is the part that comes **before** the rest of this repository. Everything
else here assumes a machine that already boots Linux, has ROCm working and
lets the GPU reach system memory. Getting to that point is six chapters, and
it is where most of the afternoons go.

| | | |
|---|---|---|
| **01** | [Before you boot anything](01-before-you-start.md) | BIOS — and the one setting that costs you a third of your machine |
| **02** | [Linux](02-linux.md) | which kernel, and why that is the only hard choice here |
| **03** | [GPU and memory](03-gpu-and-memory.md) | ROCm, the memory budget, the GTT cap |
| **04** | [Building llama.cpp](04-build-llama.md) | with the patch that is not optional on this GPU |
| **05** | [Serving a model](05-serve.md) | fetch, install, start, first token |
| **06** | [When it does not work](06-when-it-does-not-work.md) | boot, service, and the failure that is silent |

Chapters 01–03 need one reboot in the middle and are comfortably an evening.
04 is a long build you can leave running. 05 is twenty minutes.

## Is this for your machine?

    bash setup/preflight.sh

Ten seconds, reads only, needs no root. It answers the question this page
cannot: whether the GPU is gfx1151, how much memory is fitted, and — the one
that catches people — whether the BIOS is holding a large slice of it back.

**This repository is measured on Strix Halo with 128 GB.** The same silicon
ships with 32 and 64 GB; on 64 GB three of seven profiles fit, on 32 GB none
do. Nothing here is scaled down to change that, because a scaled number is a
guess and every number in this repository is not.

## Why bother, when `ollama` is one command

That is a fair question and it deserves a straight answer.

`ollama` will have something answering in five minutes. It will also, on this
hardware specifically:

* **not know about GTT**, so it cannot tell you that a model which does not
  fit will freeze the machine rather than fail — there is no separate VRAM
  here, the GPU takes system RAM, and that allocation is pinned;
* **not carry the gfx1151 patch**, so with more than one slot the output
  degenerates to `////` with no error in any log;
* **default to a small context**, which is the difference between a chat toy
  and a coding agent;
* **have no answer to the prompt-cache problem**, which on this machine was
  the difference between 140 seconds per request and 1.3.

Raw `llama.cpp` gives you every knob and no answers: `--swa-full`, `-np 1`,
`-cram`, the patch, the GTT cap are all things you would have to find
yourself, and the failures that matter here do not announce themselves.

AMD's own recipes get you to "it runs" and stop.

What this repository adds is the part after "it runs": what fits before it
freezes, what is known to be broken and how it shows, and the settings that
make a coding agent usable rather than impressive.

## Where these instructions come from

They were carried out once, on one machine — an ASUS ProArt PX13 with a Ryzen
AI Max+ 395 and 128 GB — in August 2026, and written down as they happened.
That is their strength and their limit: every command here was actually run,
and none of them has been run on your board.

Where a step is a judgement rather than a fact, it says so.

---

Next: [01 · Before you boot anything](01-before-you-start.md)
