# Contributing

This repository is about ONE machine: Strix Halo (gfx1151) with 128 GB of
shared memory. That is not a limitation waiting to be lifted — it is the
product. Every number here was measured on that configuration, and a number
measured on one machine is worth more than a number averaged over none.

So the most valuable thing you can send is not a pull request.

## The most valuable contribution is a hardware report

`setup/defects.json` knows what goes wrong on one machine and cannot learn
about a second one by itself. A ROCm regression, a board where the GTT cap
does not hold, a BIOS that behaves differently — open an issue with the output
of `bash setup/preflight.sh`. It reads only, needs no root, and prints no
tokens and no hostnames.

That is also how `setup/lib/hardware.py` grows a second verified PCI id. It
holds exactly one today, because one is what has been seen.

## What the code here is like

Four rules, and they are not style preferences — each one is written down
because breaking it cost something.

**One source of truth, everything else derived.** `lib/models.sh` for which
models exist, `lib/systemdfile.py` for systemd syntax, `lib/budget.py` for the
memory arithmetic, `lib/systemunit.py` for the system unit. Every one of those
replaced two or three copies that had already drifted apart. If you find
yourself writing a second reader for something, that is the bug.

**Comments carry the evidence, not the intention.** A line that says *why*
with a measurement and a date is worth ten that restate what the code does.
Look at any `setup/env/*.env` before writing one.

**Numbers are measured, not derived.** Every figure in this repo that was
computed from first principles turned out wrong — KV per token by a factor of
four, the per-prefix cache estimate by three to four, the Flash-Next footprint
by 30 GiB. If you cannot measure it, say the number is an estimate in the
output, every time. An estimate that does not announce itself is how a value
gets copied into five profiles.

**Nothing is scaled to a machine nobody measured.** A 64 GB owner gets the
profiles that fit as written, and `bench/` to measure their own. They do not
get invented values from a repo whose whole argument is that its values are
not invented.

## Before you open a pull request

    bash tests/run.sh        505 tests, ~8 s, no GPU and no service needed
    bash setup/check.sh      is the repo still what the running system reads?

The test suite is the contract between the parts, and that is where the bugs
in this stack actually live: the kind where nothing breaks and an effect simply
fails to appear — a `Conflicts=` line missing a model, a guard that checks the
wrong number, a cache flag copied from a profile with a different window. CI
runs the same suite on Python 3.10, 3.12 and 3.13.

If you change behaviour, the test that pins it should read like the ones
already there: it names the failure it prevents, and where possible the day it
happened.

## Two things that do not belong in this repository

**Captured request bodies.** They contain an e-mail address, a `device_id`, an
`account_uuid` and Anthropic's system prompt. `.gitignore` excludes them and
`tools/synthetic.py` produces everything the measurements need without one.

**This machine's own answers.** Where the models live, what the tunnel is
called — those go in `~/.config/llm-stack.env`, which `setup/install.sh`
writes and `.gitignore` keeps out. `tests/test_localenv.py` fails on any line
that RUNS and names one computer. Prose may describe the history; a command
somebody could paste may not.

## Everything published here is English

Including the measurement records. An earlier version of this repository kept
them in German on the argument that a translated measurement is no longer the
record — which conflated not changing the NUMBERS, which is right, with not
translating the prose around them, which does not follow.

They are translated, and the translation was checked rather than trusted:
every data line byte-identical, every measured value compared as a multiset,
the German originals kept outside the repository so any figure can be traced.
`docs/DOCUMENTS.md` has the reasoning.
