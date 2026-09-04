# Setup — the runnable configuration

The source of everything installed on the Fedora side. The truth lives here;
`/etc`, `/usr/local/bin` and `~/.claude` are copies of it.

After a reinstall this is enough:

    bash setup/install.sh          # from the repo, wherever you cloned it

The script puts profiles, gateway, model configurations and systemd units back.
For `/etc` and `/usr/local/bin` it asks for sudo once, the rest runs user-side.

---

## What this repo is measured on, and what that means for everyone else

    bash setup/preflight.sh        run this before anything

**One configuration: Strix Halo (gfx1151) with 128 GB of shared memory.** Every
number here comes from a machine of that shape — the window, the KV cost per
token, the RAM prompt cache, the GTT cap, and the memory budget's own
constants. That focus is the product: a generic stack that works everywhere
half as well is what this exists as an alternative to.

The same silicon also ships with 32 and 64 GB, and people will clone this with
one. What they get is an honest answer at the door rather than an afternoon:

| | |
|---|---|
| **128 GB** | 7 of 7 profiles fit. This is the configuration. |
| **64 GB** | 3 of 7 fit as written — `batch`, `gemma26`, `gemma31`. `qwen38` does not: its cache is 32 GiB and that number IS computed. |
| **32 GB** | none. 17.6 GiB of weights plus a 6 GiB buffer floor plus a 12 GiB host share is already over the machine, before any context at all. |

**Nothing is scaled down to make the others fit**, and that is deliberate. Every
figure in this repo was measured on one machine, and a scaled figure would be a
guess — from a repo whose whole argument is that its numbers are not guesses.
What a 64 GB owner gets is the filter above (measured values, shown or not
shown) and `bench/`, which measures their machine if they want their own.

That the list is 3 rather than 1 is not an adaptation either. `gemma26` and
`gemma31` were held back by a `-cram 32768` that had been COPIED from qwen38
and never computed — it bought 24.6 and 16.0 full windows against production's
2.2. Sizing it correctly is right at any memory size; that it also lets a
smaller machine run those two profiles is a consequence, not the reason.

## The one rule: the dependency points one way

This repo holds two things, and they are deliberately in one repo — the
findings that make it worth anything came from the seam between them. The
prompt-cache finding was a symptom at the gateway (Claude Code re-processing
every turn) with a cause at the server (sliding window attention). `-np 1`
came out of a corruption that only showed under real agent traffic. Split
them and the evidence chain gets cut in half.

    inference layer   llama.cpp build and patches, GTT, the model registry,
                      switch-model.sh, check.sh, the profiles
    gateway           llm-gateway with dialects, modes and prewarm — the one
                      entrance, consumer-agnostic: it speaks anthropic AND
                      openai, so Claude Code and DeepSeek Harness alike
    Claude harness    cc-router and the Claude Code profiles — the REFERENCE
                      CONSUMER, not a second product

**Each layer may require the one below. The inference layer must never need
the gateway, and the gateway must never need any one consumer.**

Until 26.08. it did, in two places, and both were found by asking what a
machine without a gateway would do:

* `switch-model.sh` read the gateway's port and aborted when the profile
  disagreed, restarted `cc-gateway` unconditionally, and smoked only through
  it. Without a harness the model could not be switched at all.
* `check.sh` reported six missing harness symlinks as `(not installed)`,
  which sets `DIFF=1`, and exited 1 — on a machine where nothing was wrong.

Both now ask first (`gateway_present`, either the unit or
`~/.config/llm-gateway.env`) and say plainly that the gateway is absent rather
than treating it as a defect. The smoke test still happens either way: through
the gateway where there is one, straight at `/v1/chat/completions` where there
is not — "no harness" must not come to mean "no verification".

It is the same failure class as two entries in
the two the units carry a note about: `ExecStartPost` without a leading
`-`, which let a convenience take the server down with it, and
`EnvironmentFile=` without one, which stopped the service starting when the
file went away. A companion made compulsory.

`tests/test_models.py` pins both directions.

---

## What goes where

| Here | Installed to | Content |
|---|---|---|
| `preflight.sh` | — | **run this first.** Is this repo for this machine? GPU, memory, GTT cap, distribution — and which profiles fit as written. Reads only, needs no root |
| `lib/hardware.py` | — | what machine is this: the GPU by `rocminfo` AND by PCI id, so the question can be asked before ROCm is installed |
| `local.env.template` | `~/.config/llm-stack.env`, `/etc/llm-stack.env` | **what belongs to this MACHINE** — where the models live, what the tunnel is called. Written once by `install.sh`, gitignored, and the only file a new machine has to answer |
| `env/*.env` | `/etc/llm-profile/` | **the model registry** — one file per model, and the file IS the model. `MODEL_SOURCE` says where its weights come from |
| `get-model.sh` | — | fetch what a profile needs: this stack's `ollama pull`, except the list it offers is measured on this hardware |
| `lib/models.sh` | — | which models exist, which one runs, what its profile says |
| `lib/systemdfile.py` | `~/.local/lib/llm-stack/`, `/usr/local/lib/llm-profile/` | the one reader for systemd syntax (`LLAMA_ARGS`, `Conflicts=`) |
| `lib/budget.py` | `~/.local/lib/llm-stack/`, `/usr/local/lib/llm-profile/` | **the one memory budget** — what a profile costs and whether it fits. Travels with `systemdfile.py`, which it imports by directory |
| `consumer-info.sh` | — | the four facts a consumer needs — endpoint, model names, window, token — read from the RUNNING stack, because values written into prose go stale |
| `checkroom` | `~/.local/lib/llm-stack/`, `/usr/local/bin/llm-check-room` | the budget as `ExecStartPre`: retries briefly, then refuses. A model that does not fit freezes the machine, so a service that will not start is the good outcome |
| `scripts/build-llama.sh` | — | builds the patched llama.cpp, versioned and reversible |
| `scripts/gtt.sh` | kernel command line | how much system RAM the GPU may take |
| `scripts/scout.py` | — | look at a model before downloading it: does our build know the architecture, how big is each quant, does it fit |
| `scripts/fetch-model.sh` | the model directory | resumable, size-checked GGUF download that can WAIT for a quant not yet uploaded |
| `defects.json` | — | **the defect registry** — what is known to go wrong on this hardware, as data |
| `scripts/probe.py` | `~/.local/lib/llm-stack/` | the watchdog: one question whose answer is known, every ten minutes. Turns the silent failure modes into loud ones |
| `lib/defects.py` | — | evaluates the registry against the running server and the build stamp; `check.sh` prints it |
| `lib/kernelcmdline.py` | — | editing a kernel command line without losing `root=` |
| `systemd/*.service` | `~/.config/systemd/user/`, `/etc/systemd/system/` | `llama-user@.service` (one model per instance — **the only unit file**) and `llm-watch.service` |
| `lib/systemunit.py` | `/etc/systemd/system/llama@.service` | **derives** the system unit from the user unit. Opt-in: `install.sh --system-unit`, for a host with no user session |
| `claude/*.json` | `~/.claude/profiles/` | backend profiles for Claude Code |
| `gateway/gateway.py` | `~/.local/lib/llm-stack/` | **the current gateway** (unit: `llm-gateway`). Zones, access, prefix cache |
| `gateway/{modes,tracelog}.py` | `~/.local/lib/llm-stack/` | thinking modes by model name; the trace instrument |
| `gateway/savepolicy.py` | — | when a prefix is worth saving — read from the repo by the gateway's tests and the simulator |
| `claude/cc-cachefix2.py` | — | predecessor of the gateway — kept for comparison only, not installed |
| `claude/cc-cachefix.py` | — | its predecessor in turn — superseded, see below |
| `claude/cc-router.py` | `~/.claude/bin/` | router for variant 2 (keep the subscription, subagents local) |
| `waitformodel` | `~/.local/lib/llm-stack/`, `/usr/local/bin/llm-wait-for-model` | waits for the model partition before the start |
| `../tools/prewarm.py` | `~/.local/lib/llm-stack/` | saves and restores project prefixes |
| `scripts/*.sh` | — | measurement and operating scripts, run FROM the repo. `~/llm-setup/` holds their old working copies and the logs they wrote; `install.sh` does not touch either, and the table said it did until 26.08. |
| `llmprofile` | `/usr/local/bin/llm-profile` | power profiles and backend start |
| `gateway/dialects.py` | `~/.local/lib/llm-stack/` | how a request body is read — shared by gateway and prewarm |
| `env/*.env` | `~/.config/llm-profile/` | the SAME files as symlinks: what the USER service reads |
| `patches/*.patch` | applied in `~/llama.cpp` | changes llama.cpp needs on this hardware |

## What belongs to the MACHINE, and why it is not in the repo

    setup/local.env.template   ->  ~/.config/llm-stack.env   (gitignored)

Three questions have answers that are true on one computer:

| | |
|---|---|
| `LLAMA_MODELS` | where the `.gguf` files are |
| `GATEWAY_HOST` | the public hostname of the tunnel, if there is one |
| `DOCS_LINK` | an optional shortcut to `docs/`, watched by `check.sh` |

Until 27.08. they were **defaults in the code**: `/mnt/shared/LLM` in eighteen
files, an absolute repo path in seven, and a private hostname in
`smoketest.sh`. The last of those had a consequence rather than an
inconvenience — `git clone && bash setup/smoketest.sh` sent requests at
somebody else's tunnel.

`install.sh` writes the file and derives what it can: it looks for a directory
that actually holds `.gguf` files rather than guessing a path. `GATEWAY_HOST`
stays empty, deliberately — a hostname cannot be derived, and guessing one
means aiming a test at a machine that is not yours. `smoketest.sh` then names
the remote zone as skipped instead of checking it, and a skip counts as a
deviation here.

### One file, two readers, and they must agree

    python3 setup/lib/systemdfile.py models        the model directory
    python3 setup/lib/systemdfile.py local <NAME>  any answer

    . setup/lib/models.sh; models_dir              the same, in shell
    . setup/lib/models.sh; local_var <NAME>

Both consult the same three sources in the same order: `$LLAMA_MODELS`, then
the local config, then the conventional locations (`~/models`,
`~/.cache/llama.cpp`, `~/.local/share/models`, `./models`) — and then they give
up rather than guessing. `tests/test_localenv.py` pins that the two agree,
because two readers of one file is how the three `LLAMA_ARGS` parsers began.

**No absolute path of one machine is in that convention list**, and that is
the part worth stating: a fallback containing `/mnt/shared/LLM` would be the
same hard-coding one level down. The first draft of this mechanism had it, and
it was caught by the question "why is there a static string in there".

### The consumer document says what is the same, not what your values are

    bash setup/consumer-info.sh              endpoint, models, window, token
    bash setup/consumer-info.sh --local      the same, for this machine
    bash setup/consumer-info.sh --markdown   paste-ready, to send to somebody

[`docs/CONSUMERS.md`](../docs/CONSUMERS.md) is 459 lines and about 95 % of it
is identical whether the stack is yours or somebody else's — the four Claude
Code variants, the prompt-cache rules, the settings table. The other 5 % were
VALUES, and values written into prose go stale: on 27.08. that page still said
`laguna` was available (it had been replaced two days earlier) and had claimed
"no vision" for a model with a projector until someone noticed.

So the page is generic, forks once at the top for the two kinds of reader, and
uses `$ENDPOINT` throughout. The values come from the running stack instead.

### The rule that keeps it out

`tests/test_localenv.py::TestNothingInTheRepoNamesOneMachine` scans every line
that RUNS — comments are prose and may name the history — across the whole
repo. `setup/env/*.env` has had this rule since 26.08.; it covered the one
directory whose files never contained such a path.

What is still outstanding is listed IN that test, each entry naming why:
`bench/variants/qwen38.json` (per-variant binaries) and `docs/CONSUMERS.md`. A second test fails if an entry on that
list has quietly become clean — an exception that no longer applies is worse
than no exception.

## What lives OUTSIDE this repo — and must not be forgotten

Four things the stack depends on that no `git clone` brings along. The
first one is load-bearing and fails silently, the rest fail loudly.

| Path | What it is | If it is missing |
|---|---|---|
| `~/llama.cpp` branch `master-2patches` | **both** patches, as commits, on current master — see `setup/patches/README.md`. `gfx1151-patched` is its 22-commit predecessor, kept as the way back | a `git pull` takes them with it. Without the first, **the server answers WRONG** once a second slot is used; without the second, every turn re-reads the whole previous answer. Neither says anything |
| `~/llama.cpp/build-rocm-patched` | symlink to the active build, the path `LLAMA_BIN` names | the service will not start |
| `~/llama.cpp/build-rocm-patched-<id>/` | the builds themselves, each with a `.build-stamp` | no way back to a build that worked |
| `~/.config/llm-profile/<model>.env` | symlinks to `env/*.env`, read by `llama-user@.service` | the service refuses to start (deliberately, no leading `-`) |
| `~/.config/llm-gateway.env` | gateway configuration, carries tokens | gateway falls back to defaults: no thinking modes, no system rewrite |
| `~/.cache/llama-slots/` | saved prefixes (GB-sized), plus `.owner` and parked stores per model | first request per project runs cold again |
| `~/.cache/llama-slots/.owner` | which model wrote these prefixes | `switch-model.sh` has to guess, and a wrong guess feeds one model's KV state to another |

And since 01.09.2026 the media workloads add their own — none is verified
by check.sh yet, so each fails at the NEXT FENCED RUN, after a production
stop/settle cycle has already been paid (the architecture review's first
finding: a table whose only purpose is not-forgetting had forgotten them):

| Path | What it is | If it is missing |
|---|---|---|
| `~/stable-diffusion.cpp` + `build-vulkan-<id>/` | the image/video tree and its pinned builds (`setup/scripts/build-sd.sh`) | every image/video profile's `WORKLOAD_CMD` dies with ENOENT — loud, but only inside the fence |
| `~/qwentts.cpp` + `build-vulkan-<id>/` | the TTS tree, pinned builds AND the `qwen-tts-p` adapter beside the binary (`build-qwentts.sh`) | qwen3-tts.env dies the same way |
| `~/.venvs/media-audio` + `~/.local/bin/uv` | the torch lane's venv (uv-managed CPython 3.12 — `media/audio/setup-venv.sh`, pins in `requirements.lock`) | chatterbox.env dies; rebuild from the lock, or the measured figures are claims |
| `@MODELS@/image/`, `@MODELS@/audio/`, `@MODELS@/video/` | the media weights (~64 GB; provenance per profile in `WORKLOAD_SOURCE`) | the guard refuses UNMEASURED profiles and measured ones fail at start |
| `~/.cache/huggingface/…ResembleAI--chatterbox` | chatterbox weights at the revision the wrapper pins | first fenced run re-downloads ~4 GB inside the measurement window |
| `~/SPIRV-Headers` | header fallback while `spirv-headers-devel` is not installed | the two media build scripts refuse with the fix in their message |

`bash setup/check.sh` verifies all of the llama-era ones: that the source
still carries the patch, WHICH build is active, whether the running process
comes out of it, and who owns the prefix store.

**The patches are a branch, not a diff.** That is the whole point:

    cd ~/llama.cpp && git log --oneline origin/master..master-2patches
    c799f10 server: speculation must not keep tokens past an end-of-generation token
    c92aba9 gfx1151: do not trust prop.integrated on HIP

TWO of them since 30.08.2026, and `build-llama.sh` checks a marker for each
before it builds — neither fails loudly when it goes missing, so the markers
are what makes the absence loud. The branch used to be `gfx1151-patched`; that
one grew to 22 commits carrying a local Flash-Next implementation that upstream
then merged itself, so the default moved to `master-2patches`, which is master
plus the two and nothing else. The old branch stays on disk as the way back.

An update is therefore a rebase, and a rebase either replays the patch or says
it cannot — instead of dropping it in silence. `setup/scripts/build-llama.sh`
does that, builds into a directory of its own, verifies the result and only
then moves the symlink:

    bash setup/scripts/build-llama.sh                  # build origin/master
    bash setup/scripts/build-llama.sh --ref pr/27700   # build a pull request
    bash setup/scripts/build-llama.sh --list           # what exists, what is active
    bash setup/scripts/build-llama.sh --use <id>       # go back

Building into a directory of its own is not tidiness. `llama-server` is a 12 KB
executable that maps `libggml-hip.so` out of the same `bin/`; overwriting a
mapped library is a SIGBUS in the running process, not an error in the build.

And the guard lives in build-llama.sh, NOT in cmake — a direct
`cmake --build <dir> --target llama-quantize` walks straight past it. Done
01.09.2026 to get a quantize tool: it relinked seven libraries in the
directory the PRODUCTION server had mapped, from a source tree that was
checked out on a DIFFERENT branch — resident pages stay old, re-faulted
pages read new code, and nothing says so until it crashes. The repair cost a
production stop, a clean rebuild of the same commit and copying the twin
libraries back. Build extra tools only into a build directory nothing runs
from, with the source tree checked out on that directory's own commit.

**Four built-in families plus the ones you name, and only one of them can be
served.** A build that differs from the serving one is a SUBJECT — something
to measure against it, never something to point the symlink at:

| Family | Built with | Why it is separate |
|---|---|---|
| `build-rocm-patched-<id>` | the two patches | the only family `--activate` and `--use` accept |
| `build-rocm-unpatched-<id>` | `--no-patch` | upstream as it stands, to measure whether a fix landed. Serving it answers WRONG once a second slot is used |
| `build-rocm-unroll-<id>` | `--unroll` | plus `-mllvm --amdgpu-unroll-threshold-local=600`, the workaround from llama.cpp#19984. Measured 31.08.2026: **no effect on this stack**, see `bench/reports/2026-08-31_0220_unroll-flag/` |
| `build-rocm-altsdk-<id>` | `--rocm-path DIR` | against a ROCm that is not the system's. Measured 31.08.2026 against 10.1: **+10 % decode on an empty context, gone by 64k**, see `bench/reports/2026-08-31_1219_speed-ab/` |
| `build-rocm-<name>-<id>` | `--family <name>` | a FOREIGN TREE — somebody's fork as a measurement subject. The name is lowercase letters and digits only (a hyphen would re-open the glob collision below) and never a built-in one. Instances so far, all measured 31.08.2026: `rocm-gdnfork` (RDNA 3.5 tuning fork — **nothing at the operating point**, `bench/reports/2026-08-31_1908_speed-ab/`), `rocm-engramhalo` (Flash-Next tuning fork — its MTP path decodes at 0.33-3.5 t/s here, `…_2047_speed_flashnext-B-mtpq8/`), `rocm-drluoto` (PR #27836 draft-mtp preview — **decode 16→100 t/s on copy**, `…_2158_speed_flashnext-C-mtpq8/`; the hang its early builds carried was isolated 01.09. to the pre-#27941 base and is gone on the m2 lineage, master b10743 + the PR commits, gate battery green — `flashnext-mtp-serving-shape-hangs` in the defect registry), `rocm-cachelookup` (see the row below — the one instance that is NOT a foreign tree) |
| `build-rocm-cachelookup-<id>` | `--family cachelookup` | OUR OWN two-commit change to llama.cpp's prompt-cache lookup, built as a measurement subject rather than adopted. Same lineage as the serving drluoto build plus `62850522e..6bbb3eecf`. Measured 02.09.2026: a restore into a server whose RAM cache holds a longer state costs **111.5 s against 0.33 s**, and **0.53 s with these commits** — `bench/reports/2026-09-02_2230_restore-lookup-patch/`, three builds through one reproducer. SERVED since 02.09. by an explicit `LLAMA_BIN` pin in flashnext.env, because `SESSION_RESTORE=displaced` in the gateway is unsafe without it. Reported as llama.cpp#28276 (open); retires when that issue is resolved either way — see the LLAMA_BIN block in flashnext.env for what each outcome means |

The names matter more than they look. `builds_of_backend()` globs
`build-<family>-*`, so a family named `rocm-patched-unroll` would be swept into
every list and every prune of the patched family — and `--prune` could offer to
delete it as a stale patched build. Hence `rocm-unroll`, not `rocm-patched-unroll`.

One deliberate exception to "a foreign family is never served": a PROFILE may
pin such a build by `LLAMA_BIN` — `--activate`/`--use` still refuse it, so the
serving symlink every other profile execs cannot reach it by accident, but a
profile that names the path explicitly, says why, and names its retirement
condition is a decision, not an accident. `setup/env/flashnext.env` is the
first case (the PR #27836 preview) and now the second as well: since 02.09.2026
it pins `rocm-cachelookup`, which is not a foreign tree at all but this repo's
own proposal to llama.cpp. The rule holds either way — the pin names the path,
the reason and the retirement condition — and it carries one extra obligation
that a fork preview does not: if the change is rejected upstream, the pin AND
`SESSION_RESTORE=displaced` go back together, in the same sitting. `--prune`
does not know about such pins — check the profiles before deleting a foreign
family's builds.

An alternate-SDK build additionally needs that SDK's libraries at RUN time:
ROCm 10.1's `libamdhip64` carries the SAME soname as Fedora's 7.1, so without
the SDK in RUNPATH the binary loads the system runtime and reports numbers
either way. That is why it cannot be activated, and why `bench/suites/speed-ab.py`
checks with `ldd` rather than trusting the stamp.

---

## The defect registry: `defects.json`

    python3 setup/lib/defects.py            what this machine is exposed to
    python3 setup/lib/defects.py --list     everything known, unevaluated
    bash setup/check.sh                     prints it as its last section

**Why it is data and not another section in a document.** The same defect was
written down in six places — HANDOVER, `patches/README.md`, three `.env` files
and a commit message — and they had already drifted: a profile still said "PR
#27739 is open" hours after it had been closed unmerged. Prose also cannot be
asked whether the machine in front of you is affected, and on this hardware
that is the whole question.

**The field that matters is `shows_as`**, and it decides the ordering:

| | |
|---|---|
| `silent` | wrong output, no error anywhere. The expensive kind, and on gfx1151 the common one |
| `loud` | crash, assert or refusal. Annoying, honest |
| `slow` | correct but slower — a tuning matter, not a hazard |
| `unrepeatable` | correct but not reproducible. Poisons MEASUREMENTS, which is how a wrong conclusion gets written down |

A registry that listed crashes first would put the harmless half at the top.

**What it deliberately does not do:** it does not measure. A defect whose only
honest answer is a measurement says so and names the suite. And an argument
check with no server running answers `unknown`, never `guarded` — silence must
not read as safety. `tests/test_defects.py` pins both.

Adding one: an entry needs `id`, `title`, `shows_as`, `symptom`, `measured`,
`mitigation`, `detect`, `status`. `detect.kind` is one of four — `cmdline`,
`build-flag`, `build-stamp`, `manual` — and that list is short on purpose. A
rule engine here would be a second thing to get wrong.

---

## How much memory the GPU may have: `ttm.pages_limit`

There is no separate VRAM on Strix Halo. The BIOS reserves a minimum (0.5 GiB,
deliberately) and everything else comes out of system RAM through GTT, capped
by the kernel parameter `ttm.pages_limit`.

    bash setup/scripts/gtt.sh                     what is set, what is in use
    bash setup/scripts/gtt.sh --set 108 --dry-run show the diff, change nothing
    bash setup/scripts/gtt.sh --set 108           set it (sudo, then reboot)
    bash setup/scripts/gtt.sh --verify            after the reboot: did it take?
    bash setup/scripts/gtt.sh --set 96            back to the conservative start

**It is a cap, not a reservation.** Raising it allocates nothing at boot; GTT
allocations are dynamic. The cost appears at runtime and only then, when a
model really claims it. That is why `--set` refuses a value that would leave
the host under 6 GiB and warns under 8.

Three files carry the same command line on Fedora and all three have to agree,
or the setting survives exactly until the next kernel:

    /boot/loader/entries/*.conf   what BOOTS — via grubby
    /etc/kernel/cmdline           what a NEW kernel inherits
    /etc/default/grub             inert under BLS, but what the next reader believes

The script writes all three and backs up the two it edits itself. The string
work is in `lib/kernelcmdline.py` and not in a `sed`, because `root=UUID=…` is
on the same line: getting it wrong is not a wrong number, it is a rescue stick.
`tests/test_gtt.py` drives it against this machine's real command line.

**Never set `amdgpu.gttsize`.** It is deprecated and ignored, and setting it
alongside `ttm.pages_limit` makes the driver report *"this is unusual"* — after
which ROCm sees LESS memory than with no parameter at all. The two documents
that recommended it are in `docs/archive/`.

---

## The four ceilings, and why this machine has them

**Read this before running anything that loads a second model.** It cost three
machine hangs in one day to learn, and none of them announced themselves —
GTT is pinned, so an over-large start does not page and does not get
OOM-killed. It freezes the box, takes every process with it, and leaves no log.

| | what it does |
|---|---|
| **the memory budget**, `setup/lib/budget.py` | asks whether a profile fits BEFORE it starts, and refuses if it does not. Runs as `ExecStartPre` on both units and as preflight in `switch-model.sh` — so every way of starting a model goes through it |
| **GTT cap 108 GiB** | an over-large model fails to ALLOCATE, cleanly, instead of hanging the machine. `bash setup/scripts/gtt.sh --verify` |
| **`MemoryHigh=96G` / `MemoryMax=108G`** on `llama-user@.service` | the model server is the victim, not the desktop. It is restartable; the desktop is not |
| **zram 4 GiB, no disk swap** | swap that lives in RAM cannot win against pinned GTT — it only takes memory away and delays the failure. `setup/zram-generator.conf` carries the full argument |

The first one is new since 27.08. and it is the only one that acts BEFORE
anything is allocated; the other three decide who dies once it is. All three
of them can only refuse after something has already decided to start.

### The memory budget, and why it is one file

    python3 setup/lib/budget.py --profile qwen38     what it will cost
    python3 setup/lib/budget.py --running            what is serving now
    python3 setup/lib/budget.py --observe            what it actually took
    bash setup/check.sh                              both, side by side

Until 27.08. the arithmetic existed three times and ran nowhere that mattered:

    bench/run.py           weights x 1.10, host reserve 10
    bench/sideserver.py    its own ceiling, host reserve 12
    tests/test_models.py   weights + cram, and no KV term at all

    switch-model.sh, llamaexec, systemd        nothing

Three copies of one rule is the same failure the three `LLAMA_ARGS` parsers
were (see `setup/lib/systemdfile.py`), and they had drifted the same way. The
consequence was worse: the copy that DID run charged `weights x 1.10`, where
the 10 % stands for "KV, compute buffers, loader". At the production profile's
`-c 204800 -cram 32768` that is 18 GiB against a real 79 — it would have waved
`qwen38` onto a 64 GB Strix Halo and frozen it.

The model, and the distinction that cost a hang on 26.08. when it was missed:

    gtt_need  = weights + KV + buffers      what the GPU PINS, bounded by the cap
    host_need = gtt_need + outside + cram   everything RESIDENT, bounded by RAM
    buffers   = max(6.0 GiB, (weights + KV) x 0.10)

GTT is system RAM, so it is the first term of the host side and not a column
beside it.

**The numbers are declared, not derived, and that is deliberate.** `qwen38`
records why in its own profile: the plain architecture arithmetic gives 256
KiB per token of KV and the measured figure is a quarter of that, because the
model is a hybrid. A guard that computed its own number would refuse the one
profile known to fit. So a profile carries what has been MEASURED:

| in `setup/env/*.env` | what it is |
|---|---|
| `MODEL_KV_KIB_PER_TOKEN` | KV cost per token of context. `MODEL_KV_SOURCE` says where it was measured — a test refuses one without the other |
| `MODEL_GTT_BASE_GIB` | GTT with the KV taken out, when the file size overstates it. Only Flash-Next needs it: 103.7 GiB of file, 78.1 in GTT |
| `MODEL_HOST_ANON_GIB` | what stays resident outside GTT, when it is more than the subtraction suggests. Flash-Next again: derived 16.3, measured 27.1 |

A profile with no `MODEL_KV_KIB_PER_TOKEN` is charged a pessimistic estimate
and **the output says so every time**. That is the point: an estimate that
does not announce itself is exactly what `-cram 32768` was when it got copied
into five profiles, two of which it took past what the machine has.

### The buffer term is measured, and the first version of it was wrong

It was `(weights + KV) x 1.10` until 27.08. Nine recorded measurements say that
is the wrong SHAPE — the buffers are roughly constant for a model and backend,
because they scale with `-ub` and the embedding width, not with the footprint:

| variant | `-c` | weights | KV | GTT observed | buffers |
|---|---|---|---|---|---|
| `rocm-*-spec` | 65,536 | 16.7 | 4.6 | 24.4 | **3.1** |
| `rocm-medium-spec-q8kv` | 65,536 | 16.7 | 2.3 | 22.7 | **3.6** |
| `vulkan-medium-spec` | 65,536 | 16.7 | 4.6 | 26.0 | **4.6** |
| production, live | 204,800 | 17.6 | 14.5 | 35.6 | **3.5** |

Tripling the KV moved the term by 0.4 GiB; Vulkan costs ~1.5 more than ROCm.
And 10 % of weights+KV is 1.9-3.2 GiB against a measured 3.1-4.6 — so the old
factor **under-predicted every one of those nine points**, which is the
dangerous direction. It looked right in production only because at the largest
window the percentage happened to land near the constant.

So the floor sits above the worst measurement, and the percentage is kept as an
upper branch for footprints far larger than anything measured here, where a
constant would be the optimistic answer instead.

### What `-cram` buys, which is not the same question as whether it fits

    python3 setup/lib/budget.py --cache

`-cram` was computed once, for `qwen38`, and copied into every profile that
came after. The audit of 27.08. found it 30 GiB over on Flash-Next and fixed
that one — it never asked what the flag BUYS. In full windows, where a window
is the worst case a single prefix can cost:

| | before | after |
|---|---|---|
| `batch` | **196.9** windows — 16 GiB of cache for an 8k context | 12.3 |
| `gptoss` | 7.1, on the tightest profile in the registry (11 GiB spare) | 3.6 (27 GiB spare) |
| `gemma26` | 24.6 | unchanged — generous, and 59 GiB spare |
| `gemma31` | 16.0, on an ESTIMATED KV | unchanged — measure the KV first |
| `qwen38` | 2.2 — the one that was computed | unchanged |

The two that changed were indefensible rather than merely generous. The two
that did not now carry the arithmetic in a comment, so the next reader sees a
number that was CHECKED — which is the whole complaint against this flag.
`tests/test_budget.py` fails on anything above 50 windows.

### Raising it for your own load, which is a different arithmetic

A WINDOW IS THE WRONG UNIT for this decision, and reading it as the right one
is what cost 599 s on 02.09.2026. Two things a window-count cannot see:

* every entry pays a FIXED part, whatever it holds — a 24-token health probe
  leaves a 226 MiB entry behind, so a 4 GiB budget holds at most 18 entries
  however small they are;
* a SERVED session costs more than a prefill of the same length.

Measured on Flash-Next (`bench/suites/cram-state-size.py`, report
`bench/reports/2026-09-02_1002_cram-state-size-verify/`), from llama-server's
own eviction lines and reproduced twice with different prompts, a plain prefill
is a straight line over five points:

    entry = 336.7 MiB + 39.12 KiB/token          2,000 to 90,000 tokens

**Do not size a budget from it.** A SERVED session costs more than its length,
and the surcharge grows with depth — which two points at one depth cannot show,
and which is exactly how `8704` was chosen and found wanting 90 minutes later:

     80,507 tokens   3946.260 MiB    line 3412    +534 MiB    ( 6.8 KiB/token)
    148,485 tokens   8422.274 MiB    line 6009   +2413 MiB    (16.6 KiB/token)

So size from what the machine has actually logged, not from the line:

    journalctl --user -u 'llama-user@*' | grep "making room"

Every one of those names a real entry in MiB. Take the deepest, add one probe
entry (226 MiB here) per takeover you expect between two turns, and round up.
The coefficients above are this model's; a different model, window or draft
head gives different ones, and the suite re-measures them in about four minutes
on any of them.

WHAT THE CEILING DOES, because it is silent. An entry larger than the WHOLE
limit is not shrunk and not partly stored: it is skipped, and that conversation
is uncacheable from that turn on. At 4096 MiB on this profile the line above
puts that at ~98,400 tokens for a bare prompt and ~84,400 for a served session.
Leave room above the deepest session you actually run.

CHECKING IT BEFORE THE RESTART IS HARDER THAN IT LOOKS, and both obvious ways
mislead. `--check` refuses EVERY value while the server is running, including
the one it is running with — it asks whether a second server fits beside the
first, and a restart stops the first one. A correct value was withdrawn here on
the strength of misreading that. But `--static` is not the answer either: it
prints the balance against `machine has …` and stays SILENT on the verdict, so
a value it shows without complaint can still be refused at start. The guard
weighs `available minus the 12 GiB reserve` — on this machine 116.5 − 12 =
104.5 GiB, which refused a 105.1 GiB profile that `--static` had shown plainly.

    python3 setup/lib/budget.py --profile <name> --static   # the balance
    python3 setup/lib/budget.py --profile <name> --check    # the verdict,
                                                            # server STOPPED

AND IF A START FAILS TWICE, RESET BEFORE RETRYING. systemd rate-limits with
"Start request repeated too quickly" and goes on displaying the PREVIOUS
attempt's error, so a corrected value reads as though it changed nothing:

    systemctl --user reset-failed llama-user@<profile>.service

AND AFTERWARDS, ASK THE MACHINE RATHER THAN THE ARITHMETIC. `check.sh` counts
the evictions llama-server logged in the last 24 h. Nonzero means the budget is
smaller than the load actually served; each line names the MiB it threw away.

### And the declaration is re-checked, or it is only an assertion

Every figure in this repo that was DERIVED turned out wrong — KV from the
architecture by 4x, the per-prefix cache estimate by 3-4x, the Flash-Next
footprint by 30 GiB. Every figure that was MEASURED held. A declared number
with nothing checking it drifts back into the first kind.

So `check.sh` closes the loop: it holds the prediction against what the
running server actually pinned. Only one direction is a defect — observed
comfortably below predicted means the guard is conservative, which is its job.
Observed ABOVE it means the guard under-predicts, and under-predicting is how
a machine freezes.

    = GTT predicted 35.3 GiB, observed 35.6 (+1 %) — not under-predicting

Both sides overstate on purpose: `mem_info_gtt_used` is system-wide, so the
desktop is in it, and the prediction carries its 10 % slack.

### The escape hatches correct the input; they do not skip the check

    LLM_MODEL_GIB=88            the weights the GPU will pin
    LLM_HOST_GIB=104            the host footprint — floored at the file size
    LLM_KV_KIB_PER_TOKEN=74.3   the KV figure
    LLM_NO_MEMORY_GUARD=1       off. A different and worse thing.

The first three keep every comparison running and make you write down a number
you measured. That is a different act from switching a safety off because it is
in the way, and the difference is why the machine hung twice in one day. The
floor on `LLM_HOST_GIB` is the third hang written down: `88` was once claimed
for a model with 103.7 GiB on disk, and nothing makes a file smaller.

The `BENCH_*` names bench/run.py shipped with still work, as aliases.

**The cap is a ceiling as much as a budget.** Raising it feels free, because
raising it allocates nothing. What it costs is invisible until the day
something does not fit: below the cap that is an error message, above it that
is a power cycle. This machine ran at 116 for one day, for a footprint that
measured 22 GiB smaller than predicted, and hung twice inside it.

### Never start a model beside another one by hand

    python3 bench/sideserver.py --env setup/env/<model>.env --port 8081 \
        --stop llama-user@qwen38 -- <command to run while it is up>

It arms a dead man's switch, stops the running unit, **waits for GTT to
actually fall** — not for the port to close — refuses if the weights do not
fit, starts the model in its **own cgroup with its own ceiling**, and restores
production afterwards.

Every one of those clauses is there because something without it failed:

* `kill; sleep 5` between two 87 GiB models put the second on top of an
  allocation still being torn down. GTT is released asynchronously; a dead
  process does not mean freed memory.
* A server started as a plain child inherits the CALLER's cgroup, so the
  limits above never applied to it. Only a transient systemd unit gets its own.
* `finally` is not a guarantee. When the OOM killer took the orchestrator,
  production stayed down until a human noticed. The dead man's switch is a
  systemd timer, armed BEFORE anything is stopped.
* And the memory guard checks two numbers against two limits. A measurement
  can tell you what the GPU pins; nothing makes the bytes on disk smaller, and
  `BENCH_MODEL_GIB` may correct the first and not the second.

### Foreign workloads go through the same fence

    python3 bench/sideserver.py --workload setup/workloads/sdxl.env \
        --stop llama-user@qwen38                    # the profile's own job
    python3 bench/sideserver.py --workload ... --stop ... -- <command>

Since 01.09.2026 the tenant does not have to be llama-server. A
text-to-image job (and later audio/video) pins GTT like any model, so it
gets the same dead man's switch, the same GTT-settle wait, the same weigh-in
(`budget.py --workload <name>`), and a transient unit with a derived
ceiling — plus 1 Hz peak metering whose output is the ready-to-paste
declaration for the profile's measured fields.

## The workload registry

`setup/workloads/*.env` is the list of foreign workloads, same file
discipline as the model registry below, different lifecycle: started only
through `bench/sideserver.py --workload`, never by `llama-user@`, so they
appear in no `Conflicts=` line and `switch-model.sh` does not offer them.

    bash setup/lib/models.sh workloads       what exists
    python3 setup/lib/budget.py --workload sdxl    what it costs, whether it fits

A profile's footprint fields are **born UNMEASURED**; budget.py then charges
the model files plus its buffer term and announces ESTIMATE in every
verdict. After the first fenced run the observed peaks are declared with
date + method + machine — `tests/test_workloads.py` holds that contract,
including the refusal case. The first consumers are the image profiles
(`sdxl.env`, `flux-schnell.env`, `qwen-image.env`) and the audio profiles
(`qwen3-tts.env` on the ggml/Vulkan side, `chatterbox.env` in the torch
lane); the Torch-world border that keeps the latter out of the base
install is `media/README.md`.

## The model registry

`setup/env/*.env` is the list of models. Not a list somewhere else that
happens to match the files — the files themselves:

    bash setup/lib/models.sh table      what exists, what runs
    bash setup/switch-model.sh --list   the same, plus the process command line
    bash setup/switch-model.sh <model>  switch, with a full preflight first
    bash setup/switch-model.sh <model> --dry-run    print the plan, change nothing

Every profile declares two things about itself beside its arguments:

    MODEL_TITLE=…      one line, for --list
    MODEL_SWA=yes|no   does this model have a sliding window? (see below)

systemd passes both to `llama-server`, which ignores them. A comment would be
invisible to the tooling; a variable is not.

**The one list that cannot be derived** is `Conflicts=` in
`llama-user@.service`, and the system unit derives its copy from it: systemd has no wildcard for
template instances, so every model has to be named there by hand. A model
missing from it is the worst failure this stack has — both servers start, one
loses the race for port 8080, systemd still says `active`, and the gateway
answers from whichever won. `tests/test_models.py::TestConflicts` compares the
line against the registry, and `switch-model.sh` checks it again before it
switches.

---

## The server switch that decides everything: `--swa-full`

Which models have a sliding window is a property of the MODEL and is declared
in its own profile as `MODEL_SWA`:

    bash setup/lib/models.sh table

    gemma-4-26B  1024      gemma-4-31B  1024
    laguna-s-2.1  512      gpt-oss-120b  128      qwen3.8-27b  none

With SWA, llama.cpp can only roll the KV state back inside the window. If the
point of divergence between two requests is further from the end of the prompt
than that, the server discards the entire prefix. Claude Code appends a
1,624-token block behind the user question — so always too far.

`--swa-full` allocates the KV cache of the SWA layers at full context length.
Measured on the real Claude Code body, changed question:

    without --swa-full   new=19,371  cache=0        100.2 s
    with    --swa-full   new= 1,637  cache=17,734    10.4 s

Tool conversation, four turns, without any proxy, with `--swa-full`:

    turn 1 (cold) 101.5 s · turn 2  2.0 s · turn 3  1.5 s · turn 4  1.5 s

The switch is entered in `env/laguna.env`. Counter-check:

    journalctl -u llama@laguna -n 300 | grep "full-size SWA cache"
    -> llama_kv_cache_iswa: using full-size SWA cache

The system unit does not redirect its output anywhere, so it lands in the
journal. With a manual start, correspondingly in whatever file it was
redirected to.

Memory: GTT 73.2 → 82.8 GiB of 96.

---

## cc-cachefix.py is superseded — the gateway does this now

The old version pulls **all** `system` messages out of `messages` to the start
of the prompt. For simple requests that helped (the user question slides to the
end, into the SWA window). From turn 2 on, however, Claude Code appends one
more `system` message per turn with a **changing counter**:

    <total_tokens>14981262 tokens left</total_tokens>

The old proxy pulls that to the front too — ahead of the 16,789-token tool
block, which therefore expires on every turn. Measured, with `--swa-full`
active:

    cc-cachefix.py    turns 2..4   15.3 % cache   89–90 s per turn
    no proxy          turns 2..4   99   % cache    1.5–2.0 s
    cc-cachefix2.py   turns 2..4   99   % cache    1.5–2.0 s

`gateway.py` (like its predecessor `cc-cachefix2.py`) splits the stable and
the volatile part: the stable part moves to the front, the counter stays at the
end of the history. That additionally brings the "new session, different
question" case down from 10.4 s to 0.7 s.

**Trap:** Claude Code appends the counter to the *end* of the otherwise stable
agent-types block (characters 6,979–7,028 of 7,028). Classifying the whole
block as volatile means never hoisting it and throwing the effect away.

A proxy is needed anyway: `llama-server` answers the fields `thinking`,
`context_management` and `output_config` with 400, and unbuffered forwarding is
what keeps Claude Code from aborting after 300 s of silence.

Start it and point a profile at it:

    python3 ~/.local/lib/llm-stack/gateway.py      # port 8090
    claude --settings ~/.claude/profiles/local.json

**Practical test**, warm server, wall clock of the entire `claude -p` call:

    simple question                       1.3 s
    tool conversation, 2 reads, 3 turns  13.0 s

**Note:** `claude -p` against a local model only starts reliably with
environment variables. With `--settings` it aborts with `unrecognized_model`
during title generation.

---

## Access from the LAN

The gateway is the single entrance. LAN access needs two things: an additional
bind address and named access. **Without any access configured, the gateway
rejects everything that does not come from 127.0.0.1** — there is no accidental
open state.

### Setting it up

Create a secret and put it into `~/.config/llm-gateway-tokens` (the file stays
local, mode 600, excluded from the repo via .gitignore):

    python3 -c "import secrets; print(secrets.token_urlsafe(32))"

    # one consumer per line: name<whitespace>secret
    martin-mobile   <the generated secret>

And add the LAN address in `~/.config/llm-gateway.env`:

    BIND=127.0.0.1,<your LAN address>     # ip -4 -o addr show scope global

Then `systemctl --user restart llm-gateway`.

### Firewall

The socket now listens on the LAN address, **but it is only reachable once
firewalld lets the port through**:

    sudo firewall-cmd --add-port=8090/tcp --permanent
    sudo firewall-cmd --reload

Until that has happened, the gateway listens on the address without any packets
from other machines arriving. The state can only be checked as root:

    sudo firewall-cmd --list-ports

### Tested access cases

Measured on the running gateway, all four as expected:

    local, without a token             HTTP 200   priority "local"
    via the LAN address, no token      HTTP 401   authentication_error
    via the LAN address, with a token  HTTP 200   priority "lan"
    via the LAN address, wrong token   HTTP 401

Visible in the log:

    REJECTED    192.168.178.42  lan    no valid token
    START       192.168.178.42  lan    prefix=f370acc9d8b7 warm

### Two caveats

**Docker bypasses firewalld.** Docker is installed on this machine (29.7.2,
`docker0` on 172.17.0.1/16). Docker writes its own iptables rules and can
thereby make published container ports reachable without firewalld preventing
it. Anyone containerising the stack later should not publish ports with
`-p 8090:8090` but bind to `127.0.0.1` (`-p 127.0.0.1:8090:8090`).

**Containers count as "lan".** Addresses from 172.16/12, 10/8 and 192.168/16
are classified as a private network and given priority 1. A container on
`docker0` therefore gets the same priority as a machine on the LAN — not the
local one. That is intended, but worth knowing.

---

## Access from the internet: Cloudflare tunnel

### The time limit you have to know about — and why it carries here

Cloudflare aborts a connection with **error 524** if the origin sends nothing
for 125 seconds (proxy read timeout, changeable only on the enterprise plan). A
cold start takes 100 to 180 seconds here — that would look like a knockout
criterion.

It carries anyway, because `llama-server` sends keep-alives itself in streaming
mode. Measured on a 181-second prefill with 40 tools:

    30.2 s  keep-alive    3 B    ":\n\n"
    60.2 s  keep-alive    3 B
    90.2 s  keep-alive    3 B
   120.2 s  keep-alive    3 B
   150.2 s  keep-alive    3 B
   180.2 s  keep-alive    3 B
   180.8 s  data        548 B    <- the actual start of the answer

    largest gap 30.2 s · Cloudflare limit 125 s · headroom 95 s

**But only with `stream: true`.** A non-streamed request with a long prefill
stays silent and runs into the 524. Claude Code streams by default; scripts and
`curl` tests often do not.

**And every phase in front of llama-server has to send its own sign of life,
which one did not.** On 31.08.2026 the first streamed request from another
machine hit the 524 anyway: the gateway's save-before-serve phase prefilled a
cold 22k prefix for 114.5 s without writing a byte — llama-server never saw
the request during it, so the keep-alives above never ran. Fixed the same day
(`setup/defects.json`, `save-before-serve-is-silent`): the save phase now
sends the queue phase's `:` every 30 s. The server-side table above was
re-measured on b10702-11 the same day and holds — status after 0.2 s, a ping
every 30 s across a cold 114.6 s prefill. `bench/suites/sse-ping.py` measures
any endpoint of the chain the same way.

### The separate tunnel port

When a request arrives through a tunnel, the source IP is **no longer the
client's** but `cloudflared`'s — `127.0.0.1` when run natively, `172.17.0.x` in
a container. Classifying by IP would therefore treat internet traffic as "local"
or "lan", i.e. higher than it deserves.

That is why the gateway has a second port. Whatever arrives there counts
**always** as remote and **always** needs a token — regardless of the source IP.
This works without trusting any header; a forged `CF-Connecting-IP` is of no use
to anyone.

    TUNNEL_PORT=8091
    TUNNEL_BIND=127.0.0.1,172.17.0.1

Measured, same source IP 127.0.0.1:

    via 8090                 HTTP 200   priority "local", no token
    via 8091 without token   HTTP 401
    via 8091 with a token    HTTP 200   priority "remote"

### Docker or native?

For reliability there is little in it — `--restart unless-stopped` and
`systemctl enable` do the same job. **Recommendation: Docker**, because it is
installed anyway and because `cloudflared` is the one component facing the
internet; a container limits what it can reach if it is compromised.

    docker run -d --name cloudflared --restart unless-stopped \
      cloudflare/cloudflared:latest tunnel --no-autoupdate run --token <TUNNEL-TOKEN>

Enter this as the target in the tunnel service:

    http://172.17.0.1:8091

**No `--network host`.** With it, `127.0.0.1` inside the container would be the
container itself again and the isolation would be gone. Via `172.17.0.1` the
container reaches the host binding of the tunnel port.

**Do not publish ports.** Docker writes its own iptables rules and bypasses
firewalld. `cloudflared` needs no inbound ports — it builds the connection
outwards.

### Authentication

A shared token is fine for the LAN, thin for the internet. **Put Cloudflare
Access in front** — then identity decides instead of a secret that can be passed
on. The gateway token stays as a second layer.

Whoever reaches the endpoint does not only burn GPU time: Claude Code clients
send file contents through this proxy.

---

## Managing access

Every consumer gets their **own, named token**. A single shared secret would be
neither individually revocable nor attributable in the log.

Access lives in `~/.config/llm-gateway-tokens`, mode 600, **outside the repo**:

    # one access per line:  name<whitespace>secret
    martin-mobile  xY7…
    friend-anna    kL2…

### Creating access

    python3 -c "import secrets; print(secrets.token_urlsafe(32))"
    # append a line "name <secret>", then:
    systemctl --user restart llm-gateway

### Revoking access

Delete the line, restart. Measured: the revoked access gets 401 immediately,
everyone else carries on unchanged.

### Who was that?

The log names the name, never the secret:

    START   172.19.0.2  remote  who=martin-mobile  prefix=f370acc9d8b7  warm

`GET /gateway/status` (local only) shows per prefix which accesses use it, the
warm rate and the average duration.

### Limit per access

`PER_TOKEN_MAX` (default 2) caps the concurrent requests **per access**. Above
that comes `429`. Two is a deliberate choice: Claude Code sends up to two prompt
types in parallel.

This is not rate limiting over time — whoever has a valid token can keep the GPU
busy indefinitely, just not with arbitrarily many requests at once.

### The queue: priority, but not forever

Everything above `MAX_INFLIGHT` (the slot count) waits in the gateway, ordered
`local` before `lan` before `remote`. Two settings keep that from turning into
a trap:

    QUEUE_AGE_AFTER   30   after this many seconds of waiting, a request is
                           served next whatever its zone
    QUEUE_KEEPALIVE   30   a queued STREAMING request gets a `:\n\n` this
                           often — and since 31.08. the save-before-serve
                           phase sends the same sign of life at the same pace

Both were added because both failure modes were measured. Strict priority
starves the lower zones: with four local streams — two Claude Code sessions — a
LAN caller was still waiting after 200 seconds and only got in once the load
stopped. And because nothing was written to the client while queued, a remote
caller was dropped by Cloudflare after 125 s of silence without the gateway
learning about it. After the fix the same caller is served after 31 s.

`/gateway/status` counts under `overtaken_by_age` how often ageing had to beat
priority. A climbing number means the machine is oversubscribed — that is the
moment to think about `-np 4` — but see the defect registry first:
`setup/defects.json` still lists two silent corruptions that only `-np 1`
avoids.

Non-streaming requests deliberately get no keep-alive: it would mean committing
to a status code before the upstream has answered. They also have no protection
against llama-server's own silence during a prefill, which is why
docs/CONSUMERS.md tells consumers to stream.

---

## The system unit is derived, and opt-in

    bash setup/install.sh --system-unit        generate and install it
    python3 setup/lib/systemunit.py            print what it would produce
    python3 setup/lib/systemunit.py --check    is the installed copy current?

There is ONE unit file in this repo, `systemd/llama-user@.service`. The system
unit is generated from it by `lib/systemunit.py`, which holds the whole mapping
in one list: the paths a system service may not read from `%h`, the instance
names, the target it is wanted by, and the two directives a user service cannot
carry (`User=`, `SupplementaryGroups=`). Comments are dropped rather than
translated — the substitutions are right for directives and wrong for prose,
and the reasoning belongs in the file you are told to edit.

**Why it is generated.** There used to be a second, hand-written
`llama@.service`. It had never been started here — SELinux refuses it, see
below — and by 27.08. it had rotted three ways at once:

| | user unit | the hand-written system unit |
|---|---|---|
| binary | `llamaexec` → `$LLAMA_BIN` from the profile | pinned `build-vulkan/bin/llama-server` |
| so, in practice | `build-rocm-patched` | Vulkan, **without the gfx1151 patch** |
| `MemoryHigh` / `Max` | 96G / 108G | 48G / 64G |
| `TimeoutStartSec` | 900 s, deliberately | **absent** — systemd's 45 s default, while `llm-wait-for-model` alone waits up to 120 |

The binary row is the serious one: enabling it would have served the unpatched
build, which is the one that degenerates to `////`, and the reader would have
blamed the model. The timeout row would have fired first — `llm-wait-for-model`
waits up to 120 s and systemd's default would have killed the unit at 45,
three times, after which it stays down for good.

Nothing caught any of it, because nothing ran it — the exact failure this repo
keeps describing, where nothing breaks and an effect simply fails to appear.

**What it is for.** A Strix Halo with no user session to linger on — the boxes
this hardware ships in are sold as headless inference servers. On a desktop use
the user unit and `sudo loginctl enable-linger $USER`, which is what this
machine does.

**Honest limitation.** The generated unit is installable and unit-tested
against the unit that runs every day, and it has still never been *started*,
because Fedora refuses it. "It matches the tested unit" is a different claim
from "it works", and only the first one is made.

## SELinux: why the system unit fails on Fedora

The system service `llama@laguna` aborts with `status=203/EXEC` and
`Unable to locate executable`, although the file exists and is executable. The
audit log holds the real reason:

    AVC avc: denied { execute } for comm="(llama-server)" name="llama-server"

The binary lives in `~/llama.cpp` and therefore carries the context
`unconfined_u:object_r:user_home_t:s0`. A system service may not execute
`user_home_t`. For comparison: `/usr/local/bin/llm-profile` carries `bin_t`.

### The route that works without root

`llama-user@.service` — the same unit as a **user service**. The process runs in
the user context and may execute the binary:

    systemctl --user enable --now llama-user@qwen38

It catches the same things as the system variant, in particular
`Restart=on-failure` against the GPU device loss. Limitation: it only starts on
login. For a start at boot, once:

    sudo loginctl enable-linger $USER

### The alternative with root

Either put the build in a system path (`/opt/llama.cpp`) — then
the generated system unit works unchanged — or set the context:

    sudo dnf install policycoreutils-python-utils          # for semanage
    sudo semanage fcontext -a -t bin_t "$HOME/llama.cpp/build-*/bin(/.*)?"
    sudo restorecon -Rv $HOME/llama.cpp

    # build-* rather than build-vulkan: the binary a profile starts comes
    # from its own LLAMA_BIN, and relabelling only the one you happen to
    # run today leaves the next switch failing with 203/EXEC.

**The second is still untested, but the doubt that used to stand here was the
wrong one.** It read: *"the binary loads eight libraries from the same
directory, which would then have to be labelled to match as well."* Checked
27.08.2026 — the count is exact, `ldd` shows eight objects resolved out of
that `bin/`, and every one of them carries `user_home_t` like the binary. But
the rule above ends in `(/.*)?`, which is the whole directory and its
contents, so it already reaches them. Coverage was never the problem.

What IS untested is narrower and worth naming, because it is the thing that
would actually decide it: **Fedora labels shared objects `lib_t`, not
`bin_t`** — `/usr/lib64/libc.so.6` is `system_u:object_r:lib_t:s0`, while
`/usr/local/bin/llm-profile` is `bin_t`. The rule above gives the libraries
`bin_t` along with the binary. Whether a system service may then MAP them is
the open question; it is a different question from whether it may EXECUTE the
binary, and nobody here has answered it.

Doing so needs three root steps and a persistent policy change on a home
directory, which is why it has not been done for the sake of a unit that the
user service already replaces.

**As long as the system unit is not needed it belongs switched off**, otherwise
it fails on every boot:

    sudo systemctl disable --now llama@laguna

---

## Abolishing the cold start: saving prefixes

The first call of a project costs 100 to 180 seconds — about 20,000 tokens of
system prompt and tool schemas. After every server restart that falls due again.
With saved prefixes it is **1.4 seconds**.

### How it works

Two observations, both measured:

1. A restored slot state carries pure **appending** flawlessly, but **no
   rolling back**. The slot file holds only the global layers plus the SWA
   window — 27 KiB per token instead of 102.
2. Everything up to `<user>` is identical for every request of a project: here
   22,488 of 22,499 tokens, i.e. 99.95 %. Save **exactly that section** and any
   question at all becomes an append.

What matters is rendering the prefix the way the **gateway** produces it — with
the agent-types block hoisted. Without that the 1,665-token block stays behind
the question and still costs ~10 s instead of 1.4 s.

### Using it

    # once per project configuration
    python3 tools/prewarm.py save --body body.json --name projectA

    # what is in there
    python3 tools/prewarm.py list

    # does everything sit under the id the gateway looks it up by?
    python3 tools/prewarm.py check [--repair]

Restoring runs automatically: `llama-user@.service` calls it as
`ExecStartPost`. After every restart the slots are filled before the first
request arrives.

### When the gateway saves by itself

When a cold request arrives, the gateway calls the same script in the
background — with one difference: it passes its own id through with
`--gateway-id` instead of having it recomputed. The reason is the body it hands
over. By that point it is already **corrected**, so the system field carries the
hoisted agent-types block; only that way can the prefix be rendered that later
actually arrives. But computing an id from that body yields a different value
than from the incoming request.

That is exactly what went wrong between 23 and 24 August: saving happened, the
file was there, the log reported `SAVED` — and `RESTORED` never came, because
the sidecar file sat under a key that no request ever produces. `check` finds
such files, `--repair` puts the id right (then `systemctl --user restart
llm-gateway`), and `tests/live_prefix.sh` checks the whole chain against the GPU.

### Measured, against a real service restart

> **Laguna, and a dated record.** 628 MB for 22,489 tokens is 28.6 KiB per
> token — Laguna's sliding window kept its slot states tiny. qwen38 has none,
> and a prefix costs **74.3 KiB per token** (measured 27.08.), so the same
> 22.5k-token prefix is 1.6 GiB rather than 628 MB. The TIMES below still
> describe the mechanism; the SIZES are the previous model's.

    save                        22,489 tokens, 628 MB,  237 ms
    restore                                             103-185 ms
    first question afterwards   99.6 %   1.39 s       (instead of 110 s)
    further questions           99.8 %   0.59-0.76 s
    tool turns                  99.5 %   1.25-1.36 s
    systemctl restart in total           21.5 s

### The three cache levels

The slot count does **not** limit how many projects stay warm. There are three
levels with very different reach:

| Level | Where | How many | Swap-in | Survives a restart |
|---|---|---|---|---|
| slots (`-np`) | GTT | **1** | immediate | no |
| RAM cache (`-cram 32768`) | main memory | ~32 small, **~10 real** | 0.3 s | **no** |
| disk (`--slot-save-path`) | `~/.cache/llama-slots` | unlimited | ~0.15 s | **yes** |

**Both counts were Laguna's until 27.08. and both were wrong for qwen.** The
slot row said 2; production has run `-np 1` since 26.08. The cache row said
~54, which assumed Laguna's 628 MB prefixes — qwen has no sliding window, so a
prefix holds the full f16 KV state at **74.3 KiB per token**, measured out of
the twelve prefixes actually on disk. So 32 GiB is about 32 of this project's
typical prefixes (1.0 GiB each) but only **~10 contexts the size of Claude
Code's tool head** (3.0 GiB each), and 2.2 full 204800-token windows.

With 20 active projects the RAM cache therefore does NOT cover all of them —
it covers the ten most recent and the disk takes the rest. After a restart it
is empty and the disk takes over entirely.

**The gateway reloads by itself when needed.** When a request arrives whose
prefix is cold but lies on disk, it pulls it into a free slot before forwarding.
Measured after a restart with completely cleared slots:

    projA  first question   RESTORED -> slot 0,  89 ms   then 99.6 %  1.54 s
    projB  first question   RESTORED -> slot 1,  81 ms   then 99.6 %  1.74 s
    afterwards both       99.8 %  0.91-1.06 s

`ExecStartPost` is therefore only a warm-up for the first two; everything else
comes automatically when it is needed. If more projects are active than there
are slots, they evict one another — the evicted one lands in the RAM cache and
comes back from there or from disk in fractions of a second.

### How many to keep — and what to clean up by

**A saved prefix never changes through use.** It holds only the system prompt
and the tool schemas; nothing from `<user>` on is in it. The modification date
of the `.bin` therefore only says when it was saved — as an expiry criterion it
is no good. A project used daily would carry the same date as a forgotten one.

A prefix only becomes invalid when the configuration changes: `CLAUDE.md`, MCP
servers, skills, or a Claude Code update that rewords the system prompt. Then
the id no longer matches, the file is never asked for again and lies around as a
leftover.

The usable criterion is therefore **last used**. Only the gateway knows that,
and since this version it writes it into the sidecar file — at most hourly per
prefix, so the disk does not work needlessly.

    projA   22492 tokens  628 MB  saved 15:57  last used 16:08 (1x)
    projB   22492 tokens  628 MB  saved 15:59  never used

#### The strategies

| Rule | For what | Call |
|---|---|---|
| **size limit** | the actual constraint is the disk | `AUTO_MAX_GB`, default 20, set to 100 here |
| **count** | when a fixed number is easier to keep track of | `--max-count 30` |
| **age since last use** | leftovers after configuration changes | `--ttl-days 180` |

All three delete **the longest unused first**, not the oldest files.

    python3 tools/prewarm.py cleanup --max-gb 100 --ttl-days 180 --dry-run
    python3 tools/prewarm.py cleanup --max-gb 100 --ttl-days 180

`--dry-run` only shows what would go.

#### Recommendation

**Size limit as the main rule, a generous TTL as a supplement.** At 628 MB per
prefix, 100 GB is about 150 projects — more than anyone realistically maintains in
parallel, and a fraction of the 386 GB of free disk.

The TTL should be measured in **months**, not days: persistence is meant exactly
for the project you pick up again after weeks. 180 days reliably clears out what
is dead through configuration changes without throwing away anything still
needed.

As a recurring task:

    # ~/.config/systemd/user/prefix-cleanup.timer  (weekly)
    [Timer]
    OnCalendar=weekly
    Persistent=true

#### Automatic saving

The gateway saves a prefix **by itself** as soon as it has warmed up for the
first time. Nothing has to be done by hand for that any more.

What matters here: it does **not save the slot as it stands after the request**
— that would hold prefix plus question plus answer and would then only be good
for exactly the same request again. It calls `prewarm.py save` instead, the same
version as a manual call, which produces the part up to `<user>`. Because the
slot is already warm, that costs fractions of a second:

    cold start of a new project      109.7 s
    saved automatically afterwards     0.4 s
    after a server restart            99.6 % cache

Switches: `AUTO_SAVE=0` turns it off, `AUTO_MAX_GB` (default 20, 100 here) keeps the disk
from filling up, and `AUTO_MIN_CHARS` (default 4000) keeps trivially small
prefixes out of the store. It runs in the background after the answer — the
caller never waits for it.

#### Restart detection

A watcher checks the slots every 15 seconds. If `llama-server` disappears or
every slot is empty, the gateway forgets its prefix bookkeeping.

Without that it kept considering a prefix warm after a server restart, did not
load it from disk, and the request ran into a full cold start — 109.7 s although
the file was ready. Noticed during testing.

#### Cleanup runs weekly

    prefix-cleanup.timer  ->  --max-gb 20 --ttl-days 28

Size limit as the main rule, 28 days since **last use** against leftovers after
configuration changes.

#### What is still missing

- **A per-access quota.** With named tokens it would make sense to set an upper
  limit per consumer so that one of them cannot fill the disk. Not built yet —
  relevant as soon as several foreign users are allowed to save.

### Limits

- **The prefix is bound to the configuration.** If the system prompt, a
  `CLAUDE.md` or the tool set changes, the saved state no longer matches and the
  request runs cold. The sidecar file `<name>.json` records the id so that this
  shows.
- **628 MB per prefix.** Correspondingly more with several projects; the files
  live under `~/.cache/llama-slots`.
- **More saved prefixes than slots** cannot be held at once. The rest stays on
  disk and is not loaded at start.
- The cold start does not disappear entirely — it falls due **once per project
  configuration** instead of once per restart.

---

## Two traps with the .env files

**They are systemd syntax, not bash-sourceable.** A `. file.env` in bash fails
silently, because bash reads `VAR=value command args` and tries to execute the
model name as a command. To read them from a script, join the continuation lines
yourself.

**Do not use variables.** systemd does not expand recursively in an
`EnvironmentFile` — an `M=/srv/models` with `$M/model.gguf` lands verbatim
in the call. Every path is therefore written out in full.

---

## Ports

    8080   llama-server — qwen38 | laguna | gptoss | flashnext | qwen36 |
           glm47flash | gemma26.
           The gateway asks THIS port (LLAMA_URL), so only a profile on it is
           reachable by any consumer.
    8081   the side port the measurement suites start their own server on
    8082   gemma31   (bench/sideserver.py, bench/suites/np2-candidates.py,
    8083   batch     setup/scripts/slot-test.sh), and cc-router.py in variant 2
    8090   llm-gateway (gateway.py)
    8091   llm-gateway, tunnel port (everything here counts as "remote")

**gemma26 was one of them and moved to 8080 on 04.09.2026** — it could not be
switched to at all while it sat on 8081, because `switch-model.sh` reads the
port out of `LLAMA_ARGS` and aborts when it disagrees with the gateway's. The
profile would have failed its own preflight. The same is still true of the two
below.

**The two profiles on 8082-8083 cannot currently be used.** They were given
their own ports to run ALONGSIDE the main model, and the `Conflicts=` line in
`llama-user@.service` makes that impossible — only one instance can be active,
because until 26.08. they also all wanted the whole of a 96 GiB GTT.
`switch-model.sh` refuses them in preflight rather than starting a model no
consumer can reach. Two ways to resolve it, and they lead to different places:

* move them to 8080 and accept that they are ALTERNATIVES, switchable like
  the rest; or
* take them out of `Conflicts=` and let them run in parallel — which is newly
  plausible at the 108 GiB cap (gemma26 is 13.5 GiB) but needs the gateway to
  learn about a second upstream, and needs the memory arithmetic done for two
  resident models.

Until one of them is chosen, the profiles are documentation of an intent, not
a working configuration.
