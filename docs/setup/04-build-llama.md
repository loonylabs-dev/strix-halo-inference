# 04 · Building llama.cpp

You need a build from source on this GPU, and it needs one patch. This is the
chapter with the longest wait and the shortest explanation.

## Why not a package or a container

Both are reasonable elsewhere. Here:

* the **patch below is not upstream**, and without it a second slot corrupts
  every answer with nothing in any log;
* the flags that matter changed during 2026, and a build from a few months ago
  fails or silently loses performance;
* `bench/` measures builds against each other, which needs more than one.

There are prebuilt Toolbx containers for exactly this hardware
(`kyuz0/amd-strix-halo-toolboxes`) rebuilt on every llama.cpp update, and
decoupling the inference stack from the host is a real stability argument. If
you take that route you still have to get the patch in.

## The patch, and why it is not optional

    setup/patches/hip-integrated-off.patch

One line. It stops the HIP backend from trusting `prop.integrated`. Upstream
already disables that path for CUDA — "Temporarily disabled due to issues with
corrupted output" — and re-enabled it for HIP in PR #24233.

On gfx1151 that path **corrupts the KV state as soon as a second slot is
used.** Every answer afterwards degenerates to `////` until the server is
restarted. No error, no log line, no crash.

Measured with a fresh server per cell (`bench/suites/slot-corruption.py`):

    stock build,  two slots, empty store   CORRUPT 6/6
    patched,      two slots, empty store   clean — and CORRUPT 3/6 on the
                                           NEXT start, same configuration
    patched,      one slot                 clean in every cell

Cost, on identical bodies: prefill −6.0 %, decode unchanged.

**Note what the patch does not buy.** Look at the middle line: patched with
two slots was clean on one start and corrupt on the next. Two slots are a
gamble per server start on this hardware, which is why production runs
`-np 1`. The patch removes one of two failure modes, not both.

Upstream: llama.cpp #27579 (same hardware, same symptom, trigger
`--parallel >= 2`), #27572 (root cause), #27506 (a gfx1151 regression
bisected to the commit this reverts).

## Building

Use the script. It exists because doing this by hand breaks things in three
ways that are not obvious:

    bash setup/scripts/build-llama.sh --activate     build and point the profiles at it
    bash setup/scripts/build-llama.sh --ref b10631   build a specific tag
    bash setup/scripts/build-llama.sh --list         which builds exist
    bash setup/scripts/build-llama.sh --use <id>     switch back — this is the rollback
    bash setup/scripts/build-llama.sh --dry-run      say what would happen

1. **`git pull` removes the patch and nothing says so.** The server still
   starts; the answers just turn to `////` once a second slot is used. The
   script keeps it as a **commit on a branch**, so an update is a rebase —
   which either replays it or tells you it cannot.
2. **`llama-server` is a 12 KB executable** that maps `libggml-hip.so` and
   friends out of the same `bin/`. Rebuilding in place overwrites files a
   running process has mapped, and that is a SIGBUS in the server, not an
   error in the build. Every build gets its own directory; the stable path is
   a symlink.
3. **gfx1151 collects regressions.** A new build that is slower or wrong has
   to be undoable in one command, with the old binary still on disk.

By hand, if you insist:

    cd ~/llama.cpp
    git apply "$REPO/setup/patches/hip-integrated-off.patch"

    cmake -S . -B build-rocm-patched -DCMAKE_BUILD_TYPE=Release \
          -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DGGML_HIP_GRAPHS=ON \
          -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON -DBUILD_SHARED_LIBS=ON
    cmake --build build-rocm-patched --config Release -j$(nproc)

**A flag from older guides that now breaks the build:**
`-DGGML_HIP_ROCWMMA_FATTN=ON` no longer exists — rocWMMA FlashAttention was
removed from llama.cpp in b10121 and the build option dropped in b10332. You
do not need `rocwmma-dev` either. Any guide still listing it, including
earlier versions of this one, will fail at configure time.

## Which backend

Build both if you have the disk: the answer is **not** a property of the
machine.

    ROCm     wins on prefill, and wins twice over on decode WITH speculation
    Vulkan   holds up better at full context without speculation

The production profile here is ROCm with MTP and n-gram speculation, because
that combination was measured at roughly half the wall time of Vulkan on the
same work. A model without a speculation path may well prefer Vulkan. This is
per-model and it is written down per profile — `LLAMA_BIN` in
`setup/env/*.env`, with the measurement in the comment above it.

`docs/MODELS.md` has the current answer for the models in this repository.

## Verifying it

    bash setup/preflight.sh          # says whether a patched build is present

    # is the patch still in the source?
    grep -c "gfx1151/ROCm: trusting prop.integrated" \
      ~/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu

    python3 setup/lib/defects.py     # is this build exposed to a known defect?

The last one is the check worth running after every rebuild. It reads the
build stamp and the running server's command line and says which of the known
gfx1151 defects this configuration is guarded against — the dangerous ones
here do not raise, they degrade the output.

---

Previous: [03 · GPU and memory](03-gpu-and-memory.md) ·
Next: [05 · Serving a model](05-serve.md)
