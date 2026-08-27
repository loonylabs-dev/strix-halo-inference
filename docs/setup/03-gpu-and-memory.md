# 03 · GPU and memory

This is the chapter that decides whether the machine is fast, slow, or
occasionally frozen. It has one reboot in it.

## ROCm

The good news first: ROCm comes from Fedora's own repositories and gfx1151 is
explicitly supported by Fedora's Hardware Compute SIG. No AMD repository, no
DKMS, no Secure Boot work.

    sudo dnf install rocm rocminfo rocm-smi
    sudo dnf install rocm-devel        # only if you build llama.cpp yourself
    sudo usermod -aG video,render $USER
    # log out and back in, then:
    rocminfo | grep -i gfx             # must show gfx1151
    rocm-smi

**Never install AMD's `amdgpu-dkms`.** It collides with the in-tree driver and
breaks ROCm access. If you ever do need something from `repo.radeon.com`, take
it with `--no-dkms`.

## How memory actually works here

There is no separate video memory on Strix Halo. Two mechanisms hand system
RAM to the GPU and only one of them is the one you want:

| | what it is | what this stack uses |
|---|---|---|
| **UMA** | a fixed slice carved out by the BIOS | **no** — set it to the minimum, see [01](01-before-you-start.md) |
| **GTT** | allocated dynamically out of ordinary RAM | **yes**, all of it |

`ttm.pages_limit` is the ceiling on GTT, in 4 KiB pages.

**It is a cap, not a reservation.** Raising it allocates nothing and takes
nothing away at boot. The risk is at runtime and only then: a model that
really claims 110 of 125 GiB leaves the desktop 15.

And here is the property that makes this hardware different from a discrete
GPU, and the reason half of this repository exists:

> **GTT is pinned.** A model that does not fit does not page and does not get
> OOM-killed — it freezes the machine, takes every other process with it, and
> leaves nothing in any log.

That is why `setup/lib/budget.py` refuses a start rather than attempting it,
and why the cap below is a safety feature rather than a limit: below the cap,
too large is an error message. Above it, it is a power cycle.

## Setting the cap

Use the script. It knows the page arithmetic, all three places the value has
to be written, and the `rocminfo` trap where the CPU agent's pool is reported
next to the GPU's:

    bash setup/scripts/gtt.sh                    what is set, what is in use
    bash setup/scripts/gtt.sh --set 108          set the cap, with a diff first
    bash setup/scripts/gtt.sh --set 108 --dry-run
    sudo reboot
    bash setup/scripts/gtt.sh --verify           did it take?

The ladder it climbs, with what each step leaves the host:

| GTT | `ttm.pages_limit` | host keeps | for |
|---|---|---|---|
| 96 GiB | 25165824 | 32 GiB | start here — everything up to a ~120B MoE |
| **108 GiB** | **28311552** | **20 GiB** | plus room for containers alongside |
| 116 GiB | 30408704 | 12 GiB | large models, nothing else running |
| 120 GiB | 31457280 | 8 GiB | edge cases only |

The formula is `pages = GiB × 262144`. **Start low and climb.** If the GTT
pool starves the system you boot into an unusable session, and the way back is
[06 · When it does not work](06-when-it-does-not-work.md).

This machine runs **108**, and how it got there is the argument for taking the
right-hand column seriously: it was raised to 116 for a footprint predicted at
110 GiB, measured at 87.4, and lowered again the same night. A higher cap is
more room **and less protection**.

## Two mistakes almost every guide makes

**1 · Do not set `amdgpu.gttsize`.** It is deprecated and ignored — the kernel
prints "Configuring gttsize via module parameter is deprecated, please use
ttm.pages_limit" at boot. Worse than ineffective: set both so they disagree
and the driver reports "this is unusual", after which ROCm sees *less* memory
than with no parameter at all. There is a ROCm issue with 62.2 GB instead of
120 GB from exactly this. Set only the two TTM values, and set them equal.

**2 · Do not use `amd_iommu=off`.** It disables the NPU and removes DMA
protection over USB4. If you need anything, it is `iommu=pt`.

## On Fedora, use grubby — not /etc/default/grub

Fedora has used the Boot Loader Specification since F30. The classic route
through `/etc/default/grub` plus `grub2-mkconfig` fails **silently** there;
the Fedora wiki says so explicitly. `gtt.sh` does this correctly, but if you
do it by hand:

    sudo grubby --update-kernel=ALL \
      --args="ttm.pages_limit=28311552 ttm.page_pool_size=28311552"
    sudo grubby --info=ALL | grep args

    # after the first successful boot — this step is NOT optional
    cat /proc/cmdline | sed 's/^BOOT_IMAGE=[^ ]* //' | sudo tee /etc/kernel/cmdline

The last line is what makes it survive. `grubby --update-kernel=ALL` only
changes entries that already exist; on a kernel update `kernel-install` takes
the command line from `/proc/cmdline`, and if `/etc/kernel/cmdline` exists it
wins. Without it your parameters quietly disappear at the next kernel you
install.

## After the reboot

    cat /proc/cmdline
    cat /sys/module/ttm/parameters/pages_limit
    sudo dmesg | grep "of GTT memory ready"
    rocminfo | grep -A4 "Pool 1"          # the test that actually counts

And the repository's own view of the same machine:

    bash setup/preflight.sh

    GPU         gfx1151
                AMD RYZEN AI MAX+ 395 w/ Radeon 8060S
    Memory      124.9 GiB usable
                0.5 GiB reserved by the BIOS as UMA (minimum)
    GTT         cap 108 GiB · 108 GiB visible to the driver

If `Pool 1` and the GTT cap disagree with what you set, the parameter did not
take — go back to `/etc/kernel/cmdline` above.

## What the memory has to hold

Once a model is running, four things share the machine, and knowing the shape
of them is what lets you read `budget.py`'s output:

| | |
|---|---|
| **weights** | the GGUF, pinned in GTT |
| **KV cache** | the context. Allocated in full at load, so a 200k window costs its memory whether you use it or not |
| **compute buffers** | roughly constant per model and backend — measured 3.1–4.6 GiB here |
| **RAM prompt cache** | `-cram`, host memory, so an evicted prefix comes back instead of being prefilled again |

    python3 setup/lib/budget.py --profile qwen38

        weights                        17.6 GiB   file
        KV cache                       14.5 GiB   declared
        compute buffers and loader      6.0 GiB   floor
        RAM prompt cache (-cram)       32.0 GiB   profile
        = pinned in GTT                38.1 GiB
        = resident in host RAM         70.1 GiB
        machine has                   124.9 GiB

The KV figure is **measured per model and declared in its profile**, not
derived. On this hardware the architecture arithmetic is four times wrong for
the production model — it is a hybrid, and only about every fourth layer keeps
a full KV cache. A guard computing its own number would refuse the one profile
that is known to fit.

---

Previous: [02 · Linux](02-linux.md) ·
Next: [04 · Building llama.cpp](04-build-llama.md)
