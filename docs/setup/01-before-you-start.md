# 01 · Before you boot anything

Two things happen before an installer ever runs: the BIOS gets one setting
that matters more than any other, and — if you are keeping Windows — the
existing installation has to be made safe to shrink.

Do both now. The BIOS one is easy to get wrong in a way that costs you a
third of your machine and never announces itself.

## The BIOS setting that decides everything: UMA Frame Buffer

    Advanced → AMD CBS → UMA Frame Buffer Size → 512 MB

**The minimum, not the maximum.** This is the one place where the instinct is
backwards.

On Strix Halo there is no separate video memory. The BIOS can carve out a
fixed slice of system RAM and hand it to the GPU as "VRAM" — that is the UMA
frame buffer, and under Windows it is how you give the GPU memory. On Linux
with llama.cpp it is not: the GPU reaches system memory through **GTT**, which
is allocated dynamically out of ordinary RAM.

So memory parked in UMA is lost to both sides. The GPU does not need it, and
the operating system can no longer see it. A machine with 128 GB and a 64 GB
UMA split reports about 60 GB of usable memory and behaves like a much smaller
one, for no gain at all.

`setup/preflight.sh` reads this back and says so:

    Memory      60.2 GiB usable
                64.0 GiB reserved by the BIOS as UMA  <- give it back
                124.2 GiB fitted, as far as this can be told

If you see that line, go back into the BIOS before doing anything else.

How much the GPU may then take is set later, from Linux, on the kernel command
line — [03 · GPU and memory](03-gpu-and-memory.md).

## The rest of the BIOS

| | |
|---|---|
| **Getting in** | power the device off, hold `F2`, then press power |
| **Boot menu** | `ESC` while switching on |
| **BIOS update** | do it BEFORE installing. ASUS does not ship these through fwupd/LVFS, so it needs Windows — MyASUS or EZ Flash |
| **Secure Boot** | leave it on |

**Secure Boot can stay on, and should.** Fedora's bootloader is signed by
Microsoft, and so are the kernel and every in-tree module. ROCm needs no
out-of-tree module on Fedora — it is userspace on the built-in `amdgpu`
driver. There is nothing to sign.

One caveat with timing: toggling Secure Boot changes the TPM registers and
will trigger a BitLocker recovery prompt. Decide its final state **now**,
before re-enabling BitLocker below.

## If you are keeping Windows

Worth keeping for two reasons that have nothing to do with preference: BIOS
updates need it, and it is a way back if an experiment goes wrong.

### 1 · BitLocker — get the recovery key out first

    # Admin PowerShell
    manage-bde -status
    manage-bde -protectors -get C:      # show the recovery key, and SAVE it

    # suspend it before repartitioning
    manage-bde -protectors -disable C: -RebootCount 0

Check the key at `account.microsoft.com/devices/recoverykey` as well. Print it
or put it on a different drive — **not on the SSD you are about to
repartition.**

### 2 · Turn off fast startup

    powercfg /h off

Without this the Windows partition stays in a hybrid sleep state, and Linux
mounts it read-only or reads it inconsistently. Afterwards **shut down
fully** — not "restart".

This matters again later: models often live on a partition shared with
Windows, and fast startup leaves the volume dirty in a way `ntfs3` refuses.

### 3 · Make room

`Win + X` → Disk Management → right-click `C:` → **Shrink Volume**. With
Windows' own tool, not GParted.

**At least 500 GB for Linux.** That sounds like a lot and is not: the models
alone run past 200 GB, before containers and caches. On a 2 TB drive Windows
still keeps 1.4 TB.

If Windows will not release enough:

    defrag C: /X

then temporarily disable the page file and System Protection, and try again.
**Leave the freed space unpartitioned** — the Fedora installer wants it that
way.

## What you need to hand

* A USB stick, 8 GB or more, that you do not mind erasing.
* The recovery key from step 1, somewhere that is not this machine.
* Wired network or working Wi-Fi for the installer.
* Time for one reboot between chapters 03 and 04.

---

Previous: [Index](README.md) · Next: [02 · Linux](02-linux.md)
