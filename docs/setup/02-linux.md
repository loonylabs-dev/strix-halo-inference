# 02 · Linux

One decision in this chapter matters and the rest is procedure. The decision
is the kernel.

## The only hard choice: which kernel

The surprise first — **the Fedora release barely matters, the kernel is
everything.** Fedora rebases aggressively inside a release line, so two
releases can be shipping the same kernel. What the version picks is only which
kernel is on the installer image.

The floor is **6.18.4**. Below it two KFD commits are missing and gfx1151 is
unstable. The ceiling is practical rather than principled, and this is the
part nobody tells you:

| kernel | ROCm / gfx1151 | suspend | internal speakers |
|---|---|---|---|
| 6.19.14-300.fc44 | works | > 80 % sleep score | dead |
| 7.0.x | works | 0.00 % sleep score | possible |
| 7.1.6 | works | — | possible, but colour banding and flicker |
| 7.1.8 | works | wakes itself after 2–6 s | possible |

**There is no kernel that does all of it.** The PX13's speakers need the ACP70
quirks from 7.0, and from 7.0 onward suspend on Strix Halo becomes unreliable.

*(Measured in August 2026. Kernel versions move; the shape of the trade-off is
the durable part, not the numbers.)*

**The recommendation, and it is a judgement rather than a fact:** install
Fedora 44, keep the GA kernel, block kernel updates. You get ROCm and working
suspend and no internal speakers — USB-C, Bluetooth and HDMI audio are
unaffected. Test a newer kernel later, in a snapshot-protected session, and
keep whichever you decide matters more.

That sounds like settling. It is the honest order of work: get the machine
computing first. The audio fix is its own afternoon, with firmware extraction
out of the Windows driver.

## Download and verify

    https://download.fedoraproject.org/pub/fedora/linux/releases/44/Workstation/x86_64/iso/Fedora-Workstation-Live-44-1.7.x86_64.iso
    https://download.fedoraproject.org/pub/fedora/linux/releases/44/Workstation/x86_64/iso/Fedora-Workstation-44-1.7-x86_64-CHECKSUM

On Linux:

    cd ~/Downloads
    curl -O https://fedoraproject.org/fedora.gpg
    for file in *-CHECKSUM; do
      gpgv --keyring ./fedora.gpg --output - "$file" | sha256sum -c --ignore-missing
    done
    # expected: Fedora-Workstation-Live-44-1.7.x86_64.iso: OK

On Windows:

    Get-FileHash .\Fedora-Workstation-Live-44-1.7.x86_64.iso -Algorithm SHA256

Fedora Media Writer verifies during download and saves you both steps.

## Writing the stick

Fedora Media Writer downloads, verifies and writes in one go. Or directly:

    # 1. identify the target — nothing is guessed here
    lsblk -o NAME,SIZE,MODEL,TRAN,MOUNTPOINTS

    # 2. unmount its partitions (do not format)
    sudo umount /dev/sdX?

    # 3. write — sdX is the whole disk, not a partition
    sudo dd if=Fedora-Workstation-Live-44-1.7.x86_64.iso of=/dev/sdX \
      bs=4M status=progress oflag=direct conv=fsync
    sync

Ventoy works but wants its own MOK enrolment with Secure Boot on — needless
complexity for a one-off. Rufus works if you answer **DD image mode**, not ISO
mode. Do not use UNetbootin; it breaks hybrid ISOs.

## Installing

Boot the stick (`ESC` while switching on). If the installer hangs or stays
black, press `e` in the GRUB menu and append to the `linuxefi` line, in this
order:

    iommu=pt                  # first
    amdgpu.cwsr_enable=0      # if you get GPU reset loops
    nomodeset                 # last resort, costs acceleration

**About `iommu=pt`.** A boot failure with `AMD-Vi: Event: logged IO_PAGE_FAULT`
is documented for this model, and `iommu=pt` is the workaround. On the machine
these instructions were written on it was **not** needed — installation and
continuous operation ran with no IOMMU parameter at all and the fault never
appeared. That does not refute the report; it is one firmware on one device.
Measure without it first and set it only against a demonstrated problem.

**Do not use `amd_iommu=off`**, whatever community guides say: it disables the
NPU entirely and removes DMA protection on a laptop with USB4.

### Partitioning

Anaconda has a web interface since Fedora 42. The simple path is **Install
into free space**, which uses exactly the region you freed under Windows. For
manual work: the three-dot menu top right → **Launch storage editor**. Careful
— changes there are written immediately, there is no preview.

| partition | size | type | mount |
|---|---|---|---|
| ESP — **create a new one** | 1 GiB | FAT32 / EFI System | `/boot/efi` |
| boot | 2 GiB | ext4 | `/boot` |
| root | the rest | Btrfs (optionally in LUKS2) | subvolumes `root` → `/`, `home` → `/home` |

**Make your own EFI partition; do not share Windows'.** Fedora wants an ESP of
500 MiB and ASUS/Windows typically creates 100 MB. Two ESPs on one GPT disk
are allowed and the firmware gets its own Fedora entry. The side effect is
worth the trouble on its own: **Windows updates can no longer damage your
bootloader.**

Separate `/home`? With Btrfs, no — use a subvolume. That is Fedora's default
and gives flexible space plus its own snapshot rules. Separate `/boot`? Yes,
2 GiB.

## Immediately afterwards

    # update BEFORE the kernel exclusion further down
    sudo dnf upgrade --refresh

    # faster dnf, and keep more kernels around
    sudo mkdir -p /etc/dnf/libdnf5.conf.d
    printf '[main]\nmax_parallel_downloads=10\ninstallonly_limit=5\n' \
      | sudo tee /etc/dnf/libdnf5.conf.d/80-local.conf

    # RPM Fusion
    sudo dnf install \
      https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm \
      https://mirrors.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm

    # codecs
    sudo dnf swap ffmpeg-free ffmpeg --allowerasing

    # firmware (expect no ASUS BIOS here — SSD, dock, peripherals only)
    sudo fwupdmgr refresh --force && sudo fwupdmgr update

    # the clock disagreement with Windows
    sudo timedatectl set-local-rtc 0 --adjust-system-clock

    # clipboard, needed later by Claude Code
    sudo dnf install wl-clipboard xclip

### Snapshots, before you touch the GPU stack

    sudo dnf install snapper btrfs-assistant python3-dnf-plugin-snapper

`snapper` is the backend, `btrfs-assistant` the interface. This is the way
back from everything in the next chapter, and the next chapter is where things
break.

*(`grub-btrfs` is in older guides and does not exist in Fedora 44. It only
hung snapshots into the GRUB menu — convenience. The way back through the GRUB
kernel line does not depend on it.)*

## Pinning the kernel

Now that it works, nail it down — otherwise the next routine update brings a
kernel with the suspend bug back.

    K=$(uname -r); echo "$K"                       # what is running, e.g.
                                                   # 6.19.10-300.fc44.x86_64

    # make sure THAT kernel stays installed, in case an update replaced it
    sudo dnf install "kernel-${K%.x86_64}"

    # grubby prints the exact path of every entry it knows. Take the one for
    # $K from its output rather than assembling a path by hand — /boot may or
    # may not be part of it, depending on how the machine was partitioned.
    sudo grubby --info=ALL
    sudo grubby --set-default=<the kernel path printed for $K>

**The version is yours, not ours.** Until 27.08.2026 these three lines named
`6.19.14-300.fc44` outright. Checked that day on the machine this guide was
written on: it runs 6.19.10-300.fc44, that is the only kernel installed, and
the configured repositories offer 6.19.10 and 7.1.10 — not 6.19.14. Fedora
keeps only a couple of kernels available at a time, so this says the version is
not obtainable TODAY rather than that it never existed.

Either way a version-pinned line in a setup guide rots, and read as a
requirement it sends somebody looking for a kernel they cannot install. The
table at the top of this chapter is a dated MEASUREMENT and stays as it is.
These three lines are an instruction, and what they should say is: pin the
kernel that works for YOU, which is the one `uname -r` just printed.

**Do not use `dnf versionlock` for kernels** — an open dnf5 bug makes it
ignore the lock on kernel upgrade. `excludepkgs` works.

And this is the correction that costs an afternoon if you miss it. The obvious
glob is too broad:

    # WRONG — this also blocks kernel-headers
    excludepkgs='kernel*'

`kernel-headers` is needed by `glibc-devel`, which is needed by `gcc` — so
every package pulling a build chain fails, `rocm-devel` among them, with
"requires kernel-headers, but none of the providers can be installed". And
`kernel-headers` is not tied to the running kernel at all: it is UAPI headers
for the compiler and may safely be newer.

    sudo dnf config-manager setopt \
      excludepkgs='kernel,kernel-core,kernel-modules*,kernel-devel*,kernel-tools*,kernel-uki*,kernel-debug*'

    # when you do want to update deliberately
    sudo dnf --setopt=excludepkgs= upgrade kernel

`--setopt=excludepkgs=` and not `--disableexcludes=all`: the latter is dnf4
and **dnf5 rejects it outright** — "unknown argument". Fedora 41 and later
ship dnf5, so the line that stood here until 27.08.2026 could not run on any
distribution this guide targets. Which is worse than a typo: the paragraph
above tells you to lock your kernel permanently, and this is the way back out.
Verified both ways on dnf5 5.4.2 — with the override the available kernels are
listed, without it nothing is.

**The same applies to ROCm.** The gfx1151 stack is sensitive to version drift
— `setup/defects.json` in this repository is a list of regressions specific to
this GPU. Take a Btrfs snapshot of a working state and pin ROCm too.

---

Previous: [01 · Before you boot anything](01-before-you-start.md) ·
Next: [03 · GPU and memory](03-gpu-and-memory.md)
