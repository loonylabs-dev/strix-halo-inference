#!/usr/bin/env python3
"""What machine is this? The only implementation.

    python3 setup/lib/hardware.py            a report
    python3 setup/lib/hardware.py --json
    bash setup/preflight.sh                  the same, plus what it means

Why this exists
---------------
Two things in this repo needed to know what hardware they were on and neither
could ask:

  * `setup/defects.json` carries `applies_to: {"gpu": "gfx1151"}` on nine of
    its twelve entries, and `setup/lib/defects.py::applies()` never read the
    field. On any other GPU the registry reported the same nine defects — a
    registry that cries wolf everywhere trains the reader to skip it, which is
    the failure its own docstring warns about.
  * A newcomer had no way to find out whether this repo is for their machine
    before spending an afternoon on it.

Note what this does NOT own: memory. That is setup/lib/budget.py's
`read_machine()`, and it stays there — one reader per fact.

The GPU is identified TWICE, on purpose
---------------------------------------
`rocminfo` is authoritative and gives the gfx name directly. It is also part
of ROCm, which a newcomer has not installed yet — and "is this repo for my
machine" is a question you ask BEFORE installing a GPU stack, not after. So
the PCI id is read as well, straight out of sysfs, which works on a bare
distribution with nothing but a kernel.
"""
import glob, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import budget                                                  # noqa: E402

# PCI device ids that ARE this hardware class. Kept short and sourced: these
# are the ids the amdgpu driver binds for Strix Halo (Ryzen AI Max, gfx1151).
# An id that is not here is not "wrong" — it is unknown, and the difference
# matters, because claiming to recognise hardware one has never seen is how a
# preflight starts lying.
# ONE entry, because one is what has been seen. A second id was written here
# on 27.08. as "Strix Halo, second id" and removed the same hour: nobody had
# verified it, and the paragraph above says in as many words that claiming to
# recognise hardware one has never seen is how a preflight starts lying. It
# took an hour to do it anyway.
#
# To add one: run `lspci -nn | grep -i 1002:` on the machine, take the id from
# the brackets, and say in the comment WHERE it was seen.
KNOWN_GPUS = {
    # verified 27.08.2026 on an ASUS ProArt PX13, Ryzen AI Max+ 395:
    #   c4:00.0 Display controller [0380]: ... Strix Halo ... [1002:1586]
    "1002:1586": ("gfx1151", "Strix Halo (Ryzen AI Max / Max+)"),
}

# What this repo is measured on. Every number in it — window, KV per token,
# cache, GTT cap, the memory budget's own constants — comes from one machine
# of this shape. See setup/README.md.
TARGET_GFX = "gfx1151"
# A "128 GB" machine reports ~124.9 GiB of MemTotal with the BIOS UMA split at
# its minimum. The threshold sits well below that because firmware reservations
# vary — and it is compared against MemTotal PLUS the UMA reservation, see
# machine_ram_gib().
TARGET_RAM_GIB = 110.0

# Above this, the BIOS is holding back enough memory to be worth saying so.
# 0.5 GiB is this machine's minimum setting; a few GiB is a default nobody
# changed; tens of GiB is the Windows-style UMA split, which on this stack is
# pure loss — llama.cpp reaches the GPU through GTT, which comes out of system
# RAM, so memory parked in UMA is memory neither side can use.
UMA_WORTH_MENTIONING_GIB = 4.0


def _pci_ids():
    """(vendor:device) for every DRM card, from sysfs. No ROCm needed."""
    out = []
    for dev in sorted(glob.glob("/sys/class/drm/card*/device")):
        try:
            with open(os.path.join(dev, "vendor")) as fh:
                ven = fh.read().strip()
            with open(os.path.join(dev, "device")) as fh:
                dv = fh.read().strip()
        except OSError:
            continue
        if ven and dv:
            out.append("%s:%s" % (ven.replace("0x", ""), dv.replace("0x", "")))
    return out


def _rocminfo():
    """(gfx name, marketing name) from rocminfo, or (None, None)."""
    try:
        r = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None, None
    if r.returncode != 0:
        return None, None
    gfx = re.search(r"Name:\s+(gfx\w+)", r.stdout)
    mkt = re.search(r"Marketing Name:\s+(.+)", r.stdout)
    return (gfx.group(1) if gfx else None,
            mkt.group(1).strip() if mkt else None)


def gpu():
    """What GPU is in this machine, and how confident we are about it.

    `source` is part of the answer. "pci" means the card is the right silicon
    but ROCm is not installed yet, which is a different situation from "rocm",
    where it is installed and answering — and a preflight that reported them
    the same would tell a newcomer they were ready when they were not.
    """
    gfx, marketing = _rocminfo()
    ids = _pci_ids()
    known = [KNOWN_GPUS[i] for i in ids if i in KNOWN_GPUS]
    if gfx:
        return {"gfx": gfx, "marketing": marketing, "pci": ids,
                "source": "rocm", "recognised": bool(known)}
    if known:
        return {"gfx": known[0][0], "marketing": known[0][1], "pci": ids,
                "source": "pci", "recognised": True}
    return {"gfx": None, "marketing": None, "pci": ids,
            "source": "none", "recognised": False}


def distro():
    info = {"id": None, "name": None, "version": None, "selinux": None}
    try:
        with open("/etc/os-release") as fh:
            for line in fh:
                k, _, v = line.strip().partition("=")
                v = v.strip('"')
                if k == "ID":
                    info["id"] = v
                elif k == "NAME":
                    info["name"] = v
                elif k == "VERSION_ID":
                    info["version"] = v
    except OSError:
        pass
    try:
        r = subprocess.run(["getenforce"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            info["selinux"] = r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def gtt_cap_gib():
    """The ttm.pages_limit actually in force, in GiB, or None.

    From the running kernel's command line rather than from a config file:
    the parameter only takes effect at boot, so what is written down and what
    is in force are different questions.
    """
    try:
        with open("/proc/cmdline") as fh:
            m = re.search(r"ttm\.pages_limit=(\d+)", fh.read())
    except OSError:
        return None
    return int(m.group(1)) * 4096 / 1073741824.0 if m else None


def vram_reserved_gib():
    """What the BIOS has parked as UMA, in GiB, or None.

    Not a detail. `MemTotal` is what is LEFT after the firmware takes its
    share, so on a machine whose BIOS reserves a Windows-style UMA split it
    reads far below the memory that is actually fitted — and a preflight
    reading MemTotal alone would tell a 128 GB owner that this repo is not for
    them. That is a false negative against the one user it IS for.
    """
    for path in sorted(glob.glob("/sys/class/drm/card*/device/mem_info_vram_total")):
        try:
            with open(path) as fh:
                return float(fh.read().split()[0]) / 1073741824.0
        except (OSError, ValueError, IndexError):
            pass
    return None


# "go and read it" and "I looked and do not know" are different, and with a
# default of None they were the same value — so machine_ram_gib(None, 0.5)
# quietly went and measured this machine instead of answering "unknown". A
# sentinel keeps the two apart, which matters because the unknown case is the
# one a test has to be able to express.
_READ = object()


def machine_ram_gib(mem_total=_READ, vram=_READ):
    """How much memory is FITTED, as far as it can be told from here.

    Usable memory plus the UMA reservation. Still an underestimate — firmware
    keeps a little more — which is why TARGET_RAM_GIB sits well below the
    124.9 a 128 GiB machine reports.

    Called with no arguments it reads this machine. Passing None for either
    means "not known", and the answer is then None rather than a guess.
    """
    if mem_total is _READ:
        mem_total = budget.read_machine().mem_total
    if mem_total is None:
        return None
    if vram is _READ:
        vram = vram_reserved_gib()
    return mem_total + (vram or 0.0)


def report():
    m = budget.read_machine()
    g = gpu()
    vram = vram_reserved_gib()
    fitted = machine_ram_gib(m.mem_total, vram)
    return {"gpu": g, "distro": distro(), "gtt_cap_gib": gtt_cap_gib(),
            "mem_total_gib": m.mem_total, "gtt_total_gib": m.gtt_total,
            "vram_reserved_gib": vram, "machine_ram_gib": fitted,
            "uma_is_large": (vram or 0.0) > UMA_WORTH_MENTIONING_GIB,
            "is_target_gpu": g["gfx"] == TARGET_GFX,
            "is_target_ram": (fitted or 0) >= TARGET_RAM_GIB}


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    r = report()
    if "--json" in argv:
        print(json.dumps(r, indent=2))
        return 0
    g = r["gpu"]
    print("  GPU         %s" % (g["gfx"] or "not recognised"))
    if g["marketing"]:
        print("              %s" % g["marketing"])
    print("              detected via %s%s"
          % ({"rocm": "rocminfo", "pci": "the PCI id (ROCm not installed)",
              "none": "nothing — no AMD card found"}[g["source"]],
             ", ".join(["", *g["pci"]]) if g["pci"] else ""))
    print("  Memory      %.1f GiB usable" % (r["mem_total_gib"] or 0))
    if r["vram_reserved_gib"] is not None:
        print("              %.1f GiB reserved by the BIOS as UMA%s"
              % (r["vram_reserved_gib"],
                 "  <- give it back, see below" if r["uma_is_large"] else " (minimum)"))
    if r["machine_ram_gib"]:
        print("              %.1f GiB fitted, as far as this can be told"
              % r["machine_ram_gib"])
    print("  GTT         cap %s · %s visible to the driver"
          % ("%.0f GiB" % r["gtt_cap_gib"] if r["gtt_cap_gib"] else "not set on the kernel command line",
             "%.0f GiB" % r["gtt_total_gib"] if r["gtt_total_gib"] else "unknown"))
    d = r["distro"]
    print("  System      %s %s%s"
          % (d["name"] or "unknown", d["version"] or "",
             " · SELinux %s" % d["selinux"] if d["selinux"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
