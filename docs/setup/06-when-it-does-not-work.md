# 06 · When it does not work

Three kinds of failure, and they want opposite reactions. The third one is
the reason this chapter is not just a table.

## The machine will not boot

Always the same way back, and it needs no live USB:

1. Hold `Esc` or `Shift` at boot to reach the GRUB menu
2. Highlight the entry, press `e`
3. Go to the `linux` / `linuxefi` line — `Ctrl+A` start, `Ctrl+E` end
4. **Delete or reduce `ttm.pages_limit=…`.** If the desktop hangs rather than
   the boot, also append `systemd.unit=multi-user.target` for a text session
5. `Ctrl+X` boots once. Nothing you did here is permanent

Then, from the running system:

    sudo grubby --update-kernel=ALL \
      --remove-args="ttm.pages_limit ttm.page_pool_size"

An older kernel is under **Advanced options** in GRUB. A Btrfs snapshot rolls
back through `btrfs-assistant` — which is why [02](02-linux.md) sets that up
before touching the GPU stack.

## It boots, but something is wrong

| symptom | cause / fix |
|---|---|
| installer will not boot, `IO_PAGE_FAULT` | add `iommu=pt` to the kernel line |
| **ROCm sees less memory than before you set anything** | `amdgpu.gttsize` and `ttm.pages_limit` contradict each other. `sudo dmesg \| grep "this is unusual"` |
| `rocminfo` shows no gfx1151 | not in group `video`/`render`, or kernel < 6.18.4 |
| suspend wakes up again immediately | kernel 7.1.8 — go back to 6.19.14 |
| colour banding, flicker | kernel 7.1.6 — back to 7.1.5 or 6.19.14 |
| no sound from internal speakers | expected on 6.19.x. Needs kernel ≥ 7.0 plus firmware extracted from the Windows driver |

*The kernel versions in the rows above are the ones MEASURED in August 2026
(the table in [02-linux.md](02-linux.md) has the full verdicts). Fedora keeps
only a couple of kernels available at a time, so a version named here may no
longer be installable — checked 27.08.2026: 6.19.14 was not. Take the shape of
the finding, look for the nearest kernel you can actually get, and note that
your own exclude list hides them:*

    sudo dnf --setopt=excludepkgs= list --showduplicates kernel
| the service starts and stops three times, then stays down | the model volume was not mounted. `findmnt -T <the .gguf path>` |
| the unit refuses to start and prints an arithmetic table | that is the memory guard, and it is right. See below |
| Claude Code shows a login screen despite a base URL | `ANTHROPIC_AUTH_TOKEN` is unset — it must be set even though the value does not matter |
| `400 Extra inputs are not permitted` | `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1` |
| `400` about thinking | `CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING=1` — Claude Code sends the field to unknown model names too |
| the stream dies after five minutes | Claude Code aborts after 300 s of silence. A proxy in front must not buffer SSE pings |

### Did the parameter take?

In this order. The last step is the only one that really counts.

    # 1 · does it reach the kernel at all?
    cat /proc/cmdline | tr ' ' '\n' | grep -E 'ttm|iommu|amdgpu'

    # 2 · did TTM accept it?
    cat /sys/module/ttm/parameters/pages_limit
    awk 'BEGIN{print 28311552 / 262144}'      # back to GiB

    # 3 · what did the driver say at boot?
    sudo dmesg | grep "of GTT memory ready"
    sudo dmesg | grep -i "unknown parameter"   # typo or deprecated
    sudo dmesg | grep -i "this is unusual"     # the GTT/TTM conflict

    # 4 · live
    watch -n1 'cat /sys/class/drm/card*/device/mem_info_gtt_used'

    # 5 · the test that counts: does ROCm see the memory?
    rocminfo | grep -A4 "Pool 1"
    rocm-smi --showmeminfo vram

    # 6 · and llama.cpp itself
    llama-cli --list-devices

If `rocminfo` shows far less than expected it is almost always the
`gttsize`/`pages_limit` conflict. Step 3 finds it.

Or ask the repository, which checks the same things and knows what they should
be:

    bash setup/preflight.sh
    bash setup/check.sh

## The third kind: it answers, and the answer is wrong

This is the one that costs days, and it is why `setup/defects.json` exists.

**On this GPU the dangerous failures do not raise.** They do not crash, they
do not log, and nothing in a normal stack notices. Three of them are known:

| what you see | what it is |
|---|---|
| every answer turns into `////` and stays that way until restart | the gfx1151 HIP race. The unpatched build, or a patched build with more than one slot |
| a coherent answer from **another session** | slot restore poisoning a populated context |
| answers slowly getting worse at depth | not a defect — that is the window filling. Decode reads the KV per token |

Start here:

    python3 setup/lib/defects.py

It reads the running server's command line and the build stamp and tells you
which known defects this configuration is **exposed** to, guarded against, or
unaffected by. It sorts by how the failure shows — silent first, because
listing crashes first would put the harmless half at the top.

    silent        wrong output, no error anywhere. The expensive kind.
    loud          crash, assert or refusal. Annoying, honest.
    slow          correct but slower. A tuning matter, not a hazard.
    unrepeatable  correct but not reproducible — poisons MEASUREMENTS

If you are about to report a bug against llama.cpp or against a model, run
this first. Twice on this machine the answer was neither.

## The failure that is not a failure

    REFUSING TO START qwen38: it needs about 70.1 GiB and it does not fit.
        the host has 44.2 GiB available, 12 must stay free, and ALL 70.1 GiB
        has to be resident somewhere — GTT or anonymous memory

The guard did its job. GTT is pinned: starting anyway would not have failed,
it would have frozen the machine — which happened three times in one day
before this existed.

Correct a wrong estimate with `LLM_MODEL_GIB`, `LLM_HOST_GIB` or
`LLM_KV_KIB_PER_TOKEN`, and every check still runs. `LLM_NO_MEMORY_GUARD=1`
switches it off entirely, which is a different and worse thing.

## Reporting something

The one thing this repository cannot do for itself is know what happens on
your board. `setup/defects.json` describes one machine.

Open an issue with the output of `bash setup/preflight.sh` — it reads only,
needs no root, and prints no tokens or hostnames. For anything about speed or
memory, a measurement beats a description: `python3 bench/speed.py` and
`python3 setup/lib/budget.py --profile <name>`.

---

Previous: [05 · Serving a model](05-serve.md) · [Back to the index](README.md)
