# The size of the GPU agent's global coarse-grained memory pool, in KB.
#
#   rocminfo | awk -f setup/lib/rocm-gpu-pool.awk
#
# This is the number that decides what a model may claim: what ROCm itself
# believes it can allocate, as opposed to what the kernel command line asked
# for and what the amdgpu sysfs files report. All three should agree; when
# they do not, this one is the one the loader will act on.
#
# Why it is a file and not a one-liner
# ------------------------------------
# The one-liner it replaces was
#
#     rocminfo | grep -A4 "COARSE GRAINED" | sed -n 's/.*Size: *\([0-9]*\)(.*/\1/p' | head -1
#
# and it was WRONG in a way that reads as right. rocminfo lists the CPU agent
# first, and the CPU agent also has a pool flagged COARSE GRAINED — sized at
# the whole of system RAM. So the line reported 124.9 GiB and called it the
# GPU's limit, on a machine whose GPU limit was 96 GiB at the time. A check
# that confidently prints a wrong number is worse than no check.
#
# So: walk the agents, remember which one is the GPU, and only then look at
# its pools. `Device Type: GPU` is the discriminator; on this machine agent 1
# is the CPU, agent 2 the GPU and agent 3 a DSP, and none of that is worth
# hard-coding.
#
# Output: one number, the size in KB. Nothing if no GPU agent was found.

/^Agent [0-9]+/            { is_gpu = 0; in_global = 0 }
/^ *Device Type: *GPU/     { is_gpu = 1 }
/^ *Device Type: *(CPU|DSP)/ { is_gpu = 0 }

# A pool's own header line. Only the GLOBAL, COARSE GRAINED one is the pool a
# model allocates out of — FINE GRAINED and KERNARG are different segments and
# on an APU they are sized at the whole of system RAM, which is exactly the
# trap above.
/^ *Segment:/ {
    in_global = (is_gpu && $0 ~ /GLOBAL/ && $0 ~ /COARSE GRAINED/ && $0 !~ /KERNARG/)
    next
}

in_global && /^ *Size:/ {
    # "Size:  121634816(0x7400000) KB"
    if (match($0, /[0-9]+/)) {
        print substr($0, RSTART, RLENGTH)
        exit
    }
}
