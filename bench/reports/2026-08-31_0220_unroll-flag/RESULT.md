# unroll-flag — the ROCm 7 unrolling workaround does nothing on this stack

**31 August 2026, 00:38–02:20.** Qwen3.8-27B-UD-Q4_K_XL, four interleaved
rounds, four depths, both arms in every round.

## The answer

| | reference | unroll | change |
|---|---|---|---|
| pp512 @ d0 | 246.71 t/s | 245.83 t/s | −0.4 % |
| pp512 @ d16384 | 197.77 t/s | 198.06 t/s | +0.1 % |
| pp512 @ d32768 | 160.88 t/s | 160.24 t/s | −0.4 % |
| pp512 @ d65536 | 115.85 t/s | 115.27 t/s | −0.5 % |
| tg128 @ d0 | 8.98 t/s | 8.98 t/s | −0.0 % |
| tg128 @ d16384 | 8.58 t/s | 8.58 t/s | −0.1 % |
| tg128 @ d32768 | 8.24 t/s | 8.23 t/s | −0.1 % |
| tg128 @ d65536 | 7.62 t/s | 7.64 t/s | +0.2 % |

Medians of four rounds. **Nothing moved.**

## What the numbers are worth

The difference between the arms is smaller than the spread between rounds of
the SAME arm:

| | spread within the reference arm | difference between arms |
|---|---|---|
| pp512 @ d0 | 3.1 % | −0.4 % |
| pp512 @ d16384 | 5.4 % | +0.1 % |
| pp512 @ d32768 | 5.1 % | −0.4 % |
| pp512 @ d65536 | 1.2 % | −0.5 % |

So this run can see an effect of roughly 5 % and upwards. It cannot resolve
1 %. It does not need to: llama.cpp#19984 reports **+269 %** at d32768, and
an effect of that size would have been unmissable in the first round.

Round 1 ran while a test suite was competing for the CPU — a mistake, and it
is recorded rather than dropped. It changes nothing: recomputing the medians
without round 1 moves every cell by less than 0.7 %, and the sign of two
cells flips, which is what noise does. `rounds.json` has every value.

## Why it was worth checking anyway

Both builds are commit `c799f1014` with the same two patches and the same
cmake line. The only difference is `-DCMAKE_HIP_FLAGS=-mllvm
--amdgpu-unroll-threshold-local=600`, and it is a REAL difference rather than
a flag that got dropped on the way:

- it appears in `ggml-hip.dir/flags.make` of the unroll build and not of the
  reference (checked, not assumed)
- `libggml-hip.so` differs between the two builds — 74,222,120 vs 74,230,296
  bytes, different hashes
- `libggml-base.so` is bit-identical, which is the control: the flag reaches
  the GPU kernels and nothing else

So the flag changes the generated code and the generated code performs the
same. That is a finding, not a failed experiment.

## Why the issue sees 3.7× and we see nothing

llama.cpp#19984 compares a self-built binary WITH the flag against an
OFFICIAL prebuilt one. Two variables — and the second is much larger than the
first. The official Linux ROCm binaries are built by
`.github/workflows/release.yml` as:

| | official binary | this stack |
|---|---|---|
| GPU targets | 22 architectures, gfx908…gfx1201 | `gfx1151` only |
| `GGML_HIP_MMQ_MFMA` | not set | `ON` |
| `GGML_HIP_GRAPHS` | not set | `ON` |
| `GGML_NATIVE` | `OFF` | default (on) |
| `GGML_BACKEND_DL` | `ON` | not set |

The stack already carries the optimisations the official binary does without.
The most likely reading of the issue is therefore that its 3.7× is mostly the
build CONFIGURATION rather than the unroll flag — and that we were never in
the slow regime it describes.

Not proven here. Proving it would mean building the official configuration
and measuring it, which answers a question about somebody else's binary and
changes nothing about ours. It is written down as the likely explanation, not
as a result.

## What this does NOT say

- Nothing about ROCm 10. This is ROCm 7.1.1 (Fedora packages) on both arms.
- Nothing about other unroll thresholds. 600 is the value the issue names;
  `--amdgpu-unroll-threshold-private` exists too and was not tried.
- Nothing about other models. One model, one quantisation.
- The default value of the threshold was never established, so whether 600
  raises or lowers it here is unknown. The unroll build's `libggml-hip.so` is
  8 KB SMALLER, which would fit a LOWER threshold — that is an observation
  about file size, not a claim about the setting.

## Keeping `--unroll`

The result is negative and the option stays. It costs nothing when unused,
the measurement is repeatable the next time ROCm moves underneath us, and the
family separation it needed also fixed two places where `--list` and
`--prune` were hard-wired to exactly two families. `--with-bench` is useful
on its own: llama-bench had never been built here.
