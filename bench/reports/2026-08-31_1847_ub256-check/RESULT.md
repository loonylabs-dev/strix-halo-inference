# ub256-check — one build, the flag -ub varied

model: `@MODELS@/Qwen3.8-27B-UD-Q4_K_XL.gguf`
build: `b10702-11-gc799f1014` (`c799f10147916ad58f00648b5ef0b87425f554c0`)

Screening via llama-bench: no speculation, no gateway, no
saved prefixes. A winner still has to survive the serving
profile (bench/speed.py behind bench/sideserver.py).

- **ub512**: `-ub 512`
- **ub256**: `-ub 256`

```
                           ub512         ub256
tg64 @ d0               8.96 t/s    9.00   +0%
pp2048 @ d0           248.80 t/s  253.90   +2%
tg64 @ d32768           8.39 t/s    8.14   -3%
pp2048 @ d32768       159.11 t/s  153.80   -3%

medians of 2 interleaved rounds; every round ran every arm.
Change is against the FIRST arm. A difference smaller than
the spread between rounds is not a difference — the
per-round values are in the JSON beside this.
```
