# b-check — one build, the flag -b varied

model: `@MODELS@/Qwen3.8-27B-UD-Q4_K_XL.gguf`
build: `b10702-11-gc799f1014` (`c799f10147916ad58f00648b5ef0b87425f554c0`)

Screening via llama-bench: no speculation, no gateway, no
saved prefixes. A winner still has to survive the serving
profile (bench/speed.py behind bench/sideserver.py).

- **b2048**: `-b 2048 -ub 512`
- **b512**: `-b 512 -ub 512`

```
                           b2048          b512
tg64 @ d0               8.96 t/s    8.95   -0%
pp2048 @ d0           251.88 t/s  250.04   -1%
tg64 @ d32768           8.23 t/s    8.12   -1%
pp2048 @ d32768       159.85 t/s  152.51   -5%

medians of 2 interleaved rounds; every round ran every arm.
Change is against the FIRST arm. A difference smaller than
the spread between rounds is not a difference — the
per-round values are in the JSON beside this.
```
