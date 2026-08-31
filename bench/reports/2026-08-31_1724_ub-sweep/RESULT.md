# ub-sweep — one build, the flag -ub varied

model: `@MODELS@/Qwen3.8-27B-UD-Q4_K_XL.gguf`
build: `b10702-11-gc799f1014` (`c799f10147916ad58f00648b5ef0b87425f554c0`)

Screening via llama-bench: no speculation, no gateway, no
saved prefixes. A winner still has to survive the serving
profile (bench/speed.py behind bench/sideserver.py).

- **ub512**: `-ub 512`
- **ub1024**: `-ub 1024`
- **ub2048**: `-ub 2048`

```
                           ub512        ub1024        ub2048
tg64 @ d0               9.01 t/s    8.97   -0%    8.96   -1%
pp2048 @ d0           251.85 t/s  248.83   -1%  242.06   -4%
tg64 @ d32768           8.27 t/s    8.24   -0%    8.16   -1%
pp2048 @ d32768       161.78 t/s  153.19   -5%  129.27  -20%

medians of 2 interleaved rounds; every round ran every arm.
Change is against the FIRST arm. A difference smaller than
the spread between rounds is not a difference — the
per-round values are in the JSON beside this.
```
