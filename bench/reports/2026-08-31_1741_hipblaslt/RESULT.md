# hipblaslt — one build, the env ROCBLAS_USE_HIPBLASLT varied

model: `@MODELS@/Qwen3.8-27B-UD-Q4_K_XL.gguf`
build: `b10702-11-gc799f1014` (`c799f10147916ad58f00648b5ef0b87425f554c0`)

Screening via llama-bench: no speculation, no gateway, no
saved prefixes. A winner still has to survive the serving
profile (bench/speed.py behind bench/sideserver.py).

- **off**: `ROCBLAS_USE_HIPBLASLT=0`
- **on**: `ROCBLAS_USE_HIPBLASLT=1`

```
                             off            on
tg64 @ d0               8.97 t/s    8.90   -1%
pp2048 @ d0           241.46 t/s  235.51   -2%
tg64 @ d32768           8.19 t/s    8.12   -1%
pp2048 @ d32768       133.41 t/s  128.33   -4%

medians of 2 interleaved rounds; every round ran every arm.
Change is against the FIRST arm. A difference smaller than
the spread between rounds is not a difference — the
per-round values are in the JSON beside this.
```
