# qwen36-ub — one build, the flag -ub varied

model: `@MODELS@/Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf`
build: `b10702-11-gc799f1014` (`c799f10147916ad58f00648b5ef0b87425f554c0`)

Screening via llama-bench: no speculation, no gateway, no
saved prefixes. A winner still has to survive the serving
profile (bench/speed.py behind bench/sideserver.py).

- **ub512**: `-ub 512`
- **ub2048**: `-ub 2048`
- **ub4096**: `-ub 4096`

```
                           ub512        ub2048        ub4096

medians of 2 interleaved rounds; every round ran every arm.
Change is against the FIRST arm. A difference smaller than
the spread between rounds is not a difference — the
per-round values are in the JSON beside this.
```
