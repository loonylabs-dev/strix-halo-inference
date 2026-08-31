# flashnext-ub — one build, the flag -ub varied

model: `@MODELS@/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`
build: `b10724-2-g0a716b9f2` (`0a716b9f29b23ff3bd6be854c7575597f97e9291`)

Screening via llama-bench: no speculation, no gateway, no
saved prefixes. A winner still has to survive the serving
profile (bench/speed.py behind bench/sideserver.py).

- **ub512**: `-ub 512`
- **ub1024**: `-ub 1024`
- **ub2048**: `-ub 2048`

```
                           ub512        ub1024        ub2048
tg64 @ d0              16.16 t/s   16.79   +4%   15.27   -5%
pp2048 @ d0           265.39 t/s  258.16   -3%  236.22  -11%
tg64 @ d32768          11.17 t/s    9.86  -12%   10.37   -7%
pp2048 @ d32768       154.00 t/s  158.25   +3%  149.92   -3%

medians of 2 interleaved rounds; every round ran every arm.
Change is against the FIRST arm. A difference smaller than
the spread between rounds is not a difference — the
per-round values are in the JSON beside this.
```
