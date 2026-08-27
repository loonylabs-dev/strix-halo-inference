# Measurements, ProArt PX13 — 22 August 2026

> A dated measurement record, not living documentation. Nothing here is
> edited when the world moves on. For the current state see docs/MODELS.md
> and setup/README.md.
>
> **Translated from German on 27.08.2026.** Every measured value and every
> line of program output is byte-identical to the original, which is kept out
> of this repository as `measurements.de.md` so that any figure can be checked
> against the text it came from. Only prose and headings were touched.

Raw data. All runs: llama.cpp **b10577** (commit 54ee5ee),
models from this partition, on mains power, idle desktop.
Telemetry time series separately in `telemetry-sweep.csv`.

Scripts and Linux configuration live on the Fedora side under `~/llm-setup/` —
they contain machine paths and systemd units and do not belong here.

---

## 1 · Vulkan, quiet mode

### gemma-4-26B_q4_0-it.gguf
| gemma4 26B.A4B Q4_0            |  13.43 GiB |    25.23 B | Vulkan     | 999 |   1 |           pp512 |       888.92 ± 23.79 |
| gemma4 26B.A4B Q4_0            |  13.43 GiB |    25.23 B | Vulkan     | 999 |   1 |           tg128 |         57.63 ± 1.30 |
    wall clock 14.1 s

### gemma-4-31B_q4_0-it.gguf
| gemma4 31B Q4_0                |  16.42 GiB |    30.70 B | Vulkan     | 999 |   1 |           pp512 |        176.33 ± 8.34 |
| gemma4 31B Q4_0                |  16.42 GiB |    30.70 B | Vulkan     | 999 |   1 |           tg128 |         10.26 ± 0.03 |
    wall clock 52.8 s

### Qwen3.8-27B-UD-Q4_K_XL.gguf
| qwen35 27B Q4_K - Small        |  16.68 GiB |    27.32 B | Vulkan     | 999 |   1 |           pp512 |       172.49 ± 13.15 |
| qwen35 27B Q4_K - Small        |  16.68 GiB |    27.32 B | Vulkan     | 999 |   1 |           tg128 |         10.06 ± 0.01 |
    wall clock 53.3 s

### Laguna-S-2.1-UD-Q4_K_XL-00001-of-00003.gguf
| laguna 118B.A8B Q4_K - Medium  |  68.35 GiB |   117.56 B | Vulkan     | 999 |   1 |           pp512 |        188.71 ± 8.32 |
| laguna 118B.A8B Q4_K - Medium  |  68.35 GiB |   117.56 B | Vulkan     | 999 |   1 |           tg128 |         26.78 ± 0.34 |
    wall clock 43.8 s

### gpt-oss-120b-MXFP4.gguf
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | Vulkan     | 999 |   1 |           pp512 |       393.91 ± 32.88 |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | Vulkan     | 999 |   1 |           tg128 |         45.73 ± 0.06 |
    wall clock 31.1 s


---

## 2 · ROCm, power-saving mode

### gemma-4-26B_q4_0-it.gguf
| gemma4 26B.A4B Q4_0            |  13.43 GiB |    25.23 B | ROCm       | 999 |   1 |           pp512 |       890.34 ± 56.06 |
| gemma4 26B.A4B Q4_0            |  13.43 GiB |    25.23 B | ROCm       | 999 |   1 |           tg128 |         40.40 ± 1.14 |

### gemma-4-31B_q4_0-it.gguf
| gemma4 31B Q4_0                |  16.42 GiB |    30.70 B | ROCm       | 999 |   1 |           pp512 |       210.86 ± 15.32 |
| gemma4 31B Q4_0                |  16.42 GiB |    30.70 B | ROCm       | 999 |   1 |           tg128 |          9.89 ± 0.00 |

### Qwen3.8-27B-UD-Q4_K_XL.gguf
| qwen35 27B Q4_K - Small        |  16.68 GiB |    27.32 B | ROCm       | 999 |   1 |           pp512 |       225.68 ± 12.39 |
| qwen35 27B Q4_K - Small        |  16.68 GiB |    27.32 B | ROCm       | 999 |   1 |           tg128 |          8.48 ± 0.01 |

### Laguna-S-2.1-UD-Q4_K_XL-00001-of-00003.gguf
| laguna 118B.A8B Q4_K - Medium  |  68.35 GiB |   117.56 B | ROCm       | 999 |   1 |           pp512 |        219.77 ± 5.34 |
| laguna 118B.A8B Q4_K - Medium  |  68.35 GiB |   117.56 B | ROCm       | 999 |   1 |           tg128 |         18.59 ± 0.07 |

### gpt-oss-120b-MXFP4.gguf
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | ROCm       | 999 |   1 |           pp512 |       386.09 ± 19.84 |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | ROCm       | 999 |   1 |           tg128 |         37.66 ± 0.26 |

### Laguna WITHOUT HIP_LAUNCH_BLOCKING (the cost of the workaround)
| laguna 118B.A8B Q4_K - Medium  |  68.35 GiB |   117.56 B | ROCm       | 999 |   1 |           pp512 |        222.83 ± 8.17 |
| laguna 118B.A8B Q4_K - Medium  |  68.35 GiB |   117.56 B | ROCm       | 999 |   1 |           tg128 |         19.12 ± 0.08 |
=== FERTIG ===

---

## 3 · Power-profile sweep (Vulkan)
===== PROFIL: balanced (platform_profile=balanced) =====
| gemma4 26B.A4B Q4_0            |  13.43 GiB |    25.23 B | Vulkan     | 999 |   1 |           pp512 |       1036.57 ± 7.24 |
| gemma4 26B.A4B Q4_0            |  13.43 GiB |    25.23 B | Vulkan     | 999 |   1 |           tg128 |         62.40 ± 0.51 |
| gemma4 31B Q4_0                |  16.42 GiB |    30.70 B | Vulkan     | 999 |   1 |           pp512 |        211.67 ± 4.81 |
| gemma4 31B Q4_0                |  16.42 GiB |    30.70 B | Vulkan     | 999 |   1 |           tg128 |         11.47 ± 0.03 |
| qwen35 27B Q4_K - Small        |  16.68 GiB |    27.32 B | Vulkan     | 999 |   1 |           pp512 |        211.93 ± 3.51 |
| qwen35 27B Q4_K - Small        |  16.68 GiB |    27.32 B | Vulkan     | 999 |   1 |           tg128 |         11.35 ± 0.00 |
| laguna 118B.A8B Q4_K - Medium  |  68.35 GiB |   117.56 B | Vulkan     | 999 |   1 |           pp512 |        246.68 ± 4.66 |
| laguna 118B.A8B Q4_K - Medium  |  68.35 GiB |   117.56 B | Vulkan     | 999 |   1 |           tg128 |         27.68 ± 0.04 |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | Vulkan     | 999 |   1 |           pp512 |        470.57 ± 3.14 |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | Vulkan     | 999 |   1 |           tg128 |         51.46 ± 0.40 |

===== PROFIL: performance (platform_profile=performance) =====
| gemma4 26B.A4B Q4_0            |  13.43 GiB |    25.23 B | Vulkan     | 999 |   1 |           pp512 |      1146.31 ± 19.14 |
| gemma4 26B.A4B Q4_0            |  13.43 GiB |    25.23 B | Vulkan     | 999 |   1 |           tg128 |         63.60 ± 0.39 |
| gemma4 31B Q4_0                |  16.42 GiB |    30.70 B | Vulkan     | 999 |   1 |           pp512 |        234.91 ± 5.82 |
| gemma4 31B Q4_0                |  16.42 GiB |    30.70 B | Vulkan     | 999 |   1 |           tg128 |         11.58 ± 0.00 |
| qwen35 27B Q4_K - Small        |  16.68 GiB |    27.32 B | Vulkan     | 999 |   1 |           pp512 |        233.80 ± 6.56 |
| qwen35 27B Q4_K - Small        |  16.68 GiB |    27.32 B | Vulkan     | 999 |   1 |           tg128 |         11.55 ± 0.00 |
| laguna 118B.A8B Q4_K - Medium  |  68.35 GiB |   117.56 B | Vulkan     | 999 |   1 |           pp512 |        274.50 ± 2.26 |
| laguna 118B.A8B Q4_K - Medium  |  68.35 GiB |   117.56 B | Vulkan     | 999 |   1 |           tg128 |         28.12 ± 0.10 |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | Vulkan     | 999 |   1 |           pp512 |        522.26 ± 0.86 |
| gpt-oss 120B MXFP4 MoE         |  59.02 GiB |   116.83 B | Vulkan     | 999 |   1 |           tg128 |         52.54 ± 0.41 |

=== FERTIG ===

---

## 4 · Prefill across context depth

### Gemma 4 26B-A4B (MoE, SWA 25/30, window 1024)
| model                          |       size |     params | backend    | ngl | type_k | type_v |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | --: | --------------: | -------------------: |
| gemma4 26B.A4B Q4_0            |  13.43 GiB |    25.23 B | Vulkan     | 999 |   q8_0 |   q8_0 |   1 |           pp512 |        908.88 ± 1.97 |
| gemma4 26B.A4B Q4_0            |  13.43 GiB |    25.23 B | Vulkan     | 999 |   q8_0 |   q8_0 |   1 |          pp2048 |       817.75 ± 61.14 |
| gemma4 26B.A4B Q4_0            |  13.43 GiB |    25.23 B | Vulkan     | 999 |   q8_0 |   q8_0 |   1 |          pp8192 |        687.33 ± 1.22 |
| gemma4 26B.A4B Q4_0            |  13.43 GiB |    25.23 B | Vulkan     | 999 |   q8_0 |   q8_0 |   1 |         pp32768 |        474.86 ± 0.97 |

### Qwen3.8-27B (dense, no SWA)
| model                          |       size |     params | backend    | ngl | type_k | type_v |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Small        |  16.68 GiB |    27.32 B | Vulkan     | 999 |   q8_0 |   q8_0 |   1 |           pp512 |        187.12 ± 5.10 |
| qwen35 27B Q4_K - Small        |  16.68 GiB |    27.32 B | Vulkan     | 999 |   q8_0 |   q8_0 |   1 |          pp2048 |        166.69 ± 0.30 |
| qwen35 27B Q4_K - Small        |  16.68 GiB |    27.32 B | Vulkan     | 999 |   q8_0 |   q8_0 |   1 |          pp8192 |       83.84 ± 105.99 |
| qwen35 27B Q4_K - Small        |  16.68 GiB |    27.32 B | Vulkan     | 999 |   q8_0 |   q8_0 |   1 |         pp32768 |       120.61 ± 12.18 |

> **Careful:** the `pp8192` value above (83.84 ± 105.99) is unusable — lower than
> at four times the context, with a spread larger than the mean. The cause was a
> download running in parallel. Repeated cleanly with `-r 3` on an idle system:
> **132.41 ± 8.85 t/s**. That is the value that stands.

---

## 5 · Prompt cache

20 calls, byte-identical prefix, changing suffix.
`prompt_n` = newly processed, `cache_n` = reused.

### Gemma 4 26B-A4B, with `-ctxcp 64 -cms 4096`
no.      prompt_n    cache_n    prompt_ms   cache share
------------------------------------------------------------
1            2975          0     4161.315          0.0 %
2               9       2966      286.354         99.7 %
3               9       2966      146.472         99.7 %
4               9       2966      149.699         99.7 %
5               9       2966      147.106         99.7 %
6               9       2966      148.011         99.7 %
7               9       2966      149.828         99.7 %
8               9       2966      149.416         99.7 %
9               9       2966      149.375         99.7 %
10             10       2966      319.471         99.7 %
11              9       2967      154.047         99.7 %
12              9       2967      153.467         99.7 %
13              9       2967      151.799         99.7 %
14              9       2967      150.375         99.7 %
15              9       2967      149.192         99.7 %
16              9       2967      147.998         99.7 %
17              9       2967      151.088         99.7 %
18              9       2967      148.145         99.7 %
19              9       2967      149.445         99.7 %
20             10       2966      155.245         99.7 %
0.17.308.364 I slot      release: id  3 | task 175 | stop processing: n_tokens = 2983, truncated = 0
0.17.383.843 I slot get_availabl: id  3 | task -1 | selected slot by LCP similarity, f_sim_best = 0.997 (> 0.100 thold), f_keep = 0.994
0.17.383.900 I slot launch_slot_: id  3 | task 185 | processing task, is_child = 0
0.17.689.159 I slot print_timing: id  3 | task 185 | prompt eval time =     155.25 ms /    10 tokens (   15.52 ms per token,    64.41 tokens per second)
0.17.689.165 I slot print_timing: id  3 | task 185 |        eval time =     149.82 ms /     8 tokens (   21.40 ms per token,    46.72 tokens per second)
0.17.689.166 I slot print_timing: id  3 | task 185 |       total time =     305.06 ms /    18 tokens
0.17.689.168 I slot print_timing: id  3 | task 185 |    graphs reused =        114
0.17.689.284 I slot      release: id  3 | task 185 | stop processing: n_tokens = 2983, truncated = 0

### Laguna S 2.1, without checkpoint flags
no.      prompt_n    cache_n    prompt_ms   cache share
------------------------------------------------------------
1            3777          0    19663.814          0.0 %
2              10       3767      272.345         99.7 %
3              10       3767      218.002         99.7 %
4              10       3767      218.666         99.7 %
5              10       3767       217.19         99.7 %
6              10       3767      216.812         99.7 %
7              10       3767      216.017         99.7 %
8              10       3767      218.291         99.7 %
9              10       3767      219.263         99.7 %
10             11       3767      303.709         99.7 %
11             10       3768      216.917         99.7 %
12             10       3768      216.219         99.7 %
13             10       3768      219.998         99.7 %
14             10       3768      216.906         99.7 %
15             10       3768      215.591         99.7 %
16             10       3768        212.0         99.7 %
17             10       3768      215.988         99.7 %
18             10       3768      218.291         99.7 %
19             10       3768      215.247         99.7 %
20             11       3767      228.842         99.7 %
0.51.144.112 I slot      release: id  3 | task 182 | stop processing: n_tokens = 3785, truncated = 0
0.51.223.780 I slot get_availabl: id  3 | task -1 | selected slot by LCP similarity, f_sim_best = 0.997 (> 0.100 thold), f_keep = 0.995
0.51.223.841 I slot launch_slot_: id  3 | task 192 | processing task, is_child = 0
0.51.768.403 I slot print_timing: id  3 | task 192 | prompt eval time =     228.84 ms /    11 tokens (   20.80 ms per token,    48.07 tokens per second)
0.51.768.409 I slot print_timing: id  3 | task 192 |        eval time =     315.52 ms /     8 tokens (   45.07 ms per token,    22.19 tokens per second)
0.51.768.411 I slot print_timing: id  3 | task 192 |       total time =     544.37 ms /    19 tokens
0.51.768.413 I slot print_timing: id  3 | task 192 |    graphs reused =        121
0.51.768.555 I slot      release: id  3 | task 192 | stop processing: n_tokens = 3785, truncated = 0

Both steady at 99.7 % from the second call onward. Prefill speed-up:
Gemma a factor of 28, Laguna a factor of 90. No `forcing full prompt re-processing`.

---

## 6 · Speculative decoding with eagle3 — does not work

gpt-oss-120b, same prompt, 200 tokens of output, three runs each.

### Vulkan
#### baseline
  predicted_n=200  39.24 t/s  draft=-/-
  predicted_n=200  38.52 t/s  draft=-/-
  predicted_n=200  38.33 t/s  draft=-/-
#### eagle3
  predicted_n=200  14.15 t/s  draft=86/564
  predicted_n=200  13.98 t/s  draft=86/564
  predicted_n=200  13.92 t/s  draft=86/564

### ROCm, `HIP_LAUNCH_BLOCKING=1`
#### baseline
  predicted_n=200  33.88 t/s  draft=-/-
  predicted_n=200  35.47 t/s  draft=-/-
  predicted_n=200  35.75 t/s  draft=-/-
#### eagle3
  predicted_n=200  16.07 t/s  draft=88/546
  predicted_n=200  16.26 t/s  draft=89/541
  predicted_n=200  15.94 t/s  draft=89/541

Acceptance rate 0.152 (Vulkan) and 0.165 (ROCm) — 39–59 % is the range from which
speculation pays off. Result: a factor of 2.2 to 2.7 **slower**.
Server log: `eagle3 requires ctx_other to be set`.

---

## 7 · Telemetry

`telemetry-sweep.csv` — watts, degrees, GPU load, battery and CPU clock on a
3-second grid across the whole profile sweep, written line by line with `sync`.
Peaks:

    balanced      64,1 W · 67 °C
    performance   84,1 W · 77 °C
    Entlade-Messpunkte über den gesamten Sweep: 0

---

## 8 · Prompt cache and sliding window attention — 23 August 2026

Laguna S 2.1 UD-Q4_K_XL, Vulkan-Build b10577, `-ngl 999 -fa on -ub 512 -b 2048
-c 131072 -np 2 -ctk q8_0 -ctv q8_0 --no-kv-unified -cram 32768`.
Request bodies captured from real Claude Code with a recording proxy
(2.1.241), then replayed without Claude Code straight against `llama-server`.
`neu` = newly processed tokens, `cache` = reused. (The German labels are left
as they appear in the recorded output.)

### 8.1 SWA metadata of every installed model

    Modell        Architektur   sliding_window   Muster
    ---------------------------------------------------------------------
    gemma-4-26B   gemma4        1024             sliding_window_pattern, 5 von 6
    gemma-4-31B   gemma4        1024             sliding_window_pattern, 5 von 6
    Laguna S 2.1  laguna         512             rope.freq_base_swa, rope.dimension_count_swa
    gpt-oss-120b  gpt-oss        128             —
    Qwen3.8-27B   qwen35        keine            —

### 8.2 Bisection of the Claude Code request body, without `--swa-full`

Two runs per variant, the second is measured (only the question changed).

    Variante                       Lauf 1                      Lauf 2
    ------------------------------------------------------------------------------
    V0 unveraendert                neu=19371 c=0     99,8 s    neu=19371 c=    0  100,2 s
    V1 ohne tools                  neu= 3173 c=0     15,6 s    neu= 3173 c=    0   15,7 s
    V2 ohne system-Nachricht       neu=17741 c=0     90,9 s    neu=    7 c=17734    0,3 s
    V3 system-Feld als String      neu=19371 c=0    100,1 s    neu=19371 c=    0  100,1 s
    V4 user-content als String     neu=19227 c=0     99,3 s    neu=19227 c=    0   99,3 s
    V5 ohne metadata               neu=19371 c=0     99,9 s    neu=19371 c=    0   99,8 s
    V6 ohne cache_control          neu=19371 c=0     99,8 s    neu=19371 c=    0  100,2 s

V7 ("only 3 tools") and V8 ("tools as text in the system field") are missing — a
GPU device loss (`vk::DeviceLostError`) ended the server after roughly 25 minutes
of sustained load.

### 8.3 Synthetic counter-check, without `--swa-full`

Same order of magnitude, but the question is at the end of the prompt.

    S1 synthetisch 26.507 Token, kalt   neu=26507  cache=    0   114,7 s
    S2 identisch wiederholt             neu=    1  cache=26506     0,1 s
    S3 geaenderte Frage                 neu=    7  cache=26500     0,2 s
    S4 zurueck auf die erste Frage      neu=    7  cache=26500     0,2 s

### 8.4 Rendered prompt, through `GET /slots` with `LLAMA_SERVER_SLOTS_DEBUG=1`

    Laenge des gerenderten Prompts   80.600 Zeichen
    gemeinsamer Praefix              73.522 Zeichen   =  91,2 %
    Rest danach                       7.078 Zeichen

    <system> bei 7 · "### Tools" bei 6.098 · </available_tools> bei 72.890
    </system> bei 72.908 · <user> bei 72.918 · </user> bei 73.528

    Serverlog: selected slot by LCP similarity, f_sim_best = 0.915, f_keep = 0.915
               old: ...  nur das Wort |  alpha.</user>
               new: ...  nur das Wort |  beta.</user>

### 8.5 Token count of the prompt's components (`/tokenize`)

    system-Feld                                6.081 Zeichen =  1.378 Token
    Werkzeugblock, 24 Schemata                66.764 Bytes   = 16.789 Token
    system-Nachricht NACH der Nutzerfrage      7.028 Zeichen =  1.624 Token
    Zaehler-Nachricht                             49 Zeichen =     18 Token

### 8.6 Verification matrix with `--swa-full`

    A · Einfacher Fall, Frage geaendert, OHNE Proxy
       A1 alpha (fuellt Slot)     neu=19371  cache=    0  ( 0,0 %)  101,1 s
       A2 beta  (geaendert)       neu= 1637  cache=17734  (91,5 %)   10,4 s

    B · Werkzeug-Gespraech, 4 Turns, OHNE Proxy
       B1 Turn 1 (kalt)           neu=19443  cache=    0  ( 0,0 %)  101,5 s
       B2 Turn 2                  neu=  207  cache=19443  (98,9 %)    2,0 s
       B3 Turn 3                  neu=  111  cache=19650  (99,4 %)    1,5 s
       B4 Turn 4                  neu=  112  cache=19761  (99,4 %)    1,5 s

    C · Werkzeug-Gespraech MIT cc-cachefix.py (alte Fassung)
       C1 Turn 1 (kalt)           neu=19438  cache=    0  ( 0,0 %)  101,3 s
       C2 Turn 2                  neu=16634  cache= 3007  (15,3 %)   89,2 s
       C3 Turn 3                  neu=16722  cache= 3026  (15,3 %)   90,0 s
       C4 Turn 4                  neu=16811  cache= 3045  (15,3 %)   90,4 s

    F · Einfacher Fall MIT cc-cachefix2.py
       F1 alpha (fuellt Slot)     neu=19370  cache=    0  ( 0,0 %)  101,4 s
       F2 beta  (geaendert)       neu=   30  cache=19340  (99,8 %)    0,7 s
       F3 gamma (nochmal)         neu=   30  cache=19340  (99,8 %)    0,7 s

    G · Werkzeug-Gespraech MIT cc-cachefix2.py
       G1 Turn 1 (kalt)           neu=19442  cache=    0  ( 0,0 %)  102,1 s
       G2 Turn 2                  neu=  207  cache=19442  (98,9 %)    2,0 s
       G3 Turn 3                  neu=  111  cache=19649  (99,4 %)    1,5 s
       G4 Turn 4                  neu=  112  cache=19760  (99,4 %)    1,5 s

Memory: GTT 73.2 GiB without, 82.8 GiB with `--swa-full` (of 96).
Cold-start prefill unchanged (99.8 s against 101.1 s).

### 8.7 Field test with real Claude Code

Laguna with `--swa-full`, `cc-cachefix2.py` in front, warm server. Wall clock of
the whole `claude -p` invocation including process start:

    Sag nur das Wort alpha.    1,45 s
    Sag nur das Wort beta.     1,27 s
    Sag nur das Wort gamma.    1,27 s

    Werkzeug-Gespraech, zwei Read-Aufrufe, drei Turns:  13,0 s gesamt
       serverseitig 76, 50 und 50 Token Prefill je Turn
       f_sim_best = 0.998

For comparison, from the state document, section 14: ~140 s per simple request,
~300 s per tool-heavy run.

### 8.8 Raw material

The captured request bodies are under `~/llm-setup/cachejagd/bodies/`:
`tool-001` and `tool-002` differ only in the question; `tool-006` through
`tool-009` are the four turns of a tool conversation. The logs of the
measurement series are under `~/llm-setup/cachejagd/logs/`, the detailed
record under `~/llm-setup/cachejagd/notizen/befund.md`.

### 8.9 Four projects warm at once — `-np 4` with `--swa-full`

`-c 131072 -np 4 --no-kv-unified --swa-full` (32,768 tokens per slot).
Four project prefixes built from the real Claude Code body by replacing the
working directory at its two positions in the system prompt
(characters 2,538 and 4,670 of 6,081).

    Phase 1 · nacheinander warmlaufen lassen
       P1   neu=19368  cache=    0  ( 0,0 %)   99,7 s
       P2   neu=19368  cache=    0  ( 0,0 %)  100,5 s
       P3   neu=19368  cache=    0  ( 0,0 %)  100,3 s
       P4   neu=19368  cache=    0  ( 0,0 %)  100,3 s

    Phase 2 · reihum, gleiche Frage, zwei Runden
       R1 P1  neu=1  cache=19367  (100,0 %)  0,3 s
       R1 P2  neu=1  cache=19367  (100,0 %)  0,1 s
       R1 P3  neu=1  cache=19367  (100,0 %)  0,1 s
       R1 P4  neu=1  cache=19367  (100,0 %)  0,1 s
       R2 P1  neu=1  cache=19367  (100,0 %)  0,1 s
       R2 P2  neu=1  cache=19367  (100,0 %)  0,1 s
       R2 P3  neu=1  cache=19367  (100,0 %)  0,1 s
       R2 P4  neu=1  cache=19367  (100,0 %)  0,1 s

    Phase 3 · reihum, geaenderte Frage
       P1  neu=1637  cache=17731  (91,5 %)  10,3 s
       P2  neu=1637  cache=17731  (91,5 %)  10,5 s
       P3  neu=1637  cache=17731  (91,5 %)  10,4 s
       P4  neu=1637  cache=17731  (91,5 %)  10,5 s

    GTT: 82,6 GiB vorher · 82,7 GiB nach dem Warmlaufen · 82,6 GiB am Ende

That makes the finding in section 12 of the state document ("three and four
agents do not work") obsolete — it was the same roll-back problem.

### 8.10 Slot persistence through `--slot-save-path`

    Sichern              19.371 Token, 546.240.884 Bytes      152 ms
    Wiederherstellen                                           72 ms
    identische Anfrage   neu=    1  cache=19.370  (100,0 %)   0,1 s
    geaenderte Frage     neu=19.371 cache=     0  (  0,0 %) 101,1 s

The file holds 28 KiB per token. Laguna has 48 layers, 12 of them with
`head_count = 48` and 36 with `head_count = 72`. The arithmetic for q8_0:

    alle 48 Schichten   102 KiB je Token
    nur die 12 globalen  26 KiB je Token   <- entspricht der Dateigroesse

So only the global layers plus the window are saved, not the full SWA cache —
not even with `--swa-full`. Hence the direct hit on an identical prompt and the
total failure as soon as a roll-back would be needed.

### 8.11 Two projects with real Claude Code

`--swa-full`, `cc-cachefix2.py`, `-np 2`. Wall clock of the whole invocation.

    1. /tmp/cc-jagd, Slot warm            1,4 s
    2. lanewise, erstmalig              107,1 s
    3. lanewise, zweite Frage             1,4 s
    4. zurueck nach /tmp/cc-jagd          1,4 s

Every project pays its cold start exactly once; both slots stay warm.

### 8.12 Concurrent operation — several sessions against one server

`-np 2`, `--swa-full`, `-cram 32768`. Project prefixes as in 8.9.

    Phase 1 · zwei Projekte nacheinander warmlaufen lassen
       P1   neu=19364 cache=    0 (  0,0 %)  100,1 s
       P2   neu=19364 cache=    0 (  0,0 %)  100,5 s

    Phase 2 · beide GLEICHZEITIG (Barrier), gleiche Frage
       P1   neu=    1 cache=19363 (100,0 %)    0,3 s
       P2   neu=    1 cache=19363 (100,0 %)    0,3 s

    Phase 3 · drittes, kaltes Projekt dazu, alle drei GLEICHZEITIG
       P1   neu=    1 cache=19363 (100,0 %)   97,8 s   <- Cache haelt, GPU blockiert
       P2   neu=    1 cache=19363 (100,0 %)   98,1 s
       P3   neu=18823 cache=  541 (  2,8 %)   97,8 s   <- der Kaltstart

    Phase 4 · P1 und P2 danach
       P1   neu=    1 cache=19363 (100,0 %)    0,2 s
       P2   neu=    1 cache=19363 (100,0 %)    0,1 s

    Phase 5 · kurze Fremdanfrage (Nachbildung der Titelgenerierung)
       Titel      neu=  34 cache=    3 (  8,1 %)   0,7 s
       P1 danach  neu=   1 cache=19363 (100,0 %)   0,2 s
       P2 danach  neu=   1 cache=19363 (100,0 %)   0,3 s

**The GPU bottleneck is separate from the cache bottleneck.** In phase 3, P1 and P2
hit at 100 % but take 98 s — they are queued behind P3's cold start in the
same batch loop. The cache is intact; the compute time is shared.

### 8.13 The RAM cache carries more prefixes than there are slots

Three prefixes on two slots, round robin, three rounds:

    R1 P1/P2/P3   je neu=1 cache=19363 (100,0 %)  0,09-0,22 s
    R2 P1/P2/P3   je neu=1 cache=19363 (100,0 %)  0,25 s
    R3 P1/P2/P3   je neu=1 cache=19363 (100,0 %)  0,23 s

Five prefixes on two slots:

    P1..P5        je neu=1 cache=19363 (100,0 %)  0,3 s

Memory growth per additional prefix, measured with `free -m`:

    P4  +430 MiB
    P5  +510 MiB

About 500 MiB per prefix of 19,364 tokens — the same order of magnitude as the
slot file from 8.10 (546 MB). At `-cram 32768` that is roughly **65** such
prefixes. A full-grown session of 65,536 tokens needs about 1.7 GiB
accordingly, and about 19 of those fit.

The log shows the mechanism: after P3's cold start the server picks the same
slot `by LRU` for the next request and still processes only one token — the
state came back out of the RAM cache.

    slot 0  by LRU  -> task 28  97.767 ms / 18.823 Token   (P3, kalt)
    slot 0  by LRU  -> task 26      45 ms /      1 Token   (aus dem RAM-Cache)

### 8.14 Four sessions, three projects — similar prefixes collide

Setup: S1 = project A with the full tool set, S2 = project A with **one tool
fewer** (so a different system prompt, but 88-90 % common prefix), S3 =
project B, S4 = project C. Each session runs four turns of a growing tool
conversation.

    mit -np 2                        Turn 1     Turn 2     Turn 3     Turn 4
    ------------------------------------------------------------------------
    S1 ProjA voll                    100,3 s    102,3 s    102,7 s    103,3 s
                                     ( 0,0 %)   ( 0,0 %)   ( 0,0 %)   ( 0,0 %)
    S2 ProjA -1 Tool                  11,3 s     12,6 s     13,5 s     13,9 s
                                     (90,3 %)   (89,3 %)   (88,8 %)   (88,3 %)
    S3 ProjB                         100,6 s      2,1 s      1,7 s      1,7 s
                                     ( 0,0 %)   (98,9 %)   (99,4 %)   (99,4 %)
    S4 ProjC                          97,9 s      2,1 s      1,7 s      1,7 s
                                     ( 2,8 %)   (98,9 %)   (99,4 %)   (99,4 %)

    mit -np 4                        Turn 1     Turn 2     Turn 3     Turn 4
    ------------------------------------------------------------------------
    S1 ProjA voll                    100,3 s     13,9 s     14,3 s     14,8 s
                                     ( 0,0 %)   (88,6 %)   (88,1 %)   (87,6 %)
    S2 ProjA -1 Tool                  11,3 s     12,6 s     13,6 s     13,9 s
                                     (90,3 %)   (89,3 %)   (88,8 %)   (88,3 %)
    S3 ProjB                         101,0 s      2,1 s      1,7 s      1,6 s
    S4 ProjC                         100,7 s      2,1 s      1,7 s      1,6 s
                                     ( 0,0 %)   (98,9 %)   (99,4 %)   (99,4 %)

**Different projects run perfectly** — 99.4 % from turn 3, 1.6-1.7 s.

**Two similar prefixes collide permanently**, and more slots help only so far.
Slot allocation in the -np 4 run:

    Slot 0:  0 Anfragen   <- blieb LEER
    Slot 1:  5 Anfragen   S3
    Slot 2:  5 Anfragen   S4
    Slot 3: 10 Anfragen   S1 und S2 im Wechsel

    Auswahlart: 17x by LCP similarity, 3x by LRU

A free slot stayed unused while S1 and S2 fought over slot 3.
The reason is in `server-context.cpp`: the LCP branch skips empty slots

    // skip the slot if it does not contains cached tokens
    if (tokens.empty()) { continue; }

and prefers a similar occupied one. And since the LCP path only saves to the
RAM cache at `f_keep < 0.5`, the evicted states are lost in the process.
With `-np 2`, S1 therefore goes fully cold on every turn (100 s); with `-np 4`
it at least keeps the common prefix of 17,399 tokens (14 s instead of 100 s),
but it never reaches the 99 % that dissimilar projects do.

**Practical rule:** two sessions in the same directory with different tool
sets or different MCP configuration cost ~14 s per turn permanently instead of
1.6 s. Different directories, on the other hand, are entirely unproblematic.

---

## 9 · Persistence of the prompt cache — 24 August 2026

Laguna S 2.1, `--swa-full`, `--slot-save-path`. All runs against a real service
restart, not merely against emptied slots.

### 9.1 Whole request saved — carries only partly

    sichern                     22.590 Token, 630.346.916 Bytes,  304 ms
                                = 27,2 KiB je Token
    wiederherstellen                                               59 ms

    danach identische Anfrage   neu=    1  cache=22.589 (100,0 %)   0,1 s
    danach ANHAENGEN (turn2)    neu=   85  cache=22.590 ( 99,6 %)   1,3 s
    danach GEAENDERTE Frage     neu=22.590 cache=     0 (  0,0 %) 110,1 s

    nach echtem Dienstneustart:
      wiederherstellen                                            418 ms
      turn2 (anhaengen)         neu=   85  cache=22.590 ( 99,6 %)  1,2 s

Explaining the file size: Laguna has 48 layers, 12 global and 36 with a sliding
window. A full KV would be 102 KiB per token; the file holds 27.2 — so only the
global layers plus the window. Appending needs the SWA layers only within the
window just before the append point and therefore works; rolling back would
need them further forward and fails.

### 9.2 Only the prefix up to <user> saved — carries completely

Without the hoisted agent block (straight against llama-server):

    Praefix                     20.849 Token
    wiederherstellen                              71-164 ms
    beliebige Frage danach      92,3-92,6 %       10,2-10,5 s

The 1,683 newly processed tokens are the agent-types block, which sits behind
the question.

WITH the hoisted agent block (through the gateway):

    Praefix                     22.496 Token   (99,95 % des Gesamtprompts)
    je Anfrage tatsaechlich neu     11 Token
    sichern                     628 MB, 237-247 ms
    wiederherstellen                    97-185 ms

    nach echtem Dienstneustart, verschiedene Fragen:
      "Sag alpha."              neu= 98  cache=22.489 (99,6 %)  1,39 s
      "Etwas voellig anderes."  neu= 36  cache=22.555 (99,8 %)  0,76 s
      "Und noch was."           neu= 32  cache=22.555 (99,9 %)  0,59 s
      "Vierte Frage."           neu= 34  cache=22.555 (99,8 %)  0,68 s
      Werkzeug-Turn 2           neu=119  cache=22.555 (99,5 %)  1,36 s
      Werkzeug-Turn 3           neu= 85  cache=22.674 (99,6 %)  1,25 s

    systemctl restart insgesamt                    21,5 s

### 9.3 Automatic saving by the gateway

    Kaltstart eines neuen Projekts                109,7 s
    danach automatisch gesichert                    0,4 s
    nach Serverneustart, dieselbe Konfiguration    99,6 %

The 0.4 s come from the slot already being warm: establishing the prefix is a
roll-back on a LIVE slot, and that works perfectly with `--swa-full`.

### 9.4 Reloading from disk on demand

Two projects, server restarted, all slots emptied:

    projA  erste Frage   GEHOLT -> Slot 0,  89 ms   dann 99,6 %  1,54 s
    projB  erste Frage   GEHOLT -> Slot 1,  81 ms   dann 99,6 %  1,74 s
    projA  zweite        99,8 %  1,06 s
    projB  zweite        99,8 %  0,91 s
    projA  dritte        99,8 %  0,94 s

### 9.5 The three cache levels

    Stufe                    Wo               Anzahl   Einwechseln  Neustart
    ---------------------------------------------------------------------------
    Slots (-np 2)            GTT                   2   sofort       nein
    RAM-Cache (-cram 32768)  Arbeitsspeicher     ~54   0,3 s        nein
    Platte                   ~/.cache/…    unbegrenzt   ~0,15 s      JA

628 MB per prefix; 20 projects come to 12.6 GB.
