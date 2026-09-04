> Measured at `platform_profile=performance`, verified unchanged at the end of the run.

| variant | ctx | GTT GiB | depth | cached % | pp t/s | tg prose | tg count | tg copy | draft acc. % |
|---|---|---|---|---|---|---|---|---|---|
| rocm-nospec | 65536 | 22.9 | 620 | 0 | 688.4 | 38.1 | 47.6 | 36.7 | — |
|  |  |  | 9160 | 1 | 925.3 | 35.3 | 35.2 | 34.3 | — |
|  |  |  | 36460 | 24 | 628.0 | 30.8 | 31.6 | 30.7 | — |
| vulkan-nospec | 65536 | 23.4 | 620 | 0 | 688.8 | 55.1 | 54.1 | 54.0 | — |
|  |  |  | 9160 | 1 | 958.2 | 51.5 | 51.3 | 51.3 | — |
|  |  |  | 36460 | 24 | 730.1 | 44.9 | 44.9 | 45.0 | — |
| rocm-ngram | 65536 | 23.0 | 620 | 0 | 680.3 | 39.0 | 223.3 | 105.3 | 9.4/100.0/66.7 |
|  |  |  | 9160 | 1 | 948.9 | 39.4 | 212.2 | 285.6 | 7.8/100.0/100.0 |
|  |  |  | 36460 | 24 | 669.5 | 36.0 | 178.2 | 233.0 | 12.1/100.0/100.0 |
| vulkan-ngram | 65536 | 23.4 | 620 | 0 | 728.5 | 57.0 | 213.0 | 108.9 | 0.0/100.0/83.1 |
|  |  |  | 9160 | 1 | 982.5 | 48.8 | 203.4 | 226.8 | 9.4/100.0/100.0 |
|  |  |  | 36460 | 24 | 748.1 | 41.4 | 172.6 | 112.1 | 7.8/100.0/100.0 |

Numbers that do not mean what the column says:
- rocm-ngram, depth 512, count: decode ranged 43.7-228.3 t/s over 3 runs — the median is not a number to compare against anything
- rocm-ngram, depth 512, copy: decode ranged 49.2-294.7 t/s over 3 runs — the median is not a number to compare against anything
- vulkan-ngram, depth 512, count: decode ranged 56.8-213.2 t/s over 3 runs — the median is not a number to compare against anything
- vulkan-ngram, depth 512, copy: decode ranged 53.6-138.0 t/s over 3 runs — the median is not a number to compare against anything
- vulkan-ngram, depth 32768, count: decode ranged 46.8-173.9 t/s over 3 runs — the median is not a number to compare against anything
- vulkan-ngram, depth 32768, copy: decode ranged 90.1-197.8 t/s over 3 runs — the median is not a number to compare against anything
