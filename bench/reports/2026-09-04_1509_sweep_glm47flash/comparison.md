> Measured at `platform_profile=performance`, verified unchanged at the end of the run.

| variant | ctx | GTT GiB | depth | cached % | pp t/s | tg prose | tg count | tg copy | draft acc. % |
|---|---|---|---|---|---|---|---|---|---|
| rocm-nospec | 65536 | 34.1 | 611 | 0 | 628.9 | 29.6 | 37.2 | 36.5 | — |
|  |  |  | 9151 | 6 | 420.6 | 25.3 | 25.8 | 25.3 | — |
|  |  |  | 36451 | 24 | 127.8 | 17.3 | 17.0 | 18.7 | — |
| vulkan-nospec | 65536 | 34.4 | 611 | 0 | 545.6 | 45.7 | 45.7 | 44.8 | — |
|  |  |  | 9151 | 6 | 433.0 | 34.3 | 34.1 | 37.0 | — |
|  |  |  | 36451 | 24 | 141.1 | 19.2 | 19.6 | 19.4 | — |
| rocm-ngram | 65536 | 34.1 | 611 | 0 | 619.3 | 29.9 | 263.7 | 97.7 | —/100.0/100.0 |
|  |  |  | 9151 | 6 | 416.4 | 23.5 | 23.5 | 74.2 | 0.0/0.0/100.0 |
|  |  |  | 36451 | 24 | 128.1 | 17.0 | 14.7 | 32.2 | 3.1/0.8/73.5 |
| vulkan-ngram | 65536 | 34.4 | 611 | 0 | 583.9 | 45.1 | 225.2 | 92.2 | 1.6/100.0/81.3 |
|  |  |  | 9151 | 6 | 431.5 | 29.8 | 126.7 | 119.7 | 0.0/100.0/100.0 |
|  |  |  | 36451 | 24 | 139.5 | 18.7 | 54.7 | 25.0 | 5.1/100.0/53.8 |
| rocm-mtp | 65536 | 38.9 | 611 | 0 | 528.3 | 29.7 | 191.6 | 106.3 | 82.2/95.0/97.4 |
|  |  |  | 9151 | 6 | 398.5 | 24.3 | 35.0 | 90.1 | 59.4/52.7/71.7 |
|  |  |  | 36451 | 24 | 123.3 | 15.9 | 51.3 | 38.0 | 78.1/96.6/71.7 |
| vulkan-mtp | 65536 | 39.2 | 611 | 0 | 607.6 | 39.7 | 177.7 | 94.0 | 88.2/97.5/97.3 |
|  |  |  | 9151 | 6 | 429.7 | 25.0 | 98.4 | 69.0 | 29.1/98.3/74.8 |
|  |  |  | 36451 | 24 | 136.0 | 11.2 | 26.7 | 21.8 | 33.7/61.5/56.2 |

Numbers that do not mean what the column says:
- rocm-ngram, depth 512, count: decode ranged 30.2-264.6 t/s over 3 runs — the median is not a number to compare against anything
- rocm-ngram, depth 512, copy: decode ranged 54.7-232.8 t/s over 3 runs — the median is not a number to compare against anything
- rocm-ngram, depth 8192, count: decode ranged 23.4-143.2 t/s over 3 runs — the median is not a number to compare against anything
- rocm-ngram, depth 8192, copy: decode ranged 42.7-136.9 t/s over 3 runs — the median is not a number to compare against anything
- rocm-ngram, depth 32768, count: decode ranged 13.2-58.9 t/s over 3 runs — the median is not a number to compare against anything
- vulkan-ngram, depth 512, count: decode ranged 46.5-233.4 t/s over 3 runs — the median is not a number to compare against anything
- vulkan-ngram, depth 512, copy: decode ranged 65.0-185.3 t/s over 3 runs — the median is not a number to compare against anything
- vulkan-ngram, depth 8192, copy: decode ranged 70.6-120.2 t/s over 3 runs — the median is not a number to compare against anything
- vulkan-ngram, depth 32768, count: decode ranged 16.4-55.2 t/s over 3 runs — the median is not a number to compare against anything
- rocm-mtp, depth 512, count: decode ranged 34.3-192.7 t/s over 3 runs — the median is not a number to compare against anything
- rocm-mtp, depth 512, copy: decode ranged 88.0-226.0 t/s over 3 runs — the median is not a number to compare against anything
- rocm-mtp, depth 8192, count: decode ranged 32.1-52.4 t/s over 3 runs — the median is not a number to compare against anything
- rocm-mtp, depth 32768, count: decode ranged 15.1-51.8 t/s over 3 runs — the median is not a number to compare against anything
- rocm-mtp, depth 32768, copy: decode ranged 24.8-58.2 t/s over 3 runs — the median is not a number to compare against anything
- vulkan-mtp, depth 512, count: decode ranged 37.2-182.3 t/s over 3 runs — the median is not a number to compare against anything
- vulkan-mtp, depth 512, copy: decode ranged 90.1-181.4 t/s over 3 runs — the median is not a number to compare against anything
- vulkan-mtp, depth 8192, copy: decode ranged 62.1-116.4 t/s over 3 runs — the median is not a number to compare against anything
- vulkan-mtp, depth 32768, count: decode ranged 12.0-38.9 t/s over 3 runs — the median is not a number to compare against anything
- vulkan-mtp, depth 32768, copy: decode ranged 21.0-34.4 t/s over 3 runs — the median is not a number to compare against anything
