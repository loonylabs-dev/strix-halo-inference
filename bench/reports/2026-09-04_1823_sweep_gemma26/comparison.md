all 7 cells measured

> Measured at `platform_profile=performance`, verified unchanged at the end of the run.

| variant | ctx | GTT GiB | depth | cached % | pp t/s | tg prose | tg count | tg copy | draft acc. % |
|---|---|---|---|---|---|---|---|---|---|
| rocm-nospec | 65536 | 15.9 | 626 | 0 | 797.1 | 42.0 | 41.1 | 40.6 | — |
|  |  |  | 9166 | 0 | 1087.5 | 37.7 | 37.8 | 37.5 | — |
|  |  |  | 36466 | 24 | 700.5 | 34.7 | 34.5 | 34.5 | — |
| vulkan-nospec | 65536 | 16.4 | 626 | 0 | 743.7 | 57.1 | 56.2 | 54.9 | — |
|  |  |  | 9166 | 0 | 1027.9 | 50.9 | 51.1 | 51.2 | — |
|  |  |  | 36466 | 24 | 721.2 | 44.7 | 44.8 | 45.0 | — |
| rocm-ngram | 65536 | 16.0 | 626 | 0 | 769.0 | 40.3 | 40.2 | 42.0 | 9.7/7.8/14.1 |
|  |  |  | 9166 | 0 | 1083.5 | 37.8 | 37.2 | 40.8 | 7.0/7.0/22.3 |
|  |  |  | 36466 | 24 | 694.6 | 35.4 | 34.0 | 36.8 | 15.2/9.3/100.0 |
| vulkan-ngram | 65536 | 16.5 | 626 | 0 | 789.6 | 55.1 | 55.1 | 61.3 | 8.2/13.3/33.7 |
|  |  |  | 9166 | 0 | 1028.0 | 51.2 | 49.7 | 53.5 | 16.9/10.9/29.1 |
|  |  |  | 36466 | 24 | 721.3 | 41.5 | 42.6 | 45.6 | 3.9/7.0/62.0 |
| rocm-assistant | 65536 | 16.4 | 626 | 0 | 756.0 | 86.8 | 77.6 | 78.3 | 82.1/80.9/84.7 |
|  |  |  | 9166 | 0 | 1069.1 | 64.4 | 66.0 | 75.3 | 74.6/76.9/95.1 |
|  |  |  | 36466 | 24 | 689.1 | 47.9 | 47.7 | 56.4 | 76.0/74.6/91.5 |
| vulkan-assistant | 65536 | 17.2 | 626 | 0 | 733.7 | 91.5 | 75.8 | 85.4 | 85.5/73.0/84.7 |
|  |  |  | 9166 | 0 | 997.3 | 65.9 | 81.7 | 74.9 | 73.3/82.6/80.2 |
|  |  |  | 36466 | 24 | 713.3 | 49.8 | 59.8 | 66.6 | 67.0/76.0/89.0 |
| qwen36 | — | 28.4 | 620 | 0 | 638.8 | 42.7 | 277.6 | 282.1 | 85.0/100.0/100.0 |
|  |  |  | 9160 | 1 | 861.0 | 39.0 | 263.6 | 264.9 | 42.4/100.0/100.0 |
|  |  |  | 36460 | 24 | 589.7 | 32.9 | 213.0 | 213.6 | 43.2/100.0/100.0 |

Numbers that do not mean what the column says:
- rocm-nospec, depth 512, copy: only 0.0 % of the answer was copied from the block; the model answered in the thinking channel — this is a decode rate, but not a COPY-HEAVY one. Read it with the count cell, not instead of it.
- rocm-nospec, depth 8192, copy: only 0.0 % of the answer was copied from the block; the model answered in the thinking channel — this is a decode rate, but not a COPY-HEAVY one. Read it with the count cell, not instead of it.
- vulkan-nospec, depth 512, copy: only 0.0 % of the answer was copied from the block; the model answered in the thinking channel — this is a decode rate, but not a COPY-HEAVY one. Read it with the count cell, not instead of it.
- vulkan-nospec, depth 8192, copy: only 0.0 % of the answer was copied from the block; the model answered in the thinking channel — this is a decode rate, but not a COPY-HEAVY one. Read it with the count cell, not instead of it.
- vulkan-nospec, depth 32768, copy: only 0.0 % of the answer was copied from the block; the model answered in the thinking channel — this is a decode rate, but not a COPY-HEAVY one. Read it with the count cell, not instead of it.
- rocm-ngram, depth 512, copy: only 0.0 % of the answer was copied from the block; the model answered in the thinking channel — this is a decode rate, but not a COPY-HEAVY one. Read it with the count cell, not instead of it.
- rocm-ngram, depth 8192, copy: only 0.0 % of the answer was copied from the block; the model answered in the thinking channel — this is a decode rate, but not a COPY-HEAVY one. Read it with the count cell, not instead of it.
- rocm-ngram, depth 32768, copy: only 0.0 % of the answer was copied from the block; the model answered in the thinking channel — this is a decode rate, but not a COPY-HEAVY one. Read it with the count cell, not instead of it.
- vulkan-ngram, depth 512, copy: only 0.0 % of the answer was copied from the block; the model answered in the thinking channel — this is a decode rate, but not a COPY-HEAVY one. Read it with the count cell, not instead of it.
- vulkan-ngram, depth 8192, copy: only 0.0 % of the answer was copied from the block; the model answered in the thinking channel — this is a decode rate, but not a COPY-HEAVY one. Read it with the count cell, not instead of it.
- vulkan-ngram, depth 32768, copy: only 0.0 % of the answer was copied from the block; the model answered in the thinking channel — this is a decode rate, but not a COPY-HEAVY one. Read it with the count cell, not instead of it.
- vulkan-ngram, depth 32768, copy: decode ranged 44.7-93.1 t/s over 3 runs — the median is not a number to compare against anything
- rocm-assistant, depth 512, copy: only 0.0 % of the answer was copied from the block; the model answered in the thinking channel — this is a decode rate, but not a COPY-HEAVY one. Read it with the count cell, not instead of it.
- rocm-assistant, depth 8192, copy: only 0.0 % of the answer was copied from the block; the model answered in the thinking channel — this is a decode rate, but not a COPY-HEAVY one. Read it with the count cell, not instead of it.
- rocm-assistant, depth 32768, copy: only 0.0 % of the answer was copied from the block; the model answered in the thinking channel — this is a decode rate, but not a COPY-HEAVY one. Read it with the count cell, not instead of it.
- vulkan-assistant, depth 512, copy: only 0.0 % of the answer was copied from the block; the model answered in the thinking channel — this is a decode rate, but not a COPY-HEAVY one. Read it with the count cell, not instead of it.
- vulkan-assistant, depth 8192, copy: only 0.0 % of the answer was copied from the block; the model answered in the thinking channel — this is a decode rate, but not a COPY-HEAVY one. Read it with the count cell, not instead of it.
- vulkan-assistant, depth 32768, copy: only 0.0 % of the answer was copied from the block; the model answered in the thinking channel — this is a decode rate, but not a COPY-HEAVY one. Read it with the count cell, not instead of it.
- qwen36, depth 512, count: decode ranged 78.5-283.7 t/s over 3 runs — the median is not a number to compare against anything
- qwen36, depth 512, copy: decode ranged 140.8-283.4 t/s over 3 runs — the median is not a number to compare against anything
- qwen36, depth 32768, copy: decode ranged 138.3-219.5 t/s over 3 runs — the median is not a number to compare against anything
