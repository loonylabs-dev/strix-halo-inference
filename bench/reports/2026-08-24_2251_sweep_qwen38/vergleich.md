| Variante | pass | med s→ok | Σs | out-tok | pp t/s | tg t/s | tg rewrite | tg prose |
|---|---|---|---|---|---|---|---|---|
| rocm-medium-spec | 9/9 | 51.7 | 678.8 | 11917 | — | — | 23.4 | 12.4 |
| rocm-low-spec | 9/9 | 47.8 | 581.2 | 8961 | — | — | 22.8 | 9.9 |
| rocm-none-spec | 0/9 | — | 0.0 | 0 | — | — | — | — |
| vulkan-medium-spec | 9/9 | 96.9 | 1301.4 | 11896 | — | — | 14.1 | 6.2 |
| laguna | 7/9 | 9.6 | 405.3 | 6396 | 201.2 | 15.6 | 26.9 | 26.6 |

Fehlgeschlagene Aufgaben:
- rocm-none-spec · impl-cache: request failed: HTTP Error 500: Internal Server Error
- rocm-none-spec · bugfix-intervals: request failed: HTTP Error 500: Internal Server Error
- rocm-none-spec · rewrite-modernize: request failed: HTTP Error 500: Internal Server Error
- rocm-none-spec · json-invoice: request failed: HTTP Error 500: Internal Server Error
- rocm-none-spec · sql-revenue: request failed: HTTP Error 500: Internal Server Error
- rocm-none-spec · regex-log: request failed: HTTP Error 500: Internal Server Error
- rocm-none-spec · longctx-retrieval: request failed: HTTP Error 500: Internal Server Error
- rocm-none-spec · multiturn-edit: request failed: HTTP Error 500: Internal Server Error
- rocm-none-spec · prose-cache: request failed: HTTP Error 500: Internal Server Error
- laguna · longctx-retrieval: hit the max_tokens cap (4096); no 'ANSWER: <number>' line
- laguna · prose-cache: only 0 of the keywords ['prefill', 'decode', 'cache', 'kv'] appear
