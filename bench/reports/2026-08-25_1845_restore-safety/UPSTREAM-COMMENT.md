# Draft: comment for llama.cpp issues #27579 / #27572

Paste from here — written for #27579 (same HW/model/symptoms), with a
cross-reference to #27572 (root cause analysis).

---

**Additional deterministic trigger on gfx1151/ROCm: `/slots/{id}?action=restore` during another slot's prompt processing**

Hardware: AMD Ryzen AI Max+ 395 (Strix Halo, gfx1151), 128 GB unified, ROCm
backend, build b10577 (54ee5ee). Model: Qwen3.8-27B UD-Q4_K_XL, f16 KV,
`-np 2 --no-kv-unified -ub 2048 -fa on`, with and without
`--spec-type draft-mtp,ngram-mod`.

We hit the corruption described here in production and isolated it with a
six-cell experiment (fresh server per cell, three arithmetic probes as the
detector):

| cell | result |
|---|---|
| restore into idle server (ROCm, spec on/off) | clean |
| restore during pure decode, 2 generations running (ROCm, spec on/off) | clean |
| **restore during a ~14k-token prompt processing (ROCm, spec ON)** | **corrupted** |
| **restore during a ~14k-token prompt processing (ROCm, spec OFF)** | **corrupted** |
| same restore-during-prefill scenario on **Vulkan** | clean |
| two concurrent ~14k prefills, **no restore** (ROCm) | clean (single sample) |

After the dirty cells the server returns degenerate output on every slot —
endless `/` runs, fragments of other contexts (tool names, yaml scraps) —
until a full server restart; `erase` does not heal it (matches #27068's
symptom, but our restore returns HTTP 200).

This is consistent with the `process_ubatch()` H2D-copy race analyzed in
#27572: `llama_state_seq_set_data`'s tensor writes during another slot's
multi-ubatch prompt processing act as a reliable second writer into the
unsynced stream window — which would explain why idle and decode-time
restores stay clean, and why Vulkan is unaffected. Speculative decoding is
not an ingredient.

Reproducer (self-contained, starts its own servers):
https://github.com/loonylabs-dev/strix-halo-inference/blob/main/bench/suites/restore-safety.py
