# reasoning_effort on the container backend — the vocabulary is three words, and more effort buys less answer

Question (review, 12.09.2026): the gateway offers `halogen-…-low`, `-medium`
and `-high`. Do they reach the model, and do they do anything? The integration
declared them without the measurement every llama profile owes
(`setup/gateway/modes.py`: "a profile declares only what its template reads,
and `check_modes()` refuses one that does not"), and `HALOGEN_MODES` bypassed
that guard by being a Python constant.

- backend: Halogen Flash Server, image
  `ghcr.io/peonist-ai/halogen-flash-server:0.5.6`, `HALOGEN_KV_SLOTS=1`,
  `setup/halogen/serve_api.py` mounted over the image's copy
- through the gateway (`127.0.0.1:8090`), so the single slot is respected
- decoding is **greedy**: `temperature` defaults to 0 and serve_api routes
  that to the greedy path rather than through the sampler. Everything below is
  therefore reproducible, not sampled — and the three repeats per cell came
  back byte-identical in **27 of 27** cells, which is the check on the harness
  rather than a finding.

---

## Part 1 — what the template does with a level

Rendered with the container's own tokenizer and
`/models/tokenizer/chat_template.jinja`, one user message,
`add_generation_prompt=True` (`template_render.py`).

| sent | rendered prompt |
|---|---|
| `reasoning_effort=xhigh` | system block: *"Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives…"* |
| `reasoning_effort=medium` | **no system block at all** |
| `reasoning_effort=low` | system block: *"Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion…"* |
| nothing at all | byte-identical to `xhigh` |
| `high`, `max`, `none`, `minimal` | `TemplateError: Unexpected reasoning effort …. Supported types are xhigh (default), medium, and low.` |
| `enable_thinking=false` | `<think>\n\n</think>` pre-closed in the prompt, no system block |

Three consequences, all now in the code:

1. **`high` is not a level this template has.** `HALOGEN_MODES` declared
   `high:on+high`, `xhigh:on+high`, `max:on+high` — an HTTP 500 the template
   would have raised, rescued only because `serve_api.py`'s `EFFORT_MAP`
   rewrites `high → xhigh` one layer further in. Two translations for one
   word, one of them written down. The modes now send `on+xhigh`, and
   `check_modes()` runs over the pair at gateway start like every profile's.
2. **`xhigh` is the DEFAULT.** Sending no level is the most expensive mode
   this model has. That is why the bare alias spells `enable_thinking: false`
   out instead of staying silent — silence in a tool loop would pick maximum
   thinking by accident.
3. **`none` was a synonym of the bare alias.** Both produced a byte-identical
   upstream body (measured after injection). One behaviour, two names — the
   defect `modes.names()` was written against. `none` is gone; four slugs
   remain, four behaviours.

---

## Part 2 — what the MODEL does with it

Three prompts, three levels, three repeats, `max_tokens=1536`
(`effort_effect.py`). All repeats identical, so one row per cell.

| level | prompt | completion tokens | reasoning chars | **answer chars** | finish | s |
|---|---|---|---|---|---|---|
| low | arith | **153** | 225 | 116 | stop | 3.4–4.5 |
| medium | arith | **182** | 248 | 160 | stop | 4.1–4.9 |
| high | arith | **130** | 176 | 142 | stop | 2.9–4.0 |
| low | logic | 1536 | 4887 | **0** | length | 36.2–37.8 |
| medium | logic | 1536 | 5041 | **0** | length | 36.2–37.3 |
| high | logic | 1536 | 5542 | **0** | length | 41.0–42.3 |
| low | code | 1536 | 5148 | **330** | length | 38.1–39.5 |
| medium | code | 1536 | 5483 | **0** | length | 37.2–38.8 |
| high | code | 1536 | 5720 | **0** | length | 41.3–42.6 |

### The level reaches the model, and the order is not the one the names promise

On the only task that finishes inside the budget, the three levels differ by
40 % — and **`high` is the SHORTEST**: 130 tokens against `low`'s 153 and
`medium`'s 182. `medium`, which renders no instruction at all, is the longest.
"More effort" is not "more thinking" on this model; it is a different
instruction with a different effect, and on a simple task the careful one
converges faster.

### The real risk is not a runaway. It is an empty answer.

Six of the nine cells hit `max_tokens` and **five of them returned no answer at
all** — `content` empty, the entire budget spent inside the thinking block.
`serve_api.py`'s own comment says exactly this ("a budget too small does not
shorten the answer, it removes it"), and the measurement shows how easily it is
reached: 1536 tokens is not a small budget, and a moderately hard question
exceeds it at EVERY level.

Worse, it gets worse with effort. On `code`, `low` still delivered 330
characters of answer; `medium` and `high` delivered nothing. Raising the level
under a fixed budget buys more reasoning and less answer, until there is none.

### Wall-clock at identical token counts

The capped cells all produced exactly 1536 tokens, and `high` still took
~41.7 s against `low`'s ~37.4 s — 12 % slower for the same number of tokens.
Not explained here. Candidates: denser reasoning text (5542 vs 4887 characters
for the same tokens) costing more detokenisation, or contamination (see
conditions). Recorded, not concluded.

---

## Conditions, including one that was violated

- `platform_profile` was not read at either end of this run — this harness is
  not `sweep.py` and does not record it. The figures that carry the
  conclusions are TOKEN COUNTS, which greedy decoding makes independent of how
  fast the machine was; only the seconds column depends on it.
- **Run 1 was contaminated by me.** `bash tests/run.sh` ran twice during the
  `low` and `medium` arms (it read 29.8 s against 19 s idle), and then a
  gateway restart killed the whole `high` arm mid-run with `Connection
  refused`. Run 2 (`run2.log`) re-measured `medium/code` and all of `high`
  with nothing else on the GPU. The token counts from both runs agree where
  they overlap, which is what greedy decoding predicts.
- The nine failed rows in `run1.log` are that restart, not the backend.

## Files

- `template_render.py` — part 1, runs inside the container
- `effort_effect.py` — part 2, runs against the gateway
- `run1.log`, `run2.log` — raw output, including the failed rows
