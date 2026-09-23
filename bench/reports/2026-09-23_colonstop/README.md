# Agent turns that end on their own announcement — 23.09.2026

**Verdict: the checkpoint's quality sidecar causes it.** Same token prompts,
five real stalls, 10 continuations each:

| stall (last words of the turn) | 0.8.1 + sidecar | 0.12.3 + sidecar | 0.12.3 bare |
|---|---|---|---|
| k=17 `…am Testfall-Ende:` | 0/10 | 0/10 | 0/10 |
| k=23 `…ans Ende des Templates:` | 7/10 | 10/10 | 0/10 |
| k=29 `…am Command selbst:` | 0/10 | 2/10 | 0/10 |
| k=35 `…Korrektur:` | 5/10 | 4/10 | 0/10 |
| k=47 `…mit check-docs:` | 0/10 | 1/10 | 0/10 |
| **turns ended again** | **12/50** | **17/50** | **0/50** |

Every continuation that did not end the turn opened a `<tool_call>`; no arm
produced anything else. 12 against 17 is not a difference at n=50; 0 against
either is. So it is the sidecar, not the release — upstream #89, where the
sidecar raises the rate of `<|im_end|>` as the argmax mid-text by 17x or more
over the bare file, measured there on prose.

## The failure

Claude Code (2.1.267, Windows) through the gateway's Anthropic bridge to
Halogen's `/v1/chat/completions`: 65 tools, thinking on, effort `low`,
temperature 1.0 / top_p 0.95 / top_k 20. Six turns in two sessions of one
morning announced their next step and ended — `finish_reason: stop`, no tool
call — at 120-250k tokens of context. Claude Code reads no tool call as done;
the user had to type "weiter" each time. The raw stream of two of them ends in
`:` and then `stop`; the token counts match the visible text, so nothing was
lost between server and client.

## Method

`bench/colonstop.py` (this directory's rows are its format). For each stall
k, the gateway's text-level trace gives the request that produced assistant
message k. That request is rendered once inside the running container with
the server's own `ChatReq` / `normalize_messages` / `chat_kwargs`, then the
model's own thinking and text are appended and `/v1/completions` is asked to
continue: 120 tokens, the server's sampling defaults, fresh seed each call. An
empty continuation ended the turn again; one opening `<tool_call>` is the
call the turn should have made.

The three server configurations saw byte-identical prompts: the renders were
cached once and reused, and `colonstop.py`'s render was checked against those
bytes for all five k (identical) after a `\r\n` folding bug in its first
version was found and fixed.

How the servers were swapped — production down for each, the operator's
step: a transient unit ran production's `halogenexec` with one change each —
`-e HALOGEN_CK_OVERLAY=none` for the bare arm; the setup tree of tag
`pre-halogen-0.12.3` (its halogenexec, serve_api.py, tool_parse.py) for
0.8.1 — with a trap that restarted production on any exit. Both verified
from the container log before measuring: `overlay: DISABLED` resp.
`halogen-flash-server 0.8.1` + `overlay: 723 tensors`.

## Files

| file | what |
|---|---|
| `sidecar-0.12.3.jsonl` | production as it was, 5 x 10, 12:02-12:13 |
| `bare-0.12.3.jsonl` | same image, `HALOGEN_CK_OVERLAY=none`, 11:49-12:01 |
| `sidecar-0.8.1.jsonl` | 0.8.1, sidecar as shipped then, 12:14-13:56 |
| `fix-arm-sidecar-0.12.3.jsonl` | first run, 5 x 5 x two arms: `control` 11/25 ended, `fix` (`:` + `\n\n`, the llama.cpp #19513 workaround) 0/25 |
| `other-endings-sidecar-0.12.3.jsonl` | a stall ending in `.` (a84e…:152: control 2/5 ended, fix 0/5) and two GENUINE ends (questions to the user). On those, `fix` made the model go on: 6 of 10 appended a paragraph, 1 made an unasked tool call — why the broad "nudge every text-only end" was rejected |

## Weaknesses

- **One conversation's five stalls** carry the verdict (plus one from a
  second session in the side file). The bare checkpoint was not tested on the
  `.` stall.
- **Continuations, not whole turns.** The model resumes after its own text;
  whether the bare checkpoint also stalls less often in the first place is
  implied, not measured.
- **0.8.1 had no prompt-cache hit on `/v1/completions`**: every call
  re-prefilled 123-128k tokens (~123 s). The outcome is unaffected (same
  tokens, same sampling); the wall time is why that run took 1 h 42 min of
  production.
- **Decode speed** looked the same (median 2.99 s bare vs 3.08 s sidecar for
  120 tokens, warm) — short continuations only, not a decode benchmark.
- **The quality cost is upstream's figure** (~3.8 % perplexity on prose,
  #89), not measured here.
- **`platform_profile` was not recorded** at either end of these runs.
