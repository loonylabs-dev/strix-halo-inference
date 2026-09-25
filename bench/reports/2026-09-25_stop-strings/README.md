# The gateway's end-token stop strings against Halogen's #84 guard — 25.09.2026

**Question.** An agent turn at 09:16:37 ended after 7,250 reasoning tokens
with no content and no tool call, `finish_reason: stop`, and timings 0 — the
signature of a stop-STRING finish (upstream #106). The only stop strings in
that request were the gateway's own, `<|im_end|>` and `<|endoftext|>`
(gateway.py `CONTAINER_STOPS`). Halogen 0.13.4 keeps an `<|im_end|>` written
inside the think block as literal text (upstream #84's fix). Does the
gateway's stop string then end the turn on that text?

**Method.** `bench/stopstring_ab.py`, straight against production Halogen
(`:8080`, 0.13.8 with the quality sidecar, the stack's `serve_api.py`), no
gateway in between, operator's go. Two arms that differ only in `stop`:
`gateway-stops` (the two end tokens) and `no-stops`. One prompt family — write
a ChatML chat template, reasoning with the special tokens written out — with
a fresh nonce per run, because the server answers an exact repeat with its
first answer. Thinking on, `reasoning_effort: low`, `max_tokens` 4096, no
tools declared, non-streaming. n=5 per arm, ABBA. 09:59-10:01, machine
otherwise idle, `platform_profile` performance. Rows: `runs.jsonl`; console:
`console.txt`.

**Result.**

| arm | runs | empty turn with zeroed timings | end token kept as text in reasoning |
|---|---|---|---|
| gateway-stops | 5 | **5** | 0 |
| no-stops | 5 | 0 | **5** (`end_of_turn_kept` 11-22 per turn) |

Every `gateway-stops` run ends at the same place: right after listing
`` `<|im_start|>` `` as the opener, on the opening backtick of the quoted
`<|im_end|>` — cut exactly where the model wrote the token. 89-193 tokens, 3.5-6.6 s. The
`no-stops` runs carry the token as text and run 692-1160 tokens.

**Verdict.** Confirmed: as stop strings the end tokens undo the guard. They
were removed from the gateway (`container_extras`). Nothing is lost at the
token level: `setup/halogen/serve_api.py` `get_eos_ids` registers both ids
itself, so an end token outside the think block still ends the reply.

**A second, separate finding — not the gateway's.** 3 of the 5 `no-stops`
runs ALSO ended with no content, but with real timings: each reasoning ends
on a quoted `</tool_call>`. That is the guard's documented exception (#84:
`<|im_end|>` directly after `</tool_call>` inside the block ends the turn as
a tool call). With no tools declared there is no call to make, so the turn is
empty. Reachable only by quoting a template in the reasoning; it did not
occur in real traffic here. Candidate upstream note, not filed.

**Weaknesses.** n=5 per arm, one adversarial prompt family, no tools
declared. The mechanism is shown; how often real traffic hits it is the
trace's figure (1 of 1,053 streamed turns 22.-25.09.), not this report's.
`bench/classreplay.py` still sends the old list on purpose — it replays what
the gateway sent on 24.09.
