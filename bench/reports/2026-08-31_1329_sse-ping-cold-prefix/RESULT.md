# sse-ping — the 524 was the gateway's save phase, not the tunnel

Incident: 31.08.2026 12:43, the operator's first streamed request from a
second machine (who=martin-pc2, 26,657 tokens, cold) died with a Cloudflare
524 at 124.9 s; the prewarm marched on and SAVED at 144.3 s.

- model: qwen38 (Qwen 3.8 27B UD-Q4_K_XL), production server, `-np 1`
- build: `b10702-11-gc799f1014` (master-2patches), gateway from this tree
- suite: `bench/suites/sse-ping.py` — one streamed request against one
  endpoint, a fresh cold prefix each run (nonce TOOL at position 0, see
  below), timestamps per received chunk, closed at the first data event
- body: `tools/synthetic.py --tools 40` (~22k tokens), `stream: true`,
  `max_tokens: 16`

```
endpoint                      first byte   pings        first data   largest gap
gateway :8090   (pre-fix)       114.5 s    none           120.9 s      114.5 s
llama-server :8080 direct         0.2 s    30/60/90 s     114.6 s       30.1 s
gateway :18090  (fixed)          30.0 s    30/60/90 s     122.1 s       32.0 s
Cloudflare tunnel (fixed,        30.2 s    30/60/90 s     121.1 s       30.9 s
  production, 13:34 restart)
Cloudflare tunnel, ~40k          30.4 s    8x, every       262.3 s      30.4 s
  tokens (fixed, production)                30 s
```

Cloudflare drops a connection after ~125 s without a byte from the origin.
The first row is the incident reproduced locally — no tunnel involved. The
second row clears llama-server and re-confirms setup/README.md's keep-alive
table on this build. The third row is the fix, measured against a second
gateway instance running this tree with the production environment.

**Cause.** `save_prefix_first` (the 29.08. save-before-serve design) runs the
whole prefix prefill inside the request, before the caller's response exists.
llama-server never sees the request during it, so its SSE pings cannot run;
the gateway's QUEUE_KEEPALIVE covers only the admission queue. The comment
above QUEUE_KEEPALIVE names the exact 125 s window this phase reopened.

**Fix.** The save phase now follows the queue phase's rules: for a streaming
caller the response is opened on the first 30 s slice, `:` per slice, and a
later upstream failure arrives as an SSE error event because the status is
spent. Non-streaming callers stay untouched, as documented. Guarded by
`tests/test_gateway.py::TestSavePhaseLifesign`, seen red (TypeError x3)
against the pre-fix code.

**Instrument finding, kept in the suite.** A nonce in the SYSTEM text does
not make a prompt cold here: the rendered prompt carries the tool section
first, so the divergence lands at ~90 % depth and llama reuses everything
before it — measured as f_keep 0.892 with 2,126 of 22,226 tokens recomputed,
in two runs that were believed cold and were not. The suite plants a nonce
tool at position 0 instead.

**Closed the same day — and the first closing claim was too weak, caught by
the operator.** The first tunnel run's save phase was 114.4 s, UNDER the
~120-125 s window: it proved the edge passes the comments through, not that a
lethal case now survives. The second tunnel run settles it: ~40k tokens, save
phase 253.8 s in the journal, message_start at 262.3 s — more than double any
window and well past the incident's 144 s — 8 pings at a 30 s cadence, HTTP
200, no 524. Together with the incident itself (144 s of pre-fix silence →
524, live at 12:43) both arms sit above the window: unfixed dies, fixed
survives. The gateway journal carries the second, independent witness on both
runs: the new `WAITING … sign of life while the prefix is saved` line at
exactly START+30 s, so the pings came from the fixed phase and not from the
queue path. Still open: the fix is uncommitted on `test/sse-524`.

**And the real thing, 13:53.** The operator repeated the incident from the
same second machine with real Claude Code, after the prefix was removed from
the store and llama-server restarted (RAM cache cleared): the SAME prefix
hash as the 12:43 incident, COLD, save phase 147.7 s — above the window, the
incident's own length — sign of life at START+30 s, DONE at 161.5 s with the
answer delivered. At 12:43 the identical request died at 124.9 s. Same
prefix, same machine, same path; the one variable is the fix.

Raw run output (verbatim suite lines):

```
[gateway-local]      HTTP 200 after 114.5 s; data 120.9 s; pings=0; largest_gap=114.5s
[llama-direct-cold]  HTTP 200 after   0.2 s; ':' at 30.3/60.3/90.3; data 114.6 s; largest_gap=30.1s
[gateway-fixed-cold] HTTP 200 after  30.0 s; ':' at 30.0/60.0/90.0; data 122.1 s; largest_gap=32.0s
[tunnel-fixed-cold]  HTTP 200 after  30.2 s; ':' at 30.2/60.2/90.2; data 121.1 s; largest_gap=30.9s
[tunnel-fixed-huge]  HTTP 200 after  30.4 s; ':' x8 every 30 s;     data 262.3 s; largest_gap=30.4s
```
