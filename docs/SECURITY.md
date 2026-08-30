# Security model and what has been checked about it

> **Reporting a vulnerability?** That is [SECURITY.md](../SECURITY.md) in the
> repository root — GitHub finds a `SECURITY.md` in `docs/` too, so without
> that pointer this page would be linked as the "security policy" while
> containing no way to report anything. This page is the MODEL: what the
> gateway does, and what has been measured about it.

State: 24 August 2026. Everything here is measured on the running system, not
derived.

---

## The three zones

The gateway sorts every request into one of three zones. What decides is **the
port first, then the IP**:

| Zone | Recognised by | Token | Allowed paths |
|---|---|---|---|
| `local` | 127.0.0.1 on port 8090 | no | all |
| `lan` | private IP on port 8090 | **yes** | allow list only |
| `remote` | **anything on port 8091** | **yes** | allow list only |

Why the port comes before the IP: behind a tunnel the source IP is not the
client's but `cloudflared`'s. In practice it was `172.19.0.2` — a private
address, which by the IP rule would have passed as `lan`. A separate tunnel
port classifies it correctly as `remote` without having to trust any header.

### The status path used to ignore the port

`/gateway/status` classified by source IP alone — the very rule described one
line above as insufficient. As long as `cloudflared` runs in a container the
tunnel comes from `172.19.0.2`, a private address, and the path answered
correctly with 403. If `cloudflared` ran natively it would come from
`127.0.0.1` — and `/gateway/status` would have been readable from the internet,
with prefix names, consumer names and source IPs.

The zone decision now lives once in `zone(req)`, and both paths call it. Two
versions of the same rule can no longer drift apart. Checked in
`tests/test_gateway.py`, against a tunnel port on `127.0.0.1`.

## The allow list

Remote callers reach exactly:

    /v1/messages
    /v1/messages/count_tokens
    /v1/chat/completions        (since 25.08.2026)
    /v1/models

Everything else gets **404** — even with a valid token. The order is
deliberate: allow list first, then token. A 404 does not reveal whether the
path exists.

`/v1/chat/completions` is the OpenAI dialect of the same inference, for
agents that speak it (DeepSeek Harness and most others). It is on the list
under exactly the same rules as `/v1/messages`: token required, zone
priority, per-consumer limit, and the gateway accounts for its prefix the
same way (`tests/test_gateway.py::TestZoneRemote::test_openai_dialect_is_allowed_but_still_needs_a_token`).

What stays off the list — and must stay off — is **`/completion`**: it takes
a raw prompt, bypasses the chat template entirely, and was the free
inference hole described below. The two are easy to confuse; the difference
is that `/v1/chat/completions` renders through the template and is bounded
by the same rules as any other conversation.

### Why this was necessary — the finding

Before this change the gateway passed **every path except `/v1/messages`
unchecked** to `llama-server`. Measured over the open tunnel:

    /completion            HTTP 200   free inference, token bypassed entirely
    /v1/chat/completions   HTTP 200   same
    /slots                 HTTP 200   complete prompts of every slot
    /props                 HTTP 200   server configuration
    /v1/models, /health    HTTP 200   information disclosure

The worst point is `/slots`. With `LLAMA_SERVER_SLOTS_DEBUG=1` the answer
contains the field `prompt` — the complete rendered prompt. At the time of the
measurement that was **73,676 characters** in slot 0. In multi-user operation
those would be other users' prompts including their source code.

`bench/run.py` set this switch on **every** measurement run until 24 August,
although none of its five suites reads prompts — it is needed only by
`bench/suites/replay.py`. Side effect: during a measurement run
`setup/smoketest.sh` reported an exposure that was no regression at all. Now
only someone who explicitly passes `SLOTS_DEBUG=1` sets it.

### After the fix, measured over the tunnel

    path                    without token   with token
    ---------------------------------------------------
    /slots                     404             404
    /props                     404             404
    /health                    404             404
    /completion                404             404
    /v1/chat/completions       404             404
    /v1/models                 401             200
    /v1/messages               401             200

The same block applies on the LAN. Locally everything stays open — that is the
operator's view.

---

## What is still open

### Done: `LLAMA_SERVER_SLOTS_DEBUG` is off

The variable was needed for the cache investigation and is the reason `/slots`
contained prompts at all. Since the server runs as a user service
(`llama-user@laguna`), nobody sets it any more — the unit does not. Verified:
`/slots` contains no `prompt` field.

Anyone setting it again for debugging should keep in mind that the allow list
is then the only layer protecting foreign prompts.

**And the reason to set it has largely gone away.** It was switched on twice on
30.08.2026 — once for a few minutes, once for one measurement — to find out why
a turn re-reads the previous answer. Both times it came straight off again and
`/slots` was checked for a `prompt` field afterwards. What came out of that
investigation is that the switch was not needed for it at all:
`bench/suites/slot-tail.py` answers the same question from `/slots`
`n_prompt_tokens`, `/apply-template`, `/tokenize` and the `selected slot by LCP
similarity` line, which llama-server prints at INFO level in every ordinary
journal. No debug switch, no restart, and nothing that serves a prompt over
HTTP.

That is worth stating as a rule rather than an anecdote: **before reaching for
`LLAMA_SERVER_SLOTS_DEBUG`, check whether the numbers already in the journal
answer the question.** For prompt-cache work they usually do. The one thing the
switch still adds is the identity of the tokens either side of a mismatch
(`server-context.cpp`, the `old:`/`new:` dump) — and if that is what is
wanted, it is a minutes-long, local-only measurement with the allow list as
the only remaining layer, never a setting that stays on.

### Done: the underscore in the hostname

**Do not put an underscore in a tunnel hostname.** The original name used one.
A wildcard certificate covers it — the TLS layer is happy — but not every
client is: Python's stdlib rejects it with "hostname mismatch", while `curl`
and Node accept it. The failure therefore looks like a client bug rather than
a naming one, and only for some clients.

A hyphen instead, and all three accept it. Measured 25.08.2026.

*(The hostname itself is not in this repository — it is a property of one
machine and lives in `~/.config/llm-stack.env` as `GATEWAY_HOST`. See
`bash setup/consumer-info.sh`.)*

### Cloudflare blocks Python's default user agent

Measured against the new hostname, each with a valid token:

    user agent                       result
    ---------------------------------------------------
    Python-urllib/3.14 (default)     HTTP 403, error code 1010
    no UA set (= default)            HTTP 403, error code 1010
    curl/8.5.0                       HTTP 200
    inference-stack/1.0              HTTP 200

Error 1010 is Cloudflare's browser signature check, not the gateway and not
TLS. **Any user agent of your own is enough.** Anyone running Python tools
through the tunnel — the suites in `bench/`, for instance — has to set one.

For measurements the tunnel is the wrong route anyway: it adds latency and one
more source of error. `bench/` belongs locally against 127.0.0.1.

### Named access instead of one shared token

At first there was a single token for everyone — neither individually
revocable nor attributable in the log. Now every consumer gets their own named
access in `~/.config/cc-gateway-tokens` (mode 600, outside the repo).

Measured:

    martin-mobile                         200
    test-access                           200
    invented token                        401
    after deleting the line + restart:
      test-access                         401
      martin-mobile                       200   (unaffected)

The log names the name, never the secret:

    START  172.19.0.2  remote  who=martin-mobile  prefix=f370acc9d8b7  warm

`PER_TOKEN_MAX` (default 2) caps the concurrent requests per access; above that
comes `429`. Two, because Claude Code sends up to two prompt types in parallel.

**A token still remains something you can pass on.** For an endpoint through
which Claude Code clients send file contents, **Cloudflare Access** belongs in
front — then identity decides. It can be switched on later without touching the
tunnel.

### What the gateway does not do

- **No fairness beyond ageing.** The queue guarantees that nobody waits longer
  than `QUEUE_AGE_AFTER` before being served next — it does not guarantee a
  share of the machine. A busy operator still gets most of it.
- **No rate limiting over time.** `PER_TOKEN_MAX` caps the *concurrent*
  requests per access, not the amount over a period. Whoever has a valid token
  can keep the GPU busy indefinitely — just not with arbitrarily many requests
  at once.
- **No accounting.**
- **No per-access quota for saved prefixes.** The size limit (`AUTO_MAX_GB`,
  default 20) applies to everyone together. Relevant as soon as several foreign
  users are allowed to save.
- **No log of content.** What is logged is the zone, the prefix id, the
  duration — no prompts. That is deliberate.

### Foreign credentials can arrive here

A Claude Code that is signed in to Claude sends its **subscription OAuth token**
(`sk-ant-oat01…`, 108 characters from `~/.claude/.credentials.json`) instead of
the configured `ANTHROPIC_AUTH_TOKEN`. Measured at the gateway.

For the operator that means: **never log authorization headers**, not even
prefixes of them. At that point the gateway logs only which header names were
present and how long the value was — never the value itself.

Consumers should point `CLAUDE_CONFIG_DIR` at a directory of their own; Claude
Code then does not see the stored sign-in. See CONSUMERS.md.

### The router now filters by allow list

`cc-router.py` runs at the **consumer** and separates their subscription
traffic from the local model. It used to strip `authorization` and `x-api-key`
— but not `anthropic-auth-token`, which the gateway itself lists a few lines
above as a possible carrier of the subscription token. A header nobody had
thought of would have carried the credentials to the foreign server.

In the local branch only what is on `LOCAL_ALLOWED` goes through now
(`content-type`, `accept`, `accept-language`, `anthropic-version`,
`user-agent`) plus our own access. An allow list cannot repeat that mistake:
whatever is new drops out. The difference matters, because the damage would
happen at the consumer, not at the operator — who would therefore never notice.
Checked in `tests/test_router.py`.

---

## Checked properties at a glance

| Check | Result |
|---|---|
| `llama-server` (8080) reachable from outside | no |
| gateway status (`/gateway/status`) from outside | 403, local only |
| tunnel without a token | 401 |
| tunnel with a wrong token | 401 |
| tunnel with a valid token, allowed path | 200 |
| tunnel with a valid token, blocked path | 404 |
| foreign hostname at the tunnel | 404 (ingress rule #1) |
| cold-start streaming through the tunnel | 106.8 s, no abort |
| zone classification behind the tunnel | `remote`, despite a private source IP |
| old hostname after the switch | 530, no connector |
| TLS with the hyphenated name | curl, Node and Python accept it |
| LAN without a token, empty bearer, wrong x-api-key | 401 each |
| access revoked individually | 401, the others unaffected |
| `llama-server` and tunnel port on the LAN address | unreachable |
| `/v1/messages/count_tokens` with a valid token | 200 |
| `/gateway/status` through the tunnel port | 403 |
| router forwards `anthropic-auth-token`, `x-api-key`, cookies | no |
| per-access counter after a cancelled caller | goes back down |
| gate slot after a cancellation during the reload | goes back down |
| a queued caller under sustained local load | served after 31 s, not starved |
| a queued streaming caller | gets `:\n\n` every 30 s, never silent |

The set against the running stack runs as `bash setup/smoketest.sh` — return
value 0 when everything reacts as expected. **A skipped section now counts as a
deviation**: if the token was missing, the entire remote zone used to fall away
silently and the script still reported "all checks passed". Whoever wants to
leave a zone out deliberately says so with `--local-only`.

The lines that need no running service are additionally nailed down in `tests/`
and run in under two seconds — see [../tests/README.md](../tests/README.md).
