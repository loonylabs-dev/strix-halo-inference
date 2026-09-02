# Pointing a client at this inference

Two situations, and almost all of this page is the same in both.

| | endpoint | token |
|---|---|---|
| **You run the stack** | `http://127.0.0.1:8090` | none — the local zone needs none |
| **Somebody gave you access** | what the operator told you | personal, from the operator |

Everything from *Which model name to send* onward applies to both. The one
section that does not is *Limits of your access*: it describes what a gateway
does to a GUEST, so if you are the operator, read it as what your guests get.

Throughout this page, `$ENDPOINT` stands for whichever of the two you are —
`http://127.0.0.1:8090` or `https://your-gateway`.

**The operator's own values are not written here, on purpose.** They are read
from the running stack instead, because values frozen into prose go stale —
this page carried a sentence about a model that had been replaced, and claimed
"no vision" for a model that had a projector:

    bash setup/consumer-info.sh              endpoint, models, window
    bash setup/consumer-info.sh --local      the same, for your own machine
    bash setup/consumer-info.sh --markdown   paste-ready, to send to somebody

## Which model name to send

One loaded model (Qwen 3.8 Flash-Next, text-only) serves several modes; the
gateway maps the model NAME to a thinking level. Switching names costs
nothing — no reload, and the prompt cache stays warm across switches
(measured; see docs/MODELS.md).

| Name | Thinking | Use for |
|---|---|---|
| `flashnext-low` | low | **daily default** — best correctness per second |
| `flashnext` | off | mechanical edits, title generation, quick answers |
| `flashnext-medium` | medium | the hardest problems only — it can overthink |
| `flashnext-high` | high | the deepest the template renders (it aliases `high` to `xhigh`) |

(The names above exist while the `flashnext` profile is the one being served.
Which names are live right now is what `/v1/models` answers, and what
`bash setup/consumer-info.sh` prints — a list in a document is a list from the
day it was written. This one has named an already-replaced model twice.)

**Context window:** configure your client a little BELOW the slot size, so a
long turn cannot overrun it. The server reports the real number as `n_ctx` in
`/props`, which is reachable locally but NOT through the tunnel — it is not on
the allow list. So: run `bash setup/consumer-info.sh` if the stack is yours, or
ask the operator if it is not. On the machine this page was written for the
slot holds 204,800 and the client is set to 200,000.

**`/v1/models` lists the thinking variants as separate entries** since
25.08. They are the same loaded model under different names; switching
between them costs no reload and keeps the prompt cache warm.

## First check that it works

Set the placeholder once, so every command on this page can be pasted:

    export ENDPOINT=http://127.0.0.1:8090        # you run the stack
    export ENDPOINT=https://your-gateway         # somebody gave you access

    curl $ENDPOINT/v1/messages \
      -H "Authorization: Bearer $YOUR_TOKEN" \
      -H "anthropic-version: 2023-06-01" \
      -H "content-type: application/json" \
      -d '{
            "model": "flashnext",
            "max_tokens": 64,
            "stream": true,
            "messages": [{"role": "user", "content": "Say only the word test."}]
          }'

Expected answer: an SSE stream. The **first** call can take 100 to 180 seconds
— that is the one-off cold start, see below. A `401` means the token is wrong;
a `404` means you are using a path that is blocked from outside.

A **`403` within a fraction of a second** comes from neither: that is
Cloudflare in front of the tunnel, refusing the client before it ever
reaches the gateway. Measured 25.08.: Python's `urllib` default user agent
is answered that way, `curl` is not. Any client library that sends its own
agent string should set a conventional one — the gateway never answers 403
to an inference path, so a 403 always means "blocked in front of the
house".

Reachable from outside are exactly:

    /v1/messages                Anthropic dialect (Claude Code)
    /v1/messages/count_tokens
    /v1/chat/completions        OpenAI dialect (dsh and most others)
    /v1/models


This guide is for someone running Claude Code on their **own** machine who
wants to use a locally hosted model in addition.

There are several variants. The difference between A, B and C is not the
configuration but **where your traffic runs** — read variant A to the end
before deciding. Variant D is for agents that speak OpenAI rather than
Anthropic.

---

## Variant A · Keep the subscription, add this model

**The router runs at your end, not at the operator's.**

Claude Code knows only *one* base URL. If you point it at the foreign server,
**all of your Anthropic traffic runs over that machine too** — every prompt,
every file content Claude Code sends along. You do not want that, and neither
does the operator.

The right way is a small router on your own machine: it sends requests for the
model `local/…` to the foreign server and everything else unchanged to
Anthropic.

### Setting it up

1. Take `setup/claude/cc-router.py` from this repo and put it in place:

       mkdir -p ~/.claude/bin
       cp cc-router.py ~/.claude/bin/

2. Start the router — `LLAMA_URL` points at the foreign server:

       LLAMA_URL=$ENDPOINT \
       LLAMA_API_KEY=<your token> \
       python3 ~/.claude/bin/cc-router.py

3. Create the profile `~/.claude/profiles/hybrid.json`:

   ```json
   {
     "env": {
       "ANTHROPIC_BASE_URL": "http://127.0.0.1:8090",
       "ANTHROPIC_CUSTOM_MODEL_OPTION": "local/local",
       "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": "Local (hosted elsewhere)",
       "API_TIMEOUT_MS": "1800000",
       "API_FORCE_IDLE_TIMEOUT": "0",
       "CLAUDE_CODE_ATTRIBUTION_HEADER": "0"
     }
   }
   ```

4. Start:

       claude --settings ~/.claude/profiles/hybrid.json

   The model picker now offers "Local" in addition. Everything else keeps
   running over your subscription and is billed that way.

**Important:** do not set `ANTHROPIC_API_KEY` in any profile. A key that is set
takes precedence over the subscription sign-in and is always used in `-p` mode
— the most common cause of unexpected API bills.

---

## Variant B · This model only, no Anthropic

No router needed, Claude Code points straight at the foreign server.

    ~/.claude/profiles/local.json

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "$ENDPOINT",
    "ANTHROPIC_AUTH_TOKEN": "<your token>",
    "ANTHROPIC_MODEL": "local",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "local",
    "API_TIMEOUT_MS": "1800000",
    "API_FORCE_IDLE_TIMEOUT": "0",
    "CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING": "1",
    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "200000"
  }
}
```

    claude --settings ~/.claude/profiles/local.json

> ### Important if you use `claude -p`
>
> **In non-interactive mode `--settings` does not carry.** Measured: the run
> aborts during title generation. Interactively the profile is fine; for `-p`
> set the same values as environment variables and it runs:
>
>     ANTHROPIC_BASE_URL=$ENDPOINT \
>     ANTHROPIC_AUTH_TOKEN=<your token> \
>     ANTHROPIC_MODEL=local \
>     API_TIMEOUT_MS=1800000 \
>     claude -p "Say only the word test."

> ### Important if you are signed in to Claude
>
> **A signed-in Claude Code ignores `ANTHROPIC_AUTH_TOKEN` and sends your
> subscription OAuth token instead.** Measured: instead of the 43-character
> token, `Bearer sk-ant-oat01…` with 108 characters arrived at the gateway —
> the value from `~/.claude/.credentials.json`. Sign-in fails as a result and,
> which weighs more: **your Anthropic access token is transmitted to the
> foreign server.**
>
> The remedy is a separate configuration per backend:
>
>     CLAUDE_CONFIG_DIR=~/.claude-local claude
>
> Claude Code then does not see the stored sign-in and uses the token you set.
> Measured over the tunnel: cold start 97.3 s, second question 1.4 s, tool
> conversation 12.0 s.
>
> Anyone not taking that route should at least be aware of what they transmit.

> **Trap:** `claude -p` may abort with this profile file with
> `[claude-code:unrecognized_model] … "query_source":"generate_session_title"`.
> The same values as environment variables in the call do work. It does not
> happen interactively.

---

## Variant B, scoped to a single project

Claude Code reads `.claude/settings.local.json` from the project directory,
and its `env` block applies only there. That pins ONE project to the local
model while every other project keeps running on the subscription,
untouched:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "$ENDPOINT",
    "ANTHROPIC_AUTH_TOKEN": "<your token>",
    "ANTHROPIC_MODEL": "flashnext-low",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "flashnext",
    "API_TIMEOUT_MS": "1800000",
    "API_FORCE_IDLE_TIMEOUT": "0",
    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "200000"
  }
}
```

`settings.local.json` is machine-local by convention and carries a token —
it must not land in the project's repo.

**Known trap:** a signed-in Claude Code can send its subscription OAuth
token INSTEAD of `ANTHROPIC_AUTH_TOKEN` (measured on the operator machine;
the gateway log then shows a 401 with a 108-character Authorization
header). If that hits, give the project its own config directory —
`CLAUDE_CONFIG_DIR=~/.claude-local claude` — or use variant A, whose
router mixes subscription and local model inside one session by model
name.

## Variant D · An OpenAI-speaking agent (DeepSeek Harness and friends)

The gateway serves both dialects of the same inference, so agents that
speak OpenAI's `/v1/chat/completions` get everything Claude Code gets:
zones, named tokens, prefix cache, disk reload, auto-save, and thinking
modes by model name.

Why this is interesting for a local model: Claude Code arrives with about
28k tokens of system prompt and tool schemas before you type anything, and
26.2k of that is built-in tools you cannot switch off. DeepSeek Harness
(`dsh`) starts at roughly 7.5k, and its tools are plugins you can actually
remove. The window is 200k per slot now, so this is no longer a question of
fitting at all — but every token of head is one that decode has to read
again for every token it writes.

`~/.dsh/settings.yaml`:

```yaml
llm-pi-ai:
  providers:
    localllm:                        # any name you like; it is your label
      displayName: Local LLM
      apiKeyEnv: LOCAL_LLM_TOKEN
      api: openai-completions
      baseURL: $ENDPOINT/v1
      defaultContextWindow: 200000   # slot holds 204800, see below
      models:
        - id: flashnext-low
          name: Qwen Flash-Next (low thinking)
        - id: flashnext
          name: Qwen Flash-Next (no thinking)
        - id: flashnext-medium
          name: Qwen Flash-Next (medium thinking)
```

The provider key and `displayName` are yours to choose — they are labels in
your own client. The token goes in the environment variable named under
`apiKeyEnv`, never into the file. `defaultContextWindow` has to match the server's slot size — the same number
as `CLAUDE_CODE_MAX_CONTEXT_TOKENS` in the other variants.
`bash setup/consumer-info.sh` prints it if the stack is yours; ask the operator
if it is not.

The prefix rules of the last section apply unchanged: the id is formed
from the system prompt and the tool block, so a changed plugin set means
one cold start. `dsh` is a developer preview — expect breaking changes
between versions.

## Variant C · Anthropic via the API or Vertex, plus this model

The case: you use Claude not through a subscription but through the API —
directly or via Google Cloud (Vertex / Agent Platform) or Bedrock. Typical in a
corporate setting.

### C1 · Direct Anthropic API plus this model

Works like variant A: router at your end, `cc-router.py` sends `local/*` to the
foreign server and everything else to `api.anthropic.com`. The only difference
is the sign-in — instead of the subscription sign-in you enter your API key:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8090",
    "ANTHROPIC_API_KEY": "sk-ant-...",
    "ANTHROPIC_CUSTOM_MODEL_OPTION": "local/local",
    "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": "Local (hosted elsewhere)",
    "API_TIMEOUT_MS": "1800000",
    "API_FORCE_IDLE_TIMEOUT": "0",
    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0"
  }
}
```

The router passes the `authorization` header through unchanged; only the
`local/*` requests are diverted.

### C2 · Vertex plus this model — not in the same session

There is a hurdle here that you need to know about. With

    CLAUDE_CODE_USE_VERTEX=1
    CLOUD_ML_REGION=global
    ANTHROPIC_VERTEX_PROJECT_ID=<project>

Claude Code speaks the **Vertex protocol**: different paths
(`/v1/projects/…/models/<model>:streamRawPredict`), GCP sign-in via application
default credentials. `ANTHROPIC_BASE_URL` is not evaluated at all in this mode —
so the route from variant A is not available.

**What does work reliably: two profiles that you switch between.**

    ~/.claude/settings.json         Vertex, as prescribed by the company
    ~/.claude/profiles/local.json   this model (variant B)

    claude                                          # work, via Vertex
    claude --settings ~/.claude/profiles/local.json # this model

Two separate sessions, each with its backend. That is unspectacular, but it
works today and without tinkering — and it keeps the two worlds cleanly apart,
which in a corporate context is the healthier thing anyway.

**What would work in theory but is unchecked here:** Claude Code knows
`ANTHROPIC_VERTEX_BASE_URL`, per the documentation expressly "for custom
endpoints or gateways". A router in front of it could read the model from the
*path* instead of the body, rewrite its own requests to the gateway and pass
everything else through to Vertex. Whether Claude Code's model picker offers a
foreign model at all in this mode I could **not verify** — no Vertex access is
available here. Anyone trying it should expect it not to carry.

### Two things that belong to a corporate context

- **Do not set `DISABLE_PROMPT_CACHING`.** For local inference the prompt cache
  is the entire difference between 2 and 100 seconds.
- **Settle where code is allowed to flow.** If you attach a privately hosted
  model at work, company code leaves the company infrastructure and lands on a
  foreign machine. That is a policy question, not a technical one — but it
  belongs answered before setting this up, not after.

---

## Making sure the cache bites

This is the part that decides between usable and unusable. Without hits
**every** request costs about 100 seconds instead of 2.

### The four mandatory settings

| Setting | Why |
|---|---|
| `CLAUDE_CODE_ATTRIBUTION_HEADER=0` | Without it, a block at the start of the prompt changes between requests. Prefix dead. |
| `API_TIMEOUT_MS=1800000` | The very first call takes ~100 s. With the default timeout it aborts. |
| `API_FORCE_IDLE_TIMEOUT=0` | Claude Code aborts streams that stay silent for 300 s. A cold start is exactly that. |
| `CLAUDE_CODE_MAX_CONTEXT_TOKENS` | Has to match the server's slot size, otherwise a longer session overruns the slot. Ask the operator. |

### What changes your prefix — and therefore triggers a cold start

The prefix is everything before your question: system prompt plus tool schemas,
together about 19,000 tokens. It changes when one of those changes:

- **the working directory** — it appears in two places in the system prompt
- **a `CLAUDE.md`** in the project
- **the tool set** — MCP servers, skills, plugins. That is the largest item:
  ~16,800 of the 19,000 tokens are tool schemas.

Every new project therefore pays about 100 seconds **once**. After that it
stays warm as long as the server runs — across many projects too, which the
server's RAM cache takes care of.

### The one case that hurts permanently

> **Do not start two sessions in the same directory with different MCP or skill
> configurations.**

Two prefixes that share a long start without being equal fight over the same
slot permanently and destroy each other's cache. Measured: 88 % instead of 99 %
hit rate, 14 s instead of 1.6 s per turn — and with tight slots even a full
cold start on **every** turn.

Different directories are entirely unproblematic, any number of them.

The gateway detects this case and writes it into the log:

    WARNING  prefix ce4236074506 shares its head with b2205fae3e1c —
             the two fight over one slot

### Checking that it bites

`/gateway/status` is a LOCAL path — it is not on the remote allow list, because
it would expose prefix names and consumer names to the internet
(docs/SECURITY.md). So this is a command for whoever runs the stack; a guest
asks them for the numbers:

    curl -s http://127.0.0.1:8090/gateway/status | python3 -m json.tool

    "prefixes": [ { "id": "...", "requests": 12,
                    "warm_pct": 91.7, "avg_seconds": 2.1 } ],
    "collisions": []

Healthy looks like: `warm_pct` climbing towards 100, `avg_seconds` falling
towards 1–2 s, and `collisions` staying empty.

On your side a look at the response time is enough: first request in a new
project ~100 s, every further one 1–2 s. If it stays at 100 s the cache is not
biting — then one of the four mandatory settings is wrong.

---

## If you access it with scripts instead of Claude Code

Two things that do not concern Claude Code but very much concern scripts:

**Set a user agent of your own.** Cloudflare rejects Python's default UA
(`Python-urllib/…`) with `HTTP 403, error code 1010`. Any value of your own is
enough:

    urllib.request.Request(url, headers={"User-Agent": "my-tool/1.0", ...})

**Stream.** The first call in a new project takes 100 to 180 seconds.
Cloudflare aborts after 125 seconds of silence — with `stream: true` the
server's keep-alives hold the connection open (every 30 s); without streaming a
cold request runs into a `524`.

**And reckon with the thinking block.** The model produces a `thinking` block
first. With a tight `max_tokens` the answer ends inside it, and `content`
contains no text block at all. Size it generously even for short answers.

## Limits of your access

- **Two concurrent requests per access.** Above that comes `429`. That keeps
  one user from occupying the single GPU alone.
- **Your token is personal.** It carries a name, appears in the operator's log
  and can be revoked individually without affecting anyone else. Do not pass it
  on.
- **The operator's local requests have priority.** When it is busy, you wait —
  but not indefinitely: after 30 seconds in the queue you are served next
  whatever the operator is doing. Measured under two of their Claude Code
  sessions running flat out: 31 seconds.
- **Stream, and you will never be dropped while waiting.** A queued streaming
  request gets a `:\n\n` every 30 seconds, so Cloudflare's 125-second silence
  limit never bites. Without `stream: true` you have no such protection — see
  "If you access it with scripts" above.

## What you should not expect

- **Vision is a property of the SERVED MODEL**, so check `/v1/models`
  (`capabilities`) rather than trusting this list. The model served since
  01.09. (`flashnext`) is text-only — no converted projector exists for it —
  and `/v1/models` reports `capabilities: ["completion"]` (checked
  02.09.2026). This entry has been wrong in both directions now: it said
  "no vision" while a projector was loaded (until 25.08.) and "vision works"
  while a text-only model served (until 02.09.).
- **One active user at a time.** There is one GPU and one model in memory. A
  foreign cold start blocks everyone else for its duration (measured: 98 s).
  That is why the gateway prioritises local requests over remote ones.
- **No replacement for a flagship model.** How good the model IS at your work
  is not something this repo measures — plenty of other people benchmark
  models, and a home-grown battery would age badly and be argued with. What is
  measured here is the STACK: what fits, how fast it prefills and decodes, and
  whether the answers stay correct as the window fills. See
  [MODELS.md](MODELS.md).
