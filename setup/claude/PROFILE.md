# Claude Code backend profiles

Three variants, switched via `--settings`. The base `settings.json` deliberately
stays free of backend keys: an `env` block there wins against an `export` in the
shell, so you could no longer override it.

    variant 1  subscription only  claude
    variant 2  subscription + local subagents
                                  claude --settings ~/.claude/profiles/hybrid.json
    variant 3  local only
                                  claude --settings ~/.claude/profiles/local.json

## Preconditions

Variant 3 needs a running llama-server with a matching alias:

    ~/llama.cpp/build-vulkan/bin/llama-server \
      -m "$LLAMA_MODELS/gemma-4-26B_q4_0-it.gguf" --alias gemma26 \
      -ngl 999 -fa on -ub 512 -c 32768 -ctk q8_0 -ctv q8_0 \
      --kv-unified -cram 32768 -ctxcp 64 -cms 4096 \
      --jinja --host 127.0.0.1 --port 8080

llama-server speaks the Anthropic messages API natively, no translator needed.

Variant 2 additionally needs the router from ~/.claude/bin/cc-router.py on port 8082.

## Verification

Type `/status` in Claude Code:

    no "Anthropic base URL" line              -> variant 1, straight to Anthropic
    base URL + login method with your account -> variant 2, the subscription is billed
    base URL + auth token                     -> variant 3, subscription replaced

Check the backend beforehand without Claude Code:

    curl -X POST http://127.0.0.1:8080/v1/messages \
      -H 'Authorization: Bearer dummy' -H 'anthropic-version: 2023-06-01' \
      -H 'content-type: application/json' \
      -d '{"model":"gemma26","max_tokens":1,"messages":[{"role":"user","content":"."}]}'

If the answer starts with {"id":"msg_  -> fine.

## Why ATTRIBUTION_HEADER=0

Claude Code puts a billing block in front of every request. If anything in it
changes between calls, the local prefix cache is dead — and on this machine that
cache is worth a factor of 28 to 90 (see px13-ist-stand.html, section 5).

## Not set, and deliberately so

ANTHROPIC_API_KEY appears in no profile. A key that is set takes precedence over
the subscription sign-in, and in -p mode it is always used — that is the most
common cause of unexpected API billing.
