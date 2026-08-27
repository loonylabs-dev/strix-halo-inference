#!/usr/bin/env python3
"""cc-cachefix2 — proxy between Claude Code and llama-server.

Why it is needed
----------------
Every model in the house except Qwen3.8-27B uses sliding window attention:

    gemma-4-26B  1024      gemma-4-31B  1024
    laguna-s-2.1  512      gpt-oss-120b  128      qwen3.8-27b  none

With SWA, llama.cpp can only roll the KV state back within the window.
If the point of divergence between two requests is further than `n_swa` tokens
from the end of the prompt, the server discards the entire prefix:

    forcing full prompt re-processing due to lack of cache data
    (likely due to SWA or hybrid/recurrent memory)

Claude Code appends a `system` message of about 1,600 tokens BEHIND the user
1.640 Token ("Available agent types for the Agent tool: ..."). Damit liegt jede
changed question far outside the window, and every request costs the full
vollen Prefill.

Was dieser Proxy tut
--------------------
It hoists STABLE system messages out of `messages` to the end of the `system` field.
This puts the user question back at the end of the prompt, i.e. inside the window.

The difference from the first version: VOLATILE system messages stay
stay where they are. From turn 2 on, Claude Code appends one message per turn
Form

    <total_tokens>14981262 tokens left</total_tokens>

whose number changes every time. Pulling those to the front moves changing text
IN FRONT of the 66 KB tool block and devalues it on every turn.
That is exactly where the first version failed on tool conversations.
At the end of the message history they do no harm: the history only grows
there, and pure appending needs no rolling back.

In addition the proxy removes `thinking`, `context_management` and
`output_config`, which llama-server would otherwise answer with 400, and does
not buffer the response stream — otherwise Claude Code aborts after 300 s of silence.

Start:  python3 ~/.claude/bin/cc-cachefix2.py
Env:    PORT (8090), LLAMA_URL (http://127.0.0.1:8080), LOG (leer = aus)
"""
import os, json, re, sys, time
from aiohttp import web, ClientSession, ClientTimeout

LLAMA = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")
PORT  = int(os.environ.get("PORT", 8090))
LOG   = os.environ.get("LOG", "")

HOP = {"host", "content-length", "connection", "transfer-encoding",
       "keep-alive", "accept-encoding"}
# Fields that llama-server answers with 400
DROP = ("thinking", "context_management", "output_config")

# Content that changes from turn to turn and must therefore NOT move to the front.
# Bewusst als Liste gefuehrt, damit weitere Muster nachtragbar sind.
FLUECHTIG = [
    re.compile(r"<total_tokens>\s*\d+\s*tokens left\s*</total_tokens>"),
]

def trenne(text):
    """Split a system message into its stable and its volatile part.

    Important: Claude Code appends the counter to the END of the otherwise
    stable agent-types block — so that block is not volatile as a whole, only
    in its last ~49 characters. Treating the whole block as volatile
    hoist it at all and throws the effect away.

    Returns: (stable text, list of volatile matches)
    """
    fund = []
    stabil = text
    for r in FLUECHTIG:
        fund.extend(r.findall(stabil))
        stabil = r.sub("", stabil)
    return stabil.rstrip(), fund

def blocks_to_text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(b.get("text", "") for b in c
                       if isinstance(b, dict) and b.get("type") == "text")
    return ""

def fix(p):
    msgs = p.get("messages")
    if not isinstance(msgs, list):
        return p, 0, 0
    hoch, bleiben, gesehen = [], [], set()
    n_fluechtig = 0
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "system":
            stabil, fluechtig = trenne(blocks_to_text(m.get("content")))
            if fluechtig:
                n_fluechtig += len(fluechtig)
                # stays where it is, so that the changing text sits at the
                # END of the history rather than in front of the tool block
                bleiben.append({"role": "system", "content": "".join(fluechtig)})
            # De-duplicate: the same stable block is appended again per turn
            if stabil and stabil not in gesehen:
                gesehen.add(stabil)
                hoch.append(stabil)
        else:
            bleiben.append(m)
    if not hoch:
        return p, 0, n_fluechtig
    p["messages"] = bleiben
    schwanz = "\n\n".join(hoch)
    s = p.get("system")
    if isinstance(s, list):
        p["system"] = s + [{"type": "text", "text": "\n\n" + schwanz}]
    elif isinstance(s, str):
        p["system"] = s + "\n\n" + schwanz
    else:
        p["system"] = schwanz
    return p, len(hoch), n_fluechtig

def log(*a):
    if LOG:
        print(*a, file=sys.stderr, flush=True)

async def handler(req):
    body = await req.read()
    out = None
    if req.path.startswith("/v1/messages") and body and not req.path.endswith("count_tokens"):
        try:
            p = json.loads(body)
            for k in DROP:
                p.pop(k, None)
            p, n_hoch, n_fl = fix(p)
            out = json.dumps(p).encode()
            log("[cachefix2] hoisted=%d  left volatile=%d" % (n_hoch, n_fl))
        except Exception as e:
            log("[cachefix2] passed through unchanged: %r" % (e,))
            out = None

    hdrs = {k: v for k, v in req.headers.items() if k.lower() not in HOP}
    timeout = ClientTimeout(total=None, sock_read=None, sock_connect=30)
    async with ClientSession(timeout=timeout, auto_decompress=False) as s:
        async with s.request(req.method, LLAMA + req.path_qs,
                             data=(out if out is not None else body),
                             headers=hdrs, allow_redirects=False) as up:
            rh = {k: v for k, v in up.headers.items()
                  if k.lower() not in HOP and k.lower() != "content-encoding"}
            resp = web.StreamResponse(status=up.status, headers=rh)
            await resp.prepare(req)
            async for ch in up.content.iter_any():   # kein Puffern -> SSE bleibt heil
                await resp.write(ch)
            await resp.write_eof()
            return resp

# Guarded, because importing this file STARTED A SERVER on PORT. Both
# cachefix proxies live in the directory tests/common.py is about — it
# loads cc-gateway.py, cc-router.py and prewarm.py by path and states that
# "the precondition for that is an import without consequences: no network,
# no token file, no web.run_app". Their three siblings had the guard. These
# two are superseded and no longer installed, which is why nobody noticed.
if __name__ == "__main__":
    app = web.Application(client_max_size=1024**3)
    app.router.add_route("*", "/{tail:.*}", handler)
    web.run_app(app, host="127.0.0.1", port=PORT)
