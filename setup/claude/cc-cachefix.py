#!/usr/bin/env python3
"""cc-cachefix — repairs Claude Code's prompt cache against llama-server.

Problem: Claude Code sends a message with role="system" INSIDE `messages`,
behind the user question ("Available agent types for the Agent tool: …").
The chat template treats only messages[0] as a system block; a second system
message in the middle of the history makes the whole prefix expire.

Measured on Laguna S 2.1, llama.cpp b10577:
    without the fix   changed question -> 25,146 tokens new, cache_read = 0
    with    the fix   changed question ->      7 tokens new, cache_read = 25,134

The fix pulls every system message out of `messages` to the end of the
`system` field. Identical in content, only in the right place.

Start:  python3 ~/.claude/bin/cc-cachefix.py
Env:    PORT (8090), LLAMA_URL (http://127.0.0.1:8080)
"""
import os, json
from aiohttp import web, ClientSession, ClientTimeout

LLAMA = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")
PORT  = int(os.environ.get("PORT", 8090))
HOP = {"host","content-length","connection","transfer-encoding","keep-alive","accept-encoding"}
# Fields that llama-server answers with 400
DROP = ("thinking", "context_management", "output_config")

def blocks_to_text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
    return ""

def fix(p):
    msgs = p.get("messages")
    if not isinstance(msgs, list):
        return p
    extra, keep, seen = [], [], set()
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "system":
            t = blocks_to_text(m.get("content"))
            # Entdoppeln: Claude Code haengt denselben Block pro Turn erneut an.
            # Without this the system field grows every turn and the prefix expires.
            if t.strip() and t not in seen:
                seen.add(t)
                extra.append(t)
        else:
            keep.append(m)
    if not extra:
        return p
    p["messages"] = keep
    tail = "\n\n".join(extra)
    s = p.get("system")
    if isinstance(s, list):
        p["system"] = s + [{"type": "text", "text": "\n\n" + tail}]
    elif isinstance(s, str):
        p["system"] = s + "\n\n" + tail
    else:
        p["system"] = tail
    return p

async def handler(req):
    body = await req.read()
    out = None
    if req.path.startswith("/v1/messages") and body:
        try:
            p = json.loads(body)
            for k in DROP:
                p.pop(k, None)
            out = json.dumps(fix(p)).encode()
        except Exception:
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
