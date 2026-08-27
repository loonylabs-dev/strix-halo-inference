#!/usr/bin/env python3
"""cc-router — variant 2: keep the subscription, run individual subagents locally.

Routes /v1/messages by model name:
  'local/*'  -> llama-server   (prefix is stripped, must match the --alias)
  everything else -> api.anthropic.com, byte transparent

Decisive for subscription sign-in: the anthropic-beta header is passed through
unchanged. Under OAuth it carries a capability marker; filter it out and
Anthropic answers 401.

Equally decisive: no buffering. Claude Code aborts streams that stay silent for
300 s, and SSE keep-alives have to go through immediately.

Start:  python3 ~/.claude/bin/cc-router.py
Env:    PORT (8090), LLAMA_URL (http://127.0.0.1:8080), LOCAL_PREFIX (local/)
"""
import os, json
from aiohttp import web, ClientSession, ClientTimeout

UPSTREAM = "https://api.anthropic.com"
LLAMA    = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")
PREFIX   = os.environ.get("LOCAL_PREFIX", "local/")

HOP = {"host", "content-length", "connection", "transfer-encoding",
       "keep-alive", "accept-encoding"}

# What is forwarded at all in the LOCAL branch. An allow list, not a deny list:
# previously only authorization and x-api-key were stripped, while cc-gateway
# itself knows anthropic-auth-token as a possible carrier of credentials (see
# its diagnostics branch). A header nobody thought of would therefore have
# carried the subscription token to the foreign server — and nobody would have
# noticed, because the damage happens at the consumer, not at the operator. An
# allow list cannot repeat that mistake.
LOCAL_ALLOWED = {"content-type", "accept", "accept-language",
                 "anthropic-version", "user-agent"}

async def handler(req):
    body, target, out = await req.read(), UPSTREAM, None
    local = False

    if req.path.startswith("/v1/messages"):
        try:
            p = json.loads(body)
        except Exception:
            p = None
        if isinstance(p, dict) and str(p.get("model", "")).startswith(PREFIX):
            target, local = LLAMA, True
            p["model"] = p["model"][len(PREFIX):]      # -> llama-server --alias
            # Drop fields that llama-server answers with 400.
            # ONLY in the local branch — Anthropic gets everything unchanged.
            for k in ("thinking", "context_management", "output_config"):
                p.pop(k, None)
            out = json.dumps(p).encode()

    hdrs = {k: v for k, v in req.headers.items() if k.lower() not in HOP}
    # 'local' rather than 'target is LLAMA': an identity check against a string
    # from the environment is too easy to break, and whether credentials leave
    # this machine hangs on it.
    if local:
        hdrs = {k: v for k, v in hdrs.items() if k.lower() in LOCAL_ALLOWED}
        if os.environ.get("LLAMA_API_KEY"):
            hdrs["Authorization"] = "Bearer " + os.environ["LLAMA_API_KEY"]

    timeout = ClientTimeout(total=None, sock_read=None, sock_connect=30)
    async with ClientSession(timeout=timeout, auto_decompress=False) as s:
        async with s.request(req.method, target + req.path_qs,
                             data=(out if out is not None else body),
                             headers=hdrs, allow_redirects=False) as up:
            rh = {k: v for k, v in up.headers.items()
                  if k.lower() not in HOP and k.lower() != "content-encoding"}
            resp = web.StreamResponse(status=up.status, headers=rh)
            await resp.prepare(req)
            async for chunk in up.content.iter_any():   # no buffering -> SSE stays intact
                await resp.write(chunk)
            await resp.write_eof()
            return resp

def build_app():
    """A fresh application per call — see cc-gateway.build_app()."""
    a = web.Application(client_max_size=1024**3)
    a.router.add_route("*", "/{tail:.*}", handler)
    return a

app = build_app()

if __name__ == "__main__":
    # Only start on a direct call — otherwise the router cannot be imported
    # and therefore cannot be tested.
    web.run_app(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8090)))
