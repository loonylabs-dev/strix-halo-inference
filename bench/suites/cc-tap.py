#!/usr/bin/env python3
"""cc-tap — a recording proxy in front of llama-server.

Writes every /v1/messages request body to disk, numbered, logs the usage
values of the answer and forwards unchanged (or with cc-cachefix logic).
Meant for capturing real Claude Code requests for the cache hunt.

  FIX=0   pass through unchanged  (the default, for capturing)
  FIX=1   apply the cc-cachefix logic (hoist system messages)

Env:  PORT (8090), LLAMA_URL (http://127.0.0.1:8080), OUT (./bodies), FIX (0)
"""
import os, json, time, sys
from aiohttp import web, ClientSession, ClientTimeout

LLAMA = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")
PORT  = int(os.environ.get("PORT", 8090))
OUT   = os.environ.get("OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "bodies"))
FIX   = os.environ.get("FIX", "0") == "1"
TAG   = os.environ.get("TAG", "req")

HOP = {"host","content-length","connection","transfer-encoding","keep-alive","accept-encoding"}
DROP = ("thinking", "context_management", "output_config")

os.makedirs(OUT, exist_ok=True)
N = [0]
T0 = time.time()

def log(*a):
    print("[%7.1fs]" % (time.time()-T0), *a, flush=True)

def blocks_to_text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(b.get("text","") for b in c
                       if isinstance(b, dict) and b.get("type") == "text")
    return ""

def apply_fix(p):
    msgs = p.get("messages")
    if not isinstance(msgs, list):
        return p
    extra, keep, seen = [], [], set()
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "system":
            t = blocks_to_text(m.get("content"))
            if t.strip() and t not in seen:
                seen.add(t); extra.append(t)
        else:
            keep.append(m)
    if not extra:
        return p
    p["messages"] = keep
    tail = "\n\n".join(extra)
    s = p.get("system")
    if isinstance(s, list):
        p["system"] = s + [{"type":"text","text":"\n\n"+tail}]
    elif isinstance(s, str):
        p["system"] = s + "\n\n" + tail
    else:
        p["system"] = tail
    return p

def describe(p):
    """Short profile of the request for the run log."""
    msgs = p.get("messages") or []
    roles = [m.get("role") for m in msgs if isinstance(m, dict)]
    sysmsgs = sum(1 for r in roles if r == "system")
    tools = p.get("tools") or []
    sysfield = p.get("system")
    if isinstance(sysfield, list):
        syslen = sum(len(b.get("text","")) for b in sysfield if isinstance(b, dict))
        sysblocks = len(sysfield)
    elif isinstance(sysfield, str):
        syslen, sysblocks = len(sysfield), 1
    else:
        syslen, sysblocks = 0, 0
    # cache_control-Marker zaehlen
    cc = json.dumps(p).count('"cache_control"')
    return dict(model=p.get("model"), n_msgs=len(msgs), roles=roles,
                n_sysmsgs=sysmsgs, n_tools=len(tools),
                tool_names=[t.get("name") for t in tools],
                sys_chars=syslen, sys_blocks=sysblocks,
                tools_bytes=len(json.dumps(tools)),
                cache_control=cc, total_bytes=len(json.dumps(p)))

async def handler(req):
    body = await req.read()
    out = None
    meta = None
    if req.path.startswith("/v1/messages") and body and not req.path.endswith("count_tokens"):
        N[0] += 1
        n = N[0]
        try:
            p = json.loads(body)
            # ROH speichern, vor jeder Aenderung
            raw = os.path.join(OUT, "%s-%03d-roh.json" % (TAG, n))
            with open(raw, "wb") as f:
                f.write(body)
            meta = describe(p)
            with open(os.path.join(OUT, "%s-%03d-profil.json" % (TAG, n)), "w") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            log("#%03d %d B  msgs=%d (sys=%d)  tools=%d  sysfeld=%dZ/%dBl  cc=%d  rollen=%s"
                % (n, meta["total_bytes"], meta["n_msgs"], meta["n_sysmsgs"],
                   meta["n_tools"], meta["sys_chars"], meta["sys_blocks"],
                   meta["cache_control"], ",".join(str(r) for r in meta["roles"])))
            for k in DROP:
                p.pop(k, None)
            if FIX:
                p = apply_fix(p)
            out = json.dumps(p).encode()
            with open(os.path.join(OUT, "%s-%03d-gesendet.json" % (TAG, n)), "wb") as f:
                f.write(out)
        except Exception as e:
            log("#%03d PARSE-FEHLER %r" % (N[0], e))
            out = None

    hdrs = {k: v for k, v in req.headers.items() if k.lower() not in HOP}
    timeout = ClientTimeout(total=None, sock_read=None, sock_connect=30)
    t_start = time.time()
    async with ClientSession(timeout=timeout, auto_decompress=False) as s:
        async with s.request(req.method, LLAMA + req.path_qs,
                             data=(out if out is not None else body),
                             headers=hdrs, allow_redirects=False) as up:
            rh = {k: v for k, v in up.headers.items()
                  if k.lower() not in HOP and k.lower() != "content-encoding"}
            resp = web.StreamResponse(status=up.status, headers=rh)
            await resp.prepare(req)
            buf = bytearray()
            async for ch in up.content.iter_any():
                if meta is not None and len(buf) < 4_000_000:
                    buf.extend(ch)
                await resp.write(ch)
            await resp.write_eof()
            if meta is not None:
                dt = time.time() - t_start
                usage = scrape_usage(bytes(buf))
                log("#%03d <- %.1fs  %s" % (N[0], dt, usage))
                with open(os.path.join(OUT, "%s-%03d-answer.txt" % (TAG, N[0])), "w") as f:
                    f.write("dauer_s=%.3f\nusage=%s\n" % (dt, usage))
            return resp

def scrape_usage(buf):
    """Finds usage objects in JSON or SSE answers."""
    try:
        txt = buf.decode("utf-8", "replace")
    except Exception:
        return "?"
    best = {}
    idx = 0
    while True:
        i = txt.find('"usage"', idx)
        if i < 0:
            break
        idx = i + 7
        j = txt.find("{", idx)
        if j < 0:
            break
        depth, k = 0, j
        while k < len(txt):
            if txt[k] == "{": depth += 1
            elif txt[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        try:
            u = json.loads(txt[j:k+1])
            for key, val in u.items():
                if isinstance(val, int) and val:
                    best[key] = val
        except Exception:
            pass
    return json.dumps(best) if best else "no usage"

# Guarded, because importing this file used to START THE SERVER. tests/
# common.py names that case outright — "no web.run_app" — as the precondition
# for loading a script by path. Nothing imports bench/suites/ today; that is
# the entire defence, and it is not one anybody decided to rely on.
if __name__ == "__main__":
    app = web.Application(client_max_size=1024**3)
    app.router.add_route("*", "/{tail:.*}", handler)
    log("cc-tap on :%d -> %s   FIX=%s  OUT=%s" % (PORT, LLAMA, FIX, OUT))
    web.run_app(app, host="127.0.0.1", port=PORT, print=None)
