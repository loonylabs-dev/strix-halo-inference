#!/usr/bin/env python3
"""sse-ping — when do the first bytes of a streamed answer actually arrive?

Context: a cold Claude-Code-sized prompt prefills for longer than Cloudflare's
~125 s proxy read window. setup/README.md documents that llama-server covers
that window with a 3-byte SSE comment (":\n\n") every 30 s in streaming mode —
measured on an older build. On 31.08.2026 a streamed request from a remote
client hit a 524 after 124.9 s anyway, so somewhere between llama-server and
the Cloudflare edge those bytes stopped flowing. This suite measures WHERE:
point it at llama-server directly, at the gateway, and at the tunnel URL, with
a cold prompt each time, and compare the chunk timelines.

Usage:
    python3 bench/suites/sse-ping.py --base http://127.0.0.1:8090 \
        --body <synthetic.json> [--token-file ~/.config/cc-gateway-tokens] \
        [--max-wait 240] [--label gateway-local]

The body is patched in memory, never on disk: stream -> true, max_tokens -> 16
(so the generation ends moments after the prefill), and a nonce is prepended
to system[0].text so every run is a COLD prefix. The token value is read from
the token file's first line ("name <secret>") and is never printed.

Output: one line per received chunk — seconds since send, byte count, and a
repr of the first bytes — then a summary: time to first byte, time to first
data event, number of ping comments, largest silent gap. The largest gap is
the number that decides: above ~125 s the tunnel path dies with a 524.
"""

import argparse
import http.client
import json
import ssl
import sys
import time
import urllib.parse
import uuid


def load_token(path):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                return line.split()[-1]
    raise SystemExit("token file has no usable line: %s" % path)


def patch_body(raw, model):
    body = json.loads(raw)
    body["stream"] = True
    body["max_tokens"] = 16
    if model:
        body["model"] = model
    nonce = "cold-probe %s. " % uuid.uuid4().hex
    sys0 = body.get("system")
    if isinstance(sys0, list) and sys0 and "text" in sys0[0]:
        sys0[0]["text"] = nonce + sys0[0]["text"]
    elif isinstance(sys0, str):
        body["system"] = nonce + sys0
    else:
        raise SystemExit("body has no system prompt to make cold")
    # A nonce in the system text alone does NOT make the prompt cold here:
    # the rendered prompt carries the tool section FIRST, so a changed system
    # text diverges at ~90 % depth and llama reuses everything before it
    # (measured 31.08.2026: f_keep 0.892, 2126 of 22226 tokens recomputed).
    # A nonce tool at position 0 moves the divergence to the front.
    tools = body.get("tools")
    if isinstance(tools, list):
        tools.insert(0, {"name": "cold_probe_%s" % uuid.uuid4().hex[:12],
                         "description": "cold-start marker, never call this",
                         "input_schema": {"type": "object", "properties": {}}})
    return json.dumps(body).encode()


def run(base, payload, token, max_wait, label):
    u = urllib.parse.urlsplit(base)
    if u.scheme == "https":
        conn = http.client.HTTPSConnection(
            u.netloc, timeout=max_wait, context=ssl.create_default_context())
    else:
        conn = http.client.HTTPConnection(u.netloc, timeout=max_wait)
    headers = {"content-type": "application/json",
               "accept": "text/event-stream"}
    if token:
        headers["authorization"] = "Bearer " + token

    t0 = time.monotonic()
    conn.request("POST", "/v1/messages", body=payload, headers=headers)
    resp = conn.getresponse()
    t_status = time.monotonic() - t0
    print("[%s] HTTP %s after %.1f s  (%d-byte body sent)"
          % (label, resp.status, t_status, len(payload)))

    chunks = []          # (t, nbytes, head)
    first_data = None
    last_t = t_status    # the status line is the first sign of life
    largest_gap = t_status
    try:
        while True:
            if time.monotonic() - t0 > max_wait:
                print("[%s] max-wait %.0f s reached, closing" % (label, max_wait))
                break
            piece = resp.read1(65536)
            t = time.monotonic() - t0
            if not piece:
                print("[%s] stream closed by the far side at %.1f s" % (label, t))
                break
            largest_gap = max(largest_gap, t - last_t)
            last_t = t
            head = piece[:60].decode("utf-8", "replace")
            chunks.append((t, len(piece), head))
            print("  %7.1f s  %5d B  %r" % (t, len(piece), head))
            if piece.lstrip().startswith(b"data:") or b"\ndata:" in piece:
                if first_data is None:
                    first_data = t
                    print("[%s] first data event at %.1f s — closing, "
                          "the prefill phase is what was measured" % (label, t))
                    break
    except (TimeoutError, http.client.HTTPException, OSError) as exc:
        t = time.monotonic() - t0
        print("[%s] connection error at %.1f s: %s" % (label, t, exc))
    finally:
        conn.close()

    pings = sum(1 for _, _, head in chunks if head.strip() in (":", ""))
    print("[%s] SUMMARY status=%s ttfb=%.1fs first_data=%s pings=%d "
          "largest_gap=%.1fs chunks=%d"
          % (label, resp.status, chunks[0][0] if chunks else t_status,
             ("%.1fs" % first_data) if first_data else "never",
             pings, largest_gap, len(chunks)))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="scheme://host[:port] to test, path is always /v1/messages")
    ap.add_argument("--body", required=True, help="a tools/synthetic.py file")
    ap.add_argument("--model", default="qwen38")
    ap.add_argument("--token-file", default=None,
                    help="read 'name <secret>' and send it as Bearer; value is never printed")
    ap.add_argument("--max-wait", type=float, default=240.0)
    ap.add_argument("--label", default="run")
    args = ap.parse_args()

    token = load_token(args.token_file) if args.token_file else None
    payload = patch_body(open(args.body, "rb").read(), args.model)
    return run(args.base, payload, token, args.max_wait, args.label)


if __name__ == "__main__":
    sys.exit(main())
