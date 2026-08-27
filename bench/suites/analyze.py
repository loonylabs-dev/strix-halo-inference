#!/usr/bin/env python3
"""Structural analysis of CAPTURED Claude Code request bodies.

This one genuinely needs a capture — its whole point is to look at what the
real thing contains, which is why it is the only suite that cannot be fed by
tools/synthetic.py. Captures carry an e-mail address, device_id, account_uuid
and Anthropic's system prompt, so they are excluded from the repo; put one
under bench/suites/bodies/ with bench/suites/cc-tap.py to use this.
"""
import json, sys, os, glob

W = os.path.dirname(os.path.abspath(__file__))

if not glob.glob(os.path.join(W, "bodies", "*.json")):
    print("no captured bodies under %s." % os.path.join(W, "bodies"))
    print("This suite analyses real captures — record one with cc-tap.py first.")
    print("Every other suite builds its body with tools/synthetic.py and needs none.")
    raise SystemExit(0)

def blk(b):
    if not isinstance(b, dict):
        return ("?", 0, "")
    t = b.get("type", "?")
    if t == "text":
        return (t, len(b.get("text", "")), b.get("text", "")[:60].replace("\n", "\\n"))
    if t == "tool_use":
        return (t, len(json.dumps(b.get("input", {}))), b.get("name", ""))
    if t == "tool_result":
        c = b.get("content")
        n = len(json.dumps(c)) if c is not None else 0
        return (t, n, b.get("tool_use_id", "")[:20])
    if t == "thinking":
        return (t, len(b.get("thinking", "")), "")
    return (t, len(json.dumps(b)), "")

def dump(path):
    p = json.load(open(path))
    print("=" * 78)
    print(os.path.basename(path))
    print("=" * 78)
    # top-level Felder
    print("Top-Level-Felder:", sorted(p.keys()))
    sysf = p.get("system")
    if isinstance(sysf, list):
        print("system field: array with %d blocks" % len(sysf))
        for i, b in enumerate(sysf):
            cc = "  <== cache_control %s" % b.get("cache_control") if "cache_control" in b else ""
            print("   [%d] %s %d chars  %r%s" % (i, b.get("type"), len(b.get("text", "")),
                                                   b.get("text", "")[:70].replace("\n", "\\n"), cc))
    elif isinstance(sysf, str):
        print("system field: string, %d chars" % len(sysf))
    tools = p.get("tools") or []
    print("tools: %d of them, %d bytes" % (len(tools), len(json.dumps(tools))))
    for i, t in enumerate(tools):
        cc = "  <== cache_control" if "cache_control" in t else ""
        print("   [%2d] %-28s %6d B%s" % (i, t.get("name"), len(json.dumps(t)), cc))
    msgs = p.get("messages") or []
    print("messages: %d" % len(msgs))
    for i, m in enumerate(msgs):
        role = m.get("role")
        c = m.get("content")
        if isinstance(c, str):
            print("   [%d] %-9s STRING %d Z  %r" % (i, role, len(c), c[:70].replace("\n", "\\n")))
        elif isinstance(c, list):
            print("   [%d] %-9s ARRAY %d Bloecke" % (i, role, len(c)))
            for j, b in enumerate(c):
                t, n, s = blk(b)
                cc = "  <== cache_control %s" % b.get("cache_control") if isinstance(b, dict) and "cache_control" in b else ""
                print("        (%d) %-12s %6d  %r%s" % (j, t, n, s, cc))
        else:
            print("   [%d] %-9s %r" % (i, role, c))
    for k in ("max_tokens", "temperature", "stream", "metadata", "tool_choice"):
        if k in p:
            print("%s = %s" % (k, json.dumps(p[k])[:120]))
    print()

if __name__ == "__main__":
    for f in sorted(glob.glob(os.path.join(W, "bodies", "*-roh.json"))):
        dump(f)
