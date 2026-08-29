#!/usr/bin/env python3
"""tracelog — switch the gateway's tracing on, and read what it wrote.

    python3 tools/tracelog.py on detail        while something looks odd
    python3 tools/tracelog.py on text --minutes 30    prompts too, briefly
    python3 tools/tracelog.py off
    python3 tools/tracelog.py status

    python3 tools/tracelog.py serve            a table in the browser, live
    python3 tools/tracelog.py show             the last events, compact
    python3 tools/tracelog.py lies             warm that was not warm
    python3 tools/tracelog.py prefixes         per prefix: cost and reuse
    python3 tools/tracelog.py saves            every save, restore, quarantine
    python3 tools/tracelog.py diff             where a history stopped being
                                               an append, and what it cost

The switch takes effect on the next request without restarting cc-gateway —
which matters, because restarting it clears the very prefix bookkeeping you
turned the trace on to look at.

`show`, `lies` and the rest are the questions this stack keeps asking. They
exist so that reading a trace is not an exercise in jq.
"""
import argparse, json, os, re, sys, time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "setup", "claude"))
import tracelog as TR                                            # noqa: E402


def _tr():
    return TR.Trace()


def cmd_on(a):
    t = _tr()
    level = t.set_level(a.level, minutes=a.minutes)
    print("trace: %s" % level)
    if level == "text":
        print("  PROMPTS ARE BEING WRITTEN TO %s" % t.dir)
        print("  back to `detail` in %d minutes by itself." % (a.minutes or 60))
    print("  files: %s" % t.dir)


def cmd_off(a):
    print("trace: %s" % _tr().set_level("off"))


def cmd_status(a):
    t = _tr()
    print("level: %s" % t.level)
    print("dir:   %s" % t.dir)
    if not os.path.isdir(t.dir):
        print("  (nothing written yet)")
        return
    total = 0
    for f in sorted(os.listdir(t.dir)):
        if f.startswith("trace-"):
            n = os.path.getsize(os.path.join(t.dir, f))
            total += n
            print("  %-24s %8.1f MB" % (f, n / 1e6))
    print("  %.1f MB of %.0f MB" % (total / 1e6, t.cap / 1e6))


def _load(a):
    t = _tr()
    files = sorted(f for f in os.listdir(t.dir)
                   if f.startswith("trace-") and f.endswith(".jsonl")) \
        if os.path.isdir(t.dir) else []
    if a.day:
        files = [f for f in files if a.day in f]
    out = []
    for f in files[-3:]:
        for line in open(os.path.join(t.dir, f), encoding="utf-8"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("kind") != "header":
                out.append(d)
    return out


def _clock(rec):
    return time.strftime("%H:%M:%S", time.localtime(rec.get("t", 0)))


def cmd_show(a):
    for r in _load(a)[-a.n:]:
        bits = ["%s %-9s" % (_clock(r), r.get("kind", "?"))]
        for k in ("who", "model", "prefix", "cold", "took_s", "reused",
                  "computed", "restored", "saved_s", "note"):
            if r.get(k) is not None:
                bits.append("%s=%s" % (k, r[k]))
        print("  " + "  ".join(bits))


def cmd_lies(a):
    """Where the gateway's label and the server's accounting disagree.

    The defect of 28.08.2026 in one query: a request called `warm` that
    reused nothing. Reading it out of a trace takes a second; finding it by
    hand took an evening.
    """
    bad = [r for r in _load(a)
           if r.get("kind") == "request" and r.get("cold") is False
           and isinstance(r.get("reused"), int) and isinstance(r.get("computed"), int)
           and r["reused"] * 4 < r["computed"]]
    if not bad:
        print("  nothing: every request called warm actually reused its prefix")
        return
    for r in bad:
        print("  %s prefix=%s took=%ss reused=%s computed=%s%s"
              % (_clock(r), r.get("prefix"), r.get("took_s"), r["reused"],
                 r["computed"],
                 "  (restored %s)" % r["restored"] if r.get("restored") else ""))


def cmd_prefixes(a):
    per = defaultdict(lambda: {"n": 0, "cold": 0, "took": 0.0, "reused": 0,
                               "computed": 0, "models": set()})
    for r in _load(a):
        if r.get("kind") != "request" or not r.get("prefix"):
            continue
        e = per[r["prefix"]]
        e["n"] += 1
        e["cold"] += 1 if r.get("cold") else 0
        e["took"] += float(r.get("took_s") or 0)
        e["reused"] += int(r.get("reused") or 0)
        e["computed"] += int(r.get("computed") or 0)
        if r.get("model"):
            e["models"].add(r["model"])
    print("  %-14s %5s %5s %9s %10s %10s  %s"
          % ("prefix", "reqs", "cold", "avg s", "reused", "computed", "models"))
    for pid, e in sorted(per.items(), key=lambda kv: -kv[1]["n"]):
        print("  %-14s %5d %5d %9.1f %10d %10d  %s"
              % (pid, e["n"], e["cold"], e["took"] / max(e["n"], 1),
                 e["reused"], e["computed"], " ".join(sorted(e["models"]))[:40]))


def _excerpt(msg, width=160):
    """One message reduced to something a terminal can show."""
    try:
        s = json.dumps(msg, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"))
    except Exception:
        s = repr(msg)
    s = re.sub(r"\s+", " ", s)
    return s if len(s) <= width else s[:width - 1] + "…"


def cmd_diff(a):
    """Where a conversation stopped being an append — and what it cost.

    Two consecutive requests for the same prefix normally share every message
    but the last few. When they do not, everything after the first changed
    message is re-prefilled, and that is indistinguishable in `reused` alone
    from the slot having been taken away. This reads the two apart:

        shape agrees, reuse dropped      the state was lost
        shape diverges                   the client rewrote its own history

    `msg_chars` then says which kind of rewrite: same length is a re-render,
    shorter is a truncation, longer an edit. With the trace at `text` the two
    versions of the message itself are printed underneath.
    """
    per, requests = defaultdict(list), 0
    for r in _load(a):
        if r.get("kind") != "request" or not r.get("prefix"):
            continue
        requests += 1
        if r.get("shape"):
            per[r["prefix"]].append(r)
    pairs = sum(max(len(v) - 1, 0) for v in per.values())
    if not pairs:
        # "nothing found" and "nothing could be looked at" are different
        # answers, and printing the first for the second is how a gap becomes
        # a clean bill of health. Records written before 29.08.2026 carry no
        # shape at all — the incident that motivated this command is among
        # them and can never be diffed.
        print("  nothing to compare: %d requests, none of them carrying a "
              "`shape`.\n  The trace has to be at `detail` or higher WHEN the "
              "request runs;\n  it cannot be added afterwards." % requests)
        return
    hits = 0
    for pid, rows in per.items():
        rows.sort(key=lambda r: r.get("t", 0))
        for prev, cur in zip(rows, rows[1:]):
            old, new = prev.get("shape") or [], cur.get("shape") or []
            kept = 0
            for x, y in zip(old, new):
                if x != y:
                    break
                kept += 1
            appended = kept == len(old)
            reused, computed = cur.get("reused"), cur.get("computed")
            if appended and not (isinstance(computed, int) and isinstance(reused, int)
                                 and computed > 4 * max(reused, 1) / 10):
                continue                       # a clean append that stayed cheap
            hits += 1
            print("  %s prefix=%s  msgs %d -> %d, kept %d%s"
                  % (_clock(cur), pid, len(old), len(new), kept,
                     "  (pure append)" if appended else ""))
            print("      reused=%s computed=%s took=%ss"
                  % (reused, computed, cur.get("took_s")))
            if appended:
                print("      the history agrees — a drop here is the SERVER's "
                      "state, not the client's history")
                continue
            oc = (prev.get("msg_chars") or [None] * len(old))
            nc = (cur.get("msg_chars") or [None] * len(new))
            a_chars = oc[kept] if kept < len(oc) else None
            b_chars = nc[kept] if kept < len(nc) else None
            if a_chars is not None and b_chars is not None:
                kind = ("re-rendered, same length" if a_chars == b_chars else
                        "truncated" if b_chars < a_chars else "extended")
                print("      message %d: %d -> %d characters — %s"
                      % (kept, a_chars, b_chars, kind))
            ob = (prev.get("body_full") or {}).get("messages") or []
            nb = (cur.get("body_full") or {}).get("messages") or []
            if kept < len(ob) and kept < len(nb):
                print("      was: %s" % _excerpt(ob[kept]))
                print("      now: %s" % _excerpt(nb[kept]))
            elif not ob:
                print("      (no text: run `tracelog.py on text` to see the "
                      "message itself)")
    if not hits:
        print("  nothing in %d pairs: every request for every prefix was a "
              "clean, cheap append" % pairs)
    _head_changes(_load(a))


def _head_changes(records):
    """When the PREFIX id itself changes from one request to the next.

    Grouping by prefix — which everything above does — is blind to exactly the
    most expensive failure there is, because the two requests land in
    different groups and are never compared. Measured 30.08.2026, 00:01: the
    tool list went from 13 to 21 entries mid-conversation, the prefix id
    changed with it, and 55,856 tokens of an untouched conversation were
    recomputed. 655 seconds. The message shapes agreed throughout.

    The tools and the system prompt sit IN FRONT of the messages in the
    rendered prompt, so a change there invalidates everything behind it. No
    amount of history stability helps.
    """
    per = defaultdict(list)
    for r in records:
        if r.get("kind") == "request" and r.get("who") and r.get("prefix"):
            per[r["who"]].append(r)
    fresh, returning = [], 0
    for who, rows in per.items():
        rows.sort(key=lambda r: r.get("t", 0))
        seen = set()
        for prev, cur in zip(rows, rows[1:]):
            seen.add(prev["prefix"])
            if prev["prefix"] == cur["prefix"]:
                continue
            if cur["prefix"] in seen:
                # Claude Code runs TWO prompt types side by side — the
                # conversation and a small toolless one — and they flip the
                # prefix back and forth all day. A prefix this caller has used
                # before is that, not a head that changed.
                returning += 1
                continue
            fresh.append((who, prev, cur))
    if returning:
        print("  (%d returns to a prefix this caller had used before — two "
              "prompt types alternating, not a new head)" % returning)
    if not fresh:
        print("  no caller met a prefix it had never seen before")
        return
    for who, prev, cur in sorted(fresh, key=lambda t: -(t[2].get("computed") or 0)):
        print("  %s who=%s  A HEAD IT HAD NEVER SEEN: prefix %s -> %s"
              % (_clock(cur), who, prev["prefix"], cur["prefix"]))
        print("      tools %s -> %s, prefix %s -> %s characters"
              % (prev.get("tools"), cur.get("tools"),
                 prev.get("prefix_chars"), cur.get("prefix_chars")))
        print("      cost: reused=%s computed=%s took=%ss"
              % (cur.get("reused"), cur.get("computed"), cur.get("took_s")))
        old, new = prev.get("shape") or [], cur.get("shape") or []
        if old and new:
            kept = 0
            for x, y in zip(old, new):
                if x != y:
                    break
                kept += 1
            print("      the messages: %d of %d unchanged — %s"
                  % (kept, len(old),
                     "the history was not the cause" if kept >= len(old) - 3
                     else "the history moved too"))


def cmd_saves(a):
    for r in _load(a):
        if r.get("kind") in ("save", "restore", "quarantine", "mismatch"):
            print("  %s %-11s %s" % (_clock(r), r.get("kind"),
                                     json.dumps({k: v for k, v in r.items()
                                                 if k not in ("t", "kind")},
                                                ensure_ascii=False)[:150]))


# --- the browser view --------------------------------------------------------
#
# A separate process on purpose. The gateway sits on the request path of every
# answer this machine gives; a display that can crash, block or grow must not
# live there. 127.0.0.1 only, like everything else here.
#
# The trace is append-only JSONL, so "what is new" is a byte offset — which is
# why this polls instead of holding a connection open. A poll survives a
# gateway restart, the file rolling over at midnight and a laptop lid, and each
# of those would need its own handling in a stream.

DAY_FILE = re.compile(r"^trace-\d{4}-\d{2}-\d{2}\.jsonl$")
# What a row must be clicked to reveal. The server does not send it otherwise —
# not because a browser on the same machine is a threat, but because a default
# that ships whole prompts over a socket is the kind of thing that is later
# port-forwarded by somebody in a hurry.
TEXT_FIELDS = ("system_head", "answer_tail", "prompt", "answer", "body_full")


def _safe_file(name, directory):
    """A day file inside the trace directory, or None. No traversal, no
    guessing: the name has to look exactly like what the trace writes."""
    if not name or not DAY_FILE.match(name):
        return None
    path = os.path.join(directory, name)
    if os.path.realpath(os.path.dirname(path)) != os.path.realpath(directory):
        return None
    return path if os.path.exists(path) else None


def _redact(rec):
    """Replace text by its size. The row says there is something to see; the
    click fetches it."""
    out = dict(rec)
    for k in TEXT_FIELDS:
        v = out.get(k)
        if isinstance(v, str):
            out[k] = "<%d characters — click the row>" % len(v)
        elif isinstance(v, (dict, list)):
            # `body_full` is a whole request. Measuring it means serialising it
            # again for every row of every poll, so the row says how many parts
            # there are and nothing more — the size is not worth a second pass
            # over hundreds of megabytes.
            out[k] = "<%d %s — click the row>" % (
                len(v), "messages" if isinstance(v, list) else "fields")
    return out


def read_since(path, offset, limit=2000, redact=True):
    """Records after `offset`, and where to continue. Each carries its own
    byte position, so one row can be fetched again in full."""
    recs = []
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        if offset > end:                       # truncated or rotated under us
            offset = 0
        f.seek(offset)
        while len(recs) < limit:
            at = f.tell()
            line = f.readline()
            if not line or not line.endswith(b"\n"):
                f.seek(at)                     # a half-written line: next time
                break
            try:
                rec = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            rec["_at"] = at
            recs.append(_redact(rec) if redact else rec)
        return recs, f.tell()


def read_one(path, at):
    with open(path, "rb") as f:
        f.seek(at)
        line = f.readline()
    return json.loads(line.decode("utf-8"))


def cmd_serve(a):
    import http.server, socketserver, urllib.parse
    t = _tr()
    ui = os.path.join(HERE, "traceui.html")

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            body = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            # Nothing from this page may leave the machine, and nothing from
            # outside may embed it.
            self.send_header("Content-Security-Policy",
                             "default-src 'self' 'unsafe-inline'; connect-src 'self'")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass                               # the terminal is for the operator

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(u.query)
            one = lambda k, d=None: (q.get(k) or [d])[0]
            if u.path in ("/", "/index.html"):
                with open(ui, "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if u.path == "/days":
                days = sorted(f for f in os.listdir(t.dir)
                              if DAY_FILE.match(f)) if os.path.isdir(t.dir) else []
                return self._send(200, json.dumps(
                    {"days": days, "level": t.refresh(), "dir": t.dir}))
            if u.path == "/events":
                name = one("file") or os.path.basename(t.path_for_today())
                path = _safe_file(name, t.dir)
                if not path:
                    return self._send(200, json.dumps(
                        {"file": name, "offset": 0, "records": [],
                         "level": t.refresh()}))
                recs, off = read_since(path, int(one("since", "0") or 0))
                return self._send(200, json.dumps(
                    {"file": name, "offset": off, "records": recs,
                     "level": t.refresh()}))
            if u.path == "/record":
                path = _safe_file(one("file"), t.dir)
                if not path:
                    return self._send(404, json.dumps({"error": "no such file"}))
                try:
                    return self._send(200, json.dumps(read_one(path, int(one("at", "0")))))
                except Exception as e:
                    return self._send(404, json.dumps({"error": repr(e)}))
            return self._send(404, json.dumps({"error": "no such path"}))

    with socketserver.ThreadingTCPServer(("127.0.0.1", a.port), Handler) as srv:
        srv.allow_reuse_address = True
        print("trace ui: http://127.0.0.1:%d   (level %s, %s)"
              % (a.port, t.level, t.dir))
        print("  ctrl-c to stop")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("on"); s.set_defaults(fn=cmd_on)
    s.add_argument("level", choices=[l for l in TR.LEVELS if l != "off"])
    s.add_argument("--minutes", type=int, default=None,
                   help="switch back to `detail` afterwards. Forced to 60 for "
                        "`text` if not given: prompts on disk must not be the "
                        "result of forgetting.")
    sub.add_parser("off").set_defaults(fn=cmd_off)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sv = sub.add_parser("serve"); sv.set_defaults(fn=cmd_serve)
    sv.add_argument("--port", type=int, default=8092)
    for name, fn in (("show", cmd_show), ("lies", cmd_lies),
                     ("prefixes", cmd_prefixes), ("saves", cmd_saves),
                     ("diff", cmd_diff)):
        p = sub.add_parser(name); p.set_defaults(fn=fn)
        p.add_argument("--day", help="YYYY-MM-DD; default: the last three files")
        p.add_argument("-n", type=int, default=40)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
