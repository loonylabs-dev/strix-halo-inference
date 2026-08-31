#!/usr/bin/env python3
"""prewarm — save and restore the shared prefix of a project.

The problem: the first call for a project costs 100 to 180 seconds, because
~20,000 tokens of system prompt and tool schemas have to be computed. After
every server restart that falls due again.

The solution uses two observations:

  1. A restored slot state carries pure APPENDING flawlessly. It does not
     carry rolling back — the slot file holds only the global layers plus the
     SWA window (27 KiB per token instead of 102).
  2. Everything up to <user> is identical for every request of a project.
     Save exactly that section and any question becomes an append.

Measured against a real service restart, 22,497 tokens:

    save                        628 MB, 247 ms
    restore                              97 ms
    any question afterwards     99.6-99.8 %   0.85-1.56 s   (instead of 110 s)
    tool turns afterwards       99.5-99.6 %   1.43-1.58 s

Important: the prefix has to be rendered the way the gateway produces it —
that is, WITH the agent-types block hoisted. Without that the block stays
behind the question, has to be recomputed for every new question and still
costs ~10 s instead of 0.85 s.

Usage:

    # from a captured or a synthetic body
    python3 tools/prewarm.py save --body body.json --name projectA

    # pull every saved prefix into the slots (e.g. after a start)
    python3 tools/prewarm.py restore

    # what is in there?
    python3 tools/prewarm.py list
"""
import argparse, hashlib, json, os, re, sys, time, urllib.request

# dialects.py is the single source of truth for how a request body is read —
# shared with gateway.py. Duplicating that logic here is exactly how the
# id contract broke once before, and tests/test_dialects.py exists because of
# it: it holds the ONE reading of a request body for both dialects. In the
# repo it sits in setup/gateway/; installed, the symlinks put it side by side
# with this file in ~/.local/lib/llm-stack.
for _cand in (os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "setup", "gateway"),
              os.path.dirname(os.path.abspath(__file__))):
    if os.path.exists(os.path.join(_cand, "dialects.py")):
        sys.path.insert(0, _cand)
        break
import dialects as DIA                                    # noqa: E402

SRV        = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")
SLOT_PATH   = os.environ.get("SLOT_PATH", os.path.expanduser("~/.cache/llama-slots"))
VOLATILE  = [re.compile(r"<total_tokens>\s*\d+\s*tokens left\s*</total_tokens>")]

def req(path, payload=None, method=None, t=1800):
    d = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(SRV + path, data=d, method=method,
                               headers={"content-type": "application/json"})
    with urllib.request.urlopen(r, timeout=t) as x:
        b = x.read().decode()
        return json.loads(b) if b.strip().startswith(("{", "[")) else b

def blocks_to_text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(b.get("text", "") for b in c
                       if isinstance(b, dict) and b.get("type") == "text")
    return ""

def build_prefix(body, dialect=DIA.ANTHROPIC, hoist=None):
    """Render the part up to <user> — the way the gateway produces it.

    Mirrors its correction via the shared module: stable system messages are
    hoisted to the front, volatile ones (the <total_tokens> counter) are
    dropped, because they do not belong in the stable prefix anyway. Works
    for both dialects; the tool block is converted where needed.

    `hoist` MUST match what the gateway does, or this writes a file holding a
    prefix no request ever sends. Since 30.08.2026 the gateway can be told to
    leave mid-conversation system messages alone (HOIST_SYSTEM=0), and for a
    few minutes that day this function did not know: it would have saved the
    hoisted rendering while the gateway sent the other one, the restore would
    have matched nothing, and the file would have been quarantined for a
    defect that was in this line. None = read the same environment variable
    the gateway reads, so a hand-run and a gateway-run agree by default; the
    gateway passes it explicitly anyway.
    """
    if hoist is None:
        hoist = os.environ.get("HOIST_SYSTEM", "1") == "1"
    body = json.loads(json.dumps(body))          # never mutate the caller's
    if hoist:
        body, _ = DIA.hoist_system_messages(body, dialect, VOLATILE)
    full = req("/apply-template",
               DIA.template_payload(body, dialect), t=300)["prompt"]
    cut = full.find("<user>")
    if cut < 0:
        # A different chat template: fall back to the last user marker
        cut = full.rfind("X")
    return full[:cut]

def render_hash(prefix):
    return hashlib.sha256(prefix.encode("utf-8")).hexdigest()[:12]

def gateway_id(body, dialect=DIA.ANTHROPIC):
    """The same id that the gateway computes from a request.

    Both sides call dialects.prefix_id now, so they cannot drift apart any
    more. The basis is the RAW body (system head plus tools), not the
    rendered prefix.

    Only a stopgap for manual use, where a raw body is at hand. When the
    gateway calls automatically it passes its id through with --gateway-id,
    because its body is already corrected by then: the system head carries
    the hoisted blocks, and recomputing would produce a key that no incoming
    request ever creates.
    """
    return DIA.prefix_id(body, dialect)[0]

def wait_until_ready(t=900):
    """Wait until the server serves the slot interface.

    Do not check via /props or /health: /props already answers while the model
    is still loading, and /slots then returns 503. So ask for exactly what is
    needed afterwards — /slots itself.
    """
    for _ in range(t // 3):
        try:
            req("/slots", t=5)
            return True
        except Exception:
            time.sleep(3)
    return False

# There used to be a busy_slot() here that took the FIRST slot with content.
# That was a mistake: /completion may have used a different slot, and then a
# foreign state was saved — 22,767 instead of 22,488 tokens in testing, and
# the restored state matched no request. The slot number now comes from the
# answer of /completion itself.

def save(a):
    if not wait_until_ready():
        raise SystemExit("server did not become ready in time")
    with open(a.body, encoding="utf-8") as f:
        body = json.load(f)
    dialect = getattr(a, "dialect", DIA.ANTHROPIC)
    told = getattr(a, "hoist", None)
    prefix = build_prefix(body, dialect,
                          hoist=None if told is None else told == "1")
    h = render_hash(prefix)
    n_tok = len(req("/tokenize", {"content": prefix}, t=300)["tokens"])
    print("prefix: %d characters, %d tokens, id %s" % (len(prefix), n_tok, h))

    print("precomputing …")
    t0 = time.time()
    answer = req("/completion", {"prompt": prefix, "n_predict": 1,
                                  "cache_prompt": True})
    # THE SPLIT, because "done in 51.4 s" alone cannot say whether the slot
    # already held half of this. llama.cpp reports it per request and this was
    # dropping it on the floor; the trace's Cache column then stood empty for
    # every save while requests beside it showed 99 %. Asked 30.08.2026.
    tm = answer.get("timings") or {}
    print("  done in %.1f s, reused %s of %s tokens"
          % (time.time() - t0, tm.get("cache_n"),
             (tm.get("cache_n") or 0) + (tm.get("prompt_n") or 0)))

    sid = answer.get("id_slot")
    if sid is None:
        raise SystemExit("/completion named no id_slot")

    def held(slot_id):
        return next((x.get("n_prompt_tokens") for x in req("/slots")
                     if x["id"] == slot_id), None)

    have = held(sid)
    print("  slot %d holds %s tokens (prefix: %d)" % (sid, have, n_tok))
    if have is not None and have > n_tok + 2:
        # The slot already carried a whole conversation and /completion only
        # appended to it. Saving now writes the SESSION, not the prefix —
        # and restoring that later puts a foreign user turn in front of the
        # model, which then answers the old question with no error anywhere.
        # Reported from a second machine on 26.08. as "another session's
        # answer, verbatim". This used to print a warning and save anyway.
        print("  slot holds more than the prefix (%s > %d) — erasing it and"
              % (have, n_tok))
        print("  recomputing, so the state saved is the prefix and nothing else")
        req("/slots/%d?action=erase" % sid, {}, "POST", 300)
        answer = req("/completion", {"prompt": prefix, "n_predict": 1,
                                      "cache_prompt": True})
        sid = answer.get("id_slot", sid)
        have = held(sid)
        print("  slot %d now holds %s tokens" % (sid, have))
        if have is not None and have > n_tok + 2:
            raise SystemExit(
                "refusing to save: slot %d still holds %s tokens against a "
                "prefix of %d. A state that carries a session answers the "
                "SESSION's question after a restore, silently."
                % (sid, have, n_tok))
    filename = "%s.bin" % a.name
    r = req("/slots/%d?action=save" % sid, {"filename": filename}, "POST", 900)
    print("saved: %s — %d tokens, %.0f MB, %.0f ms"
          % (filename, r["n_saved"], r["n_written"] / 1e6, r["timings"]["save_ms"]))

    # WHAT WAS WRITTEN HAS TO BE THE PREFIX, and this is the only moment it
    # can be checked cheaply. The slot was inspected BEFORE the save; the
    # checks above cannot see a request that took it in between, and on
    # 28.08.2026 two files in this store were written by exactly that race —
    # one holding 34 tokens, one holding 14957 tokens of something else.
    #
    # A wrong count is not "a bit off", it is WORTHLESS, and that is measured
    # rather than assumed: bench/reports/2026-08-29_restore-semantics. A
    # restored state is reused only where it is a PREFIX of the incoming
    # prompt — a state carrying anything beyond it is discarded WHOLE, not
    # trimmed back to the common part. Restoring a 14998-token state whose
    # first 14967 tokens were the prompt exactly still recomputed all 14969.
    #
    # So a file that does not carry exactly the prefix costs a full prefill on
    # every request that ever hits it, and buys nothing. It is deleted here
    # rather than published — the caller can save again, which costs seconds.
    n_saved = r.get("n_saved")
    if not isinstance(n_saved, int) or abs(n_saved - n_tok) > 2:
        try:
            os.remove(os.path.join(SLOT_PATH, filename))
        except OSError:
            pass
        raise SystemExit(
            "\nrefusing to publish %s: %s tokens were written where the prefix "
            "is %d.\n  Something took the slot during the save. Such a file is "
            "not slightly wrong,\n  it is unusable — a restored state is only "
            "reused where it is a PREFIX of\n  the request (measured: "
            "bench/reports/2026-08-29_restore-semantics).\n  The .bin has been "
            "deleted; nothing was written to the store."
            % (filename, n_saved, n_tok))

    os.makedirs(SLOT_PATH, exist_ok=True)
    passed_in = getattr(a, "gateway_id", None)
    gk = passed_in or gateway_id(body, dialect)
    with open(os.path.join(SLOT_PATH, "%s.json" % a.name), "w") as f:
        json.dump({"name": a.name, "render_id": h,
                   "gateway_id": gk,
                   "token": r["n_saved"],
                   "bytes": r["n_written"], "saved_at": time.strftime("%Y-%m-%d %H:%M")},
                  f, indent=2)
    print("sidecar: %s.json — gateway id %s (%s)"
          % (a.name, gk, "passed in" if passed_in else "computed from the body"))

def restore(a):
    if not wait_until_ready():
        raise SystemExit("server did not become ready in time")
    slots = req("/slots")
    free = [s["id"] for s in slots if not s.get("n_prompt_tokens")]
    # By LAST USED, not by name. Sorted alphabetically there was a prefix of
    # seven tokens ahead of two real projects here, and with only two slots it
    # would have evicted one of them. Files without a .bin drop out right
    # away instead of running into a failing restore.
    available = [d["name"] for d in
             sorted(_inventory(), key=lambda d: d["_sortkey"], reverse=True)
             if d["_present"]]
    if not available:
        print("nothing saved under %s" % SLOT_PATH); return
    # Remember n_free BEFORE the loop: free is emptied inside it, and the
    # closing message used to compute from the remainder. It therefore stayed
    # away as soon as len(available) <= len(slots) — skipped prefixes went
    # unnoticed, and the caller is ExecStartPost, where nobody reads along.
    n_free = len(free)
    print("%d saved prefixes, %d free of %d slots"
          % (len(available), n_free, len(slots)))
    restored = 0
    for name in available[:n_free]:
        sid = free.pop(0)
        t0 = time.time()
        try:
            r = req("/slots/%d?action=restore" % sid, {"filename": "%s.bin" % name},
                    "POST", 900)
            print("  slot %d <- %-16s %d tokens, %.0f ms"
                  % (sid, name, r["n_restored"], r["timings"]["restore_ms"]))
            restored += 1
        except Exception as e:
            print("  slot %d <- %-16s ERROR %s" % (sid, name, str(e)[:60]))
    left = available[n_free:]
    if left:
        print("  NOT restored (%d), only %d of %d slots were free: %s"
              % (len(left), n_free, len(slots), ", ".join(left)))
    print("  %d of %d restored" % (restored, len(available)))
    return restored

def _inventory():
    """Every saved prefix with its size and last use."""
    aus = []
    if not os.path.isdir(SLOT_PATH):
        return aus
    for f in sorted(os.listdir(SLOT_PATH)):
        if not f.endswith(".json"):
            continue
        # An unreadable sidecar file must not break through here: restore()
        # runs as ExecStartPost of llama-user@.service, and a failure there
        # holds up the model server at start. Better to skip this one prefix
        # and say so.
        try:
            with open(os.path.join(SLOT_PATH, f), encoding="utf-8") as fh:
                d = json.load(fh)
            name = d["name"]
        except Exception as e:
            print("  %-16s skipped, sidecar unreadable: %s"
                  % (f, str(e)[:60]))
            continue
        b = os.path.join(SLOT_PATH, name + ".bin")
        # A quarantined prefix keeps its bytes under another name. It is NOT
        # `_present` — it cannot be restored — but it very much occupies the
        # disk this tool prunes, and the cleanup that could not see it left a
        # gigabyte behind while deleting the sidecar that said why.
        q = b + ".unusable"
        d["_unusable"] = os.path.exists(q)
        d["_bytes_disk"] = (os.path.getsize(b) if os.path.exists(b)
                            else os.path.getsize(q) if d["_unusable"] else 0)
        d["_present"] = os.path.exists(b)
        # Without a recorded use, fall back to the time it was saved.
        # Read the pre-rename German keys too — sidecar files written before
        # August 2026 still carry them, and 628 MB per prefix is too much to
        # orphan over a spelling.
        d.setdefault("gateway_id", d.get("gateway_kennung"))
        d.setdefault("saved_at", d.get("zeitpunkt"))
        d.setdefault("last_used", d.get("zuletzt_benutzt"))
        d.setdefault("used_total", d.get("benutzt_gesamt", 0))
        d["_sortkey"] = d.get("last_used") or d.get("saved_at") or ""
        aus.append(d)
    return aus

def _days_ago(stamp):
    try:
        return (time.time() - time.mktime(time.strptime(stamp, "%Y-%m-%d %H:%M"))) / 86400
    except Exception:
        return 1e9

# The three limits, with the name a user types. Kept as data because the
# refusal below has to name the flag, not the attribute.
LIMITS = (("--max-gb", "max_gb"), ("--max-count", "max_count"),
          ("--ttl-days", "ttl_days"))


def check_limits(a):
    """Refuse a limit that could mean two things, rather than picking one.

    `--max-gb 0` in a tool whose job is deleting reads as "keep nothing". This
    code read it as "no limit", because `if a.max_gb:` is false at zero — so
    the command ran, printed "nothing to delete", and did nothing at all. That
    happened on 28.08.2026: the instruction was given, executed twice, and the
    9 GB it was supposed to free were still there. Nothing failed. Nothing
    said anything.

    Both readings are defensible, which is exactly why neither may be chosen
    silently. Same rule as budget._num, one directory over: a number that
    cannot be trusted is refused rather than fallen back from, "because
    silently falling back would hide a mistake in the one place where a number
    is being trusted instead of measured".

    So: omit the flag for no limit, pass --purge to keep nothing, and 0 is an
    error that says both.
    """
    if getattr(a, "purge", False):
        named = [flag for flag, attr in LIMITS if getattr(a, attr, None) is not None]
        if named:
            raise SystemExit(
                "\n--purge means keep NOTHING, %s means keep that much.\n"
                "  Both cannot be meant. Drop one." % ", ".join(named))
    for flag, attr in LIMITS:
        v = getattr(a, attr, None)
        if v is None:
            continue
        if v < 0:
            raise SystemExit("\n%s cannot be negative (%s given)." % (flag, v))
        if v == 0:
            raise SystemExit(
                "\n%s 0 is ambiguous and will not be guessed at.\n"
                "  It could mean KEEP NOTHING     -> use --purge\n"
                "  or it could mean NO LIMIT      -> omit %s\n"
                "  Until 28.08.2026 this was read as 'no limit' and deleted "
                "nothing, silently." % (flag, flag))


def cleanup(a):
    """Delete by rules — the longest UNUSED first.

    Why not by file age: a prefix never changes through use, so the file date
    only says when it was saved. A project used daily would carry the same
    date as a forgotten one.
    """
    check_limits(a)
    inv = _inventory()
    if not inv:
        print("nothing under %s" % SLOT_PATH); return
    doomed = []

    # QUARANTINED FIRST, and under no rule at all. The gateway sets a prefix
    # aside when a restore of it demonstrably carried nothing (see
    # saved-prefix-holds-a-foreign-state): the file cannot be restored, cannot
    # become useful again, and is about a gigabyte. Keeping it until an LRU
    # limit happens to reach it would be keeping rubbish by seniority. The
    # sidecar goes with it — its reason has been read by then or never will be.
    for d in inv:
        if d.get("_unusable"):
            doomed.append((d, "quarantined — the restore carried nothing"))

    if getattr(a, "purge", False):
        doomed = [(d, "purge") for d in inv]

    if a.ttl_days:
        for d in inv:
            if _days_ago(d["_sortkey"]) > a.ttl_days:
                doomed.append((d, "unused for %.0f days" % _days_ago(d["_sortkey"])))

    keep = [d for d in inv if d not in [w[0] for w in doomed]]
    keep.sort(key=lambda d: d["_sortkey"], reverse=True)   # last used first

    if a.max_count and len(keep) > a.max_count:
        for d in keep[a.max_count:]:
            doomed.append((d, "over the limit of %d" % a.max_count))
        keep = keep[:a.max_count]

    if a.max_gb:
        limit = a.max_gb * 1e9
        total = 0
        kept = []
        for d in keep:
            if total + d["_bytes_disk"] > limit:
                doomed.append((d, "over %g GB" % a.max_gb))
            else:
                total += d["_bytes_disk"]; kept.append(d)
        keep = kept

    if not doomed:
        print("nothing to delete (%d prefixes, %.1f GB)"
              % (len(inv), sum(d["_bytes_disk"] for d in inv) / 1e9))
        return
    free = sum(d["_bytes_disk"] for d, _ in doomed)
    for d, reason in doomed:
        print("  %s %-16s %5.0f MB  %s"
              % ("deleting " if not a.dry_run else "would go:", d["name"],
                 d["_bytes_disk"] / 1e6, reason))
        if not a.dry_run:
            for e in (".bin", ".bin.unusable", ".json"):
                try: os.remove(os.path.join(SLOT_PATH, d["name"] + e))
                except FileNotFoundError: pass
    print("  %s %.1f GB, %d prefixes remain"
          % ("freed:" if not a.dry_run else "would free:", free / 1e9, len(keep)))

def list_saved(a):
    """One line per saved prefix.

    Goes through _inventory() rather than reading the files itself — that is
    the one place that knows about the pre-rename German keys, and a second
    reader would drift away from it.
    """
    inv = _inventory()
    if not inv:
        print("nothing under %s" % SLOT_PATH); return
    for d in inv:
        lu = d.get("last_used")
        print("  %-16s %6d tokens  %5.0f MB  saved %s  %s%s"
              % (d["name"], d.get("token", 0), d.get("bytes", 0) / 1e6,
                 d.get("saved_at") or "?",
                 ("last used %s (%dx)" % (lu, d.get("used_total", 0)))
                 if lu else "never used",
                 "" if d["_present"] else "  <== .bin missing"))
    print("  %.1f GB in total under %s"
          % (sum(d["_bytes_disk"] for d in inv) / 1e9, SLOT_PATH))

def check(a):
    """Check whether the sidecar files match the id the gateway will look
    them up by.

    Background: until August 2026 save() recomputed the gateway_id from the
    body the gateway handed it — and that body is already corrected by then.
    The key therefore matched no incoming request: automatically saved
    prefixes were never restored and were saved again on every cold start.
    This is detectable without the original body, because the gateway names
    its files after the id: if a file is named like an id, gateway_id has to
    be exactly that.
    """
    inv = _inventory()
    if not inv:
        print("nothing under %s" % SLOT_PATH); return 0
    affected, missing = [], []
    for d in inv:
        if not d["_present"]:
            missing.append(d)
        if re.fullmatch(r"[0-9a-f]{12}", d["name"] or "") \
                and d.get("gateway_id") != d["name"]:
            affected.append(d)
    for d in missing:
        print("  %-16s .bin missing — sidecar without content" % d["name"])
    for d in affected:
        print("  %-16s gateway_id=%s, should be %s"
              % (d["name"], d.get("gateway_id"), d["name"]))
        if a.repair:
            path = os.path.join(SLOT_PATH, d["name"] + ".json")
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
            raw["gateway_id"] = d["name"]
            raw["repariert"] = time.strftime("%Y-%m-%d %H:%M")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(raw, f, indent=2, ensure_ascii=False)
            print("      repaired")
    if not affected and not missing:
        print("  all %d sidecar files are fine" % len(inv))
    elif not a.repair and affected:
        print("  %d affected — 'prewarm.py check --repair' puts the id right, "
              "then restart llm-gateway" % len(affected))
    return len(affected)

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="befehl", required=True)
    s1 = sub.add_parser("save")
    s1.add_argument("--body", required=True)
    s1.add_argument("--name", required=True)
    # The gateway passes its own id through instead of having it recomputed
    # here — its body is already corrected. See gateway_id().

    s1.add_argument("--gateway-id", default=None, dest="gateway_id",
                    help="take the gateway's id instead of recomputing it")
    # NO `type=` HERE. argparse applies type conversion BEFORE checking
    # choices, so `type=lambda v: v == "1"` turned "0" into False and then
    # rejected False as an invalid choice — every automatic save failed with a
    # usage message, live, until it was noticed 30.08.2026. The string is
    # converted where it is used instead.
    s1.add_argument("--hoist", default=None, choices=("0", "1"),
                    help="hoist mid-conversation system messages to the front, "
                         "as the gateway's HOIST_SYSTEM does. It MUST match, or "
                         "this writes a prefix no request ever sends. Default: "
                         "read HOIST_SYSTEM from the environment.")
    s1.add_argument("--dialect", default=DIA.ANTHROPIC,
                    choices=(DIA.ANTHROPIC, DIA.OPENAI),
                    help="which shape the body has (the gateway passes the "
                         "dialect of the endpoint the request came in on)")
    s1.set_defaults(fn=save)
    s2 = sub.add_parser("restore");  s2.set_defaults(fn=restore)
    s3 = sub.add_parser("list");    s3.set_defaults(fn=list_saved)
    s4 = sub.add_parser("cleanup", help="delete by size, count or age")
    s4.add_argument("--purge", action="store_true", dest="purge",
                    help="delete every saved prefix. The explicit way to say "
                         "'keep nothing' — a limit of 0 is refused instead of "
                         "guessed at. Works with --dry-run.")
    s4.add_argument("--max-gb",     type=float, default=None, dest="max_gb")
    s4.add_argument("--max-count", type=int,   default=None, dest="max_count")
    s4.add_argument("--ttl-days",  type=int,   default=None, dest="ttl_days")
    s4.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="only show what would go, delete nothing")
    s4.set_defaults(fn=cleanup)
    s5 = sub.add_parser("check", help="check sidecar files against the id")
    s5.add_argument("--repair", action="store_true",
                    help="put a wrong gateway_id right")
    s5.set_defaults(fn=check)
    return ap.parse_args(argv)

def main(argv=None):
    a = parse_args(argv)
    return a.fn(a)

if __name__ == "__main__":
    main()
