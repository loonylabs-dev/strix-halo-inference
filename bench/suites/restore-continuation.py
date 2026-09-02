#!/usr/bin/env python3
"""Does a restored slot state serve the CONTINUATION of its own conversation?

    mkdir -p /tmp/restore-cont
    python3 bench/sideserver.py --env setup/env/flashnext.env --port 8081 \\
        --stop "llama-user@$(bash setup/lib/models.sh serving)" \\
        --extra "--slot-save-path /tmp/restore-cont/ -cram 0" -- \\
        python3 bench/suites/restore-continuation.py --url http://127.0.0.1:8081

The unit to stop is DERIVED, not written down: a hard-wired name would stop
the wrong unit after a model switch, and with `Conflicts=` that silently swaps
the serving model (review, 01.09.2026). This is a production interruption —
it needs the operator's go, and production comes back on teardown.

THE QUESTION, and why it is not the one already answered.

bench/reports/2026-08-29_restore-semantics/ measured whether a slot saved AS
IT STANDS after an answer can serve the next request, and concluded:

  > A restored state is only reused when it is a PREFIX of the incoming
  > prompt. A state carrying anything beyond that — even the question it was
  > saved with — is discarded whole, not trimmed back to the common part.

That verdict is not disputed here. What is disputed is that the experiment
ever tested the shape a Claude Code session has. Read its own scripts:
`postanswer.py` saves a state holding `prefix+Q1+A1` and then sends
`SYN.body(question=Q2)` — a FRESH body carrying a different first question.
The saved state is not a prefix of that prompt; it diverges at Q1. The same
holds for `turnproof.py`, whose three turns are three separate one-question
bodies. Both cells are the report's own control B seen from the wrong side.

A session does something else. Turn N+1 is turn N plus the model's answer plus
a new user message — the saved state IS a true prefix of it, which is exactly
the condition the report says has to hold. So the case that decides whether a
session can be persisted at all has never been run.

WHAT COULD MAKE THIS MEASUREMENT LIE, and what stops it

  the RAM prompt cache answers instead of the file
      A reuse after a restore proves nothing while `-cram` is non-zero: the
      state may have come back from llama.cpp's own cache, which the restore
      never touched. Run with `-cram 0`; passed through sideserver's --extra
      it lands after the profile's own value and wins, because arg.cpp
      assigns `cache_ram_mib` rather than accumulating it (unlike
      `--spec-type`, which does accumulate — HANDOVER, 01.09.2026).
      `/props` does not report the effective value, so the flag is not read
      back. It does not have to be: with a cache in play cell 4 comes back
      warm and the run aborts there.

  the displacement does not displace
      Cell 4 sends the continuation with NO restore in front of it. If that
      cell is already warm, nothing was displaced and every later number is
      meaningless. The run aborts there rather than reporting.

  the instrument cannot see "warm" at all
      Cell 8 is an ordinary follow-up turn. If it does not come back warm,
      a zero in cell 7 says nothing about restores — see bench/README, "a
      comparison that measures nothing in both arms".

  the build changed underneath the old verdict
      Cell 9 rebuilds the 29.08. shape on today's binary. It must come back
      cold. If it does not, the prefix rule itself has moved and cell 7 is
      good for a reason this suite did not test.

Written 02.09.2026, not yet run: flashnext serves without `--slot-save-path`,
so every slots action answers "This server does not support slots action".
"""
import argparse, json, os, sys, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "setup", "gateway"))
import synthetic as SYN                                          # noqa: E402
import dialects as DIA                                           # noqa: E402

STATE = "restore-continuation.bin"


def post(url, payload, timeout=1800):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def get(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class Run:
    """The server under test, plus the two numbers every cell is judged on."""

    def __init__(self, url, tools, mid_system):
        self.url, self.tools, self.mid_system = url, tools, mid_system
        self.rows = []

    def ask(self, body, tag, max_tokens=24, fatal=True):
        """Send one request; return (seconds, reused, computed, answer text).

        `reused` is llama.cpp's own count of what it did NOT recompute. It is
        read from the answer rather than from a log line: the 29.08. report
        found `f_sim_best = 0.997` on a request that then recomputed all
        14,969 tokens, so the slot-selection log and the reuse are not the
        same mechanism and only the second one is the subject here.

        `fatal=False` turns a failed cell into a recorded result instead of
        the end of the run — bench/suites/restore-safety.py lost its last two
        cells to a propagating timeout three times, and every one of those
        reports looked complete. Cells that others depend on stay fatal;
        cells that only carry their own answer do not.
        """
        b = json.loads(json.dumps(body))          # never mutate the caller's
        b["model"] = "local"
        b["max_tokens"] = max_tokens
        b["stream"] = False
        b["temperature"] = 0
        if self.mid_system:
            # Some templates reject a system message that is not at position 0
            # with HTTP 500 (Qwen 3.8, 24.08.2026). The gateway rewrites those
            # into user text blocks; this does the same, so the suite measures
            # the cache and not a template's opinion. Length and position are
            # preserved, so the prefix identity is untouched.
            b, _ = DIA.mid_system_to_user(b, DIA.ANTHROPIC)
        t0 = time.time()
        try:
            d = post(self.url + "/v1/messages", b)
        except Exception as e:
            detail = e.read().decode("utf-8")[:300] if hasattr(e, "read") else str(e)
            msg = "  %-30s FAILED: %s" % (tag, detail)
            self.rows.append({"cell": tag, "failed": detail})
            if fatal:
                raise SystemExit(msg + "\n  (a cell later cells depend on)")
            print(msg, flush=True)
            return None, None, None, "", None
        secs = time.time() - t0
        u = d.get("usage") or {}
        reused = u.get("cache_read_input_tokens")
        computed = u.get("input_tokens")
        text = " ".join(c.get("text", "") for c in d.get("content", [])
                        if isinstance(c, dict))
        print("  %-30s %7.1f s   reused=%-7s computed=%-7s"
              % (tag, secs, reused, computed), flush=True)
        self.rows.append({"cell": tag, "s": round(secs, 1),
                          "reused": reused, "computed": computed})
        return secs, (reused or 0), (computed or 0), text, d

    # ---- the conversation, with the model's REAL answers written back ----
    def conversation(self, project, turns):
        """A body whose assistant turns are what the model actually said.

        synthetic.py's own `turns=` builds a synthetic assistant turn
        ("Let me look." plus a tool call). That is fine for shaping a prompt
        and wrong for this measurement: the slot after an answer holds the
        GENERATED tokens, so a body carrying anything else is not a
        continuation of the state and the test would fail for a reason that
        has nothing to do with restores.
        """
        body = SYN.body(project=project, n_tools=self.tools,
                        question="Reply with the single word: start.")
        for i in range(turns):
            _, _, computed, text, _ = self.ask(body, "build turn %d" % (i + 1))
            if i == 0:
                # The cold first turn computes the whole prompt, which at this
                # depth is almost entirely head (system + tools). Control 9
                # needs that number: a state "trimmed back to the common part"
                # would return roughly this much, and a state discarded whole
                # returns nothing. Without it the two are one bucket.
                self.head_tokens = computed
            body["messages"] = body["messages"] + [
                {"role": "assistant", "content": [
                    {"type": "text", "text": text}]},
                {"role": "user", "content": [
                    {"type": "text", "text":
                     "Continue. Reply with the single word: step%d." % (i + 1)}]},
            ]
        return body

    def slots(self):
        try:
            return [(s.get("id"), s.get("n_prompt_tokens"))
                    for s in get(self.url + "/slots")]
        except Exception as e:                              # /slots may be off
            return [("?", "unreadable: %s" % e)]

    def save(self, tag="save"):
        t0 = time.time()
        try:
            d = post(self.url + "/slots/0?action=save", {"filename": STATE})
        except Exception as e:
            detail = e.read().decode("utf-8")[:300] if hasattr(e, "read") else str(e)
            raise SystemExit(
                "  %s FAILED: %s\n\n  If this says 'does not support slots "
                "action', the server was started without --slot-save-path. "
                "That is the state production is in (flashnext.env, 02.09.), "
                "so it has to be passed to the side server explicitly."
                % (tag, detail))
        wall = time.time() - t0
        print("  %-30s %7.1f s   n_saved=%s  %.0f MB  server=%.0f ms"
              % (tag, wall, d.get("n_saved"), (d.get("n_written") or 0) / 1e6,
                 (d.get("timings") or {}).get("save_ms", 0)), flush=True)
        self.rows.append({"cell": tag, "s": round(wall, 1),
                          "n_saved": d.get("n_saved"),
                          "n_written": d.get("n_written"),
                          "save_ms": (d.get("timings") or {}).get("save_ms")})
        return d

    def restore(self, tag="restore"):
        t0 = time.time()
        d = post(self.url + "/slots/0?action=restore", {"filename": STATE})
        wall = time.time() - t0
        print("  %-30s %7.1f s   n_restored=%s  %.0f MB  server=%.0f ms"
              % (tag, wall, d.get("n_restored"), (d.get("n_read") or 0) / 1e6,
                 (d.get("timings") or {}).get("restore_ms", 0)), flush=True)
        self.rows.append({"cell": tag, "s": round(wall, 1),
                          "n_restored": d.get("n_restored"),
                          "restore_ms": (d.get("timings") or {}).get("restore_ms")})
        return d


def follow_up(body, text, question):
    """Turn N+1: the state's own conversation plus one more user message."""
    return dict(body, messages=body["messages"] + [
        {"role": "assistant", "content": [{"type": "text", "text": text}]},
        {"role": "user", "content": [{"type": "text", "text": question}]},
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081",
                    help="the SIDE server, not production")
    ap.add_argument("--tools", type=int, default=24)
    ap.add_argument("--turns", type=int, default=3,
                    help="conversation depth before the save")
    ap.add_argument("--no-mid-system", action="store_true",
                    help="do not rewrite mid-conversation system messages")
    ap.add_argument("--out", default=None, help="write the rows as JSON")
    a = ap.parse_args()

    if "8080" in a.url:
        raise SystemExit("refusing: 8080 is production. Start a side server "
                         "with bench/sideserver.py and pass its port.")

    print(__doc__.split("THE QUESTION")[0].strip(), "\n")
    print("REQUIRES: the server started with `--slot-save-path <dir>` AND")
    print("          `-cram 0`. Without the first, every slots action errors;")
    print("          without the second, a warm cell 7 proves nothing.\n")

    R = Run(a.url, a.tools, not a.no_mid_system)

    print("=== 1. build the conversation, %d turns, real answers written back"
          % a.turns)
    body = R.conversation("/tmp/restore-cont", a.turns)
    _, _, _, last, _ = R.ask(body, "settle (state is now S)")
    print("     slots:", R.slots(), flush=True)

    print("=== 2. save the slot as it stands")
    saved = R.save()
    n_saved = saved.get("n_saved") or 0

    other = SYN.body(project="/tmp/restore-cont-other", n_tools=a.tools,
                     question="Reply with the single word: other.")

    print("=== 3. displace it")
    R.ask(other, "displacing request")
    print("     slots:", R.slots(), flush=True)

    print("=== 4. CONTROL, cold: the continuation with NO restore in front")
    c1 = follow_up(body, last, "Continue. Reply with the single word: c-one.")
    _, reused_cold, _, _, _ = R.ask(c1, "continuation, no restore")
    if reused_cold > n_saved * 0.5:
        raise SystemExit(
            "\nABORTED — the displacement did not displace: this cell came "
            "back warm (reused=%d of a %d-token state), so a warm cell 7 "
            "would prove nothing. Check that -cram is 0 and that cell 3 "
            "really took the slot." % (reused_cold, n_saved))

    print("=== 5. displace again")
    R.ask(other, "displacing request")

    print("=== 6. restore S")
    got = R.restore()
    n_restored = got.get("n_restored") or 0
    print("     slots:", R.slots(), flush=True)

    print("=== 7. THE MEASUREMENT: the continuation, after the restore")
    c2 = follow_up(body, last, "Continue. Reply with the single word: c-two.")
    _, reused_hot, _, hot_text, _ = R.ask(c2, "continuation, restored",
                                          fatal=False)

    print("=== 8. CONTROL, warm: an ordinary follow-up, nothing in between")
    c3 = follow_up(c2, hot_text, "Continue. Reply with the single word: c-three.")
    _, reused_warm, _, _, _ = R.ask(c3, "follow-up, warm", fatal=False)

    print("=== 9. CONTROL, the 29.08. shape on today's build")
    R.ask(other, "displacing request", fatal=False)
    R.restore("restore (for the old shape)")
    old = SYN.body(project="/tmp/restore-cont", n_tools=a.tools,
                   question="Reply with the single word: different.")
    _, reused_old, _, _, _ = R.ask(old, "fresh body, other question",
                                   fatal=False)

    print("\n=== verdict")
    head = getattr(R, "head_tokens", 0) or 0
    if reused_hot is None or reused_warm is None:
        print("  a cell did not return — the verdict below is only as good as")
        print("  the cells that ran. Missing cells read as 0, which is NOT")
        print("  the same as a measured zero. See the rows.")
    reused_hot = reused_hot or 0
    reused_warm = reused_warm or 0
    reused_old = reused_old or 0
    carries = reused_hot > n_restored * 0.5
    instrument_ok = reused_warm > n_restored * 0.5
    # Three outcomes, not two. `discarded whole` is the 29.08. rule; `trimmed`
    # would mean the rule has softened to the common prefix, which is a
    # different world and must not be read as the old one.
    trimmed = head and reused_old > head * 0.5
    old_rule_holds = not trimmed and reused_old < max(head, 1) * 0.5
    print("  restore carried the continuation : %s  (reused %d of %d restored)"
          % ("YES" if carries else "NO", reused_hot, n_restored))
    print("  instrument can see warm          : %s  (control 8 reused %d)"
          % ("YES" if instrument_ok else "NO", reused_warm))
    print("  29.08. prefix rule reproduced    : %s  (control 9 reused %d, "
          "head is %d)"
          % ("YES" if old_rule_holds else ("NO, TRIMMED" if trimmed else "NO"),
             reused_old, head))
    if not instrument_ok:
        print("\n  INADMISSIBLE: control 8 did not come back warm, so nothing "
              "here separates 'the restore failed' from 'the run measured "
              "nothing'. Do not report cell 7.")
    elif carries and old_rule_holds:
        print("\n  The prefix rule holds AND a session continuation satisfies "
              "it. Session persistence is worth measuring further — next is "
              "the save duration over depth, which decides the cooldown.")
    elif not carries and old_rule_holds:
        print("\n  The rule is stricter than the 29.08. report states: even a "
              "true prefix is discarded. Session persistence is dead on this "
              "build; nothing downstream needs building.")
    else:
        print("\n  Control 9 did not reproduce the old rule, so the restore "
              "semantics moved with the build. Cell 7 may be good for a "
              "reason this suite did not test — re-read before believing it.")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"url": a.url, "tools": a.tools, "turns": a.turns,
                       "n_saved": n_saved, "n_restored": n_restored,
                       "rows": R.rows}, f, indent=1)
        print("\n  rows -> %s" % a.out)


if __name__ == "__main__":
    main()
