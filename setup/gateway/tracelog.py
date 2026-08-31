#!/usr/bin/env python3
"""tracelog — what went in, what came back, and what the cache really did.

    from tracelog import Trace
    T = Trace()                                   # off unless switched on
    T.record("request",
             summary={"prefix": ident, "cold": cold},
             detail={"tools": 24, "head_chars": 70892},
             text={"system": head, "answer": answer})

One JSON line per event, one file per day. Which of the three field groups
reach the file is decided by the LEVEL, and the level is read from a control
file so it can be changed while the gateway is serving — the moment something
looks odd is the moment you want it on, and a restart resets the very
bookkeeping you are trying to observe.

    off       nothing at all. The default, and what a fresh install has.
    summary   numbers and ids. ~300 bytes a request.
    detail    plus hashes, lengths, timings, tool names. ~1 KB.
    text      plus the actual prompt and answer. ~70 KB.

WHY `text` IS ARMED SEPARATELY AND EXPIRES BY ITSELF. It writes complete
prompts to disk in the clear. docs/SECURITY.md calls the /slots prompt
exposure the worst finding of this project; this is the same material, only
persistent. So it cannot be reached by the normal switch, it carries an
expiry, and the file says in its own first line what it contains.

The directory is 0700 and every file 0600, outside the repo, capped, and
oldest-first pruned. Nothing here belongs in git, and nothing here should
outlive the debugging session it was written for.
"""
import json, os, time

# NOT called `trace`: that is a standard library module, and a file of that
# name beside a script hides it for every import in the same directory —
# tests/test_scripts.py catches exactly this, and caught this.
LEVELS = ("off", "summary", "detail", "text")
DEFAULT_DIR = os.path.expanduser("~/.cache/llm-gateway-trace")
# Total across all days. A debugging session is megabytes; a month of `text`
# would be gigabytes, and the point at which somebody notices is the point at
# which the disk is full.
DEFAULT_CAP_BYTES = 200 * 1024 * 1024
CONTROL = "level"


def _rank(level):
    try:
        return LEVELS.index(level)
    except ValueError:
        return 0


class Trace:
    """Writes if switched on, and is cheap when it is not.

    The level is re-read from the control file when that file changes, so a
    `tools/tracelog.py on detail` takes effect on the next request without
    touching the service. The check is a stat(), not a read.
    """

    def __init__(self, directory=None, cap_bytes=DEFAULT_CAP_BYTES, now=time.time):
        self.dir = directory or os.environ.get("TRACE_DIR") or DEFAULT_DIR
        self.cap = cap_bytes
        self._now = now
        self._level = "off"
        self._expires = 0.0
        self._seen = None            # mtime of the control file
        self._full = False           # cap reached, said once
        self.refresh()

    # --- the switch --------------------------------------------------------
    def _control_path(self):
        return os.path.join(self.dir, CONTROL)

    def refresh(self):
        """Re-read the level if the control file changed. Called per event."""
        try:
            state = os.stat(self._control_path()).st_mtime_ns
        except OSError:
            self._level, self._expires, self._seen = "off", 0.0, None
            return self._level
        if state != self._seen:
            self._seen = state
            self._level, self._expires = self._read_control()
        # `text` gives itself back. An operator who forgets is the normal
        # case, not the exception, and the cost of forgetting is a disk full
        # of prompts.
        if self._level == "text" and self._expires and self._now() > self._expires:
            self._level = "detail"
        return self._level

    def _read_control(self):
        try:
            with open(self._control_path(), encoding="utf-8") as f:
                d = json.load(f)
            level = d.get("level", "off")
            return (level if level in LEVELS else "off"), float(d.get("expires", 0) or 0)
        except Exception:
            return "off", 0.0

    def set_level(self, level, minutes=None):
        """Write the control file. `minutes` arms an expiry — required for
        `text`, which must not be able to run for a week by accident."""
        if level not in LEVELS:
            raise ValueError("unknown level %r, one of %s" % (level, ", ".join(LEVELS)))
        if level == "text" and not minutes:
            minutes = 60
        os.makedirs(self.dir, mode=0o700, exist_ok=True)
        try:
            os.chmod(self.dir, 0o700)
        except OSError:
            pass
        payload = {"level": level,
                   "expires": (self._now() + minutes * 60) if minutes else 0,
                   "set_at": time.strftime("%Y-%m-%d %H:%M")}
        path = self._control_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.chmod(path, 0o600)
        self._seen = None
        return self.refresh()

    @property
    def level(self):
        return self._level

    def on(self, level="summary"):
        return _rank(self._level) >= _rank(level) and self._level != "off"

    # --- the writing -------------------------------------------------------
    def path_for_today(self):
        return os.path.join(self.dir, "trace-%s.jsonl"
                            % time.strftime("%Y-%m-%d", time.localtime(self._now())))

    def _total_bytes(self):
        total = 0
        for f in os.listdir(self.dir):
            if f.startswith("trace-") and f.endswith(".jsonl"):
                try:
                    total += os.path.getsize(os.path.join(self.dir, f))
                except OSError:
                    pass
        return total

    def _prune(self):
        """Oldest day first, until under the cap. Returns what it removed."""
        files = sorted(f for f in os.listdir(self.dir)
                       if f.startswith("trace-") and f.endswith(".jsonl"))
        gone = []
        while len(files) > 1 and self._total_bytes() > self.cap:
            victim = files.pop(0)
            try:
                os.remove(os.path.join(self.dir, victim))
                gone.append(victim)
            except OSError:
                break
        return gone

    def record(self, kind, summary=None, detail=None, text=None):
        """One event. Returns the line written, or None when off or capped."""
        if self.refresh() == "off":
            return None
        rec = {"t": round(self._now(), 3), "kind": kind}
        rec.update(summary or {})
        if _rank(self._level) >= _rank("detail"):
            rec.update(detail or {})
        if _rank(self._level) >= _rank("text"):
            rec.update(text or {})
        line = json.dumps(rec, ensure_ascii=False, default=str)
        try:
            os.makedirs(self.dir, mode=0o700, exist_ok=True)
            path = self.path_for_today()
            fresh = not os.path.exists(path)
            if self._total_bytes() > self.cap:
                self._prune()
                if self._total_bytes() > self.cap:
                    if not self._full:
                        self._full = True
                    return None
            self._full = False
            with open(path, "a", encoding="utf-8") as f:
                if fresh:
                    f.write(json.dumps({
                        "t": round(self._now(), 3), "kind": "header",
                        "note": "llm-gateway trace. `text` records contain "
                                "COMPLETE PROMPTS in the clear — treat this "
                                "file like the conversations it describes.",
                        "level": self._level}) + "\n")
                f.write(line + "\n")
            if fresh:
                os.chmod(path, 0o600)
        except Exception:
            # A trace that breaks the thing it observes is worse than no
            # trace. Nothing here may reach the request path.
            return None
        return line
