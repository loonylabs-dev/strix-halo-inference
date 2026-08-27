#!/usr/bin/env python3
"""tasklib — the task battery for the model decision, and its checkers.

Why this exists: t/s numbers alone cannot decide between models. A model that
decodes at 30 t/s but thinks for 4,000 tokens loses against one that decodes
at 22 t/s and answers directly. The metric that decides is **seconds until a
verified-correct answer** — so every task here carries a checker that says
pass or fail without a human in the loop.

The battery is deliberately mixed so that the speculative-decoding question
gets an honest answer. MTP/ngram speculation thrives on repetitive output
(file rewrites, counting, agent loops) and does nothing for novel prose. A
battery of only rewrites would flatter it. The free-prose end of that range
moved to bench/speed.py on 27.08. as the `prose` workload, where it belongs:
it is a property of a CONFIGURATION, not a judgement of a model, and this
file's last prose task carried a quality CHECKER into a repo that stopped
judging model quality on 26.08. Tags:

    code-novel      write new code             — speculation helps a little
    code-repetitive rewrite given code         — speculation's best case
    struct          JSON / SQL / regex         — short, correctness-focused
    longctx         retrieval at ~14k tokens   — measures prefill at depth
    agentic         multi-turn edit loop       — warm cache, the CC workload

Import stays without consequences (tests/common.py loads by path). Checkers
that execute model code do so in a subprocess with a timeout — this runs
generated code on the local machine, which is acceptable for a benchmark on
the operator's own box, and nowhere else.

The checker contract: checker(text, task) -> (passed: bool, reason: str).
`text` is the assistant's visible content, thinking already stripped.
"""
import json, os, re, sqlite3, subprocess, sys, tempfile

# ---------------------------------------------------------------- extraction

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

def strip_think(text):
    """Remove inline <think> blocks; returns (clean_text, thought_chars).

    Most templates put thinking into `reasoning_content`, but a template
    without a reasoning parser leaves it inline — a checker would then judge
    the thoughts, and every task would fail on a formality.
    """
    if not text:
        return "", 0
    thought = sum(len(m.group(0)) for m in THINK_RE.finditer(text))
    return THINK_RE.sub("", text).strip(), thought

def extract_block(text, lang=None):
    """The LAST fenced code block, or None.

    The last one, not the first: models that think aloud often show a draft
    first and the final version at the end. Judging the draft would punish
    exactly the careful answers.
    """
    fence = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.DOTALL)
    hits = [(m.group(1).lower(), m.group(2)) for m in fence.finditer(text or "")]
    if lang:
        typed = [b for l, b in hits if l == lang]
        if typed:
            return typed[-1].strip()
    if hits:
        return hits[-1][1].strip()
    return None

def extract_code(text):
    """Python source out of an answer: last ```python block, else the whole
    text if it happens to compile (some models answer with bare code)."""
    block = extract_block(text, "python")
    if block is not None:
        return block
    t = (text or "").strip()
    try:
        compile(t, "<answer>", "exec")
        return t
    except SyntaxError:
        return None

def extract_json(text):
    """A JSON object out of an answer: fenced block first, then the outermost
    {...} span. Returns the parsed object or None."""
    for candidate in (extract_block(text, "json"), text):
        if not candidate:
            continue
        s = candidate.strip()
        a, b = s.find("{"), s.rfind("}")
        if a < 0 or b <= a:
            continue
        try:
            return json.loads(s[a:b + 1])
        except ValueError:
            continue
    return None

def extract_sql(text):
    block = extract_block(text, "sql") or extract_block(text)
    s = (block if block is not None else (text or "")).strip().rstrip(";").strip()
    return s or None

# ---------------------------------------------------- running generated code

def run_python(code, test_code, timeout=20):
    """Execute model code together with a test suite in a subprocess.

    Returns (passed, reason). The tests are plain asserts in __main__ — no
    unittest discovery, so a syntax error in the model code is reported as
    exactly that and not as '0 tests ran'.
    """
    with tempfile.TemporaryDirectory(prefix="bench-task-") as d:
        path = os.path.join(d, "candidate.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code + "\n\n# ---- checker-owned tests ----\n" + test_code)
        try:
            r = subprocess.run([sys.executable, path], capture_output=True,
                               text=True, timeout=timeout, cwd=d)
        except subprocess.TimeoutExpired:
            return False, "test run exceeded %ds" % timeout
    if r.returncode == 0:
        return True, "ok"
    tail = ((r.stderr or "") + (r.stdout or "")).strip().splitlines()
    return False, ("; ".join(tail[-2:]) or "nonzero exit")[:300]

# ------------------------------------------------------------------ checkers

def check_python_tests(text, task):
    code = extract_code(text)
    if code is None:
        return False, "no python code found in the answer"
    return run_python(code, task["tests"])

def check_rewrite(text, task):
    code = extract_code(text)
    if code is None:
        return False, "no python code found in the answer"
    for banned in task.get("banned_substrings", []):
        if banned in code:
            return False, "still contains %r" % banned
    for needed in task.get("required_substrings", []):
        if needed not in code:
            return False, "missing %r" % needed
    return run_python(code, task["tests"])

def check_json_fields(text, task):
    got = extract_json(text)
    if got is None:
        return False, "no parseable JSON object in the answer"
    want = task["expected"]
    for key, val in want.items():
        if key == "positionen":
            continue
        if key not in got:
            return False, "field %r missing" % key
        if isinstance(val, (int, float)):
            try:
                if abs(float(got[key]) - float(val)) > 0.01:
                    return False, "%s=%r, expected %r" % (key, got[key], val)
            except (TypeError, ValueError):
                return False, "%s=%r is not a number" % (key, got[key])
        elif str(got[key]).strip() != str(val):
            return False, "%s=%r, expected %r" % (key, got[key], val)
    pos = got.get("positionen")
    want_pos = want["positionen"]
    if not isinstance(pos, list) or len(pos) != len(want_pos):
        return False, "positionen: expected %d entries" % len(want_pos)
    for i, (g, w) in enumerate(zip(pos, want_pos)):
        try:
            if int(g.get("menge")) != w["menge"] or \
               abs(float(g.get("einzelpreis")) - w["einzelpreis"]) > 0.01:
                return False, "position %d: %r" % (i + 1, g)
        except (TypeError, ValueError, AttributeError):
            return False, "position %d is malformed: %r" % (i + 1, g)
        if w["stichwort"].lower() not in str(g.get("bezeichnung", "")).lower():
            return False, "position %d: %r without %r" % (i + 1, g, w["stichwort"])
    return True, "ok"

def check_sql(text, task):
    sql = extract_sql(text)
    if not sql:
        return False, "no SQL in the answer"
    if ";" in sql:
        return False, "more than one statement"
    # WITH counts: a CTE is still one read-only SELECT statement, and window
    # questions are answered that way more often than not.
    if not sql.lower().lstrip().startswith(("select", "with")):
        return False, "not a SELECT"
    db = sqlite3.connect(":memory:")
    try:
        db.executescript(task["schema"])
        want = db.execute(task["reference_sql"]).fetchall()
        try:
            got = db.execute(sql).fetchall()
        except sqlite3.Error as e:
            return False, "sqlite: %s" % e
        if got == want:
            return True, "ok"
        # Same rows in a different order only counts when no order was asked.
        if not task.get("ordered", True) and sorted(got) == sorted(want):
            return True, "ok"
        return False, "rows %r, expected %r" % (got[:4], want[:4])
    finally:
        db.close()

def check_regex(text, task):
    block = extract_block(text)
    pattern = (block if block is not None else (text or "")).strip().strip("`")
    pattern = pattern.splitlines()[-1].strip() if pattern else ""
    if not pattern:
        return False, "no pattern in the answer"
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return False, "does not compile: %s" % e
    for line, groups in task["must_match"]:
        m = rx.fullmatch(line) or rx.match(line)
        if not m:
            return False, "does not match: %r" % line
        for g, val in groups.items():
            try:
                if m.group(g) != val:
                    return False, "group %r on %r: %r != %r" % (g, line,
                                                               m.group(g), val)
            except IndexError:
                return False, "group %r missing" % g
    for line in task["must_not_match"]:
        if rx.fullmatch(line):
            return False, "must not match, but does: %r" % line
    return True, "ok"

def check_answer_line(text, task):
    hits = re.findall(r"ANSWER:\s*(-?\d+)", text or "")
    if not hits:
        return False, "no 'ANSWER: <number>' line"
    if int(hits[-1]) != task["expected"]:
        return False, "ANSWER %s, expected %d" % (hits[-1], task["expected"])
    return True, "ok"

CHECKERS = {
    "python_tests": check_python_tests,
    "rewrite":      check_rewrite,
    "json_fields":  check_json_fields,
    "sql":          check_sql,
    "regex":        check_regex,
    "answer_line":  check_answer_line,
}

# ---------------------------------------------------------- longctx material

def longctx_module(n_functions=520, target_index=364):
    """A deterministic ~14k-token module with one target function buried in it.

    No randomness on purpose: the same text on every machine and every run,
    so prefill numbers stay comparable across reports.
    """
    lines = ['"""Generated utility module for the retrieval task."""', ""]
    for i in range(n_functions):
        a, b, m = (i * 7 + 3) % 97 + 2, (i * 13 + 5) % 89 + 1, (i % 9) + 11
        if i == target_index:
            lines += [
                "def compute_checksum_v3(x):",
                '    """Checksum used by the export path (v3)."""',
                "    acc = x * 31 + 7",
                "    acc = acc ^ (x << 2)",
                "    return acc % 8191",
                "",
            ]
        lines += [
            "def helper_fn_%04d(x):" % i,
            '    """Helper %d for the pipeline stage %d."""' % (i, i % 12),
            "    return (x * %d + %d) %% %d" % (a, b, m),
            "",
        ]
    return "\n".join(lines)

def longctx_expected(x=1723):
    acc = x * 31 + 7
    acc = acc ^ (x << 2)
    return acc % 8191

# ------------------------------------------------------------ task material

BUGGY_MERGE = '''def merge_intervals(intervals):
    """Merge overlapping or touching intervals, given as (start, end) tuples
    with start <= end. Result: sorted, non-overlapping tuples."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda iv: iv[1])
    out = [list(intervals[0])]
    for lo, hi in intervals[1:]:
        if lo < out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [tuple(x) for x in out]
'''

MERGE_TESTS = '''assert merge_intervals([]) == []
assert merge_intervals([(1, 3)]) == [(1, 3)]
assert merge_intervals([(1, 2), (4, 5)]) == [(1, 2), (4, 5)]
assert merge_intervals([(1, 2), (2, 3)]) == [(1, 3)], "touching intervals merge"
assert merge_intervals([(1, 10), (2, 3)]) == [(1, 10)]
assert merge_intervals([(5, 6), (1, 4), (2, 5)]) == [(1, 6)], "unsorted input"
assert merge_intervals([(1, 4), (0, 0)]) == [(0, 0), (1, 4)]
print("OK")
'''

CACHE_TESTS = '''c = BoundedTTLCache(maxsize=2, ttl=10)
c.put("a", 1, now=0)
c.put("b", 2, now=1)
assert c.get("a", now=5) == 1
assert c.get("a", now=10) == 1, "entry is valid up to and including now=put+ttl"
assert c.get("a", now=11) is None, "expired"
assert c.get("missing", now=0) is None
c2 = BoundedTTLCache(maxsize=2, ttl=100)
c2.put("a", 1, now=0)
c2.put("b", 2, now=1)
assert c2.get("a", now=2) == 1          # touches a -> b is now least recent
c2.put("c", 3, now=3)                    # evicts b, not a
assert c2.get("b", now=4) is None, "b was least recently used"
assert c2.get("a", now=4) == 1
assert c2.get("c", now=4) == 3
c2.put("a", 99, now=5)                   # overwrite refreshes value and time
assert c2.get("a", now=104) == 99
c3 = BoundedTTLCache(maxsize=1, ttl=5)
c3.put("x", 1, now=0)
c3.put("y", 2, now=0)
assert c3.get("x", now=0) is None and c3.get("y", now=0) == 2
print("OK")
'''

LEGACY_MODULE = '''"""Reporting helpers (legacy style)."""

def formatHeader(title, width):
    return "%s" % title.center(width, "=")

def formatRow(name, value, width):
    return "%-*s %s" % (width, name, value)

def formatPercent(part, total):
    if total == 0:
        return "0.0%"
    return "%.1f%%" % (100.0 * part / total)

def formatBytes(n):
    if n < 1024:
        return "%d B" % n
    if n < 1048576:
        return "%.1f KiB" % (n / 1024.0)
    return "%.1f MiB" % (n / 1048576.0)

def formatDuration(seconds):
    minutes = int(seconds) // 60
    rest = seconds - 60 * minutes
    return "%dm %04.1fs" % (minutes, rest)

def formatCountLabel(count, singular, plural):
    if count == 1:
        return "%d %s" % (count, singular)
    return "%d %s" % (count, plural)

def buildSummaryLine(name, count, total):
    return "%s: %s of %s" % (name, count, total)

def buildErrorLine(code, message):
    return "E%04d: %s" % (code, message)
'''

REWRITE_TESTS = '''assert format_header("Report", 12) == "===Report===", format_header("Report", 12)
assert format_row("cpu", "97%", 6) == "cpu    97%"
assert format_percent(1, 8) == "12.5%"
assert format_percent(3, 0) == "0.0%"
assert format_bytes(512) == "512 B"
assert format_bytes(2048) == "2.0 KiB"
assert format_bytes(3145728) == "3.0 MiB"
assert format_duration(75.25) == "1m 15.2s" or format_duration(75.25) == "1m 15.3s"
assert format_count_label(1, "file", "files") == "1 file"
assert format_count_label(3, "file", "files") == "3 files"
assert build_summary_line("tests", 90, 90) == "tests: 90 of 90"
assert build_error_line(7, "boom") == "E0007: boom"
print("OK")
'''

INVOICE_TEXT = """Betreff: Ihre Bestellung vom 10.08.2026 — Rechnung

LoonyParts GmbH · Werkstr. 14 · 44227 Dortmund

Rechnung
Rechnungsnummer: RE-2026-0817
Rechnungsdatum: 12.08.2026
Kundennummer: K-5521

Wir berechnen Ihnen wie folgt:

  2 x Temperatursensor TS-400        je 320,00 EUR      640,00 EUR
  5 x Anschlusskabel 3m geschirmt    je  40,00 EUR      200,00 EUR
  1 x IoT-Gateway GW-Edge            je 400,00 EUR      400,00 EUR

Nettobetrag:                                          1.240,00 EUR
zzgl. 19% MwSt.:                                        235,60 EUR
Rechnungsbetrag:                                      1.475,60 EUR

Zahlbar ohne Abzug bis zum 26.08.2026.
"""

INVENTORY_FILE = '''"""Tiny inventory bookkeeping."""

class Inventory:
    def __init__(self):
        self._items = {}

    def add(self, name, qty, unit_price):
        if qty <= 0:
            raise ValueError("qty must be positive")
        entry = self._items.setdefault(name, {"qty": 0, "unit_price": unit_price})
        entry["qty"] += qty
        entry["unit_price"] = unit_price

    def quantity(self, name):
        return self._items.get(name, {}).get("qty", 0)
'''

INVENTORY_TESTS = '''inv = Inventory()
inv.add("bolt", 100, 0.10)
inv.add("plate", 4, 12.50)
assert inv.quantity("bolt") == 100
assert abs(inv.total_value() - 60.0) < 1e-9, inv.total_value()
inv.remove("bolt", 30)
assert inv.quantity("bolt") == 70
try:
    inv.remove("bolt", 999)
    raise SystemExit("remove beyond stock must raise ValueError")
except ValueError:
    pass
try:
    inv.remove("missing", 1)
    raise SystemExit("remove of unknown item must raise ValueError")
except ValueError:
    pass
assert inv.low_stock(5) == ["plate"], inv.low_stock(5)
assert inv.low_stock(1000) == sorted(["bolt", "plate"])
print("OK")
'''

SQL_SCHEMA = """CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT);
CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER,
                     amount REAL, created TEXT);
INSERT INTO customers VALUES (1,'Krause','Dortmund'),(2,'Ilgen','Bochum'),
                             (3,'Weber','Essen'),(4,'Sato','Dortmund');
INSERT INTO orders VALUES (1,1,80.0,'2026-08-01'),(2,1,45.5,'2026-08-03'),
  (3,2,300.0,'2026-08-04'),(4,3,20.0,'2026-08-10'),(5,3,30.0,'2026-08-11'),
  (6,3,49.0,'2026-08-12'),(7,4,101.0,'2026-08-13'),(8,2,12.0,'2026-08-14');
"""

# ------------------------------------------------------------------- battery

CODE_RULES = ("Reply with the complete solution in ONE ```python code block. "
              "No usage examples, no tests of your own.")

TASKS = [
    {
        "id": "impl-cache", "tags": ["code-novel"], "max_tokens": 8192,
        "checker": "python_tests", "tests": CACHE_TESTS,
        "system": "You are a careful senior Python engineer.",
        "user": (
            "Implement a Python class `BoundedTTLCache`.\n\n"
            "Constructor: `BoundedTTLCache(maxsize, ttl)`.\n"
            "Methods:\n"
            "- `put(key, value, now)` stores the value with timestamp `now` "
            "(a number, passed in explicitly — do not read the clock). "
            "Overwriting an existing key refreshes value and timestamp.\n"
            "- `get(key, now)` returns the value, or `None` if the key is "
            "missing or expired. An entry is valid while `now <= stored_time "
            "+ ttl`. A successful get makes the key most-recently used.\n"
            "- When `put` would exceed `maxsize` distinct keys, evict the "
            "least-recently-used key first (puts and successful gets both "
            "count as use).\n\n" + CODE_RULES),
    },
    {
        "id": "bugfix-intervals", "tags": ["code-novel"], "max_tokens": 6144,
        "checker": "python_tests", "tests": MERGE_TESTS,
        "system": "You are a careful senior Python engineer.",
        "user": (
            "This function is buggy:\n\n```python\n" + BUGGY_MERGE + "```\n\n"
            "Observed failure: `merge_intervals([(5, 6), (1, 4), (2, 5)])` "
            "returns `[(1, 6), (5, 6)]` — expected `[(1, 6)]`. Touching "
            "intervals such as `(1, 2), (2, 3)` must merge to `(1, 3)`.\n\n"
            "Fix the function. Keep its name and signature.\n\n" + CODE_RULES),
    },
    {
        "id": "rewrite-modernize", "tags": ["code-repetitive"],
        "max_tokens": 6144, "checker": "rewrite", "tests": REWRITE_TESTS,
        "banned_substrings": ["formatHeader", "formatRow", "formatPercent",
                              "formatBytes", "formatDuration",
                              "formatCountLabel", "buildSummaryLine",
                              "buildErrorLine"],
        "required_substrings": ["def format_header", "def format_row",
                                "def format_percent", "def format_bytes",
                                "def format_duration", "def format_count_label",
                                "def build_summary_line", "def build_error_line"],
        "system": "You are a careful senior Python engineer.",
        "user": (
            "Modernize this module. Rename every function from camelCase to "
            "snake_case, replace ALL %-formatting with f-strings, and keep "
            "the behavior byte-for-byte identical (including widths and "
            "rounding).\n\n```python\n" + LEGACY_MODULE + "```\n\n"
            "Reply with the COMPLETE rewritten file in one ```python block."),
    },
    {
        "id": "json-invoice", "tags": ["struct"], "max_tokens": 4096,
        "checker": "json_fields",
        "expected": {
            "rechnungsnummer": "RE-2026-0817", "datum": "2026-08-12",
            "netto": 1240.00, "mwst_satz": 19, "brutto": 1475.60,
            "positionen": [
                {"menge": 2, "einzelpreis": 320.00, "stichwort": "sensor"},
                {"menge": 5, "einzelpreis": 40.00,  "stichwort": "kabel"},
                {"menge": 1, "einzelpreis": 400.00, "stichwort": "gateway"},
            ],
        },
        "system": "You extract structured data. You answer with JSON only.",
        "user": (
            "Extract from this German invoice:\n\n" + INVOICE_TEXT + "\n"
            "Return exactly one JSON object with the keys: rechnungsnummer "
            "(string), datum (ISO YYYY-MM-DD), netto (number), mwst_satz "
            "(number), brutto (number), positionen (array of objects with "
            "bezeichnung, menge, einzelpreis). Numbers as JSON numbers, "
            "not strings."),
    },
    {
        "id": "sql-revenue", "tags": ["struct"], "max_tokens": 4096,
        "checker": "sql", "schema": SQL_SCHEMA, "ordered": True,
        "reference_sql": (
            "SELECT c.name, SUM(o.amount) AS total FROM customers c "
            "JOIN orders o ON o.customer_id = c.id GROUP BY c.id, c.name "
            "HAVING SUM(o.amount) > 100 ORDER BY total DESC"),
        "system": "You write SQLite SQL. You answer with one SQL statement only.",
        "user": (
            "Schema and data:\n\n```sql\n" + SQL_SCHEMA + "```\n\n"
            "Write ONE SQLite SELECT that returns customer name and their "
            "total order amount, only for customers whose total exceeds 100, "
            "ordered by total descending. Columns: name, total."),
    },
    {
        "id": "regex-log", "tags": ["struct"], "max_tokens": 4096,
        "checker": "regex",
        "must_match": [
            ('2026-08-24 21:14:03 [ERROR] code=500 msg="upstream timeout"',
             {"date": "2026-08-24", "time": "21:14:03", "level": "ERROR",
              "code": "500", "msg": "upstream timeout"}),
            ('2026-01-02 03:04:05 [WARN] code=42 msg=""',
             {"level": "WARN", "code": "42", "msg": ""}),
            ('2025-12-31 23:59:59 [INFO] code=0 msg="ok, done."',
             {"code": "0", "msg": "ok, done."}),
        ],
        "must_not_match": [
            '2026-08-24 21:14:03 ERROR code=500 msg="x"',
            '2026-08-24 21:14:03 [ERROR] code= msg="x"',
            '2026-08-24 21:14:03 [ERROR] code=500 msg=unquoted',
        ],
        "system": "You write Python regular expressions.",
        "user": (
            "Log lines look like this:\n\n"
            '    2026-08-24 21:14:03 [ERROR] code=500 msg="upstream timeout"\n\n'
            "Write ONE Python regex with the named groups date, time, level, "
            "code and msg (msg = the text between the quotes, may be empty). "
            "The level is one of INFO, WARN, ERROR. The regex must match the "
            "whole line and reject lines where the level has no brackets, "
            "code is empty, or msg is unquoted.\n\n"
            "Reply with the bare pattern on the last line of a ``` block — "
            "no re.compile(), no quotes around it."),
    },
    {
        "id": "longctx-retrieval", "tags": ["longctx"], "max_tokens": 4096,
        "checker": "answer_line", "expected": longctx_expected(1723),
        "system": "You answer questions about the given source file.",
        "user": (
            "Here is a generated Python module:\n\n```python\n"
            + longctx_module() + "```\n\n"
            "What does `compute_checksum_v3(1723)` return? Compute it exactly. "
            "Finish with a line of the form `ANSWER: <integer>`."),
    },
    {
        "id": "multiturn-edit", "tags": ["agentic"], "max_tokens": 6144,
        "checker": "python_tests", "tests": INVENTORY_TESTS,
        "system": "You are a careful senior Python engineer.",
        "turns": [
            ("Here is a file:\n\n```python\n" + INVENTORY_FILE + "```\n\n"
             "Add a method `total_value(self)` that returns the sum of "
             "qty * unit_price over all items. Reply with the COMPLETE "
             "updated file in one ```python block."),
            ("Now add `remove(self, name, qty)`: reduce the stored quantity; "
             "raise ValueError when the item is unknown or the stock is "
             "insufficient. Reply with the COMPLETE updated file in one "
             "```python block."),
            ("Finally add `low_stock(self, threshold)`: the alphabetically "
             "sorted list of item names with quantity strictly below the "
             "threshold. Reply with the COMPLETE updated file in one "
             "```python block."),
        ],
    },
]

# ------------------------------------------------------------- hard battery
# The core battery could not separate the thinking levels: every ROCm cell
# passed 9/9. These tasks exist to answer ONE question — does thinking buy
# correctness — so they are chosen to punish pattern-matching without
# reasoning. Wherever a reference is needed, the test code carries its own
# brute force; no hand-computed constant can be silently wrong.

HARD_RULES = ("Reply with the complete solution in ONE ```python code block. "
              "No usage examples, no tests of your own.")

WEIGHTED_TESTS = '''import itertools
def _ok(sel):
    sel = sorted(sel)
    return all(a[1] <= b[0] for a, b in zip(sel, sel[1:]))
def _brute(jobs):
    best = 0
    for r in range(len(jobs) + 1):
        for combo in itertools.combinations(jobs, r):
            if _ok(list(combo)):
                best = max(best, sum(j[2] for j in combo))
    return best
CASES = [
    [],
    [(1, 3, 5)],
    [(1, 3, 5), (2, 4, 6)],
    [(1, 3, 5), (3, 5, 6)],
    [(1, 10, 10), (2, 3, 4), (4, 5, 4), (6, 7, 4)],
    [(1, 4, 3), (3, 5, 2), (0, 6, 4), (4, 7, 2), (3, 8, 1),
     (6, 9, 5), (5, 9, 4), (8, 10, 2)],
    [(0, 2, 1), (2, 4, 1), (4, 6, 1), (0, 6, 2)],
    [(5, 6, 7), (1, 2, 3), (3, 4, 2), (2, 3, 9)],
]
for jobs in CASES:
    got = max_weight_schedule(list(jobs))
    want = _brute(jobs)
    assert got == want, "jobs=%r: got %r, expected %r" % (jobs, got, want)
print("OK")
'''

ROTATED_TESTS = '''CASES = [
    ([4, 5, 6, 7, 0, 1, 2], 0), ([4, 5, 6, 7, 0, 1, 2], 3),
    ([6, 7, 8, 1, 2, 3, 4, 5], 8), ([6, 7, 8, 1, 2, 3, 4, 5], 6),
    ([1], 1), ([1], 0), ([1, 2, 3, 4, 5], 4), ([2, 1], 1), ([2, 1], 2),
    ([5, 6, 7, 8, 9, 1, 2, 3], 9), ([5, 6, 7, 8, 9, 1, 2, 3], 4),
]
for arr, target in CASES:
    got = find_rotated(list(arr), target)
    want = arr.index(target) if target in arr else -1
    assert got == want, "arr=%r target=%r: got %r, expected %r" % (
        arr, target, got, want)
print("OK")
'''

NEXTPERM_TESTS = '''CASES = [
    ([1, 2, 3], [1, 3, 2]), ([3, 2, 1], [1, 2, 3]), ([1, 1, 5], [1, 5, 1]),
    ([1, 3, 2], [2, 1, 3]), ([2, 3, 1], [3, 1, 2]), ([1], [1]),
    ([1, 5, 1], [5, 1, 1]), ([2, 2, 2], [2, 2, 2]),
    ([1, 2, 2, 3], [1, 2, 3, 2]),
]
for seq, want in CASES:
    got = next_permutation(list(seq))
    assert list(got) == want, "%r: got %r, expected %r" % (seq, got, want)
print("OK")
'''

HARD_SQL_SCHEMA = SQL_SCHEMA + """
INSERT INTO orders VALUES (9,1,80.0,'2026-08-02'),(10,4,101.0,'2026-08-01'),
  (11,2,300.0,'2026-08-20'),(12,1,95.0,'2026-08-05');
"""

def hard_count_expected():
    """4-digit numbers with strictly increasing digits and digit sum
    divisible by 5 — by definition, not by arithmetic."""
    n = 0
    for x in range(1000, 10000):
        d = [int(c) for c in str(x)]
        if all(a < b for a, b in zip(d, d[1:])) and sum(d) % 5 == 0:
            n += 1
    return n

HARD = [
    {
        "id": "hard-weighted-intervals", "tags": ["hard", "code-novel"],
        "max_tokens": 8192, "checker": "python_tests",
        "tests": WEIGHTED_TESTS,
        "system": "You are a careful senior Python engineer.",
        "user": (
            "Implement `max_weight_schedule(jobs)`.\n\n"
            "`jobs` is a list of `(start, end, weight)` tuples with "
            "`start < end` and positive weights. Return the maximum total "
            "weight of a subset of jobs where no two selected jobs overlap. "
            "A job may begin exactly when another ends (touching is "
            "allowed). The optimum matters — greedy by weight or by end "
            "time alone is wrong for some inputs.\n\n" + HARD_RULES),
    },
    {
        "id": "hard-rotated-search", "tags": ["hard", "code-novel"],
        "max_tokens": 8192, "checker": "python_tests",
        "tests": ROTATED_TESTS,
        "system": "You are a careful senior Python engineer.",
        "user": (
            "Implement `find_rotated(arr, target)`: `arr` is a sorted array "
            "of DISTINCT integers that has been rotated an unknown number "
            "of positions (possibly zero). Return the index of `target`, or "
            "-1 if absent, in O(log n) — a linear scan does not count. "
            "Handle arrays of length 1 and the not-rotated case.\n\n"
            + HARD_RULES),
    },
    {
        "id": "hard-next-permutation", "tags": ["hard", "code-novel"],
        "max_tokens": 8192, "checker": "python_tests",
        "tests": NEXTPERM_TESTS,
        "system": "You are a careful senior Python engineer.",
        "user": (
            "Implement `next_permutation(seq)`: return the next permutation "
            "of the list `seq` in lexicographic order as a new list. If "
            "`seq` is already the last permutation, return the first "
            "(sorted ascending). Duplicate elements must be handled "
            "correctly (each distinct arrangement appears exactly once in "
            "the cycle).\n\n" + HARD_RULES),
    },
    {
        "id": "hard-sql-window", "tags": ["hard", "struct"],
        "max_tokens": 8192, "checker": "sql", "schema": HARD_SQL_SCHEMA,
        "ordered": True,
        "reference_sql": (
            "SELECT name, amount, rn FROM ("
            "SELECT c.name AS name, o.amount AS amount, ROW_NUMBER() OVER ("
            "PARTITION BY c.id ORDER BY o.amount DESC, o.created ASC) AS rn "
            "FROM customers c JOIN orders o ON o.customer_id = c.id) "
            "WHERE rn <= 2 ORDER BY name ASC, rn ASC"),
        "system": "You write SQLite SQL. You answer with one SQL statement only.",
        "user": (
            "Schema and data:\n\n```sql\n" + HARD_SQL_SCHEMA + "```\n\n"
            "Write ONE SQLite SELECT returning, for every customer, their "
            "top 2 orders by amount (descending; ties broken by earlier "
            "`created` first). Columns exactly: name, amount, rn — where rn "
            "is 1 for the largest order and 2 for the second. Order the "
            "result by name ascending, then rn ascending. Customers with a "
            "single order appear once."),
    },
    {
        "id": "hard-count-numbers", "tags": ["hard", "struct"],
        "max_tokens": 8192, "checker": "answer_line",
        "expected": hard_count_expected(),
        "system": "You are a careful mathematician. Show your reasoning "
                  "briefly, then give the final line.",
        "user": (
            "How many 4-digit numbers (1000-9999) have strictly increasing "
            "digits AND a digit sum divisible by 5? Count exactly. Finish "
            "with a line of the form `ANSWER: <integer>`."),
    },
]

TASKS_ALL = TASKS + HARD
TASK_INDEX = {t["id"]: t for t in TASKS_ALL}

def get_checker(task):
    return CHECKERS[task["checker"]]
