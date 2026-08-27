"""Tests for bench/tasklib.py — the task battery and its checkers.

Why tests here: the battery decides the model question, and a checker that
quietly accepts a wrong answer (or rejects a right one) forges exactly the
number the decision rests on. Every checker is therefore driven with a
known-good and a known-bad answer. The known-good answers double as proof
that the tasks are actually solvable as specified — a task nobody can pass
measures nothing.

The subprocess checkers really spawn python; that costs ~1 s across the file
and is the point: run_python's contract (asserts, exit code, timeout) is what
every code task stands on.
"""
import unittest

import common

T = common.load("bench/tasklib.py", "tasklib")


GOOD_CACHE = '''```python
from collections import OrderedDict

class BoundedTTLCache:
    def __init__(self, maxsize, ttl):
        self.maxsize, self.ttl = maxsize, ttl
        self._d = OrderedDict()

    def put(self, key, value, now):
        if key in self._d:
            self._d.pop(key)
        elif len(self._d) >= self.maxsize:
            self._d.popitem(last=False)
        self._d[key] = (value, now)

    def get(self, key, now):
        item = self._d.get(key)
        if item is None:
            return None
        value, t = item
        if now > t + self.ttl:
            return None
        self._d.move_to_end(key)
        return value
```'''

# Same shape, one contract broken: get() does not refresh recency.
BAD_CACHE = GOOD_CACHE.replace("        self._d.move_to_end(key)\n", "")

GOOD_MERGE = '''```python
def merge_intervals(intervals):
    if not intervals:
        return []
    ivs = sorted(intervals)
    out = [list(ivs[0])]
    for lo, hi in ivs[1:]:
        if lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [tuple(x) for x in out]
```'''

GOOD_REWRITE = '''```python
def format_header(title, width):
    return title.center(width, "=")

def format_row(name, value, width):
    return f"{name:<{width}} {value}"

def format_percent(part, total):
    return "0.0%" if total == 0 else f"{100.0 * part / total:.1f}%"

def format_bytes(n):
    if n < 1024:
        return f"{n} B"
    if n < 1048576:
        return f"{n / 1024.0:.1f} KiB"
    return f"{n / 1048576.0:.1f} MiB"

def format_duration(seconds):
    minutes = int(seconds) // 60
    rest = seconds - 60 * minutes
    return f"{minutes}m {rest:04.1f}s"

def format_count_label(count, singular, plural):
    return f"{count} {singular}" if count == 1 else f"{count} {plural}"

def build_summary_line(name, count, total):
    return f"{name}: {count} of {total}"

def build_error_line(code, message):
    return f"E{code:04d}: {message}"
```'''

GOOD_JSON = '''Here is the extraction:
```json
{"rechnungsnummer": "RE-2026-0817", "datum": "2026-08-12",
 "netto": 1240.0, "mwst_satz": 19, "brutto": 1475.60,
 "positionen": [
   {"bezeichnung": "Temperatursensor TS-400", "menge": 2, "einzelpreis": 320.0},
   {"bezeichnung": "Anschlusskabel 3m geschirmt", "menge": 5, "einzelpreis": 40.0},
   {"bezeichnung": "IoT-Gateway GW-Edge", "menge": 1, "einzelpreis": 400.0}]}
```'''

GOOD_SQL = ('```sql\nSELECT c.name, SUM(o.amount) AS total\n'
            'FROM orders o JOIN customers c ON c.id = o.customer_id\n'
            'GROUP BY c.name HAVING SUM(o.amount) > 100\n'
            'ORDER BY total DESC\n```')

GOOD_REGEX = ('```\n'
              r'(?P<date>\d{4}-\d{2}-\d{2}) (?P<time>\d{2}:\d{2}:\d{2}) '
              r'\[(?P<level>INFO|WARN|ERROR)\] code=(?P<code>\d+) '
              r'msg="(?P<msg>[^"]*)"'
              '\n```')


def task(tid):
    return T.TASK_INDEX[tid]


def check(tid, text):
    t = task(tid)
    return T.get_checker(t)(text, t)


class TestExtraction(unittest.TestCase):
    def test_the_last_block_wins(self):
        text = "Draft:\n```python\nx = (\n```\nFinal:\n```python\nx = 1\n```"
        self.assertEqual(T.extract_code(text), "x = 1")

    def test_bare_code_counts_when_it_compiles(self):
        self.assertEqual(T.extract_code("x = 1"), "x = 1")
        self.assertIsNone(T.extract_code("This is prose, not code."))

    def test_json_with_prose_around_it(self):
        self.assertEqual(T.extract_json('Sure! {"a": 1} Hope that helps.'),
                         {"a": 1})
        self.assertIsNone(T.extract_json("no json here"))

    def test_strip_think_removes_and_counts(self):
        clean, n = T.strip_think("<think>hmm hmm</think>answer")
        self.assertEqual(clean, "answer")
        self.assertEqual(n, len("<think>hmm hmm</think>"))

    def test_sql_extraction_drops_the_trailing_semicolon(self):
        self.assertEqual(T.extract_sql("```sql\nSELECT 1;\n```"), "SELECT 1")


class TestCodeCheckers(unittest.TestCase):
    def test_a_correct_cache_passes(self):
        ok, reason = check("impl-cache", GOOD_CACHE)
        self.assertTrue(ok, reason)

    def test_a_missing_lru_refresh_fails(self):
        ok, reason = check("impl-cache", BAD_CACHE)
        self.assertFalse(ok)
        self.assertIn("least recently used", reason)

    def test_the_fixed_merge_passes_and_the_buggy_one_fails(self):
        ok, reason = check("bugfix-intervals", GOOD_MERGE)
        self.assertTrue(ok, reason)
        # The exact code from the task statement must fail its own tests —
        # otherwise the task tests nothing.
        ok, _ = check("bugfix-intervals", "```python\n%s```" % T.BUGGY_MERGE)
        self.assertFalse(ok)

    def test_rewrite_passes_and_unrenamed_module_fails_on_banned_names(self):
        ok, reason = check("rewrite-modernize", GOOD_REWRITE)
        self.assertTrue(ok, reason)
        ok, reason = check("rewrite-modernize",
                           "```python\n%s```" % T.LEGACY_MODULE)
        self.assertFalse(ok)
        self.assertIn("formatHeader", reason)

    def test_no_code_is_a_clean_fail_not_a_crash(self):
        ok, reason = check("impl-cache", "I would rather not.")
        self.assertFalse(ok)
        self.assertIn("no python code", reason)


class TestStructCheckers(unittest.TestCase):
    def test_a_correct_invoice_passes(self):
        ok, reason = check("json-invoice", GOOD_JSON)
        self.assertTrue(ok, reason)

    def test_a_wrong_amount_fails(self):
        ok, reason = check("json-invoice", GOOD_JSON.replace("1475.60", "1475.0"))
        self.assertFalse(ok)
        self.assertIn("brutto", reason)

    def test_numbers_as_strings_fail(self):
        ok, _ = check("json-invoice", GOOD_JSON.replace('"netto": 1240.0',
                                                        '"netto": "1240,00 EUR"'))
        self.assertFalse(ok)

    def test_an_equivalent_sql_formulation_passes(self):
        ok, reason = check("sql-revenue", GOOD_SQL)
        self.assertTrue(ok, reason)

    def test_sql_without_having_fails(self):
        ok, _ = check("sql-revenue", GOOD_SQL.replace(
            "HAVING SUM(o.amount) > 100\n", ""))
        self.assertFalse(ok)

    def test_sql_with_two_statements_fails(self):
        ok, reason = check("sql-revenue",
                           "```sql\nSELECT 1; DROP TABLE orders\n```")
        self.assertFalse(ok)

    def test_the_reference_regex_passes(self):
        ok, reason = check("regex-log", GOOD_REGEX)
        self.assertTrue(ok, reason)

    def test_a_regex_without_named_groups_fails(self):
        ok, _ = check("regex-log", "```\n.*\n```")
        self.assertFalse(ok)


class TestLongctx(unittest.TestCase):
    def test_the_module_is_deterministic(self):
        self.assertEqual(T.longctx_module(), T.longctx_module())

    def test_the_expected_answer_matches_the_embedded_function(self):
        scope = {}
        exec(T.longctx_module(), scope)          # noqa: S102 - own generator
        self.assertEqual(scope["compute_checksum_v3"](1723),
                         T.longctx_expected(1723))

    def test_the_checker_reads_the_last_answer_line(self):
        good = "thinking...\nANSWER: 999\nwait, no.\nANSWER: %d" \
               % T.longctx_expected(1723)
        ok, reason = check("longctx-retrieval", good)
        self.assertTrue(ok, reason)
        ok, _ = check("longctx-retrieval", "ANSWER: 1")
        self.assertFalse(ok)


GOOD_WEIGHTED = '''```python
import bisect

def max_weight_schedule(jobs):
    jobs = sorted(jobs, key=lambda j: j[1])
    ends = [j[1] for j in jobs]
    dp = [0] * (len(jobs) + 1)
    for i, (s, e, w) in enumerate(jobs):
        k = bisect.bisect_right(ends, s, 0, i)
        dp[i + 1] = max(dp[i], dp[k] + w)
    return dp[-1]
```'''

GREEDY_WEIGHTED = '''```python
def max_weight_schedule(jobs):
    total, taken = 0, []
    for s, e, w in sorted(jobs, key=lambda j: -j[2]):
        if all(e <= s2 or e2 <= s for s2, e2, w2 in taken):
            taken.append((s, e, w)); total += w
    return total
```'''

GOOD_ROTATED = '''```python
def find_rotated(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        if arr[lo] <= arr[mid]:
            if arr[lo] <= target < arr[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:
            if arr[mid] < target <= arr[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1
```'''

GOOD_NEXTPERM = '''```python
def next_permutation(seq):
    a = list(seq)
    i = len(a) - 2
    while i >= 0 and a[i] >= a[i + 1]:
        i -= 1
    if i < 0:
        return sorted(a)
    j = len(a) - 1
    while a[j] <= a[i]:
        j -= 1
    a[i], a[j] = a[j], a[i]
    a[i + 1:] = list(reversed(a[i + 1:]))
    return a
```'''

GOOD_WINDOW_SQL = ('```sql\nWITH ranked AS (SELECT c.name AS name, o.amount '
                   'AS amount, ROW_NUMBER() OVER (PARTITION BY c.id ORDER BY '
                   'o.amount DESC, o.created ASC) AS rn FROM customers c '
                   'JOIN orders o ON o.customer_id = c.id) SELECT name, '
                   'amount, rn FROM ranked WHERE rn <= 2 ORDER BY name, rn\n```')


class TestHardCheckers(unittest.TestCase):
    """The hard battery separates the thinking levels — but only if its
    checkers accept a correct solution (task is solvable) and reject the
    canonical shortcut (task punishes non-reasoning)."""

    def test_the_dp_solution_passes(self):
        ok, reason = check("hard-weighted-intervals", GOOD_WEIGHTED)
        self.assertTrue(ok, reason)

    def test_the_greedy_shortcut_fails(self):
        ok, _ = check("hard-weighted-intervals", GREEDY_WEIGHTED)
        self.assertFalse(ok, "greedy by weight must not survive the cases")

    def test_rotated_search_passes_and_handles_the_edges(self):
        ok, reason = check("hard-rotated-search", GOOD_ROTATED)
        self.assertTrue(ok, reason)

    def test_next_permutation_with_duplicates_passes(self):
        ok, reason = check("hard-next-permutation", GOOD_NEXTPERM)
        self.assertTrue(ok, reason)

    def test_the_window_sql_passes_in_a_different_formulation(self):
        ok, reason = check("hard-sql-window", GOOD_WINDOW_SQL)
        self.assertTrue(ok, reason)

    def test_count_answer_is_by_brute_force_not_by_hand(self):
        ok, reason = check("hard-count-numbers",
                           "ANSWER: %d" % T.hard_count_expected())
        self.assertTrue(ok, reason)
        ok, _ = check("hard-count-numbers", "ANSWER: 1")
        self.assertFalse(ok)


class TestBatteryShape(unittest.TestCase):
    def test_ids_are_unique_and_checkers_exist(self):
        ids = [t["id"] for t in T.TASKS_ALL]
        self.assertEqual(len(ids), len(set(ids)))
        for t in T.TASKS_ALL:
            self.assertIn(t["checker"], T.CHECKERS, t["id"])
            self.assertTrue(t.get("user") or t.get("turns"), t["id"])
            self.assertGreaterEqual(t["max_tokens"], 1024, t["id"])

    def test_every_tag_family_is_represented(self):
        tags = {tag for t in T.TASKS for tag in t["tags"]}
        # "prose" is deliberately absent since 27.08. The free-prose task was
        # the last QUALITY checker in this file — it judged whether an
        # explanation was good — and it moved to bench/speed.py as a decode
        # workload, which is what its only surviving caller wanted from it.
        for needed in ("code-novel", "code-repetitive", "struct", "longctx",
                       "agentic"):
            self.assertIn(needed, tags)


if __name__ == "__main__":
    unittest.main()
