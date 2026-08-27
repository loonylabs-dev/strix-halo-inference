"""Tests for bench/measure.py — the place where measurements come into being.

Why tests here of all places: the numbers from bench/ are the basis of every
statement in docs/. A measurement that silently yields a wrong rate is worse
than one that aborts — it travels into the documentation as a finding and
afterwards cannot be told apart from a real measurement.
"""
import unittest

import common

MH = common.load("bench/measure.py", "measure")


class TestEvaluate(unittest.TestCase):
    def test_computes_the_documented_rate(self):
        # The numbers behind --swa-full: setup/README.md, "The server switch
        # that decides everything", and docs/measurements/cache-hunt-finding.md.
        m = MH.evaluate({"usage": {"input_tokens": 1637,
                                    "cache_read_input_tokens": 17734}}, 10.4)
        self.assertEqual(m["new"], 1637)
        self.assertEqual(m["cached"], 17734)
        self.assertEqual(m["rate"], 91.5)
        self.assertEqual(m["seconds"], 10.4)

    def test_a_cold_run_is_zero_percent(self):
        m = MH.evaluate({"usage": {"input_tokens": 19371,
                                    "cache_read_input_tokens": 0}})
        self.assertEqual(m["rate"], 0.0)
        self.assertNotIn("seconds", m)

    def test_missing_usage_aborts_instead_of_guessing(self):
        """This used to produce -0.0 % — which reads like a finding."""
        for antwort in ({}, {"usage": {}}, {"usage": None},
                        {"usage": {"cache_read_input_tokens": 5}}, None, "broken"):
            with self.subTest(antwort=antwort):
                with self.assertRaises(MH.NoMeasurement):
                    MH.evaluate(antwort)

    def test_the_servers_error_is_passed_along(self):
        with self.assertRaises(MH.NoMeasurement) as k:
            MH.evaluate({"error": {"message": "context window exceeded"}})
        self.assertIn("context window exceeded", str(k.exception))

    def test_zero_tokens_is_no_measurement(self):
        with self.assertRaises(MH.NoMeasurement):
            MH.evaluate({"usage": {"input_tokens": 0,
                                    "cache_read_input_tokens": 0}})


class TestRequired(unittest.TestCase):
    def test_returns_the_value(self):
        self.assertEqual(MH.required({"input_tokens": 42}), 42)

    def test_raises_instead_of_minus_one(self):
        for u in ({}, None, {"cache_read_input_tokens": 3}):
            with self.subTest(u=u):
                with self.assertRaises(MH.NoMeasurement):
                    MH.required(u)


class TestGtt(unittest.TestCase):
    def test_returns_a_number_or_none_but_never_raises(self):
        # On a machine without amdgpu the result must be None, not an error.
        w = MH.gtt_gib()
        self.assertTrue(w is None or isinstance(w, float))


if __name__ == "__main__":
    unittest.main()
