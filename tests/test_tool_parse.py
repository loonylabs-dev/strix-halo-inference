"""test_tool_parse — offline tests for setup/halogen/tool_parse.py.

Verifies incremental tool argument streaming, parameter JSON construction,
marker holdback safety, and protocol compatibility without needing GPU or container.
"""
import json
import pathlib
import re
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common

REPO = common.REPO
TP = common.load("setup/halogen/tool_parse.py", "tool_parse_patched")


class TestToolParseHeaderAndBaseSha(unittest.TestCase):
    """The vendored copy must name its origin image and base sha256."""

    PATH = REPO / "setup" / "halogen" / "tool_parse.py"

    def test_it_names_the_image_it_was_cut_from(self):
        head = self.PATH.read_text(encoding="utf-8")[:4000]
        self.assertRegex(
            head, r"halogen-flash-server:\d+\.\d+\.\d+",
            "the vendored tool_parse.py does not say which image tag it patches")
        self.assertRegex(
            head, r"BASE_SHA256\s*[:=]\s*[0-9a-f]{64}",
            "without the base file's hash nothing can notice that the image "
            "moved underneath this copy")
        m = re.search(r"BASE_SHA256\s*[:=]\s*([0-9a-f]{64})", head)
        self.assertEqual(
            m.group(1),
            "8868cd2ff18d727bc8f95eb827f8dd1f8c68be209cf0058414240d5da0d25de4",
            "BASE_SHA256 does not match the cut copy in the pinned image")


class TestIncrementalToolStream(unittest.TestCase):
    """ToolStream must stream arguments incrementally chunk-by-chunk."""

    def test_incremental_deltas_on_long_generation(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "write",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"}
                    }
                }
            }
        }]
        stream = TP.ToolStream(tools, make_id=lambda: "call_fixed_id")

        tokens = [
            "I'll generate the game:\n\n",
            "<tool_call>\n",
            "<function=write>\n",
            "<parameter=path>\n",
            "game.html",
            "\n</parameter>\n",
            "<parameter=content>\n",
            "<!DOCTYPE html>\n",
            "<html>\n",
            "<head><title>Test Game</title></head>\n",
            "<body>\n",
            "<canvas id=\"c\"></canvas>\n",
            "<script>\n",
            "console.log('init');\n",
            "</script>\n",
            "</body>\n",
            "</html>\n",
            "</parameter>\n",
            "</function>\n",
            "</tool_call>"
        ]

        acc = ""
        emitted_deltas = []
        emitted_content = ""
        for t in tokens:
            acc += t
            cd, evs = stream.push(acc)
            if cd:
                emitted_content += cd
            for e in evs:
                args = e.get("function", {}).get("arguments", "")
                if args:
                    emitted_deltas.append(args)

        self.assertEqual(emitted_content, "I'll generate the game:\n\n")
        # Must have streamed multiple incremental chunks rather than buffering all at once
        self.assertGreater(len(emitted_deltas), 5,
                           "tool arguments were buffered rather than streamed incrementally")

        # Reconstructed JSON must be 100% valid and match expected parameters
        full_args = "".join(emitted_deltas)
        parsed = json.loads(full_args)
        self.assertEqual(parsed["path"], "game.html")
        self.assertIn("<canvas id=\"c\"></canvas>", parsed["content"])
        self.assertIn("console.log('init');", parsed["content"])
        self.assertTrue(stream.any_calls())

    def test_marker_holdback_safety(self):
        """Partial tags at the tail must be held back and not leaked into argument strings."""
        tools = [{
            "type": "function",
            "function": {
                "name": "code",
                "parameters": {
                    "type": "object",
                    "properties": {"body": {"type": "string"}}
                }
            }
        }]
        stream = TP.ToolStream(tools, make_id=lambda: "call_fixed_id")

        # Step 1: Tool starts
        _, evs = stream.push("<tool_call>\n<function=code>\n<parameter=body>\nfunction test() {")
        chunks = [e.get("function", {}).get("arguments", "") for e in evs]
        self.assertTrue(any("function test() {" in c for c in chunks))

        # Step 2: Emit partial closing tag '</par'
        _, evs2 = stream.push("<tool_call>\n<function=code>\n<parameter=body>\nfunction test() {</par")
        chunks2 = [e.get("function", {}).get("arguments", "") for e in evs2]
        # '</par' must NOT be in chunks2
        self.assertFalse(any("</par" in c for c in chunks2),
                         "partial tag marker was leaked into arguments delta")

        # Step 3: Closing tag completes
        _, evs3 = stream.push("<tool_call>\n<function=code>\n<parameter=body>\nfunction test() {</parameter>\n</function>\n</tool_call>")
        all_chunks = chunks + chunks2 + [e.get("function", {}).get("arguments", "") for e in evs3]
        full = "".join(all_chunks)
        parsed = json.loads(full)
        self.assertEqual(parsed["body"], "function test() {")

    def test_typed_parameters(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "calc",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expr": {"type": "string"},
                        "round": {"type": "boolean"},
                        "digits": {"type": "integer"}
                    }
                }
            }
        }]
        stream = TP.ToolStream(tools, make_id=lambda: "call_calc")
        text = (
            "<tool_call>\n"
            "<function=calc>\n"
            "<parameter=expr>\n1 + 2\n</parameter>\n"
            "<parameter=round>\ntrue\n</parameter>\n"
            "<parameter=digits>\n4\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        _, evs = stream.push(text)
        args_str = "".join(e.get("function", {}).get("arguments", "") for e in evs)
        parsed = json.loads(args_str)
        self.assertEqual(parsed, {"expr": "1 + 2", "round": True, "digits": 4})

    def test_non_tool_turn(self):
        stream = TP.ToolStream()
        cd1, evs1 = stream.push("Hello ")
        cd2, evs2 = stream.push("Hello world!")
        self.assertEqual(cd1, "Hello ")
        self.assertEqual(cd2, "world!")
        self.assertEqual(evs1, [])
        self.assertEqual(evs2, [])
        self.assertFalse(stream.any_calls())


class TestSplitToolCallsAndNormalize(unittest.TestCase):
    """Non-streaming split_tool_calls and normalize_messages must remain intact."""

    def test_split_tool_calls_complete(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "write",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}}
                }
            }
        }]
        text = "Prefix<tool_call>\n<function=write>\n<parameter=path>\napp.py\n</parameter>\n</function>\n</tool_call>Suffix"
        content, calls, truncated = TP.split_tool_calls(text, tools)
        self.assertEqual(content, "PrefixSuffix")
        self.assertFalse(truncated)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "write")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"path": "app.py"})

    def test_normalize_messages_handles_json_arguments(self):
        msgs = [
            {"role": "assistant", "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "test", "arguments": '{"a": 1}'}
            }]}
        ]
        norm = TP.normalize_messages(msgs)
        self.assertEqual(norm[0]["tool_calls"][0]["function"]["arguments"], {"a": 1})


if __name__ == "__main__":
    unittest.main()
