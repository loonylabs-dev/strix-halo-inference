"""Tests for anthropic_bridge.py — protocol conversion between Anthropic and OpenAI.

Zero external dependencies: runs fully offline without server, GPU, or container.
"""

import json
import unittest

import common

AB = common.load("setup/gateway/anthropic_bridge.py", "anthropic_bridge")


class TestRequestTranslation(unittest.TestCase):
    def test_plain_chat_with_string_system(self):
        req = {
            "model": "flashnext",
            "system": "You are a helpful assistant.",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
            "max_tokens": 1024,
            "temperature": 0.7,
            "stream": True
        }
        oai = AB.anthropic_to_openai_request(req, target_model="halogen-qwen3.8")
        self.assertEqual(oai["model"], "halogen-qwen3.8")
        self.assertTrue(oai["stream"])
        self.assertEqual(oai["max_tokens"], 1024)
        self.assertEqual(oai["temperature"], 0.7)
        self.assertEqual(len(oai["messages"]), 2)
        self.assertEqual(oai["messages"][0], {"role": "system", "content": "You are a helpful assistant."})
        self.assertEqual(oai["messages"][1], {"role": "user", "content": "Hello!"})

    def test_system_as_blocks(self):
        req = {
            "system": [
                {"type": "text", "text": "Base instructions. "},
                {"type": "text", "text": "Additional rules."}
            ],
            "messages": [{"role": "user", "content": "Ping"}]
        }
        oai = AB.anthropic_to_openai_request(req)
        self.assertEqual(len(oai["messages"]), 2)
        self.assertEqual(oai["messages"][0]["role"], "system")
        self.assertEqual(oai["messages"][0]["content"], "Base instructions. Additional rules.")

    def test_tools_declaration_mapping(self):
        req = {
            "tools": [
                {
                    "name": "Bash",
                    "description": "Execute a bash command",
                    "input_schema": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"]
                    }
                }
            ],
            "tool_choice": {"type": "tool", "name": "Bash"},
            "messages": [{"role": "user", "content": "Run ls"}]
        }
        oai = AB.anthropic_to_openai_request(req)
        self.assertIn("tools", oai)
        self.assertEqual(len(oai["tools"]), 1)
        t0 = oai["tools"][0]
        self.assertEqual(t0["type"], "function")
        self.assertEqual(t0["function"]["name"], "Bash")
        self.assertEqual(t0["function"]["parameters"]["properties"]["command"]["type"], "string")
        self.assertEqual(oai["tool_choice"], {"type": "function", "function": {"name": "Bash"}})

    def test_tool_use_and_tool_result_roundtrip(self):
        req = {
            "messages": [
                {"role": "user", "content": "List files"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Running ls..."},
                        {"type": "tool_use", "id": "call_123", "name": "Bash", "input": {"command": "ls -la"}}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "call_123", "content": "file1.txt\nfile2.txt"}
                    ]
                }
            ]
        }
        oai = AB.anthropic_to_openai_request(req)
        msgs = oai["messages"]
        self.assertEqual(len(msgs), 3)

        # Assistant message with tool call
        self.assertEqual(msgs[1]["role"], "assistant")
        self.assertEqual(msgs[1]["content"], "Running ls...")
        self.assertEqual(len(msgs[1]["tool_calls"]), 1)
        self.assertEqual(msgs[1]["tool_calls"][0]["id"], "call_123")
        self.assertEqual(msgs[1]["tool_calls"][0]["function"]["name"], "Bash")
        self.assertEqual(json.loads(msgs[1]["tool_calls"][0]["function"]["arguments"]), {"command": "ls -la"})

        # User message with tool result mapped to role: tool
        self.assertEqual(msgs[2]["role"], "tool")
        self.assertEqual(msgs[2]["tool_call_id"], "call_123")
        self.assertEqual(msgs[2]["content"], "file1.txt\nfile2.txt")

    def test_image_attachment_mapping(self):
        req = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}
                    ]
                }
            ]
        }
        oai = AB.anthropic_to_openai_request(req)
        msgs = oai["messages"]
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertIsInstance(msgs[0]["content"], list)
        self.assertEqual(msgs[0]["content"][0], {"type": "text", "text": "What is in this image?"})
        self.assertEqual(msgs[0]["content"][1]["type"], "image_url")
        self.assertEqual(msgs[0]["content"][1]["image_url"]["url"], "data:image/png;base64,iVBORw0KGgo=")


class TestResponseTranslation(unittest.TestCase):
    def test_non_streaming_text_response(self):
        oai_resp = {
            "id": "chatcmpl-999",
            "model": "halogen-qwen3.8",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello there!"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 12,
                "prompt_tokens_details": {"cached_tokens": 40}
            }
        }
        ant = AB.openai_to_anthropic_response(oai_resp)
        self.assertEqual(ant["type"], "message")
        self.assertEqual(ant["role"], "assistant")
        self.assertEqual(ant["stop_reason"], "end_turn")
        self.assertEqual(len(ant["content"]), 1)
        self.assertEqual(ant["content"][0], {"type": "text", "text": "Hello there!"})
        self.assertEqual(ant["usage"]["input_tokens"], 50)
        self.assertEqual(ant["usage"]["output_tokens"], 12)
        self.assertEqual(ant["usage"]["cache_read_input_tokens"], 40)

    def test_non_streaming_tool_call_response(self):
        oai_resp = {
            "id": "chatcmpl-999",
            "model": "halogen-qwen3.8",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "Read", "arguments": "{\"path\":\"README.md\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 30, "completion_tokens": 15}
        }
        ant = AB.openai_to_anthropic_response(oai_resp)
        self.assertEqual(ant["stop_reason"], "tool_use")
        self.assertEqual(len(ant["content"]), 1)
        self.assertEqual(ant["content"][0]["type"], "tool_use")
        self.assertEqual(ant["content"][0]["id"], "call_abc")
        self.assertEqual(ant["content"][0]["name"], "Read")
        self.assertEqual(ant["content"][0]["input"], {"path": "README.md"})


class TestStreamTranslation(unittest.TestCase):
    def test_stream_text_deltas(self):
        st = AB.StreamTranslator(model_name="halogen-qwen3.8")
        
        # Chunk 1: Role / initial
        e1 = st.feed_chunk({
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": None}],
            "usage": {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 90}}
        })
        # Should emit message_start, content_block_start, content_block_delta
        text1 = "".join(e1)
        self.assertIn("event: message_start\n", text1)
        self.assertIn("event: content_block_start\n", text1)
        self.assertIn('"type": "text_delta", "text": "Hello"', text1)
        self.assertIn('"cache_read_input_tokens": 90', text1)

        # Chunk 2: Continuation
        e2 = st.feed_chunk({
            "choices": [{"index": 0, "delta": {"content": " world!"}, "finish_reason": None}]
        })
        text2 = "".join(e2)
        self.assertIn("event: content_block_delta\n", text2)
        self.assertIn('"type": "text_delta", "text": " world!"', text2)

        # Chunk 3: Finish
        e3 = st.feed_chunk({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 5}
        })
        efin = st.finish()
        all_trailing = "".join(e3 + efin)
        self.assertIn("event: content_block_stop\n", all_trailing)
        self.assertIn("event: message_delta\n", all_trailing)
        self.assertIn('"stop_reason": "end_turn"', all_trailing)
        self.assertIn("event: message_stop\n", all_trailing)

    def test_stream_tool_calls(self):
        st = AB.StreamTranslator(model_name="halogen-qwen3.8")
        
        # Chunk 1: tool call init
        e1 = st.feed_chunk({
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_xyz",
                        "type": "function",
                        "function": {"name": "Bash", "arguments": "{\"cmd\":"}
                    }]
                },
                "finish_reason": None
            }]
        })
        text1 = "".join(e1)
        self.assertIn("event: message_start\n", text1)
        self.assertIn("event: content_block_start\n", text1)
        self.assertIn('"type": "tool_use"', text1)
        self.assertIn('"name": "Bash"', text1)
        self.assertIn('"partial_json": "{\\"cmd\\":"', text1)

        # Chunk 2: tool call args continuation
        e2 = st.feed_chunk({
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": " \"echo hi\"}"}
                    }]
                },
                "finish_reason": None
            }]
        })
        text2 = "".join(e2)
        self.assertIn("event: content_block_delta\n", text2)
        self.assertIn('"partial_json": " \\"echo hi\\"}"', text2)

        # Chunk 3: finish reason tool_calls
        e3 = st.feed_chunk({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]
        })
        efin = st.finish()
        all_trailing = "".join(e3 + efin)
        self.assertIn("event: content_block_stop\n", all_trailing)
        self.assertIn('"stop_reason": "tool_use"', all_trailing)


class TestUsageSurvivesTheRealChunkOrder(unittest.TestCase):
    """The order the upstream server actually sends, which is not the one the
    first tests mocked.

    serve_api.py with include_usage puts `"usage": null` in EVERY content
    chunk and the numbers in ONE extra chunk with `choices: []`, right before
    [DONE] — OpenAI's own shape. Anthropic's message_start is emitted on the
    FIRST chunk, so at that moment the input accounting does not exist yet.

    Measured live against the gateway on 12.09.2026: every streaming turn
    reached Claude Code as `input_tokens: 0`, with no cache_read_input_tokens
    anywhere in the stream — while the gateway's own trace recorded
    reused=27361 for the same request. The consumer's context and cache
    display were wrong for every request against this backend, and the test
    that was supposed to cover it passed because its mock put the usage in
    the first chunk.
    """

    REAL_ORDER = [
        {"id": "c", "model": "halogen", "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": "Hi"},
             "finish_reason": None}], "usage": None},
        {"id": "c", "choices": [
            {"index": 0, "delta": {"content": " there"},
             "finish_reason": "stop"}], "usage": None},
        {"id": "c", "choices": [], "usage": {
            "prompt_tokens": 27366, "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 27361}}},
    ]

    def stream(self):
        t = AB.StreamTranslator("halogen")
        out = []
        for c in self.REAL_ORDER:
            out += t.feed_chunk(c)
        out += t.finish()
        return "".join(out)

    def events(self):
        got = {}
        for block in self.stream().split("\n\n"):
            if not block.strip():
                continue
            name = [l[7:] for l in block.splitlines() if l.startswith("event: ")]
            data = [l[6:] for l in block.splitlines() if l.startswith("data: ")]
            if name and data:
                got.setdefault(name[0], []).append(json.loads(data[0]))
        return got

    def test_the_final_usage_carries_the_input_accounting(self):
        ev = self.events()
        self.assertIn("message_delta", ev)
        usage = ev["message_delta"][-1]["usage"]
        self.assertEqual(usage.get("input_tokens"), 27366,
                         "the input tokens never reach the consumer: "
                         "message_start was sent before the numbers existed "
                         "and nothing corrects it")
        self.assertEqual(usage.get("cache_read_input_tokens"), 27361)
        self.assertEqual(usage.get("output_tokens"), 5)

    def test_a_stream_without_any_usage_still_closes(self):
        """A backend that sends no usage at all must not make this raise or
        invent numbers."""
        t = AB.StreamTranslator("halogen")
        out = "".join(t.feed_chunk(
            {"choices": [{"index": 0, "delta": {"content": "x"},
                          "finish_reason": "stop"}]}) + t.finish())
        self.assertIn("message_stop", out)
        self.assertNotIn("cache_read_input_tokens", out)


class TestWhatAnAnthropicClientSendsAndGetsBack(unittest.TestCase):
    def test_stop_sequences_are_translated(self):
        """Anthropic spells them `stop_sequences`, OpenAI spells them `stop`.
        Untranslated they were dropped without a word, and a client that
        relies on one to end a turn gets a turn that does not end."""
        oai = AB.anthropic_to_openai_request(
            {"messages": [], "stop_sequences": ["\n\nHuman:"]})
        self.assertEqual(oai.get("stop"), ["\n\nHuman:"])

    def test_top_k_survives(self):
        oai = AB.anthropic_to_openai_request({"messages": [], "top_k": 40})
        self.assertEqual(oai.get("top_k"), 40)

    def test_a_failed_tool_result_stays_marked(self):
        """`is_error` is the difference between "the tool said this" and "the
        tool failed". Dropping it hands the model a failure dressed as
        output."""
        oai = AB.anthropic_to_openai_request({"messages": [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1",
                 "is_error": True, "content": "boom"}]}]})
        tool_msg = [m for m in oai["messages"] if m.get("role") == "tool"][0]
        self.assertIn("boom", tool_msg["content"])
        self.assertIn("error", tool_msg["content"].lower())

    def test_an_upstream_error_becomes_an_anthropic_error(self):
        """An OpenAI error envelope reaching an Anthropic client is an
        unparsed blob: the SDK looks for error.type and error.message."""
        err = AB.openai_error_to_anthropic(
            400, '{"error": {"message": "bad effort", "type": "invalid_request_error"}}')
        self.assertEqual(err["type"], "error")
        self.assertEqual(err["error"]["type"], "invalid_request_error")
        self.assertIn("bad effort", err["error"]["message"])

    def test_a_body_that_is_not_json_still_becomes_one(self):
        err = AB.openai_error_to_anthropic(502, "<html>bad gateway</html>")
        self.assertEqual(err["type"], "error")
        self.assertTrue(err["error"]["message"])


class TestTokenCount(unittest.TestCase):
    def test_token_count_positive(self):
        body = {
            "system": "System instructions here",
            "messages": [{"role": "user", "content": "How many tokens is this?"}]
        }
        res = AB.anthropic_token_count_response(body)
        self.assertIn("input_tokens", res)
        self.assertGreater(res["input_tokens"], 0)


class TestGatewayAnthropicBridgeIntegration(unittest.IsolatedAsyncioTestCase):
    """End-to-end integration test of Gateway + Anthropic Bridge against an OpenAI upstream."""

    async def asyncSetUp(self):
        from unittest import mock
        from aiohttp import web, ClientSession
        from aiohttp.test_utils import TestServer

        self.GW = common.load("setup/gateway/gateway.py", "gateway")
        self.received_upstream_requests = []
        self.upstream_stream_chunks = []
        self.upstream_non_stream_response = {"choices": [{"message": {"role": "assistant", "content": "mock"}}]}

        async def upstream_handler(request):
            if request.path == "/health":
                return web.json_response({"status": "ok"})
            if request.path == "/v1/chat/completions":
                body = await request.json()
                self.received_upstream_requests.append({
                    "path": request.path,
                    "headers": dict(request.headers),
                    "body": body,
                })
                if body.get("stream"):
                    resp = web.StreamResponse(
                        status=200,
                        headers={"Content-Type": "text/event-stream; charset=utf-8"},
                    )
                    await resp.prepare(request)
                    for chunk in self.upstream_stream_chunks:
                        payload = f"data: {json.dumps(chunk)}\n\n"
                        await resp.write(payload.encode("utf-8"))
                    await resp.write(b"data: [DONE]\n\n")
                    await resp.write_eof()
                    return resp
                else:
                    return web.json_response(self.upstream_non_stream_response)
            return web.json_response({"error": "not found"}, status=404)

        up_app = web.Application()
        up_app.router.add_route("*", "/{tail:.*}", upstream_handler)
        self.upstream_server = TestServer(up_app)
        await self.upstream_server.start_server()

        gw_port = await common.free_port()
        self.backup = {
            k: getattr(self.GW, k) for k in
            ("LLAMA", "TUNNEL_PORT", "TOKENS", "PREFIXES", "SAVED",
             "IN_FLIGHT_PER_TOKEN", "GATE", "PER_TOKEN_MAX", "AUTO_SAVE",
             "BACKEND_FLAVOR")
        }
        self.GW.LLAMA = str(self.upstream_server.make_url("")).rstrip("/")
        self.GW.TUNNEL_PORT = None
        self.GW.TOKENS = {"test-key": "claude-client"}
        self.GW.PREFIXES, self.GW.SAVED, self.GW.IN_FLIGHT_PER_TOKEN = {}, {}, {}
        self.GW.GATE = self.GW.PriorityGate(2)
        self.GW.PER_TOKEN_MAX = 2
        self.GW.AUTO_SAVE = False
        self.GW.BACKEND_FLAVOR = "openai"

        self.log_lines = []
        self.log_patch = mock.patch.object(
            self.GW, "log", lambda *a: self.log_lines.append(" ".join(map(str, a)))
        )
        self.log_patch.start()

        self.gw_server = TestServer(self.GW.build_app(), port=gw_port, **self.GW.RUNNER_KWARGS)
        await self.gw_server.start_server()
        self.gw_url = f"http://127.0.0.1:{gw_port}"
        self.session = ClientSession()

    async def asyncTearDown(self):
        self.log_patch.stop()
        await self.session.close()
        await self.gw_server.close()
        await self.upstream_server.close()
        for k, v in self.backup.items():
            setattr(self.GW, k, v)

    async def test_streaming_chat_completion_translation(self):
        self.upstream_stream_chunks = [
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": None}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 100}}
            },
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {"content": " from Halogen!"}, "finish_reason": None}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 4}
            },
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 4}
            }
        ]

        ant_req = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 256,
            "stream": True
        }

        async with self.session.post(
            f"{self.gw_url}/v1/messages",
            headers={
                "x-api-key": "test-key",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json=ant_req
        ) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("text/event-stream", resp.headers.get("content-type", ""))
            body = await resp.text()

        # 1. Verify upstream received OpenAI format
        self.assertEqual(len(self.received_upstream_requests), 1)
        up_req = self.received_upstream_requests[0]
        self.assertEqual(up_req["path"], "/v1/chat/completions")
        self.assertTrue(up_req["body"]["stream"])
        self.assertEqual(up_req["body"]["messages"][0], {"role": "user", "content": "Say hello"})

        # 2. Verify downstream received Anthropic SSE events
        self.assertIn("event: message_start\n", body)
        self.assertIn('"cache_read_input_tokens": 100', body)
        self.assertIn("event: content_block_start\n", body)
        self.assertIn('"text": "Hello"', body)
        self.assertIn('"text": " from Halogen!"', body)
        self.assertIn("event: content_block_stop\n", body)
        self.assertIn("event: message_delta\n", body)
        self.assertIn('"stop_reason": "end_turn"', body)
        self.assertIn("event: message_stop\n", body)

    async def test_tool_calling_streaming_translation(self):
        self.upstream_stream_chunks = [
            {
                "id": "chatcmpl-2",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "Bash", "arguments": "{\"command\":"}
                        }]
                    },
                    "finish_reason": None
                }]
            },
            {
                "id": "chatcmpl-2",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": " \"uptime\"}"}
                        }]
                    },
                    "finish_reason": None
                }]
            },
            {
                "id": "chatcmpl-2",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]
            }
        ]

        ant_req = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Check uptime"}],
            "tools": [
                {
                    "name": "Bash",
                    "description": "Run shell command",
                    "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}
                }
            ],
            "stream": True
        }

        async with self.session.post(
            f"{self.gw_url}/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json=ant_req
        ) as resp:
            self.assertEqual(resp.status, 200)
            body = await resp.text()

        # Check upstream received mapped OpenAI tools
        self.assertEqual(len(self.received_upstream_requests), 1)
        up_tools = self.received_upstream_requests[0]["body"].get("tools")
        self.assertIsNotNone(up_tools)
        self.assertEqual(up_tools[0]["function"]["name"], "Bash")

        # Check downstream received Anthropic tool_use SSE blocks
        self.assertIn("event: content_block_start\n", body)
        self.assertIn('"type": "tool_use"', body)
        self.assertIn('"name": "Bash"', body)
        self.assertIn('"partial_json": "{\\"command\\":', body)
        self.assertIn('"partial_json": " \\"uptime\\"}"', body)
        self.assertIn("event: content_block_stop\n", body)
        self.assertIn('"stop_reason": "tool_use"', body)

    async def test_a_non_streaming_turn_is_accounted_for_too(self):
        """The translated non-streaming path returned a freshly built JSON
        response and never touched the sniff buffer, so reuse, output and
        stop_reason were all blank for it — the request appeared in the trace
        as a row with a duration and nothing else. Streaming filled them; the
        two paths have to agree."""
        self.upstream_non_stream_response = {
            "id": "chatcmpl-3",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "four"}}],
            "usage": {"prompt_tokens": 52, "completion_tokens": 3,
                      "prompt_tokens_details": {"cached_tokens": 47}},
        }
        async with self.session.post(
            f"{self.gw_url}/v1/messages",
            headers={"x-api-key": "test-key",
                     "anthropic-version": "2023-06-01"},
            json={"model": "claude-3-5-sonnet-20241022", "max_tokens": 16,
                  "messages": [{"role": "user", "content": "2+2?"}]},
        ) as resp:
            self.assertEqual(resp.status, 200)
            got = await resp.json()
        self.assertEqual(got["content"][0]["text"], "four")
        self.assertEqual(got["usage"]["cache_read_input_tokens"], 47)
        done = [l for l in self.log_lines if l.startswith("DONE")]
        self.assertTrue(done, "the request never finished")
        self.assertIn("reused=47 computed=5", done[-1],
                      "the non-streaming path reports no accounting at all: "
                      + done[-1])

    async def test_a_non_streaming_turn_invents_no_time_to_first_token(self):
        """There is no first token in an answer that arrives whole. Stamping
        one would put ttft a hair under took, and the derived write rate is
        output divided by that difference — a number with no meaning and no
        tilde to warn anybody."""
        self.upstream_non_stream_response = {
            "choices": [{"finish_reason": "stop",
                         "message": {"role": "assistant", "content": "x"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1}}
        async with self.session.post(
            f"{self.gw_url}/v1/messages",
            headers={"x-api-key": "test-key",
                     "anthropic-version": "2023-06-01"},
            json={"model": "claude-3-5-sonnet-20241022", "max_tokens": 4,
                  "messages": [{"role": "user", "content": "hi"}]},
        ) as resp:
            self.assertEqual(resp.status, 200)
            await resp.read()
        src = (common.REPO / "setup" / "gateway" / "gateway.py").read_text(
            encoding="utf-8")
        arm = src[src.index("if target_path is not None and translate_stream is None:"):]
        arm = arm[:arm.index("if resp is None:")]
        code = "\n".join(l for l in arm.splitlines()
                         if not l.lstrip().startswith("#"))
        self.assertNotIn("first_token_at", code,
                         "the non-streaming arm stamps a time to first token")

    async def test_count_tokens_endpoint(self):
        ant_req = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Count me"}],
        }

        async with self.session.post(
            f"{self.gw_url}/v1/messages/count_tokens",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json=ant_req
        ) as resp:
            self.assertEqual(resp.status, 200)
            res = await resp.json()
            self.assertIn("input_tokens", res)
            self.assertGreater(res["input_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
