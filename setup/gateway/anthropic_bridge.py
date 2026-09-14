#!/usr/bin/env python3
"""anthropic_bridge — Protocol translation between Anthropic Messages and OpenAI APIs.

Enables consumers that speak the Anthropic dialect (such as Claude Code)
to interact transparently with OpenAI-only backends (like Halogen Flash Server),
including tool calling, image inputs, streaming SSE, and prompt cache reporting.

Zero external dependencies: pure Python standard library.
"""

import json
import re
import uuid
from typing import Any, Dict, Generator, List, Optional, Tuple


def _msg_id() -> str:
    return "msg_" + uuid.uuid4().hex[:24]


def _tool_id() -> str:
    return "toolu_" + uuid.uuid4().hex[:24]


def blocks_to_text(content: Any) -> str:
    """Extract plain text from content, whether string or block list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict):
                if b.get("type") == "text":
                    parts.append(b.get("text", ""))
                elif b.get("type") == "tool_result":
                    parts.append(blocks_to_text(b.get("content", "")))
        return "".join(parts)
    return ""


def anthropic_to_openai_request(body: dict, target_model: Optional[str] = None) -> dict:
    """Translate an Anthropic /v1/messages request body to an OpenAI /v1/chat/completions body.
    
    Handles:
      - system prompt (string or blocks) -> messages[0] with role: "system"
      - conversational messages (user, assistant, tool_use, tool_result)
      - image attachments (base64 -> data: URL)
      - tool declarations (Anthropic input_schema -> OpenAI parameters)
      - tool choice mapping
      - sampling and token limits
    """
    out: Dict[str, Any] = {}
    
    # Model name
    if target_model:
        out["model"] = target_model
    elif "model" in body:
        out["model"] = body["model"]
    else:
        out["model"] = "default"

    # Streaming and token limits
    if "stream" in body:
        out["stream"] = bool(body["stream"])
        if out["stream"]:
            out["stream_options"] = {"include_usage": True}
    if "max_tokens" in body:
        out["max_tokens"] = body["max_tokens"]
    if "temperature" in body:
        out["temperature"] = body["temperature"]
    if "top_p" in body:
        out["top_p"] = body["top_p"]
    if "top_k" in body:
        out["top_k"] = body["top_k"]
    # Anthropic spells them `stop_sequences`, OpenAI spells them `stop`. The
    # two are the same field under different names, and until 12.09.2026 the
    # Anthropic one was read by nobody: a client relying on one to end a turn
    # got a turn that did not end, with nothing anywhere saying why.
    stops = body.get("stop_sequences")
    if isinstance(stops, str):
        stops = [stops]
    if stops:
        out["stop"] = list(stops)

    oai_messages: List[Dict[str, Any]] = []

    # 1. System prompt
    system = body.get("system")
    if system:
        sys_text = blocks_to_text(system)
        if sys_text:
            oai_messages.append({"role": "system", "content": sys_text})

    # 2. Conversational turns
    messages = body.get("messages") or []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")

        if isinstance(content, str):
            oai_messages.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            oai_messages.append({"role": role, "content": ""})
            continue

        # Content is a list of blocks
        text_parts = []
        tool_calls = []
        tool_results = []
        image_parts = []

        for b in content:
            if not isinstance(b, dict):
                continue
            b_type = b.get("type")
            if b_type == "text":
                text_parts.append(b.get("text", ""))
            elif b_type == "image":
                src = b.get("source") or {}
                if src.get("type") == "base64" and src.get("data"):
                    media = src.get("media_type", "image/jpeg")
                    image_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{media};base64,{src['data']}"}
                    })
            elif b_type == "tool_use":
                t_id = b.get("id") or _tool_id()
                t_name = b.get("name", "")
                t_input = b.get("input")
                args_str = json.dumps(t_input) if isinstance(t_input, dict) else str(t_input or "{}")
                tool_calls.append({
                    "id": t_id,
                    "type": "function",
                    "function": {"name": t_name, "arguments": args_str}
                })
            elif b_type == "tool_result":
                res_content = b.get("content", "")
                res_text = blocks_to_text(res_content) if isinstance(res_content, (list, dict)) else str(res_content)
                # `is_error` is the difference between "the tool said this"
                # and "the tool failed", and the OpenAI tool message has no
                # field for it — so it has to survive as text or not at all.
                # Dropping it hands the model a failure dressed as output,
                # which is the one thing a tool loop must not be told.
                if b.get("is_error"):
                    res_text = "Error: " + res_text if res_text else "Error"
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": b.get("tool_use_id", ""),
                    "content": res_text
                })

        # Assemble OpenAI messages
        if role == "assistant":
            m: Dict[str, Any] = {"role": "assistant"}
            combined_text = "".join(text_parts)
            m["content"] = combined_text if combined_text else None
            if tool_calls:
                m["tool_calls"] = tool_calls
            oai_messages.append(m)

        elif role == "user":
            # If tool results are present, emit them as separate tool messages first
            for tr in tool_results:
                oai_messages.append(tr)

            # Any accompanying user text or images
            if image_parts or text_parts:
                u_content: List[Dict[str, Any]] = []
                for t in text_parts:
                    if t:
                        u_content.append({"type": "text", "text": t})
                u_content.extend(image_parts)

                if len(u_content) == 1 and u_content[0]["type"] == "text":
                    oai_messages.append({"role": "user", "content": u_content[0]["text"]})
                elif u_content:
                    oai_messages.append({"role": "user", "content": u_content})
            elif not tool_results:
                oai_messages.append({"role": "user", "content": ""})

    out["messages"] = oai_messages

    # 3. Tools definition
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        oai_tools = []
        for t in tools:
            if not isinstance(t, dict) or not t.get("name"):
                continue
            oai_tools.append({
                "type": "function",
                "function": {
                    "name": t.get("name"),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or {"type": "object", "properties": {}}
                }
            })
        if oai_tools:
            out["tools"] = oai_tools

    # 4. Tool choice
    tc = body.get("tool_choice")
    if isinstance(tc, dict):
        tc_type = tc.get("type")
        if tc_type == "auto":
            out["tool_choice"] = "auto"
        elif tc_type == "any":
            out["tool_choice"] = "required"
        elif tc_type == "tool" and tc.get("name"):
            out["tool_choice"] = {"type": "function", "function": {"name": tc["name"]}}

    # 5. Thinking / reasoning parameters
    thinking = body.get("thinking")
    if isinstance(thinking, dict):
        if thinking.get("type") == "enabled":
            out["enable_thinking"] = True
            budget = thinking.get("budget_tokens")
            if budget:
                out["max_thinking_tokens"] = budget
                out["reasoning_effort"] = "high"
        elif thinking.get("type") == "disabled":
            out["enable_thinking"] = False
    else:
        # Default in Anthropic is disabled unless explicitly requested
        out["enable_thinking"] = False

    if "max_thinking_tokens" in body:
        out["max_thinking_tokens"] = body["max_thinking_tokens"]

    return out


def openai_to_anthropic_response(resp: dict, model: str = "halogen") -> dict:
    """Translate an OpenAI non-streaming response to Anthropic /v1/messages JSON."""
    choices = resp.get("choices") or []
    first = choices[0] if choices else {}
    msg = first.get("message") or {}
    finish_reason = first.get("finish_reason")

    content_blocks = []

    # Thinking / Reasoning content
    reasoning = msg.get("reasoning_content")
    if reasoning:
        content_blocks.append({"type": "thinking", "thinking": reasoning})

    # Text content
    text = msg.get("content")
    if text:
        content_blocks.append({"type": "text", "text": text})

    # Tool calls
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args_str = fn.get("arguments", "{}")
        try:
            parsed_args = json.loads(args_str)
        except Exception:
            parsed_args = {"raw": args_str}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id") or _tool_id(),
            "name": fn.get("name", ""),
            "input": parsed_args
        })

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    # Map stop reason
    stop_map = {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "length": "max_tokens"
    }
    stop_reason = stop_map.get(finish_reason, "end_turn")

    # Usage
    u = resp.get("usage") or {}
    in_tokens = u.get("prompt_tokens", 0)
    out_tokens = u.get("completion_tokens", 0)
    cached_tokens = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)

    usage_dict = {
        "input_tokens": in_tokens,
        "output_tokens": out_tokens
    }
    if cached_tokens > 0:
        usage_dict["cache_read_input_tokens"] = cached_tokens

    return {
        "id": _msg_id(),
        "type": "message",
        "role": "assistant",
        "model": resp.get("model") or model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage_dict
    }


def openai_error_to_anthropic(status: int, raw: Any) -> dict:
    """An upstream error, in the shape an Anthropic SDK can read.

    Every Anthropic client looks for `error.type` and `error.message`; an
    OpenAI envelope passed through unchanged arrives as an unparsed blob, and
    a non-JSON body (a proxy's HTML, an empty 502) arrives as nothing at all.
    Both happen: Halogen answers 400 with its own envelope for an unsupported
    reasoning_effort, and a container that is still loading answers with
    neither.
    """
    text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw or "")
    kind, message = None, None
    try:
        obj = json.loads(text)
        err = obj.get("error") if isinstance(obj, dict) else None
        if isinstance(err, dict):
            kind = err.get("type")
            message = err.get("message")
        elif isinstance(err, str):
            message = err
    except Exception:
        pass
    if not message:
        # The raw body, trimmed. Saying "upstream error" and throwing the
        # body away is how a clear 400 becomes an unexplained failure.
        message = " ".join(text.split())[:500] or ("upstream returned %d" % status)
    return {"type": "error",
            "error": {"type": kind or ANTHROPIC_ERROR_TYPES.get(
                status, "api_error"),
                      "message": message}}


ANTHROPIC_ERROR_TYPES = {
    400: "invalid_request_error", 401: "authentication_error",
    403: "permission_error", 404: "not_found_error",
    413: "request_too_large", 429: "rate_limit_error",
    500: "api_error", 502: "api_error", 503: "overloaded_error",
    529: "overloaded_error",
}


# Characters per token. HEURISTIC, NOT MEASURED on this tokenizer — the
# familiar English rule of thumb, and the traffic here is code, JSON tool
# schemas and diffs, which run denser. Kept because the alternative is no
# answer at all: neither backend exposes a tokenizer over HTTP, and the
# consumer asking is deciding when to compact a conversation.
#
# The honest way to replace it is a /tokenize endpoint on the backend, or a
# measurement of this ratio against recorded traffic. Until then the number
# is a guess and this comment is what says so.
CHARS_PER_TOKEN = 4


def anthropic_token_count_response(body: dict) -> dict:
    """An ESTIMATED token count for /v1/messages/count_tokens.

    See CHARS_PER_TOKEN: this counts characters and divides. It is not a
    tokenisation and must not be used for anything that has to be exact.
    """
    total_chars = len(blocks_to_text(body.get("system", "")))
    for m in body.get("messages") or []:
        total_chars += len(blocks_to_text(m.get("content", "")))
    for t in body.get("tools") or []:
        total_chars += len(json.dumps(t))
    tokens = max(1, total_chars // CHARS_PER_TOKEN)
    return {"input_tokens": tokens}


def format_sse(event: str, data: dict) -> str:
    """Format an Anthropic SSE event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


class StreamTranslator:
    """Translates an OpenAI SSE chunk stream into Anthropic SSE events.
    
    Maintains block indices and state across streaming delta chunks.
    """

    def __init__(self, model_name: str = "halogen"):
        self.model_name = model_name
        self.message_id = _msg_id()
        self.started = False
        self.ended = False
        
        self.current_block_type: Optional[str] = None  # "text" or "tool_use"
        self.current_block_index: int = -1
        
        # Tool call tracking by OpenAI tool call index:
        # tool_calls_map[tc_index] = {"id": ..., "name": ..., "block_index": ...}
        self.tool_calls_map: Dict[int, Dict[str, Any]] = {}

        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.cached_tokens: int = 0
        self.finish_reason: Optional[str] = None

    def feed_chunk(self, chunk: dict) -> List[str]:
        """Process one parsed OpenAI chunk dict and return list of formatted SSE event strings."""
        events: List[str] = []

        if not self.started:
            self.started = True
            # Optional initial usage in chunk
            u = chunk.get("usage") or {}
            self.input_tokens = u.get("prompt_tokens", 0)
            self.cached_tokens = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)

            msg_start = {
                "type": "message_start",
                "message": {
                    "id": self.message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": chunk.get("model") or self.model_name,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": self.input_tokens,
                        "output_tokens": 1
                    }
                }
            }
            if self.cached_tokens > 0:
                msg_start["message"]["usage"]["cache_read_input_tokens"] = self.cached_tokens
            events.append(format_sse("message_start", msg_start))

        # Check for usage in final or intermediate chunk
        if chunk.get("usage"):
            u = chunk["usage"]
            if u.get("prompt_tokens"):
                self.input_tokens = u["prompt_tokens"]
            if u.get("completion_tokens"):
                self.output_tokens = u["completion_tokens"]
            if u.get("prompt_tokens_details", {}).get("cached_tokens"):
                self.cached_tokens = u["prompt_tokens_details"]["cached_tokens"]

        choices = chunk.get("choices") or []
        if not choices:
            return events

        choice = choices[0]
        delta = choice.get("delta") or {}
        if choice.get("finish_reason"):
            self.finish_reason = choice["finish_reason"]

        # 0. Reasoning content delta
        reasoning = delta.get("reasoning_content")
        if reasoning:
            if self.current_block_type != "thinking":
                if self.current_block_type in ("text", "tool_use"):
                    events.append(format_sse("content_block_stop", {"type": "content_block_stop", "index": self.current_block_index}))
                
                self.current_block_index += 1
                self.current_block_type = "thinking"
                events.append(format_sse("content_block_start", {
                    "type": "content_block_start",
                    "index": self.current_block_index,
                    "content_block": {"type": "thinking", "thinking": ""}
                }))

            events.append(format_sse("content_block_delta", {
                "type": "content_block_delta",
                "index": self.current_block_index,
                "delta": {"type": "thinking_delta", "thinking": reasoning}
            }))

        # 1. Text content delta
        text = delta.get("content")
        if text:
            if self.current_block_type != "text":
                if self.current_block_type in ("thinking", "tool_use"):
                    events.append(format_sse("content_block_stop", {"type": "content_block_stop", "index": self.current_block_index}))
                
                self.current_block_index += 1
                self.current_block_type = "text"
                events.append(format_sse("content_block_start", {
                    "type": "content_block_start",
                    "index": self.current_block_index,
                    "content_block": {"type": "text", "text": ""}
                }))

            events.append(format_sse("content_block_delta", {
                "type": "content_block_delta",
                "index": self.current_block_index,
                "delta": {"type": "text_delta", "text": text}
            }))

        # 2. Tool calls delta
        tool_calls = delta.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                tc_idx = tc.get("index", 0)
                fn = tc.get("function") or {}
                fn_name = fn.get("name")
                fn_args = fn.get("arguments")

                if tc_idx not in self.tool_calls_map:
                    # New tool call
                    if self.current_block_type in ("text", "thinking", "tool_use"):
                        events.append(format_sse("content_block_stop", {"type": "content_block_stop", "index": self.current_block_index}))

                    self.current_block_index += 1
                    self.current_block_type = "tool_use"
                    t_id = tc.get("id") or _tool_id()
                    self.tool_calls_map[tc_idx] = {
                        "id": t_id,
                        "name": fn_name or "",
                        "block_index": self.current_block_index
                    }

                    events.append(format_sse("content_block_start", {
                        "type": "content_block_start",
                        "index": self.current_block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": t_id,
                            "name": fn_name or "",
                            "input": {}
                        }
                    }))

                # Emit partial arguments delta
                if fn_args:
                    b_idx = self.tool_calls_map[tc_idx]["block_index"]
                    events.append(format_sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": b_idx,
                        "delta": {"type": "input_json_delta", "partial_json": fn_args}
                    }))

        return events

    def finish(self) -> List[str]:
        """Close any open blocks and emit final message_delta and message_stop."""
        if self.ended:
            return []
        self.ended = True
        events: List[str] = []

        if not self.started:
            events.extend(self.feed_chunk({"choices": []}))

        # Close open block
        if self.current_block_type is not None:
            events.append(format_sse("content_block_stop", {
                "type": "content_block_stop",
                "index": self.current_block_index
            }))
            self.current_block_type = None

        stop_map = {
            "stop": "end_turn",
            "tool_calls": "tool_use",
            "length": "max_tokens"
        }
        stop_reason = stop_map.get(self.finish_reason, "end_turn")
        if self.tool_calls_map and stop_reason == "end_turn":
            stop_reason = "tool_use"

        # THE INPUT ACCOUNTING BELONGS HERE, not only in message_start.
        #
        # message_start goes out on the first chunk, and with
        # stream_options.include_usage the upstream server sends its numbers
        # in ONE extra chunk at the very end — so at message_start time there
        # is nothing to report and `input_tokens: 0` is what the consumer
        # sees. Measured live 12.09.2026: every streaming turn reached Claude
        # Code as zero input tokens and no cache hit, while the gateway's own
        # trace recorded reused=27361 for the same request.
        #
        # Anthropic's message_delta carries usage, so repeating the complete
        # figures here is the supported place to correct it. They are
        # cumulative totals, not a delta, which is what the API says they are.
        usage_dict = {
            "output_tokens": max(1, self.output_tokens)
        }
        if self.input_tokens:
            usage_dict["input_tokens"] = self.input_tokens
        if self.cached_tokens:
            usage_dict["cache_read_input_tokens"] = self.cached_tokens
        events.append(format_sse("message_delta", {
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": None
            },
            "usage": usage_dict
        }))

        events.append(format_sse("message_stop", {
            "type": "message_stop"
        }))

        return events
