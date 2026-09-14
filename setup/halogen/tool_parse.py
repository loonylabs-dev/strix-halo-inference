#!/usr/bin/env python3
# ===========================================================================
# VENDORED AND PATCHED — this is NOT this repository's code.
#
#   cut from  ghcr.io/peonist-ai/halogen-flash-server:0.8.1
#             image digest sha256:d444524bffb6f487650cb1c29e3dae67fcebf0c6a22abe5a6342f0e4315567e0
#   base file /halogen/tools/tool_parse.py
#   BASE_SHA256 = 8868cd2ff18d727bc8f95eb827f8dd1f8c68be209cf0058414240d5da0d25de4
#
# setup/halogenexec mounts this file OVER the one in the image and verifies
# BASE_SHA256 against the image's own copy before it does. That check is the
# whole reason the hash is written down: without it a bumped image keeps its
# new tag in `podman ps` while this old copy silently reverts every upstream
# change to this one file — the shape of defect this repository keeps
# finding, and the reason setup/patches/ exists for llama.cpp.
#
# WHAT IS CHANGED:
#   Upstream 0.6.1 officially adopted the incremental tool-call parameter
#   streaming mechanism (_scan_params, _stream_safe, and ToolStream JSON chunking)
#   addressing public issue #36. This copy is re-cut from 0.8.1.
# ===========================================================================
"""tools/tool_parse.py — Qwen3.8 tool-call parsing.

Engine-free and GPU-free ON PURPOSE. Every rule in here is a property of
`chat_template.jinja`, not of a kernel, so this module is importable and
testable on a laptop — `tools/gate-tools.py --offline` runs the whole battery
with no ROCm, no checkpoint and no daemon. A parser that can only be exercised
by standing up 27B of weights does not get exercised.

THE FORMAT IS NOT JSON. This template asks for, and the model emits,

    <tool_call>
    <function=NAME>
    <parameter=KEY>
    VALUE
    </parameter>
    </function>
    </tool_call>

VALUES ARE UNTYPED TEXT. The template renders a string RAW and
unquoted and everything else through `tojson`, and the model copies that
exactly. So `3` is six identical bytes whether the parameter is an integer or
a string, and only the tool's declared JSON Schema tells them apart —
`split_tool_calls` therefore takes `tools`, and the schema-less path is
documented as lossy rather than pretended to be exact.
"""

import json
import re
import uuid

TC_OPEN = "<tool_call>"
TC_CLOSE = "</tool_call>"
PARAM_END = "</parameter>"

_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_FUNC_RE = re.compile(r"<function=([^>\n]*)>[ \t]*\n?")
_PARAM_RE = re.compile(r"<parameter=([^>\n]*)>[ \t]*\n?")
# What may legally follow a value's closing tag. Used as a LOOKAHEAD so that a
# value containing the literal text '</parameter>' (a tool that writes source
# code about tools will produce one) closes at the right occurrence.
_AFTER_END_RE = re.compile(r"\s*(<parameter=|</function>|</tool_call>)")


def new_call_id():
    return "call_" + uuid.uuid4().hex[:24]


def schema_types(tools):
    """-> {function_name: {param_name: json_type_or_None}}

    Accepts both OpenAI's `{"type": "function", "function": {...}}` wrapper and
    a bare `{"name": ..., "parameters": ...}`. A parameter whose schema has no
    plain `type` (anyOf/oneOf/$ref) maps to None and takes the lossy path.
    """
    out = {}
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        name = fn.get("name")
        if not isinstance(name, str):
            continue
        props = (fn.get("parameters") or {}).get("properties")
        types = {}
        if isinstance(props, dict):
            for k, v in props.items():
                types[k] = norm_type(v.get("type") if isinstance(v, dict)
                                     else None)
        out[name] = types
    return out


# Real tool definitions are not written by JSON Schema pedants: MCP servers
# and hand-rolled agent tools declare `int`, `str`, `dict`, `list`, `bool`
# constantly. vLLM's parsers for this same XML family carry the same repair
# table (`repair_param_type`), and a type this does not recognize falls
# through to None rather than being taken literally.
_TYPE_ALIAS = {"str": "string", "text": "string", "varchar": "string",
               "char": "string", "enum": "string", "bool": "boolean",
               "binary": "boolean", "list": "array", "sequence": "array",
               "tuple": "array", "arr": "array", "dict": "object",
               "map": "object", "mapping": "object", "json": "object",
               "none": "null"}


def norm_type(t):
    if not isinstance(t, str):
        return None
    t = t.strip().lower()
    if t in ("string", "integer", "number", "boolean", "array", "object",
             "null"):
        return t
    if t in _TYPE_ALIAS:
        return _TYPE_ALIAS[t]
    if t.startswith(("int", "uint", "long", "short", "unsigned")):
        return "integer"
    if t.startswith(("num", "float", "double", "decimal")):
        return "number"
    return None


def coerce(raw, jtype):
    """One wire value + its declared type -> the Python value to put in
    `arguments`.

    `string` is returned VERBATIM and is never JSON-decoded, never stripped:
    a legal string value is a whole source file, quotes and trailing newline
    included, and json.loads over it either throws or silently mangles it.
    Everything else was written by `tojson` and is stripped before parsing.

    A literal that does not parse is kept as its RAW TEXT rather than dropped
    — a client's own validator complaining about a bad integer is more useful
    than an argument that silently vanished. `int()`/`float()` are tried ahead
    of json.loads because they accept what a model actually writes (`+3`,
    ` 7 `) and JSON does not, and a bare true/false is matched
    case-insensitively for the same reason.
    """
    if jtype == "string":
        return raw
    v = raw.strip()
    if jtype is None:
        # Unknown/absent schema: guess, and only when the guess is
        # unambiguous. A parse that yields a str means the model quoted a
        # string, which this template never does — keep the raw text then.
        # NOTE this DIVERGES from vLLM, which defaults an unschema'd
        # parameter to string. It is deliberate: under this template an
        # unquoted `{"a": 1}` was written by `tojson`, and handing a tool the
        # 8-character text of an object it declared as an object is a worse
        # answer than handing it the object.
        try:
            p = json.loads(v)
        except Exception:
            return raw
        return raw if isinstance(p, str) else p
    if v.lower() == "null":
        return None
    if jtype == "boolean" and v.lower() in ("true", "false"):
        return v.lower() == "true"
    if jtype == "integer":
        try:
            return int(v)
        except Exception:
            pass
    if jtype == "number":
        try:
            return float(v)
        except Exception:
            pass
    try:
        return json.loads(v)
    except Exception:
        return raw


def _value_end(body, vstart, final):
    """-> (index the value ends at, index to resume scanning), or (None, None).

    The next `<parameter=` is a HARD LIMIT on the search. Without it, a call
    that omits one closing tag —

        <parameter=a>\n1\n<parameter=b>\n2\n</parameter>

    — lets parameter `a` swallow `b` by claiming b's closing tag. Bounding the
    window first is what makes the missing-tag tolerance safe rather than
    destructive.

    `final` = the text is complete. Mid-stream, a closing tag with nothing
    after it yet is NOT accepted: more value may still be arriving, and an
    argument emitted early is unrecoverable on a streaming wire.
    """
    nxt = body.find("<parameter=", vstart)
    limit = len(body) if nxt < 0 else nxt
    i, last = body.find(PARAM_END, vstart), None
    while 0 <= i < limit:
        last = i
        # Lookahead, not first-match: a value may legitimately CONTAIN the
        # text '</parameter>' (any tool that writes code about tools will
        # produce one), and the closing tag is the occurrence followed by the
        # next parameter or the end of the function.
        if _AFTER_END_RE.match(body[i + len(PARAM_END):]):
            return i, i + len(PARAM_END)
        i = body.find(PARAM_END, i + 1)
    if last is not None and (final or nxt >= 0):
        return last, last + len(PARAM_END)
    if nxt >= 0:                 # closing tag omitted; the next one ends it
        return nxt, nxt
    if final:
        return len(body), len(body)
    return None, None


def _scan_params(body, final=True):
    """-> ([(name, raw_value)] for every COMPLETE parameter in a function
    body, (name, text_so_far) for the one still being written or None)."""
    out, pos = [], 0
    while True:
        m = _PARAM_RE.search(body, pos)
        if not m:
            return out, None
        end, nxt = _value_end(body, m.end(), final)
        if end is None:
            return out, (m.group(1).strip(), body[m.end():])
        raw = body[m.end():end]
        if raw.endswith("\n"):      # the template's own separator, not data
            raw = raw[:-1]
        out.append((m.group(1).strip(), raw))
        pos = nxt


def _parse_params(body, final=True):
    """-> [(name, raw_value)] for every COMPLETE parameter in a function body."""
    return _scan_params(body, final)[0]


def _stream_safe(sofar):
    """The prefix of a parameter value still being written that is CERTAIN
    to be a prefix of the value `_scan_params` will finally return, so it can
    go on the wire now (public issue #36).

    Three things are held back. Everything from the LAST '</parameter>' on:
    a value may contain that text (code about tools), and `_value_end`
    resolves which occurrence closes it only when the text after one is a
    legal follower, or, at the end, by taking the last one, so nothing after
    the last occurrence is known to be value. A tail that could be the start
    of a tag ('<', '</par', '<param'), for the same reason. And trailing
    whitespace, because the separator newline before the closing tag is not
    data and a block that ends without its tags is stripped by the block
    regex; either way it is released the moment a non-blank character
    follows it. Holding more is always safe, it only delays.
    """
    j = sofar.rfind(PARAM_END)
    if j >= 0:
        sofar = sofar[:j]
    else:
        k = sofar.rfind("<")
        if k >= 0 and (PARAM_END.startswith(sofar[k:])
                       or "<parameter=".startswith(sofar[k:])):
            sofar = sofar[:k]
    return sofar.rstrip()


def _parse_block(block, types, final=True):
    """One <tool_call> body -> (name, [(param, raw)], pending) or None if
    malformed. `pending` is the parameter still being written, see
    `_scan_params`; always None once the block is complete."""
    m = _FUNC_RE.search(block)
    if not m:
        return None
    name = m.group(1).strip()
    if not name:
        return None
    body = block[m.end():]
    cut = body.find("</function>")
    if cut >= 0:
        body = body[:cut]
    elif not final:
        params, pending = _scan_params(body, final=False)
        return name, params, pending  # mid-stream: the call is still arriving
    # A complete <tool_call>...</tool_call> whose </function> the model
    # forgot is still a call. The closing tool_call tag already proves the
    # block ended, so refusing it here would drop a real call over a typo.
    return name, _parse_params(body, final=final), None


def build_arguments(name, params, types):
    d = {}
    fn_types = types.get(name, {})
    for k, raw in params:
        d[k] = coerce(raw, fn_types.get(k))
    return json.dumps(d, ensure_ascii=False)


def split_tool_calls(text, tools=None, make_id=new_call_id):
    """-> (content_or_None, [tool_call, ...], truncated)

    `content` is every span OUTSIDE a well-formed <tool_call> block, joined and
    stripped — None when empty, which is what OpenAI puts on a pure tool turn.
    A block that never closed, or that carries no <function=...>, is NOT
    invented into a tool call: its raw text stays in content and `truncated`
    goes true, so a run cut off by max_tokens reads as truncated instead of as
    a call the model did not finish making.
    """
    types = schema_types(tools)
    calls, gaps, pos, truncated = [], [], 0, False
    for m in _BLOCK_RE.finditer(text):
        parsed = _parse_block(m.group(1), types)
        if parsed is None:
            truncated = True
            continue                # leave the raw block inside `content`
        gaps.append(text[pos:m.start()])
        pos = m.end()
        name, params, _ = parsed
        calls.append({"id": make_id(), "type": "function",
                      "function": {"name": name,
                                   "arguments": build_arguments(name, params,
                                                                types)}})
    tail = text[pos:]
    if TC_OPEN in tail:
        truncated = True
    gaps.append(tail)
    content = "".join(gaps).strip()
    return (content or None), calls, truncated


class ToolStream:
    """Incremental splitter for SSE: content deltas out one side, OpenAI
    `delta.tool_calls` events out the other.

    Driven by the FULL accumulated content string rather than by deltas, the
    same way ThinkSplit is, because a marker can arrive split across two
    tokens and only the full text is unambiguous.

    GRANULARITY (public issue #36, and the mechanism behind #3). Until 0.6.1
    nothing left this splitter before '</function>' arrived, so a call's
    id, name and every argument went out together in one burst: a 150-char
    bash command was 1.4 s of dead air, a file-writing call was minutes, and
    Node's undici (a 300 s inactivity timer, no client setting) hung up on
    the long ones. Now the id+name event goes out as soon as the function
    tag closes, a non-string parameter goes out when its value closes (the
    wire carries values untyped, so an integer cannot be typed before its
    last digit), and a STRING parameter streams: its JSON opening `"key": "`
    on the parameter tag, then the value as it arrives, JSON-escaped in
    pieces (the escaping is per character, so pieces concatenate to exactly
    `json.dumps` of the whole), then the closing quote. What is held back
    while a value is open is decided by `_stream_safe`. The concatenation of
    every fragment is byte-identical to the non-streaming `arguments`, which
    `tools/smoke-toolstream.py` asserts on a battery of hostile values. Only
    a parameter whose schema says `string` streams; an untyped one takes the
    lossy guess path at the end, as before.

    A run cut off by max_tokens inside a call now leaves that call's partial
    arguments on the wire with `finish_reason: "length"`, which is what
    OpenAI does; the non-streaming path keeps dropping the unfinished call.
    """

    def __init__(self, tools=None, make_id=new_call_id):
        self.types = schema_types(tools)
        self.make_id = make_id
        self.sent = ""      # content already emitted
        # per-index: {name, n: params emitted whole, closed,
        #             open: key of the string parameter mid-stream or None,
        #             v: chars of its value already on the wire}
        self.calls = []

    @staticmethod
    def _esc(s):
        """JSON-escape a piece of a string value: json.dumps minus the quotes.
        Per-character, so pieces concatenate to the whole's escaping."""
        return json.dumps(s, ensure_ascii=False)[1:-1]

    def _outside(self, text):
        """Text that is definitely NOT part of a tool call, held back at the
        tail so a half-arrived '<tool_call>' never leaks out as content."""
        gaps, pos = [], 0
        for m in _BLOCK_RE.finditer(text):
            gaps.append(text[pos:m.start()])
            pos = m.end()
        tail = text[pos:]
        k = tail.find(TC_OPEN)
        if k >= 0:
            gaps.append(tail[:k])
        else:
            hold = 0
            for n in range(len(TC_OPEN) - 1, 0, -1):
                if tail.endswith(TC_OPEN[:n]):
                    hold = n
                    break
            gaps.append(tail[:len(tail) - hold])
        return "".join(gaps)

    def _blocks(self, text):
        """-> [(body, complete)] including the block still being written."""
        out, pos = [], 0
        for m in _BLOCK_RE.finditer(text):
            out.append((m.group(1), True))
            pos = m.end()
        tail = text[pos:]
        k = tail.find(TC_OPEN)
        if k >= 0:
            out.append((tail[k + len(TC_OPEN):], False))
        return out

    def push(self, full):
        """-> (content_delta, [tool_call_delta, ...])"""
        out = self._outside(full)
        if out.startswith(self.sent):
            delta = out[len(self.sent):]
        else:                       # retokenization: emit only the difference
            n = 0
            while n < len(self.sent) and n < len(out) and self.sent[n] == out[n]:
                n += 1
            delta = out[n:]
        self.sent = out

        events = []
        for i, (body, complete) in enumerate(self._blocks(full)):
            parsed = _parse_block(body, self.types, final=complete)
            if parsed is None:
                continue
            name, params, pending = parsed
            while len(self.calls) <= i:
                self.calls.append(None)
            st = self.calls[i]
            if st is None:
                st = self.calls[i] = {"name": name, "n": 0, "closed": False,
                                      "open": None, "v": 0}
                events.append({"index": i, "id": self.make_id(),
                               "type": "function",
                               "function": {"name": name, "arguments": ""}})
            if st["closed"]:
                continue
            fn_types = self.types.get(name, {})

            def emit(frag):
                if frag:
                    events.append({"index": i,
                                   "function": {"arguments": frag}})
            for k, raw in params[st["n"]:]:
                if st["open"] is not None:
                    # the parameter that was streaming just closed: the
                    # rest of its value, then the quote. What went out is
                    # raw[:v] by construction (_stream_safe, and the text
                    # this is driven by only ever grows).
                    emit(self._esc(raw[st["v"]:]) + '"')
                    st["open"], st["v"] = None, 0
                else:
                    emit(("{" if st["n"] == 0 else ", ")
                         + json.dumps(k, ensure_ascii=False) + ": "
                         + json.dumps(coerce(raw, fn_types.get(k)),
                                      ensure_ascii=False))
                st["n"] += 1
            if pending is not None and fn_types.get(pending[0]) == "string":
                k, sofar = pending
                if st["open"] is None:
                    emit(("{" if st["n"] == 0 else ", ")
                         + json.dumps(k, ensure_ascii=False) + ': "')
                    st["open"], st["v"] = k, 0
                safe = _stream_safe(sofar)
                if len(safe) > st["v"]:
                    emit(self._esc(safe[st["v"]:]))
                    st["v"] = len(safe)
            if complete:
                events.append({"index": i,
                               "function": {"arguments":
                                            "{}" if st["n"] == 0 else "}"}})
                st["closed"] = True
        return delta, events

    def any_calls(self):
        return any(c and c["closed"] for c in self.calls)


def normalize_messages(messages):
    """Make an OpenAI-shaped history renderable by THIS template.

    Two fixes, both of which are the difference between a tool loop that runs
    and a 400 on turn two:

    F3 — OpenAI's `arguments` is a JSON STRING; the template does
    `arguments|items` and raises `Can only get item pairs from a mapping` on
    anything that is not a dict. Every client echoes our own response back, so
    this fires on the very first round trip.

    F5 — the template renders tool results POSITIONALLY and never emits
    `tool_call_id`. A client that answers two parallel calls out of order gets
    them silently transposed, so a run of tool messages is reordered to match
    the preceding assistant's `tool_calls` when — and only when — every id
    lines up. Anything less and the order is left exactly as the client sent
    it, because guessing here would corrupt rather than repair.

    Raises ValueError with a client-readable message; the caller turns that
    into a 400.
    """
    msgs = []
    for m in messages:
        if not isinstance(m, dict):
            raise ValueError("each message must be an object")
        m = dict(m)
        if m.get("role") == "function":     # legacy client spelling
            m["role"] = "tool"
        # Public issue #32: OpenAI's `developer` role is what clients send
        # for a reasoning model's instructions (pi does, for every model it
        # marks as reasoning), and this template only knows `system`. They
        # mean the same thing here; the template rejected it with a 400.
        if m.get("role") == "developer":
            m["role"] = "system"
        tcs = m.get("tool_calls")
        if m.get("role") == "assistant" and isinstance(tcs, list):
            fixed = []
            for tc in tcs:
                tc = dict(tc) if isinstance(tc, dict) else {}
                fn = tc.get("function")
                fn = dict(fn) if isinstance(fn, dict) else tc
                args = fn.get("arguments")
                if isinstance(args, str):
                    s = args.strip()
                    try:
                        args = json.loads(s) if s else {}
                    except Exception as e:
                        raise ValueError(
                            f"tool_calls[].function.arguments for "
                            f"'{fn.get('name')}' is not valid JSON: {e}")
                    if not isinstance(args, dict):
                        raise ValueError(
                            f"tool_calls[].function.arguments for "
                            f"'{fn.get('name')}' must decode to an object")
                    fn["arguments"] = args
                elif args is None:
                    fn["arguments"] = {}
                tc["function"] = fn
                fixed.append(tc)
            m["tool_calls"] = fixed
        msgs.append(m)

    i = 0
    while i < len(msgs):
        if msgs[i].get("role") != "tool":
            i += 1
            continue
        j = i
        while j < len(msgs) and msgs[j].get("role") == "tool":
            j += 1
        prev = msgs[i - 1] if i else None
        order = {}
        if prev and isinstance(prev.get("tool_calls"), list):
            for n, tc in enumerate(prev["tool_calls"]):
                cid = tc.get("id") if isinstance(tc, dict) else None
                if isinstance(cid, str):
                    order[cid] = n
        run = msgs[i:j]
        ids = [r.get("tool_call_id") for r in run]
        if order and all(x in order for x in ids) and len(set(ids)) == len(ids):
            msgs[i:j] = sorted(run, key=lambda r: order[r["tool_call_id"]])
        i = j
    return msgs
