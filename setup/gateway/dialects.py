#!/usr/bin/env python3
"""dialects — the one place that knows how a request body is built.

Two dialects reach the same llama-server, and the stack has to read the same
things out of both:

    anthropic   POST /v1/messages            Claude Code, cc-router
                system: str | [ {type:text,text} ]      separate field
                tools:  [ {name, description, input_schema} ]
    openai      POST /v1/chat/completions    DeepSeek Harness (dsh), most others
                system prompt = messages[0] with role "system"
                tools:  [ {type:function, function:{name,description,parameters}} ]

Why a module of its own, and why both gateway.py and prewarm.py import it:
the prefix id decides whether a saved state is ever found again. That logic
used to exist TWICE — once in the gateway, once in prewarm — and the two
drifted apart: prewarm recomputed the id from the already-corrected body and
wrote a key that no incoming request ever produced. Everything looked fine
(SAVED in the log, file on disk, 628 MB) and RESTORED simply never appeared
for two days. One module, one truth, and tests that hold both callers against
each other (tests/test_dialects.py, tests/test_gateway.py::TestIdContract).

Design rules for anything added here:
  * the ENDPOINT decides the dialect, never a guess at the body — a body can
    look like both, a path cannot;
  * every function takes the dialect explicitly, no hidden default magic;
  * nothing here talks to the network or reads configuration. It is pure
    body-in/body-out so it can be tested without a GPU and without a server.
"""
import hashlib
import json

ANTHROPIC = "anthropic"
OPENAI = "openai"

# Paths that carry a prompt worth accounting for. count_tokens is deliberately
# absent: it costs no inference and must not occupy a slot.
INFERENCE_PATHS = ("/v1/messages", "/v1/chat/completions")

# How much of the prefix the head id covers. Part of the id contract, so it
# lives with the id and not in one of the two callers — the gateway and
# prewarm have to agree on it byte for byte.
HEAD_BYTES = 8192


def detect(path):
    """Which dialect an endpoint speaks. The path decides, never the body."""
    clean = (path or "").split("?", 1)[0].rstrip("/")
    return OPENAI if clean.startswith("/v1/chat/completions") else ANTHROPIC


def is_inference(path):
    """True for paths that actually run a prompt through the model."""
    clean = (path or "").split("?", 1)[0].rstrip("/")
    if clean.endswith("count_tokens"):
        return False
    return any(clean.startswith(p) for p in INFERENCE_PATHS)


def blocks_to_text(content):
    """Text of a content field, whatever shape it has.

    Covers all four that occur in the wild: a plain string, Anthropic blocks,
    OpenAI content parts (same {type,text} shape), and None.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content
                       if isinstance(b, dict) and b.get("type") == "text")
    return ""


def system_head(body, dialect):
    """The stable system text — the front of what the model will see."""
    if dialect == OPENAI:
        msgs = body.get("messages")
        if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict) \
                and msgs[0].get("role") == "system":
            return blocks_to_text(msgs[0].get("content"))
        return ""
    return blocks_to_text(body.get("system"))


def set_system_head(body, text, dialect):
    """Replace the stable system text in place, creating it if needed."""
    if dialect == OPENAI:
        msgs = body.setdefault("messages", [])
        if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
            msgs[0]["content"] = text
        else:
            msgs.insert(0, {"role": "system", "content": text})
        return body
    s = body.get("system")
    if isinstance(s, list):
        # Keep the existing blocks (they carry cache_control) and put the
        # addition in its own block — rewriting them would drop the
        # cache_control markers Claude Code sets.
        head = blocks_to_text(s)
        extra = text[len(head):]
        if extra:
            body["system"] = s + [{"type": "text", "text": extra}]
    else:
        body["system"] = text
    return body


def tools_signature(body):
    """Stable text of the tool block. Dialect-independent on purpose: the
    shapes differ, so the ids differ — which is right, because the rendered
    prompts differ too and must not share a slot."""
    return json.dumps(body.get("tools") or [], sort_keys=True)


def template_kwargs_signature(body):
    """The chat template kwargs, canonically — or "" when there are none.

    Sorted, because two dicts with the same pairs in a different order are the
    same request and must not be two keys. Absent and `{}` both render as ""
    for the same reason: both mean "whatever the server was started with".
    """
    kw = body.get("chat_template_kwargs")
    if not isinstance(kw, dict) or not kw:
        return ""
    return json.dumps(kw, sort_keys=True, separators=(",", ":"))


def prefix_text(body, dialect):
    """The stable start of a request: system head, tools, and the mode.

    Exactly what decides whether llama.cpp can reuse a slot — and until
    28.08.2026 this function got that wrong by leaving out the third term.

    chat_template_kwargs are not metadata. The served Qwen template puts
    `Reasoning effort is set to …` at the FRONT of the first system block,
    before the tools, so the gateway's three model names for one loaded model
    render three different prompts diverging at character 19. Measured against
    the running server:

        off (server default)  sha fe8d7ee8   1108 chars
        think  (low)          sha 677f3ace   1235 chars
        deep   (medium)       sha aa6c7b7d   1097 chars

    They shared one id. So prewarm saved one rendering, the daily driver asked
    for another, the restore diverged at token ~5, and the gateway logged
    RESTORED and counted it warm — a prefill wearing a cache hit's clothes.
    """
    return (system_head(body, dialect) + "\x00" + tools_signature(body)
            + "\x00" + template_kwargs_signature(body))


def prefix_id(body, dialect, head_bytes=HEAD_BYTES):
    """(full id, head id) of the stable prompt start.

    The head id catches prefixes that share a beginning and would therefore
    fight over the same slot; the gateway warns about those.
    """
    import hashlib
    full = prefix_text(body, dialect)
    return (hashlib.sha256(full.encode("utf-8")).hexdigest()[:12],
            hashlib.sha256(full[:head_bytes].encode("utf-8")).hexdigest()[:12])


def iter_system_messages(body):
    """(index, message) of every system message inside `messages`."""
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return
    for i, m in enumerate(msgs):
        if isinstance(m, dict) and m.get("role") == "system":
            yield i, m


def hoist_system_messages(body, dialect, volatile_patterns):
    """Move the STABLE part of system messages to the front of the prompt.

    Claude Code appends a system block BEHIND the user question and glues a
    counter to its end. Left in place, the counter changes every turn, the
    prefix id changes with it, and every request runs cold. Hoisting the
    stable part and leaving only the volatile remainder behind keeps the
    prefix identical across turns.

    Returns (body, number of volatile fragments left in place). Bodies
    without mid-conversation system messages come back untouched, so this is
    safe to call for every request in either dialect.
    """
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return body, 0
    hoisted, keep, seen = [], [], set()
    n_volatile = 0
    for i, m in enumerate(msgs):
        is_system = isinstance(m, dict) and m.get("role") == "system"
        # messages[0] IS the system prompt in OpenAI — it is already at the
        # front and must stay where it is.
        if not is_system or (dialect == OPENAI and i == 0):
            keep.append(m)
            continue
        text = blocks_to_text(m.get("content"))
        stable = text
        volatile = []
        for rx in volatile_patterns:
            volatile.extend(rx.findall(stable))
            stable = rx.sub("", stable)
        stable = stable.rstrip()
        if volatile:
            n_volatile += len(volatile)
            keep.append({"role": "system", "content": "".join(volatile)})
        if stable and stable not in seen:
            seen.add(stable)
            hoisted.append(stable)
    if not hoisted:
        return body, n_volatile
    body["messages"] = keep
    set_system_head(body, (system_head(body, dialect) + "\n\n"
                           + "\n\n".join(hoisted)).strip("\n"), dialect)
    return body, n_volatile


def mid_system_to_user(body, dialect):
    """Turn non-leading system messages into user messages.

    Some chat templates reject them outright (Qwen 3.8: "System message must
    be at the beginning", HTTP 500 for every Claude Code request). Position
    and text are preserved, so the prompt keeps its length and the prefix id
    is untouched — only the role marker changes. The content SHAPE stays
    dialect-correct: Anthropic wants blocks, OpenAI wants the string.
    """
    n = 0
    for i, m in iter_system_messages(body):
        if i == 0:
            continue
        content = m.get("content")
        m["role"] = "user"
        if dialect == ANTHROPIC and isinstance(content, str):
            m["content"] = [{"type": "text", "text": content}]
        n += 1
    return body, n


def to_openai_tools(body, dialect):
    """The tool block in OpenAI shape — what /apply-template expects."""
    tools = body.get("tools") or []
    if dialect == OPENAI:
        return tools
    return [{"type": "function",
             "function": {"name": t.get("name", ""),
                          "description": t.get("description", ""),
                          "parameters": t.get("input_schema", {})}}
            for t in tools if isinstance(t, dict)]


def model_listing_arrays(listing):
    """Every array in a /v1/models answer that lists models.

    llama-server answers with BOTH `models` (Ollama-flavoured) and `data`
    (OpenAI standard) in the same body. Extending only the first one — as
    this stack did until 26.08. — leaves an OpenAI client seeing exactly
    one model, which is the dialect the extension was written for.
    """
    return [k for k in ("models", "data")
            if isinstance(listing.get(k), list) and listing[k]]


def template_payload(body, dialect, probe="X"):
    """The body for llama-server's /apply-template: system head, tools, and
    one throwaway user turn whose marker says where the prefix ends."""
    payload = {"messages": [{"role": "system",
                             "content": system_head(body, dialect)},
                            {"role": "user", "content": probe}],
               "tools": to_openai_tools(body, dialect)}
    # The mode travels with it, or prewarm renders the server default whatever
    # mode the request that triggered the save was in — and saves a state that
    # the request can never match. See prefix_text.
    kw = body.get("chat_template_kwargs")
    if isinstance(kw, dict) and kw:
        payload["chat_template_kwargs"] = kw
    return payload


# --- what the server actually did -------------------------------------------
#
# Until 28.08.2026 the gateway's warm/cold label was a CLAIM: "I have seen this
# prefix id before". Whether llama.cpp then reused anything was never asked,
# and on that day a restore from disk was logged warm, cost a full 14960-token
# prefill, and nothing anywhere disagreed — see
# `saved-prefix-holds-a-foreign-state` in setup/defects.json.
#
# The answer was in the response the whole time. llama-server reports, per
# request, how many prompt tokens came out of a cache and how many it had to
# compute. Three shapes carry it, and a gateway that speaks two dialects meets
# all three:
#
#   llama.cpp OAI    "timings": {"cache_n": 14957, "prompt_n": 4}
#   OpenAI usage     "usage": {"prompt_tokens": 14961,
#                              "prompt_tokens_details": {"cached_tokens": 14957}}
#   Anthropic        "usage": {"cache_read_input_tokens": 14957,
#                              "input_tokens": 4}
#
# `timings` is preferred where present: it is llama.cpp's own accounting rather
# than a translation of it.

def _pair(reused, evaluated):
    ok = (isinstance(reused, int) and isinstance(evaluated, int)
          and reused >= 0 and evaluated >= 0)
    return (reused, evaluated) if ok else None


def reuse_from_object(obj):
    """(reused, evaluated) from ONE parsed response object, or None.

    Anthropic's streaming carries the input accounting in `message_start`,
    which is the FIRST event, not the last — so a caller sniffing a stream has
    to offer this function the head as well as the tail.
    """
    if not isinstance(obj, dict):
        return None
    t = obj.get("timings")
    if isinstance(t, dict):
        got = _pair(t.get("cache_n"), t.get("prompt_n"))
        if got:
            return got
    for u in (obj.get("usage"),
              (obj.get("message") or {}).get("usage")
              if isinstance(obj.get("message"), dict) else None):
        if not isinstance(u, dict):
            continue
        got = _pair(u.get("cache_read_input_tokens"), u.get("input_tokens"))
        if got:
            return got
        det = u.get("prompt_tokens_details")
        if isinstance(det, dict) and isinstance(u.get("prompt_tokens"), int):
            cached = det.get("cached_tokens")
            if isinstance(cached, int):
                got = _pair(cached, u["prompt_tokens"] - cached)
                if got:
                    return got
    return None


def reuse_from_text(text):
    """(reused, evaluated) from a response body or an SSE fragment, or None.

    Tolerant on purpose. It is fed the head and the tail of a proxied stream,
    so it will see truncated JSON, half events and keep-alive comments, and it
    must answer "I do not know" rather than raise — a gateway that dies over
    its own bookkeeping is worse than one that cannot label a request.
    """
    if not text:
        return None
    best = None
    for chunk in text.split("\n"):
        line = chunk.strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        got = reuse_from_object(obj)
        if got:
            # The LAST complete answer wins: in an Anthropic stream the head
            # carries message_start's accounting, and nothing later contradicts
            # it; in an OAI stream the final chunk carries `timings`, which is
            # the better source.
            best = got
    if best is None:
        try:
            best = reuse_from_object(json.loads(text))
        except Exception:
            best = None
    return best


def output_from_object(obj):
    """How many tokens the model WROTE, from one parsed object, or None.

    The counterpart to reuse_from_object, and it comes from the same three
    shapes: llama.cpp's own `timings.predicted_n`, Anthropic's
    `usage.output_tokens`, OpenAI's `usage.completion_tokens`.

    Anthropic streams it in `message_delta`, at the END — the opposite end
    from where that dialect puts the input accounting, which is why a caller
    sniffing a stream has to keep both its head and its tail.
    """
    if not isinstance(obj, dict):
        return None
    t = obj.get("timings")
    if isinstance(t, dict) and isinstance(t.get("predicted_n"), int):
        return t["predicted_n"]
    for u in (obj.get("usage"),
              (obj.get("message") or {}).get("usage")
              if isinstance(obj.get("message"), dict) else None):
        if not isinstance(u, dict):
            continue
        for key in ("output_tokens", "completion_tokens"):
            if isinstance(u.get(key), int):
                return u[key]
    return None


def output_from_text(text):
    """Written tokens from a response body or SSE fragment, or None.

    The LAST number wins: an Anthropic stream reports output_tokens in
    message_start as well, where it is 1 or 2 and means nothing yet — the
    figure that counts is the one in message_delta at the end.
    """
    if not text:
        return None
    best = None
    for chunk in text.split("\n"):
        line = chunk.strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        got = output_from_object(obj)
        if got is not None:
            best = got
    if best is None:
        try:
            best = output_from_object(json.loads(text))
        except Exception:
            best = None
    return best


def rates_from_text(text):
    """(reading, writing) in tokens per second, from `timings`, or None.

    llama.cpp measures both phases itself and reports them as
    `prompt_per_second` (the prefill — reading) and `predicted_per_second`
    (the decode — writing). They are two different machines: measured on this
    stack, reading runs at ~200 t/s and writing at ~10-15.

    ONLY from `timings`, and never derived. A request's total duration cannot
    be split into the two phases afterwards — a rate computed from it would be
    neither of them, and a number that looks like a measurement but is a guess
    is worse than an empty column.

    That also means these are absent for the Anthropic route: llama.cpp's
    to_json_anthropic() (server-task.cpp:779) builds id, type, role, content,
    model, stop_reason, stop_sequence and usage — no timings. So Claude Code
    traffic has no rates and this returns None, which is the honest answer.
    """
    if not text:
        return None
    best = None
    for chunk in text.split("\n"):
        line = chunk.strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line.startswith("{"):
            continue
        try:
            t = json.loads(line).get("timings")
        except Exception:
            continue
        if not isinstance(t, dict):
            continue
        r, w = t.get("prompt_per_second"), t.get("predicted_per_second")
        ok = lambda v: isinstance(v, (int, float)) and v > 0
        if ok(r) or ok(w):
            best = (round(r, 1) if ok(r) else None,
                    round(w, 1) if ok(w) else None)
    return best


def message_shape(body, dialect=ANTHROPIC):
    """One short hash per message, in order — the SHAPE of a conversation.

    Two requests for the same prefix normally share every message but the last
    few: the client appends. When they do not, the client has rewritten its own
    history — a compaction, a truncated tool result, a re-ordered turn — and
    everything after the change has to be computed again.

    That is indistinguishable, from the outside, from the slot having been
    taken away: both show up as a drop in `reused`. On 29.08.2026 it cost two
    rounds of blaming the wrong thing — the watchdog was measured at 209,587
    re-prefilled tokens a week, and the arithmetic later said the client had
    edited its own history while the probe stood next to it.

    Comparing the two shapes tells them apart:

        the shapes agree, reuse dropped   -> the state was lost
        the shapes diverge, reuse dropped -> the client rewrote

    Per MESSAGE, not per token: a change inside one long message marks the
    whole message. That is enough for the distinction and cheap enough for the
    request path — a few dozen hashes over text already in memory.
    """
    return [h for h, _ in message_fingerprints(body, dialect)]


def message_fingerprints(body, dialect=ANTHROPIC):
    """(hash, characters) per message — the shape, plus what each one weighs.

    The size is what turns a LOCATED divergence into a diagnosis, and the hash
    alone cannot give it: a message that changed and kept its length is a
    re-render (a timestamp, a counter, a reordered block), one that shrank is a
    truncation, one that grew is an edit. On 29.08.2026 the difference decided
    whether an 18,450-token re-prefill was the client rewriting one line or
    compacting a third of the conversation, and there was no way to tell.

    Cheap for the same reason `message_shape` is: the canonical form is built
    once and its length is already in hand.
    """
    out = []
    for m in (body.get("messages") or []):
        try:
            canon = json.dumps(renderable(m), sort_keys=True,
                               separators=(",", ":"), ensure_ascii=False)
        except Exception:
            canon = repr(m)
        out.append((hashlib.sha1(canon.encode("utf-8", "replace")).hexdigest()[:8],
                    len(canon)))
    return out


# Fields that ride along in a request and never reach the rendered prompt.
# `cache_control` is Anthropic's own cache breakpoint, and Claude Code MOVES it
# forward as a conversation grows — so a message whose text has not changed by
# one character arrives with a different JSON on the next turn.
#
# Measured 30.08.2026: two consecutive turns, message 1 identical in all 5,100
# characters of its text, differing ONLY by the presence of
# {"cache_control": {"type": "ephemeral"}}. The shape reported a rewritten
# history, `msgs_kept` fell to 1 of 2, and the reading pointed at the wrong
# message for the re-prefill that had actually happened at the assistant turn.
# An instrument that says "the client rewrote its history" because a cache hint
# moved is worse than no instrument: it is confidently wrong, in a comparison
# whose entire purpose is to tell two causes apart.
IGNORED_IN_SHAPE = ("cache_control",)


def renderable(message):
    """A message stripped of what the chat template never sees.

    Conservative on purpose: only fields KNOWN not to render are dropped. A
    field wrongly kept costs a false "rewritten"; a field wrongly dropped hides
    a real one, and that is the worse mistake of the two.
    """
    try:
        m = json.loads(json.dumps(message))
    except Exception:
        return message
    if isinstance(m, dict):
        for k in IGNORED_IN_SHAPE:
            m.pop(k, None)
        c = m.get("content")
        if isinstance(c, list):
            for bl in c:
                if isinstance(bl, dict):
                    for k in IGNORED_IN_SHAPE:
                        bl.pop(k, None)
    return m


def shapes_agree(old, new):
    """How many leading messages are unchanged."""
    n = 0
    for a, b in zip(old or [], new or []):
        if a != b:
            break
        n += 1
    return n
