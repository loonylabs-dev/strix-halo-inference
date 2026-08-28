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

Why a module of its own, and why both cc-gateway.py and prewarm.py import it:
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
