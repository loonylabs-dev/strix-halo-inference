#!/usr/bin/env python3
# ===========================================================================
# VENDORED AND PATCHED — this is NOT this repository's code.
#
#   cut from  ghcr.io/peonist-ai/halogen-flash-server:0.8.1
#             image digest sha256:d444524bffb6f487650cb1c29e3dae67fcebf0c6a22abe5a6342f0e4315567e0
#   base file /halogen/tools/serve_api.py
#   BASE_SHA256 = 018c899b43e496bb27c082a7ffe2f1da17d898144e1f5832607d797478d55cb6
#
# setup/halogenexec mounts this file OVER the one in the image and verifies
# BASE_SHA256 against the image's own copy before it does. That check is the
# whole reason the hash is written down: without it a bumped image keeps its
# new tag in `podman ps` while this old copy silently reverts every upstream
# change to this one file — the shape of defect this repository keeps
# finding, and the reason setup/patches/ exists for llama.cpp.
#
# WHAT IS CHANGED — four hunks against the base above. Hunks 1-3 are the
# vendored end-token fix this file has carried since 0.6.x; hunk 4 is
# repository policy, added 14.09.2026 and NOT an upstream behaviour.
#
#   1. get_eos_ids()  The model has TWO end tokens, <|im_end|> (248046) and
#      <|endoftext|> (248044), and the base registered only tok.eos_token_id.
#      A generation that emitted the other one was not stopped: it decoded on
#      to max_tokens and repeated itself. Filed as halogen-second-eos-token-
#      unregistered in setup/defects.json.
#   2. the same function takes the request's stop STRINGS and registers the
#      ones that are single tokens as EOS ids, so a client's stop list is
#      honoured by the engine rather than only by the text matcher.
#   3. app.state bindings: exposes app.state.run and app.state.serve for tests.
#   4. the fit-to-room gate (FIT_TO_ROOM / FIT_TO_ROOM_FLOOR, and the clamp in
#      serve()): a max_tokens that does not fit the window the prompt left is
#      clamped to that room when HALOGEN_FIT_TO_ROOM=1, and refused with the
#      numbers as before when the gate is off or the room is below
#      HALOGEN_FIT_TO_ROOM_FLOOR=1024. Enabled PER UNIT — every halogen unit
#      carries `HALOGEN_FIT_TO_ROOM=1` as an Environment= line (so the knob
#      survives a rebuild and a switch-model to the 27B) — and the shared
#      template and the bench side servers deliberately do not.
#
# RETIREMENT: hunks 1-3 go when an image ships them — check the upstream
# changelog on every bump. Hunk 4 goes only if upstream ships an equivalent
# clamp; until then it stays even on an image whose other hunks have landed.
# The mount check refuses the start when the base hash moves, and says so.
# ===========================================================================
"""serve_api.py — the OpenAI-compatible front-end.

Owns everything text-shaped: the BPE tokenizer, Qwen's Jinja2 chat
template, the OpenAI request/response schema, and SSE framing. Talks to
the engine daemon (`halogen --serve`) over the newline-framed integer
protocol in src/serve.cpp — the engine only ever sees token ids.

This is `tools/`-shaped by design: it never touches a
kernel, and the engine stays framework-free and CLI-testable. A C++
tokenizer is the recorded future path, after the eval harness exists.

Endpoints: /v1/models, /v1/completions, /v1/chat/completions,
/v1/responses (all four support stream=true), /health.

Run (box, in the container — tools/run-serve.sh wraps both processes):
    python3 tools/serve_api.py --tokenizer <snapshot> --engine 127.0.0.1:8730
"""

import argparse
import asyncio
import base64
import binascii
import io
import math
import itertools
import json
import os
import random
import sys
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, model_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tool_parse import (ToolStream, normalize_messages,      # noqa: E402
                        split_tool_calls)

SSE_PROF = {"t": 0.0, "n": 0}
# This string is what `/v1/models` lists and what every OpenAI client echoes
# back in `model`, so it must name the model actually being served: a client
# routing by model id has nothing else to go on. Verify it by calling the
# endpoint, not by reading the engine, which never sees it.
# Overridable so one host can run two stacks without the ids colliding.
MODEL_ID = os.environ.get("HALOGEN_MODEL_ID", "halogen-qwen3.8-flash-next")

# OpenAI's error envelope. FastAPI's native shape is {"detail": "..."}, which
# every OpenAI SDK misreads: they look for error.message and fall back to the
# raw body, so a perfectly clear 400 arrives at the user as an unparsed blob.
# Nothing in this repo consumed `detail`, so this replaces it rather than
# adding beside it.
ERROR_TYPES = {400: "invalid_request_error", 401: "authentication_error",
               403: "permission_error", 404: "not_found_error",
               429: "rate_limit_error"}



# ── vision: decode, resize, and hand the engine raw RGB ──────────────────────
#
# The engine patchifies and normalises; this side only has to produce pixels
# at a size the tower's geometry accepts. That split keeps the front-end free
# of a tensor framework -- it needs a decoder and nothing else -- and puts the
# patch-order rule in the one file that documents it.
#
# ONE APPROXIMATION, stated rather than hidden: the reference resizes in float
# after rescaling, and this resizes in 8-bit and sends bytes. The difference is
# at most half a level in 255, which is the same order as the bf16 rounding the
# tower already carries, and it buys a wire payload of H*W*3 instead of four
# times that.
VIS_PATCH = 16
VIS_MERGE = 2
VIS_UNIT = VIS_PATCH * VIS_MERGE          # both sides must be a multiple of 32
VIS_MIN_PIXELS = 256 * 256
# 2560x1440, measured rather than assumed: end to end one image costs
# 5.5 / 11.8 / 25.3 / 105.8 s at 1280x800 / 1080p / 1440p / 4K and reads no
# better above 1440p. Oversize is DOWNSCALED here, not refused.
VIS_MAX_PIXELS = int(os.environ.get("HALOGEN_VISION_MAX_PIXELS", 2560 * 1440))
# The image stamps its release into HALOGEN_IMAGE_VERSION (the Containerfile
# sets it from the same VERSION build arg as the OCI label). Public issue #26
# was an api at 0.4.4 in front of an engine at 0.5.6, and nothing in either
# container's log or in /health said a version, so a user reading both had no
# way to see they disagreed. The engine reports its own on the INFO line and
# this front-end compares; see version_status().
API_VERSION = (os.environ.get("HALOGEN_IMAGE_VERSION") or "unknown").strip() \
    or "unknown"
# Resolved from the tokenizer at startup, never hardcoded: the id is 248056 on
# this checkpoint and a literal would be one more fork leftover waiting to
# happen.
IMAGE_TOKEN_ID = None
# PUBLIC ISSUE #39: A MENTION OF THE PLACEHOLDER IS NOT A PLACEHOLDER.
#
# The tokenizer parses added tokens out of ordinary text, so a user who
# types `<|image_pad|>` (a pasted log, a quoted traceback, a bug report
# about this server) or a model that prints it produces the SPECIAL id, the
# same id the chat template writes for an image part. The engine refuses
# that id when no image covers it (public issue #26), and the refusal is
# right for a rendered placeholder with no pixels behind it and wrong for a
# mention: the next turn re-tokenizes the history, so one mention anywhere
# in a conversation, by either side, made every later turn a 400.
#
# A mention now becomes the ordinary tokens of its own spelling (`<`, `|`,
# `image`, `_pad`, `|`, `>` on this tokenizer), decided on IDS after
# tokenization, so the model reads the same characters and the engine never
# sees an uncovered placeholder. Which pads are the template's is decided by
# what the template writes -- see textify_pad_mentions(). The rest of the
# vision family is resolved here for that test; the text spellings are
# computed from the tokenizer, and CHECKED to decode back to the name.
VIDEO_TOKEN_ID = None
VISION_START_ID = None
VISION_END_ID = None
PAD_TEXT_IDS = {}


def pillow_ready():
    """Can this front-end decode an image at all? Answered by IMPORTING it.

    /health reports the answer, so it has to be the same question the request
    path asks -- a version check or a package listing would be a second
    belief about the same thing.
    """
    global _PILLOW
    if _PILLOW is None:
        try:
            from PIL import Image           # noqa: F401
            _PILLOW = True
        except Exception:
            _PILLOW = False
    return _PILLOW


_PILLOW = None


def vis_resolve_token(tok):
    global IMAGE_TOKEN_ID, VIDEO_TOKEN_ID, VISION_START_ID, VISION_END_ID

    def one(name):
        i = tok.convert_tokens_to_ids(name)
        return i if isinstance(i, int) and i >= 0 else None
    IMAGE_TOKEN_ID = one("<|image_pad|>")
    VIDEO_TOKEN_ID = one("<|video_pad|>")
    VISION_START_ID = one("<|vision_start|>")
    VISION_END_ID = one("<|vision_end|>")
    PAD_TEXT_IDS.clear()
    for name, sid in (("<|image_pad|>", IMAGE_TOKEN_ID),
                      ("<|video_pad|>", VIDEO_TOKEN_ID)):
        if sid is None:
            continue
        # The placeholder's spelling as TEXT: what the tokenizer produces
        # when told not to parse special tokens. Verified rather than
        # trusted: the pieces must decode back to the name and none may
        # be the special id itself, or a "mention" would still be a
        # placeholder. A tokenizer that fails this keeps the old behaviour
        # (the engine's refusal), and says so at startup.
        try:
            plain = list(tok(name, add_special_tokens=False,
                             split_special_tokens=True)["input_ids"])
        except TypeError:
            plain = []
        if plain and sid not in plain and tok.decode(plain) == name:
            PAD_TEXT_IDS[sid] = plain
        else:
            print(f"serve_api: {name} has no text spelling on this "
                  f"tokenizer; a message that mentions it will be refused "
                  f"by the engine (issue #39)", flush=True)
    return IMAGE_TOKEN_ID


def textify_pad_mentions(ids, snap=0, snap2=0, n_images=0):
    """-> (ids, snap, snap2, n): every vision placeholder id that is only
    MENTIONED becomes the ordinary tokens of its spelling (issue #39).

    Which pads are the template's, decided by what the template writes: an
    image part renders as `<|vision_start|><|image_pad|><|vision_end|>` and
    nothing else in the template emits a pad. So when the request carries
    images, a pad flanked by that pair is the template's and every other
    pad is a mention; when it carries none, every pad is a mention. A video
    pad is always a mention, because a `video` content part is refused
    before the template runs (vis_collect), so no template ever wrote one.

    The one shape this cannot tell apart: a user who types the whole triple
    AND attaches images in the same request. That pad is counted as the
    template's, the count no longer matches the images, and
    expand_image_pads() refuses with both numbers -- a 400 that names
    itself, never a silently misplaced image.

    The tokenization at a mention is `' '`, `'<'`, `'|'`, ... rather than
    the `' <|'`, ... a plain-text tokenization of the same characters would
    give, because the special token was a hard boundary for its
    neighbours. The characters are identical; only the split differs, and
    only at the mention. Every request without a mention is untouched.

    SNAP and SNAP2 are token COUNTS into `ids`, so growth before either
    shifts it. Both are read from their ORIGINAL values: the shifted value
    is never compared against the running index, which is the mistake
    expand_image_pads() used to make.
    """
    if not PAD_TEXT_IDS or not any(v in PAD_TEXT_IDS for v in ids):
        return ids, snap, snap2, 0
    out, grown, n = [], 0, 0
    snap0, snap20, last = snap, snap2, len(ids) - 1
    for i, v in enumerate(ids):
        plain = PAD_TEXT_IDS.get(v)
        templ = (v == IMAGE_TOKEN_ID and n_images > 0 and 0 < i < last
                 and ids[i - 1] == VISION_START_ID
                 and ids[i + 1] == VISION_END_ID)
        if plain is not None and not templ:
            out.extend(plain)
            grown += len(plain) - 1
            n += 1
        else:
            out.append(v)
        if snap0 and i == snap0 - 1:
            snap = snap0 + grown
        if snap20 and i == snap20 - 1:
            snap2 = snap20 + grown
    return out, snap, snap2, n


def expand_image_pads(ids, snap, snap2, images):
    """One `<|image_pad|>` per image becomes `ntok` of them.

    The template writes `<|vision_start|><|image_pad|><|vision_end|>`, one
    pad per image, and the engine needs one token per merged patch. The
    expansion happens on IDS, after tokenization, because the pad is a
    special token the tokenizer never merges across, so its positions are
    exact.

    SNAP and SNAP2 are token COUNTS into this same list, so every
    expansion before them shifts them; getting that wrong would point the
    prompt cache's snapshot into the middle of an image and silently cost
    every warm turn.
    """
    if not images:
        return ids, snap, snap2, []
    pads = [i for i, v in enumerate(ids) if v == IMAGE_TOKEN_ID]
    if len(pads) != len(images):
        raise HTTPException(400, f"the chat template emitted {len(pads)} "
                                 f"image placeholders for {len(images)} "
                                 f"images")
    out, placed, grown, k = [], [], 0, 0
    # Read the ORIGINAL counts. This used to compare the running index
    # against the SHIFTED snap, so once a shift had happened the test
    # fired again at the new value and added the growth a second time,
    # and a third, for as long as the prompt lasted. Latent for an
    # image, whose growth (63 tokens at the smallest size) outruns the
    # few-token assistant opener that follows the history point, so the
    # shifted value was never reached; found when issue #39's five-token
    # growth per mention did not outrun it.
    snap0, snap20 = snap, snap2
    for i, v in enumerate(ids):
        if k < len(pads) and i == pads[k]:
            H, W, _rgb, ntok = images[k]
            placed.append((len(out), H, W))
            out.extend([IMAGE_TOKEN_ID] * ntok)
            grown += ntok - 1
            k += 1
        else:
            out.append(v)
        if snap0 and i == snap0 - 1:
            snap = snap0 + grown
        if snap20 and i == snap20 - 1:
            snap2 = snap20 + grown
    return out, snap, snap2, placed


def vis_smart_resize(height, width, factor=VIS_UNIT,
                     min_pixels=VIS_MIN_PIXELS, max_pixels=VIS_MAX_PIXELS):
    """transformers' qwen2_vl smart_resize, which cannot be imported here:
    its module pulls in torchvision and this image carries no framework."""
    if min(height, width) <= 0:
        raise ValueError("image has a zero dimension")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("image aspect ratio must be under 200:1")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def vis_decode_part(url):
    """An OpenAI image_url -> (H, W, rgb_bytes, n_tokens).

    Only `data:` URLs and bare base64 are accepted. Fetching an http(s) URL
    would make the server issue arbitrary outbound requests on a client's say
    so, which is a different feature with a different threat model; it is
    refused with a message that says which.
    """
    if not isinstance(url, str) or not url:
        raise ValueError("image_url.url must be a non-empty string")
    if url.startswith(("http://", "https://")):
        raise ValueError("this server does not fetch image URLs; inline the "
                         "image as a data: URL "
                         "(data:image/png;base64,...)")
    payload = url
    if url.startswith("data:"):
        head, _, payload = url.partition(",")
        if "base64" not in head:
            raise ValueError("only base64 data: URLs are supported")
    try:
        raw = base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError):
        raise ValueError("image_url.url is not valid base64")
    if not raw:
        raise ValueError("image payload is empty")
    try:
        from PIL import Image
    except ImportError:
        raise ValueError("this server was built without image support "
                         "(Pillow is not installed)")
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
    except Exception as e:
        raise ValueError(f"cannot decode the image: {e}")
    im = im.convert("RGB")
    W0, H0 = im.size
    if W0 * H0 > VIS_MAX_PIXELS * 4:
        # Refuse before decoding cost becomes the attack, not after.
        raise ValueError(f"image is {W0}x{H0}; the limit before resizing is "
                         f"{VIS_MAX_PIXELS * 4} pixels")
    H, W = vis_smart_resize(H0, W0)
    if (W, H) != (W0, H0):
        from PIL import Image as _I
        im = im.resize((W, H), _I.BICUBIC)
    rgb = im.tobytes()
    if len(rgb) != H * W * 3:
        raise ValueError("decoded image is not 3-channel")
    ntok = (H // VIS_UNIT) * (W // VIS_UNIT)
    return H, W, rgb, ntok


def vis_collect(msgs):
    """Every image in the conversation, in template render order."""
    out = []
    for m in msgs:
        c = m.get("content")
        if not isinstance(c, list):
            continue
        for item in c:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image_url" or "image_url" in item:
                u = item.get("image_url")
                u = u.get("url") if isinstance(u, dict) else u
                out.append(vis_decode_part(u))
            elif item.get("type") == "image" or "image" in item:
                v = item.get("image")
                out.append(vis_decode_part(v if isinstance(v, str) else ""))
            elif item.get("type") == "video" or "video" in item:
                # Refused HERE, before the template writes its
                # `<|video_pad|>`: the engine has no video path and would
                # refuse the pad on the D line, which on a streaming request
                # arrives after the headers (issue #39). Refusing every
                # template video pad up front is also what lets
                # textify_pad_mentions() treat every video pad as a mention.
                raise ValueError("this server has no video path; a `video` "
                                 "content part cannot be served")
    return out

def oai_error(status, message, code=None):
    return JSONResponse(status_code=status, content={"error": {
        "message": message, "type": ERROR_TYPES.get(status, "server_error"),
        "param": None, "code": code}})


# the timeouts that keep ONE bad request from wedging the server.
# At ONE slot the engine lock is the whole server: anything that can block
# while holding it can stop every other client indefinitely, which is what
# happened on 2026-08-24 (busy=true, GPU 0%, queued piling up behind a request
# that had already gone away). an earlier change makes that a per-request failure instead
# of an outage -- with N slots a wedged request costs one slot, and the
# batched path deliberately does NOT drop the shared socket on timeout,
# because other requests are riding it.
# How long /health waits for the engine to say PONG before calling it
# unresponsive. The engine answers between decode rounds (~25 ms) and between
# prefill chunks (~1.4 s at the shipped chunk, ~14 s for a 16k chunk in 1M
# mode), so this is generous by an order of magnitude on purpose: the cost of
# a false alarm is a restarted container.
ENGINE_PING_S = float(os.environ.get("HALOGEN_ENGINE_PING_S", 30.0))

FIRST_TOKEN_S = 1800.0   # waits out PREFILL; a cold 262K prompt is ~19 min
NEXT_TOKEN_S = 300.0     # between tokens a round is sub-second; minutes = wedge
ABORT_DRAIN_S = 10.0     # resync is an optimization; reconnecting is correct

# PUBLIC ISSUE #25: HOW LONG AN IDLE POOLED CONNECTION LIVES.
#
# uvicorn's default is 5 seconds and this file never set it, so every client
# that idled longer than that between turns found its pooled socket closed and
# its next request failing. The reporter bracketed it precisely: reuse at 1, 2,
# 3 and 4 s idle all 200, reuse at 5, 6 and 8 s all RemoteDisconnected, with
# back-to-back reuse at zero idle passing, which is the control that rules out
# a server closing after every response.
#
# Five seconds is a web default and this is not a web server. The normal client
# here is an agent that idles for the length of a tool call, a file write, or a
# person reading the last answer, and `POST` is not idempotent so undici will
# not transparently retry on a fresh socket the way it would for a `GET`. The
# failure therefore reaches the user rather than being absorbed.
#
# An idle HTTP connection holds a file descriptor and a task. It holds no
# engine slot and no KV, so it is uncoupled from HALOGEN_KV_SLOTS and there is
# no reason for this to be stingy. 300 s is the reporter's suggestion and is
# arbitrary past a few minutes; the KNOB is the fix, not the number. The race
# is inherent to HTTP keep-alive either way -- a server may close at the moment
# a client writes -- so a client that pools connections should still retry on a
# fresh socket. What the 5 s default did was make a rare race constant.
KEEPALIVE_S = float(os.environ.get("HALOGEN_KEEPALIVE_TIMEOUT", 300.0))

# PUBLIC ISSUES #3 AND #36: A STREAM THAT GOES QUIET GETS HUNG UP ON.
#
# Node's undici, which Pi and Qwen Code are built on, kills a response body
# that is silent for 300 s (`bodyTimeout`, no setting in either client), and
# #3's reporter showed that re-chunking the identical bytes through a proxy
# made every failure disappear. Two things made this server silent for that
# long: tool-call arguments were flushed as one burst (#36, fixed in
# tool_parse.ToolStream), and a prefill sends nothing until its first token
# (a cold 262k prompt is minutes). An SSE comment line (`: keepalive`) is
# invisible to every SSE parser and resets every inactivity timer, so one
# goes out whenever nothing else has for this many seconds. 0 disables it.
SSE_KEEPALIVE_S = float(os.environ.get("HALOGEN_SSE_KEEPALIVE_S", 10.0))


def schema_text(rf):
    """The schema the engine gets, from either wire's spelling of
    response_format: chat nests it (`{"type": "json_schema", "json_schema":
    {"name", "schema", "strict"}}`), the Responses `text.format` is flat
    (`{"type": "json_schema", "name", "schema", "strict"}`; recorded from the
    Codex source). `json_object` is the literal "json".
    Compact JSON text, so the engine's cache keys on the bytes."""
    if rf.get("type") == "json_object":
        return "json"
    schema = rf.get("schema")
    if schema is None and isinstance(rf.get("json_schema"), dict):
        schema = rf["json_schema"].get("schema")
    if not isinstance(schema, dict):
        raise HTTPException(400, "response_format json_schema needs a "
                                 "`schema` object (OpenAI's `json_schema: "
                                 "{name, schema}` on chat, `text.format: "
                                 "{type, name, schema}` on /v1/responses)")
    return json.dumps(schema, separators=(",", ":"), ensure_ascii=False)


# The byte-level BPE remap (GPT-2's bytes_to_unicode), inverted: a token's
# vocabulary string is unicode; its BYTES are what the engine's grammar
# matches. Byte-exact, so a token that is half of a multi-byte character
# stays half of one.
def _bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_VOCAB_BLOB = None


def vocab_blob(tok):
    """The VOCAB payload (0.8.0): `HGV1`, u32 n, then per id u8 flags
    (1 = an added/special token, never text under a grammar), u16 length,
    the token's bytes. Built once per process from this front end's own
    tokenizer, so the engine's grammar sees exactly the tokens this file
    will detokenize."""
    global _VOCAB_BLOB
    if _VOCAB_BLOB is not None:
        return _VOCAB_BLOB
    import struct
    u2b = {u: b for b, u in _bytes_to_unicode().items()}
    vocab = tok.get_vocab()
    added = set(tok.get_added_vocab().values()) if hasattr(tok, "get_added_vocab") else set()
    n = max(vocab.values()) + 1
    inv = {i: t for t, i in vocab.items()}
    out = bytearray(b"HGV1" + struct.pack("<I", n))
    for i in range(n):
        t = inv.get(i)
        if t is None:
            out += struct.pack("<BH", 1, 0)
        elif i in added:
            b = t.encode("utf-8")
            out += struct.pack("<BH", 1, len(b)) + b
        else:
            b = bytes(u2b[ch] for ch in t)
            out += struct.pack("<BH", 0, len(b)) + b
    _VOCAB_BLOB = (n, bytes(out))
    return _VOCAB_BLOB


class Metrics:
    """0.8.0: the totals behind `/metrics`, llama-server's Prometheus names
    so a dashboard built for llama.cpp (or behind llama-swap) reads this
    server unchanged. Every number is the engine's own D line, the same
    fields `timings` copies, summed once per request; nothing is measured
    here. The two rate gauges are over the requests since the last scrape,
    as llama-server's are (it resets its bucket on every /metrics read), so
    a scraper sees the rate of the interval it asked about."""
    def __init__(self):
        self.prompt_tokens = 0          # processed prompt tokens (prompt minus the cached prefix)
        self.prompt_seconds = 0.0
        self.predicted_tokens = 0
        self.predicted_seconds = 0.0
        self.requests = 0
        self.cache_tokens = 0           # prompt tokens the cache covered
        self.draft_tokens = 0
        self.draft_accepted = 0
        self.structured_requests = 0
        self.b_prompt_tokens = 0        # the bucket since the last scrape
        self.b_prompt_seconds = 0.0
        self.b_predicted_tokens = 0
        self.b_predicted_seconds = 0.0

    def record(self, t, structured=False):
        """`t` is `timings(d)` for one finished request."""
        pn = int(t.get("prompt_n", 0)); gn = int(t.get("predicted_n", 0))
        ps = float(t.get("prompt_ms", 0.0)) / 1000.0
        gs = float(t.get("predicted_ms", 0.0)) / 1000.0
        self.prompt_tokens += pn; self.prompt_seconds += ps
        self.predicted_tokens += gn; self.predicted_seconds += gs
        self.b_prompt_tokens += pn; self.b_prompt_seconds += ps
        self.b_predicted_tokens += gn; self.b_predicted_seconds += gs
        self.requests += 1
        self.cache_tokens += int(t.get("cache_n", 0))
        self.draft_tokens += int(t.get("draft_n", 0))
        self.draft_accepted += int(t.get("draft_n_accepted", 0))
        self.structured_requests += 1 if structured else 0

    def render(self, processing, deferred, kv_reserved, kv_pool):
        """The Prometheus text exposition, and the bucket resets."""
        b_ps = (self.b_prompt_tokens / self.b_prompt_seconds
                if self.b_prompt_seconds > 0 and self.b_prompt_tokens > 0 else 0.0)
        b_gs = (self.b_predicted_tokens / self.b_predicted_seconds
                if self.b_predicted_seconds > 0 else 0.0)
        self.b_prompt_tokens = 0; self.b_prompt_seconds = 0.0
        self.b_predicted_tokens = 0; self.b_predicted_seconds = 0.0
        rows = [
            ("counter", "llamacpp:prompt_tokens_total", "Number of prompt tokens processed.", self.prompt_tokens),
            ("counter", "llamacpp:prompt_seconds_total", "Prompt process time.", self.prompt_seconds),
            ("counter", "llamacpp:tokens_predicted_total", "Number of generation tokens processed.", self.predicted_tokens),
            ("counter", "llamacpp:tokens_predicted_seconds_total", "Predict process time.", self.predicted_seconds),
            ("gauge", "llamacpp:prompt_tokens_seconds", "Average prompt throughput in tokens/s. (since the last scrape)", b_ps),
            ("gauge", "llamacpp:predicted_tokens_seconds", "Average generation throughput in tokens/s. (since the last scrape)", b_gs),
            ("gauge", "llamacpp:requests_processing", "Number of requests processing.", processing),
            ("gauge", "llamacpp:requests_deferred", "Number of requests deferred.", deferred),
        ]
        if kv_pool:
            rows += [
                ("gauge", "llamacpp:kv_cache_tokens", "KV-cache tokens (positions reserved by the requests in flight: prompt + max_tokens each).", kv_reserved),
                ("gauge", "llamacpp:kv_cache_usage_ratio", "KV-cache usage. 1 means 100 percent usage. (reserved positions over the pool)", min(1.0, kv_reserved / kv_pool)),
            ]
        rows += [
            ("counter", "halogen:requests_total", "Requests completed.", self.requests),
            ("counter", "halogen:prompt_tokens_cached_total", "Prompt tokens the prompt cache covered (not processed).", self.cache_tokens),
            ("counter", "halogen:draft_tokens_total", "Tokens proposed by the draft head and prompt lookup.", self.draft_tokens),
            ("counter", "halogen:draft_tokens_accepted_total", "Of those, accepted.", self.draft_accepted),
            ("counter", "halogen:structured_requests_total", "Requests decoded under a JSON schema.", self.structured_requests),
        ]
        out = []
        for kind, name, help_, value in rows:
            out.append("# HELP %s %s" % (name, help_))
            out.append("# TYPE %s %s" % (name, kind))
            out.append("%s %s" % (name, ("%d" % value) if isinstance(value, int) else ("%.6g" % value)))
        return "\n".join(out) + "\n"


class EngineBusy(HTTPException):
    def __init__(self):
        super().__init__(status_code=503,
                         detail="timed out waiting for the engine: "
                                "halogen serves one request at a time "
                                "(batch-1) and the queue did not clear")


def engine_refusal(parts):
    """The message for an engine `D <req> error ...` line.

    The engine now writes its reason after the six fixed fields, so
    the client reads the same sentence the engine log does. Before that the
    reason went only to stderr and this text had to GUESS: the first version
    blamed prompt length for what was usually an unsupported field, and the
    second refused to guess at all. An older engine still sends the bare
    line, and gets the no-guess text.
    """
    # 0.8.1 (issue #53): a request that ends mid-generation carries the D
    # line's statistics between the fixed fields and the reason, so the
    # reason starts at the first field that is not a number. Before this
    # the statistics were the message: "the engine refused this request:
    # 0 0 0 0 0 0 0 0".
    tail = parts[7:]
    i = 0
    while i < len(tail):
        try:
            float(tail[i])
        except ValueError:
            break
        i += 1
    why = " ".join(tail[i:]).strip()
    if why:
        return "the engine refused this request: " + why
    return ("the engine rejected this request. Check /health for what this "
            "build supports, and that the prompt fits the context.")


class Engine:
    """Client for the src/serve.cpp line protocol.

    Slice 1 is batch 1 by construction, so a single connection is held and
    serialized behind a lock — a second concurrent request gets 503 rather
    than silently interleaving into one KV state. Batching is handled by the scheduler.
    """

    def __init__(self, host, port):
        self.host, self.port = host, port
        self.r = self.w = None
        # an earlier change: a SEMAPHORE, not a lock. Sized from the engine's INFO after
        # connect (kv_slots); 1 reproduces the batch-1 behaviour exactly,
        # including `.locked()` meaning "no capacity right now".
        self.slots = asyncio.Semaphore(1)
        self.n_slots = 1
        self.req = 0
        self.active = None   # batch-1 path only: the in-flight req id
        self.waiting = 0     # callers queued for a slot
        self.inflight = {}   # slot key -> monotonic start, for /health
        self.busy_since = None   # oldest live hold; None when idle
        self.streams = {}    # req id -> asyncio.Queue, the DEMUX
        self.wlock = asyncio.Lock()   # serialises WRITES onto the one socket
        # Public issue #3: serialises CONNECTS. `_ensure()` is called by every
        # request, so a socket that goes away with N callers in flight raced
        # all N into connect() at once. Measured before this lock: six
        # concurrent _ensure() calls opened SIX connections, each one leaking
        # the previous socket, spawning another _demux on the same stream (the
        # one thing that function's docstring says must never happen), and
        # replacing both the stream table and the slot semaphore underneath
        # live holders. tools/smoke-reconnect.py is the falsifier.
        self.clock = asyncio.Lock()
        self.reader = None   # background demux task (batched engines only)
        self.cstat = asyncio.Queue()  # CSTAT replies, via the demux
        self.info = {}       # engine capabilities, from INFO at connect
        self.tokenizer = None   # 0.8.0: bound by build_app, for VOCAB
        self.metrics = Metrics()   # 0.8.0: the /metrics totals
        self.reserved = {}         # slot key -> positions the request holds (prompt + max_tokens)
        self._live = (True, 0.0, "")   # last probe_live result
        self._live_at = None           # when it was taken (monotonic)

    async def connect(self):
        async with self.clock:
            await self._connect_locked()

    async def _connect_locked(self):
        # Re-checked INSIDE the lock: callers queued behind the connect that
        # just succeeded must use it rather than replace it. Without this the
        # lock would only serialise the damage instead of preventing it.
        if self.w is not None and not self.w.is_closing():
            return
        # Wake anything still waiting on the old socket BEFORE the table it is
        # registered in goes away, or it waits out its whole token budget and
        # reports the engine as silent. Then stop the old reader, so exactly
        # one coroutine ever owns the stream.
        for q in list(self.streams.values()):
            q.put_nowait(None)
        self.streams = {}
        if self.reader is not None and not self.reader.done():
            self.reader.cancel()
        self.reader = None
        old_w = self.w
        self.r, self.w = await asyncio.open_connection(self.host, self.port)
        if old_w is not None:
            try:
                old_w.close()          # it was replaced, not merely dropped
            except Exception:
                pass
        self.info = await self._probe()
        if self.info.get("grammar"):
            await self._send_vocab()
        n = max(1, int(self.info.get("kv_slots", 1)))
        # The semaphore is only rebuilt when its SIZE changes. Replacing it on
        # every reconnect discards the count of slots currently held, so a
        # caller that later released into the new object would raise the
        # capacity above what the engine actually has.
        if n != self.n_slots or self.slots is None:
            self.n_slots = n
            self.slots = asyncio.Semaphore(n)
        if self.n_slots > 1:
            # One reader owns the socket and fans lines out by req id. Nothing
            # else may read it: two coroutines reading one stream is how a
            # response ends up spliced into another request's tokens.
            self.reader = asyncio.create_task(self._demux(self.streams))

    async def _send_vocab(self):
        """0.8.0: give the engine the tokenizer's tokens as BYTES.

        The engine is ids-only and owns no tokenizer; a grammar needs to know
        which token ids spell which bytes, so it gets exactly this front end's
        vocabulary, once per connection, and never a second tokenizer that
        could drift. The blob is built once per process (~2.6 MB, base64 on
        one line like an image). A failure leaves `grammar` off with the
        reason, so /health and the 400 say why rather than the engine refusing
        a GEN later.
        """
        try:
            if self.tokenizer is None:
                raise RuntimeError("no tokenizer bound to the engine client")
            blob = vocab_blob(self.tokenizer)
            self.w.write(b"VOCAB %d %s\n" % (blob[0], base64.b64encode(blob[1])))
            await self.w.drain()
            raw = await asyncio.wait_for(self.r.readline(), timeout=60)
            p = raw.decode().split(None, 2)
            if len(p) >= 2 and p[0] == "V" and p[1] == "ok":
                return
            self.info["grammar"] = False
            self.info["grammar_error"] = (" ".join(p[2:]) or raw.decode().strip()
                                          or "no reply to VOCAB")
        except Exception as e:
            self.info["grammar"] = False
            self.info["grammar_error"] = f"VOCAB failed: {e}"
        print(f"serve_api: structured output OFF: {self.info['grammar_error']}",
              flush=True)

    async def probe_live(self, timeout=None):
        """Is the engine's loop TURNING? -> (ok, seconds, detail).

        A CONNECT proves nothing. The kernel completes it into the listen
        backlog with no help from the engine, so a process whose GPU queue has
        been aborted, or that is stopped outright, answers every connect and
        no request. Measured on the shipped 0.4.4 image with the engine
        SIGSTOPped: the published healthcheck succeeded, /health said "ok",
        and every request hung to its timeout. That is the reported failure.

        PONG is the property a backlog cannot fake, and the engine answers it
        on a QUEUED connection between decode rounds and between prefill
        chunks, so this stays true while the engine is busy. A SEPARATE short
        lived connection is used rather than the multiplexed session socket:
        it cannot interleave with a generation, it needs no lock, and it is
        the same question the container healthcheck asks.

        The result is cached for a second so a hammering client cannot turn
        /health into a connection storm.
        """
        now = time.monotonic()
        if self._live_at is not None and now - self._live_at < 1.0:
            return self._live
        t = float(timeout if timeout is not None else ENGINE_PING_S)
        t0 = time.monotonic()
        w = None

        async def ask():
            nonlocal w
            r, w = await asyncio.open_connection(self.host, self.port)
            w.write(b"PING\n")
            await w.drain()
            return await r.readline()

        try:
            # ONE BUDGET FOR THE WHOLE PROBE. Per-step timeouts are not a
            # request budget: connect, drain and read each got the full value,
            # so a stopped engine (whose port still accepts) could hold
            # /health for twice it and the caller timed out first with no
            # answer at all. Measured: a 45 s client gave up on a 30 s
            # setting.
            line = await asyncio.wait_for(ask(), timeout=t)
            ok = line.strip() == b"PONG"
            out = (ok, round(time.monotonic() - t0, 3),
                   "" if ok else f"unexpected reply {line[:40]!r}")
        except asyncio.TimeoutError:
            out = (False, round(time.monotonic() - t0, 3),
                   f"no PONG within {t:g}s")
        except Exception as e:
            out = (False, round(time.monotonic() - t0, 3), repr(e)[:120])
        finally:
            if w is not None:
                try:
                    w.close()
                except Exception:
                    pass
        self._live, self._live_at = out, time.monotonic()
        return out

    async def _demux(self, streams=None):
        """Fan req-tagged lines into per-request queues.

        The protocol was built for this: every T/D/X line carries <req>. On
        socket close, every waiting stream is woken with a sentinel rather
        than left hanging -- an orphaned reader is precisely the wedge,
        one layer up.
        """
        try:
            while True:
                raw = await self.r.readline()
                if not raw:
                    break
                parts = raw.decode().split()
                if parts and parts[0] == "C":
                    self.cstat.put_nowait(parts)   # /cache, not a generation
                    continue
                if len(parts) < 2 or parts[0] not in ("T", "D"):
                    continue
                try:
                    rid = int(parts[1])
                except ValueError:
                    continue
                q = self.streams.get(rid)
                if q is not None:
                    q.put_nowait(parts)
        except Exception:
            pass
        finally:
            # The table THIS reader served, captured at creation. Reading
            # self.streams here would wake whatever table happens to be
            # current, which after a reconnect is a different one, leaving the
            # callers this reader was responsible for to time out instead.
            for q in list((streams if streams is not None else self.streams).values()):
                q.put_nowait(None)          # sentinel: engine went away

    async def _probe(self):
        """Ask the engine what the loaded checkpoint can do.

        /health should report capabilities rather than guess them. An engine
        that predates INFO ignores the verb entirely, so this must never
        block forever — on timeout we fall back to 'serial only', which is
        the safe assumption.
        """
        try:
            self.w.write(b"INFO\n")
            await self.w.drain()
            raw = await asyncio.wait_for(self.r.readline(), timeout=5)
            p = raw.decode().split()
            if p and p[0] == "I" and len(p) >= 6:
                return {"mtp": p[1] == "1", "draft_head": p[2] == "1",
                        "ctx": int(p[3]), "spec_rows": int(p[4]),
                        "default": int(p[5]),
                        # trailing: the drafter's weights are loaded and
                        # ready. In this fork the MTP head IS the drafter, so
                        # the engine answers this with the MTP head's
                        # readiness. It was a hardcoded 0 until the field was
                        # fixed: the value is inherited from an engine with a
                        # separate draft model, where it meant that model's
                        # weights were present, and nothing here ever set it.
                        "drafter_weights": len(p) >= 7 and p[6] == "1",
                        # trailing: a separate shortlist draft model is
                        # SELECTABLE (weights present AND the verify apparatus
                        # wired). This fork has no such model, so the engine
                        # sends a literal 0 here and this is always False.
                        "dflash2": len(p) >= 8 and p[7] == "1",
                        # trailing, an earlier change: prompt-cache cap (MB, 0 = off) and
                        # the snapshot alignment. Alignment is reported, not
                        # buried, because it is what decides whether a warm
                        # answer is bitwise identical to a cold one (an earlier change).
                        "cache_mb": int(p[8]) if len(p) >= 9 else 0,
                        "cache_align": int(p[9]) if len(p) >= 10 else 0,
                        # trailing, an earlier change: the engine's KV slot pool. The
                        # front-end sizes its own concurrency from this rather
                        # than guessing -- admitting more than there are slots
                        # just rebuilds the queue one layer up. Absent (older
                        # engine) means one slot, i.e. batch-1, which is the
                        # safe assumption for the same reason 'serial only' is.
                        "kv_slots": int(p[10]) if len(p) >= 11 else 1,
                        "slot_ctx": int(p[11]) if len(p) >= 12 else 0,
                        # The prompt cache MODE. 1 = snapshot on a
                        # prefill-chunk boundary, which is what makes a warm
                        # answer byte-identical to a cold one; 2 = snapshot at
                        # every request end, faster at every prompt length and
                        # NOT byte-identical. Absent (older engine) means 1,
                        # because that is all an older engine could do.
                        "cache_mode": int(p[12]) if len(p) >= 13 else 1,
                        # The engine SAMPLES: SAMPLE / PENALTY / BIAS /
                        # LOGPROBS are honoured on both drafters. Read from
                        # the engine rather than assumed, because /health
                        # once advertised nine sampling parameters this
                        # engine did not have. Absent (older engine) = no.
                        "sampling": len(p) >= 14 and p[13] == "1",
                        # The RoPE scaling factor in force (1 = the native
                        # context). Read from the engine so /health names
                        # the numeric arm that is actually serving.
                        "rope_factor": float(p[14]) if len(p) >= 15 else 1.0,
                        # 0.3: the KV pool's size in positions, shared by the
                        # slots; a request reserves prompt + max_tokens of it.
                        # Absent (older engine) = one context per slot.
                        "kv_pool": int(p[15]) if len(p) >= 16 else 0,
                        # Whether the engine holds a VISION TOWER. Read from
                        # the engine because /health cannot see it any other
                        # way, and a server that says it takes images and
                        # then refuses them is the lying-field defect this
                        # project has now shipped four times. Absent (an
                        # older engine) = no, which is what an older engine
                        # is.
                        "vision": len(p) >= 17 and p[16] == "1",
                        # The engine's release, from ITS image's stamp (#26).
                        # Absent = an engine older than the field, which is
                        # itself the answer: it predates this front-end.
                        "version": p[17] if len(p) >= 18 else "unknown",
                        # Prompt lookup beside the head: the n-gram length
                        # matched against the request's own context and the
                        # chain length proposed; 0 = off. Absent (an older
                        # engine) = none.
                        "pld_n": int(p[18]) if len(p) >= 20 else 0,
                        "pld_k": int(p[19]) if len(p) >= 20 else 0,
                        # Where the trunk came from: "hgn" (the engine's own
                        # checkpoint), "gguf" (a llama.cpp file repacked at
                        # this start) or "gguf-cache" (that repack read back
                        # from its cache file). Absent (an older engine) =
                        # hgn, which is all an older engine could load.
                        "checkpoint_format": p[20] if len(p) >= 21 else "hgn",
                        # 0.8.0: the engine accepts a SCHEMA on the GEN line
                        # (constrained decoding) once this front
                        # end has sent it the vocabulary. 0 = HALOGEN_GRAMMAR=0;
                        # absent = an engine older than structured output.
                        "grammar": len(p) >= 22 and p[21] == "1"}
        except Exception:
            pass
        return {"mtp": False, "draft_head": False, "default": 0,
                "kv_slots": 1, "slot_ctx": 0, "cache_mode": 1,
                "sampling": False, "rope_factor": 1.0, "kv_pool": 0,
                "vision": False, "version": "unknown"}

    async def _ensure(self):
        if self.w is None or self.w.is_closing():
            await self.connect()

    async def close(self):
        try:
            if self.w is not None:
                self.w.close()
        except Exception:
            pass
        self.r = self.w = None
        self.active = None

    @staticmethod
    def _done_dict(parts):
        """Parse a D line's trailing stats. ONE copy, used by both paths --
        the field indices are exactly what drifts when a trailing field gets
        added to the serial reader and not the demux."""
        d = {"reason": parts[2], "n_prompt": int(parts[3]),
             "n_gen": int(parts[4]), "prefill_ms": float(parts[5]),
             "decode_ms": float(parts[6])}
        if len(parts) >= 10:            # an earlier change trailing stats
            d["drafter"] = int(parts[7])
            d["rounds"] = int(parts[8])
            d["commit"] = int(parts[9])
        if len(parts) >= 11:            # an earlier change prompt cache
            d["n_cached"] = int(parts[10])
        if len(parts) >= 12:            # issue #22: tokens produced beside
            d["shared"] = int(parts[11])   # other streams (inside commit)
        if len(parts) >= 14:            # prompt-lookup rounds (among
            d["pld_rounds"] = int(parts[12])   # rounds) and the drafts they
            d["pld_accepted"] = int(parts[13])  # accepted (inside commit)
        if len(parts) >= 15:            # 0.7.0, #45: the chain tokens those
            d["pld_drafted"] = int(parts[14])   # rounds proposed (1..K each)
        return d

    async def _gen_batched(self, req, line):
        """One of many concurrent generations over the shared socket.

        The socket is read ONLY by _demux(); this coroutine waits on its own
        queue. The two timeout budgets are unchanged and still apply per
        request, because a wedge is still a wedge -- what changed is that it
        now takes down ONE request instead of the server.
        """
        q = asyncio.Queue()
        self.streams[req] = q
        finished = False
        try:
            async with self.wlock:
                self.w.write(line.encode())
                await self.w.drain()
            first = True
            while True:
                try:
                    parts = await asyncio.wait_for(
                        q.get(), timeout=FIRST_TOKEN_S if first else NEXT_TOKEN_S)
                except asyncio.TimeoutError:
                    # Do NOT drop the shared socket: other requests are riding
                    # it. Fail this one. That is the whole point of batching --
                    # one bad request stops being everyone's problem.
                    raise HTTPException(
                        504, "engine went silent for %ds%s"
                             % (FIRST_TOKEN_S if first else NEXT_TOKEN_S,
                                " while prefilling" if first else " mid-decode"))
                if parts is None:
                    raise HTTPException(502, "engine closed the connection")
                if parts[0] == "T":
                    first = False
                    yield (int(parts[2]), None,
                           float(parts[3]) if len(parts) > 3 else None)
                elif parts[0] == "D":
                    if parts[2] == "error":
                        raise HTTPException(400, engine_refusal(parts))
                    finished = True
                    yield None, self._done_dict(parts), None
                    return
        finally:
            self.streams.pop(req, None)
            if not finished:
                # THE GENERATOR KNOWS ITS OWN REQ, which is why cancellation
                # lives here rather than in Engine.abort(): with N concurrent
                # requests a single self.active cannot say which one went
                # away. Without the cancel the engine decodes to max_tokens
                # into a slot nobody is reading -- an earlier change's failure, except it
                # now costs one slot instead of the server.
                #
                # Fire-and-forget: the D line comes back to a stream that no
                # longer exists and _demux drops it, which is correct. We must
                # NOT wait for it, because that would be the unbounded drain
                # an earlier change removed.
                try:
                    self.w.write(f"X {req}\n".encode())
                except Exception:
                    pass

    async def generate(self, ids, max_tokens, eos, drafter=None, sample=None,
                       penalty="", snap=0, snap2=0, images=None, schema=None,
                       after=None, escape=(), think=None):   # noqa: E301
        """Yields (token_id, None) per token, then (None, done_dict).

        `drafter` is the trailing wire field (0 serial / 1 MTP); None omits
        it and lets the engine apply its own default.

        `sample` is (temp, top_k, top_p, min_p, seed) or None. It goes on the
        wire as a KEYED suffix, not more positional fields, so an engine built
        before sampling landed still parses the line -- it simply never sees the
        keyword. None omits it entirely and the engine decodes greedy.
        """
        await self._ensure()
        self.req += 1
        req = self.req
        # VISION: one VIMG line per image, BEFORE the GEN that uses it and
        # keyed by the same request id. Base64 on its own line, because the
        # engine's framing is line-based and a raw binary payload behind the
        # GEN line would be split on whatever 0x0a bytes it contained.
        for first_tok, H, W, rgb in (images or []):
            self.w.write(("VIMG %d %d %d %d %s\n" %
                          (req, first_tok, H, W,
                           base64.b64encode(rgb).decode())).encode())
        line = (f"GEN {req} {max_tokens} {len(eos)} "
                + (" ".join(str(e) for e in eos) + " " if eos else "")
                + f"{len(ids)} " + " ".join(str(i) for i in ids)
                + (f" {drafter}" if drafter is not None else "")
                # %.9g, not %g. %g carries 6 significant digits, which
                # renders top_p=0.9999999 as "1" -- and the engine reads
                # top_p >= 1 as NO NUCLEUS FILTER, so an unusually tight
                # request would silently become an unfiltered one. 9 digits
                # round-trips a float exactly, which is what the engine parses
                # these into. Exponential notation is fine: C++ operator>>
                # reads "1e-05" correctly.
                + (" SAMPLE %.9g %d %.9g %.9g %d" % sample if sample else "")
                + (penalty or "")
                # SNAP k: where the prompt cache should snapshot (the length
                # of the prefix the next turn will repeat). Chat requests set
                # it to the end of the rendered history, before the assistant
                # opener; see stable_prefix_len().
                + (f" SNAP {snap}" if snap else "")
                # SNAP2 k: a second, shorter snapshot point (the end of the
                # system prompt), so requests sharing it and asking different
                # things resume from it. Mode 2 only; the engine ignores it
                # otherwise.
                + (f" SNAP2 {snap2}" if snap2 else "")
                # IMG <count> <first_tok> <H> <W> ... : the engine checks this
                # against the VIMG lines it received and REFUSES a mismatch.
                # A dropped image must not become a prompt prefilled with the
                # embedding of the pad token repeated a thousand times, which
                # reads as a confident answer about an image nobody sent.
                + ((" IMG %d " % len(images)) +
                   " ".join("%d %d %d" % (f, h, w) for f, h, w, _ in images)
                   if images else "")
                # 0.8.0: SCHEMA <b64 of the JSON schema | json>, AFTER
                # <id> (constrain only after this id: the </think> token when
                # the request thinks), ESCAPE <n> <ids> (ids legal at the very
                # start beside the grammar; choosing one ends the constraint:
                # the <tool_call> token on a request with tools, because the
                # schema binds the final text and a call is not text).
                + ((" SCHEMA json" if schema == "json" else
                    " SCHEMA " + base64.b64encode(schema.encode()).decode())
                   if schema else "")
                + (f" AFTER {after}" if schema and after is not None else "")
                + ((" ESCAPE %d " % len(escape)) + " ".join(str(e) for e in escape)
                   if schema and escape else "")
                # 0.8.1, #56: THINK <budget> <end id> <k> <close ids>: the
                # engine closes the think block itself at the budget.
                + ((" THINK %d %d %d " % (think[0], think[1], len(think[2])))
                   + " ".join(str(t) for t in think[2]) if think else "")
                + "\n")
        if self.n_slots > 1:
            async for item in self._gen_batched(req, line):
                yield item
            return
        self.w.write(line.encode())
        await self.w.drain()
        self.active = req
        # TIMEOUTS ON THE MAIN READ, which had none. A bare readline() here
        # means a silent engine wedges the request FOREVER while holding the
        # batch-1 lock, so one bad request takes the whole server down rather
        # than failing by itself.
        #
        # Two different budgets, because the two waits are nothing alike:
        #   FIRST token waits out PREFILL, which at 262K context is minutes
        #     (a cold 262K prompt is estimated ~19 min), so this has to be
        #     generous or long prompts break.
        #   LATER tokens arrive every round -- a few hundred ms even deep in
        #     context -- so a gap of minutes is not slowness, it is a wedge.
        # A single timeout cannot serve both: sized for prefill it never fires
        # on a hang, sized for decode it kills every long prompt.
        first = True
        while True:
            try:
                raw = await asyncio.wait_for(
                    self.r.readline(),
                    timeout=FIRST_TOKEN_S if first else NEXT_TOKEN_S)
            except asyncio.TimeoutError:
                # Drop the socket: the stream's position is now unknown, and
                # a half-read stream desynchronizes the NEXT request.
                await self.close()
                raise HTTPException(
                    504, "engine went silent for %ds%s. The connection was "
                         "dropped and the next request will reconnect"
                         % (FIRST_TOKEN_S if first else NEXT_TOKEN_S,
                            " while prefilling" if first else " mid-decode"))
            if not raw:
                self.active = None
                raise HTTPException(502, "engine closed the connection")
            parts = raw.decode().split()
            if not parts:
                continue
            if parts[0] == "T" and int(parts[1]) == req:
                # THREE elements, not two. The second is reserved for the
                # terminal `done` dict and the loop below tests it with
                # `is not None`, so putting a per-token logprob there would
                # make every token look like the end of the stream.
                # parts[3] is the an earlier change logprob column, present only when
                # LOGPROBS was sent -- absent for every older engine and every
                # greedy request, hence a length check and not a version.
                first = False
                yield (int(parts[2]), None,
                       float(parts[3]) if len(parts) > 3 else None)
            elif parts[0] == "D" and int(parts[1]) == req:
                self.active = None
                if parts[2] == "error":
                    raise HTTPException(400, engine_refusal(parts))
                yield None, self._done_dict(parts), None
                return

    async def cache_stats(self):
        """Live prompt-cache counters. Under the same lock as a
        generate: the engine reads one line at a time, so interleaving a
        CSTAT into an in-flight request would desynchronize the stream."""
        await self._ensure()
        if self.n_slots > 1:
            # NEVER read self.r here: _demux() owns it. Two coroutines reading
            # one stream is how a CSTAT reply ends up spliced into some
            # request's token sequence.
            async with self.wlock:
                self.w.write(b"CSTAT\n")
                await self.w.drain()
            p = await asyncio.wait_for(self.cstat.get(), timeout=10)
        else:
            async with self.slots:
                self.w.write(b"CSTAT\n")
                await self.w.drain()
                raw = await asyncio.wait_for(self.r.readline(), timeout=10)
            p = raw.decode().split()
        if not p or p[0] != "C" or len(p) < 10:
            return None
        # 0.6.1 (issue #31): `reserved_bytes` and `cap_bytes` were stand-ins
        # from when the cache held one entry (and so were `entries` and
        # `evicted`, which are real now). Field 3 is the size of one entry,
        # the last stored; field 4 is HALOGEN_CACHE_ENTRIES. Named for what
        # they are; an engine older than 0.6.1 still fills the positions.
        d = {"entries": int(p[1]), "bytes": int(p[2]),
             "last_entry_bytes": int(p[3]), "max_entries": int(p[4]),
             "hits": int(p[5]), "misses": int(p[6]), "stores": int(p[7]),
             "evicted": int(p[8]), "prompt_tokens_saved": int(p[9])}
        if len(p) >= 14:   # trailing: the cache's own copy cost, an earlier change
            d.update(store_ms_total=float(p[10]),
                     restore_ms_total=float(p[11]),
                     last_store_ms=float(p[12]),
                     last_restore_ms=float(p[13]))
        if len(p) >= 15:   # stores the machine was too short to take
            d["refused"] = int(p[14])
        if len(p) >= 17:
            # A hit whose rows are in another region copies them instead of
            # running cold. Free on an idle host and throttled by the driver
            # on a loaded one, so `rows_copied_ms` climbing while hit counts
            # look healthy is a memory-pressure signature rather than a cache
            # miss. Only readable at shutdown before this.
            d["rows_copied"] = int(p[15])
            d["rows_copied_ms"] = float(p[16])
        if len(p) >= 18:
            # 0.8.1 (issue #54): a region keeps its anchor and its newest
            # leaf; a store that extends the leaf replaces it. Counted here,
            # not under `evicted`, because nothing a later request could
            # have hit was lost.
            d["superseded"] = int(p[17])
        return d

    async def abort(self):
        """Stop the in-flight request AND resynchronize the stream.

        Both halves matter. Without the cancel the engine keeps decoding to
        max_tokens after the client is gone — minutes of GPU held while
        every other request 503s. Without draining through the D line the
        leftover T lines desynchronize the NEXT request. On any failure the
        connection is dropped so the next call reconnects clean rather than
        inheriting a half-read stream.
        """
        if self.n_slots > 1:
            # an earlier change: per-request cancellation is done by the generator that
            # owns the req (see _gen_batched). A shared-socket abort here
            # would have to guess WHICH request vanished, and dropping the
            # socket would take down every other request riding it.
            return
        req = self.active
        if req is None:
            return
        try:
            if self.w is not None and not self.w.is_closing():
                self.w.write(f"X {req}\n".encode())
                await self.w.drain()
            # BOUNDED AS A WHOLE, not per read. The first version used
            # `while True` with a 180 s timeout on each readline, which bounds
            # nothing: every line that arrives resets the clock, so a stream
            # that keeps talking keeps this loop alive indefinitely. And this
            # runs BEFORE the lock is released, so an unbounded drain is an
            # unbounded lock hold -- observed 2026-08-24 as busy=true with the
            # GPU at 0% and every later request queued behind it.
            #
            # Resynchronizing is an OPTIMIZATION (it saves a reconnect);
            # dropping the socket is always correct, because _ensure()
            # reconnects and the engine's accept loop takes a fresh client. So
            # give it a short deadline and fall back to the correct thing.
            deadline = time.monotonic() + ABORT_DRAIN_S
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                raw = await asyncio.wait_for(self.r.readline(), timeout=left)
                if not raw:
                    break
                p = raw.decode().split()
                if p and p[0] == "D" and int(p[1]) == req:
                    self.active = None
                    return
        except Exception:
            pass
        await self.close()


# OpenAI effort vocabulary -> what this template actually accepts.
EFFORT_MAP = {"minimal": "low", "low": "low", "medium": "medium",
              "high": "xhigh", "xhigh": "xhigh"}
# PUBLIC ISSUE #50: `reasoning_effort: "none"` is the OpenAI-compatible
# spelling for thinking OFF, and agent clients send it on a real path (Hermes
# turns thinking off for the continuation after a turn that spent its whole
# budget thinking, and for title generation). It is not an effort the template
# knows; it is `enable_thinking: false`, and it is treated as exactly that
# wherever a request's thinking state is read (chat_kwargs, thinking_on).
EFFORT_NONE = "none"
EFFORT_VALUES = list(EFFORT_MAP) + [EFFORT_NONE]


def effort_is_none(req):
    e = getattr(req, "reasoning_effort", None)
    return isinstance(e, str) and e.strip().lower() == EFFORT_NONE


def thinking_on(req):
    """Whether this request's prompt opens a think block: false when the
    request (or the server default) says enable_thinking false, or when the
    request says reasoning_effort "none" (#50). The ThinkSplit must be told
    the same thing chat_kwargs tells the template, or a no-marker reply
    misfiles as reasoning."""
    if effort_is_none(req):
        return False
    return server_default(req, "enable_thinking") is not False


# PUBLIC ISSUE #30: SERVER-SIDE DEFAULTS FOR WHAT A CLIENT MAY OMIT.
#
# Most agent clients send no sampling fields at all, and this server's
# built-in default is greedy because greedy is what every identity gate
# certifies. Qwen's own operating point for this model is SAMPLED in every
# mode (generation_config.json ships do_sample: true, T=1.0, top_p=0.95,
# top_k=20), so until now an operator who wanted the model card's settings
# needed every client to cooperate. These variables give the OPERATOR a
# default for each field a request may leave out. The rules:
#
#   * A field the request SENDS always wins; a default only fills a field the
#     request left out, one field at a time. That is the whole contract.
#   * A request that sends `temperature: 0` decodes greedy and takes NONE of
#     the sampling defaults: it chose greedy, and greedy has no sampler for a
#     top_p or a penalty to apply to. A request that omits temperature gets
#     the server's, if set, and samples.
#   * A bad value refuses to START, with the variable named, rather than
#     serving something else. An out-of-range request value is a 400 today
#     for the same reason; an out-of-range default would silently make every
#     request that 400, which is worse.
#   * Setting HALOGEN_TEMPERATURE moves every unadorned request off the greedy
#     path, and the greedy path is the only one the bitwise guarantees are
#     made on. /health says which path the default takes, so a client can
#     read it rather than infer it.
#
# Read once at import, so that a misconfiguration is found when the api
# starts (`--check-defaults` lets the entrypoint find it BEFORE the engine
# spends minutes pinning the checkpoint).
def _env_number(name, lo, hi, kind):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        v = kind(raw)
    except ValueError:
        raise SystemExit(f"serve_api: {name}={raw!r} is not a "
                         f"{'whole number' if kind is int else 'number'}")
    if not (lo <= v <= hi):
        raise SystemExit(f"serve_api: {name}={raw} is outside {lo}..{hi}; "
                         f"values are not clamped, fix the variable")
    return v


def _env_effort(name):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    e = EFFORT_MAP.get(raw.strip().lower())
    if e is None:
        if raw.strip().lower() == EFFORT_NONE:
            raise SystemExit(f"serve_api: {name}=none: the default for "
                             f"thinking OFF is HALOGEN_ENABLE_THINKING=0; "
                             f"{name} takes {'|'.join(EFFORT_MAP)}")
        raise SystemExit(f"serve_api: {name}={raw!r} unsupported; use "
                         f"{'|'.join(EFFORT_MAP)}")
    return e


def _env_bool(name):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    v = raw.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise SystemExit(f"serve_api: {name}={raw!r} is not a boolean; use 1 or 0")


DEFAULTS = {
    "max_tokens": _env_number("HALOGEN_MAX_TOKENS_DEFAULT", 1, 1 << 30, int),
    "temperature": _env_number("HALOGEN_TEMPERATURE", 0.0, 2.0, float),
    "top_p": _env_number("HALOGEN_TOP_P", 0.0, 1.0, float),
    "top_k": _env_number("HALOGEN_TOP_K", 0, 1 << 30, int),
    "min_p": _env_number("HALOGEN_MIN_P", 0.0, 1.0, float),
    "presence_penalty": _env_number("HALOGEN_PRESENCE_PENALTY", -2.0, 2.0,
                                    float),
    "frequency_penalty": _env_number("HALOGEN_FREQUENCY_PENALTY", -2.0, 2.0,
                                     float),
    "reasoning_effort": _env_effort("HALOGEN_REASONING_EFFORT"),
    # 0.6.1: the ninth default. The template thinks unless told not to, and
    # a client that cannot send `enable_thinking` (the Responses API has no
    # such field) had no way to run this server in instruct mode. Same rule
    # as the other eight: a request that names it wins.
    "enable_thinking": _env_bool("HALOGEN_ENABLE_THINKING"),
    # 0.8.1 (issue #56): the thinking budget's default, see ChatReq.
    "max_thinking_tokens": _env_number("HALOGEN_MAX_THINKING_TOKENS", 1, 1 << 30, int),
}
# The built-in budgets when HALOGEN_MAX_TOKENS_DEFAULT is unset. Chat and
# Responses share one (the model reasons before it answers, and a budget too
# small removes the answer rather than shortening it, see ChatReq); the
# legacy completions route keeps OpenAI's small convention. ONE variable
# overrides both, because an operator raising "the default budget" means
# every route they use.
CHAT_MAX_TOKENS = DEFAULTS["max_tokens"] or 8192
COMPLETION_MAX_TOKENS = DEFAULTS["max_tokens"] or 128
# The fit-to-room gate, read once at import. Default OFF: the "does not fit"
# case stays a refusal that names the numbers (see the clamp note in serve()).
#
# Set on the interactive Claude Code unit only. There, a 400 one token under
# the window strands a session that could otherwise run, and to that client a
# refusal reads the same as any other API hiccup, so the operator cannot tell
# "the model stopped" from "the request did not fit". The gate clamps the
# request to the room the prompt actually left instead of refusing, so a
# session reaches the true edge of the context — and the operator owns that
# boundary, which is what they asked for.
#
# It is deliberately NOT in the shared llm-stack.env template and NOT set on
# any bench side server: a budget quietly shrunk under a benchmark is exactly
# the failure this file already documents (the old clamp that lost a run's
# results). Only a unit whose operator wants thin tail output sets it.
FIT_TO_ROOM = bool(_env_bool("HALOGEN_FIT_TO_ROOM"))
# Below this remaining room the clamp gives up and the hard refusal stays:
# a budget of a few dozen tokens is the silent-truncation case this file
# argues against, not a useful answer. The floor makes "the prompt has eaten
# the window" an honest error rather than a request that emits one token.
FIT_TO_ROOM_FLOOR = int(os.environ.get("HALOGEN_FIT_TO_ROOM_FLOOR", "1024"))
SAMPLING_DEFAULT_FIELDS = ("temperature", "top_p", "top_k", "min_p",
                           "presence_penalty", "frequency_penalty")


def server_default(req, field):
    """The request's value if it sent one, else the server's default.

    Keyed on the VALUE being None rather than on `model_fields_set`, because
    /v1/responses builds its ChatReq with every field passed explicitly (as
    None when the client omitted it), which pydantic counts as "set".
    """
    v = getattr(req, field, None)
    return v if v is not None else DEFAULTS.get(field)


def defaults_summary():
    """The defaults an operator set, for the startup line and /health."""
    names = {"max_tokens": "HALOGEN_MAX_TOKENS_DEFAULT",
             "temperature": "HALOGEN_TEMPERATURE", "top_p": "HALOGEN_TOP_P",
             "top_k": "HALOGEN_TOP_K", "min_p": "HALOGEN_MIN_P",
             "presence_penalty": "HALOGEN_PRESENCE_PENALTY",
             "frequency_penalty": "HALOGEN_FREQUENCY_PENALTY",
             "reasoning_effort": "HALOGEN_REASONING_EFFORT",
             "enable_thinking": "HALOGEN_ENABLE_THINKING",
             "max_thinking_tokens": "HALOGEN_MAX_THINKING_TOKENS"}
    return {names[k]: v for k, v in DEFAULTS.items() if v is not None}


class CompletionReq(BaseModel):
    model: str = MODEL_ID
    prompt: str = ""
    max_tokens: int = COMPLETION_MAX_TOKENS
    stream: bool = False
    stop: list[str] | str | None = None
    # an earlier change: {"include_usage": true} appends a final chunk carrying usage,
    # per OpenAI. Without it a streaming client has no token accounting at
    # all — the non-streamed body is the only place usage appears.
    stream_options: dict | None = None
    # an earlier change: which drafter decodes this request (serial | mtp | dflash2).
    # Output is IDENTICAL whichever is picked — spec commit is trunk-argmax
    # equality — so this only changes speed, which is what makes it a clean
    # live A/B.
    drafter: str | None = None
    # THE SAMPLING FIELDS, AND WHY THEY WERE MISSING.
    #
    # `/v1/completions` returned 500 on EVERY request, in every published
    # image from the release that added sampling onward. The endpoint calls
    # `check_sampling(req)` and `sample_spec(req)`, both of which read
    # `req.temperature`; when sampling shipped, the fields were added to
    # ChatReq and this model was not touched, so the very first attribute
    # access raised AttributeError and the request became an opaque
    # "internal error".
    #
    # It survived because nothing exercised it: the release gate's endpoint
    # smoke tests `/v1/chat/completions` only, and the two scripts that do
    # reference this route are not in that gate. Found by trying to USE the
    # endpoint to look at raw model output, which is the same way every other
    # defect of this shape has been found here.
    #
    # Declared to match ChatReq, because the code behind this endpoint has
    # always been written as though they were here.
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict | None = None
    logprobs: int | bool | None = None
    top_logprobs: int | None = None
    n: int | None = None
    # Public issue #14, reported by @hvico: declared so it can be
    # REFUSED. Undeclared, pydantic dropped it and the request returned 200
    # with prose, so a client sending a schema could not tell that nothing
    # enforced it -- and `json.loads(content)` then fails on markdown. The
    # README's stated policy for anything this build cannot honour is a 400
    # naming the field, which is what 0.5.0 already does for an image with no
    # tower. This field was the one exception.
    response_format: dict | None = None


# The template controls this server honours, in either spelling. Module
# scope rather than a class attribute because pydantic reads an unannotated
# class attribute as a field default.
TEMPLATE_KWARGS = ("reasoning_effort", "enable_thinking", "preserve_thinking")


class ChatReq(BaseModel):
    model: str = MODEL_ID
    messages: list[dict]
    # This model reasons before it answers and those tokens count against the
    # budget, so a budget too small does not shorten the answer, it removes it:
    # generation stops inside the thinking block, `parts()` finds no closing
    # marker, and the caller gets content "" with the whole reply in
    # reasoning_content. 8192 clears an ordinary request with headroom while
    # still bounding how long one request can hold a slot. Callers who need
    # more ask for more; the ceiling is separate policy (--max-tokens-cap).
    max_tokens: int = CHAT_MAX_TOKENS
    # OpenAI deprecated `max_tokens` for Chat Completions when reasoning models
    # shipped: generated tokens began including reasoning the caller never sees,
    # so the bound needed a name for what it actually bounds.
    # `max_completion_tokens` is the current Chat Completions field and
    # `max_output_tokens` is the Responses API's. All three mean one thing here,
    # an upper bound on reasoning + content, so all three are declared and
    # `_resolve_budget` folds them into max_tokens for everything downstream.
    # Declared rather than left to pydantic's `ignore`, so that a caller using
    # the current spelling is honored instead of quietly getting the default.
    max_completion_tokens: int | None = None
    max_output_tokens: int | None = None
    stream: bool = False
    stop: list[str] | str | None = None
    stream_options: dict | None = None
    drafter: str | None = None
    # Qwen3.8 chat-template controls (an earlier change — template kwargs, no engine work).
    # The template accepts low | medium | xhigh and raises on anything else,
    # so OpenAI's vocabulary is mapped onto it in chat_kwargs().
    reasoning_effort: str | None = None
    # 0.8.1 (issue #56): the THINKING BUDGET. At most this many generated
    # tokens inside the think block; if `</think>` has not come by then the
    # engine closes the block itself (Qwen's own budget sentence, then
    # `</think>`) and the answer follows, in the same stream, on the same
    # state. Greedy at 100k+ can loop inside the block and spend the whole
    # max_tokens there; the card's sampling is the cure, this bounds the
    # damage. None = no budget (the shipped behaviour). Also
    # HALOGEN_MAX_THINKING_TOKENS as the server default; request wins.
    max_thinking_tokens: int | None = None
    enable_thinking: bool | None = None
    preserve_thinking: bool | None = None
    tools: list[dict] | None = None
    # an earlier change. 'auto' (default) | 'none' | 'required' |
    # {"type": "function", "function": {"name": ...}}.
    tool_choice: str | dict | None = None
    parallel_tool_calls: bool | None = None
    # PUBLIC ISSUE #24: THE OTHER SPELLING OF THE THREE FIELDS ABOVE.
    #
    # vLLM and SGLang take template controls NESTED, as
    # `chat_template_kwargs: {"enable_thinking": false}`, and that is what most
    # agentic clients send because that is what they were written against.
    # This field was not declared, so pydantic's default `extra="ignore"`
    # dropped it with no warning and no log line, and a caller who asked for
    # thinking off got a 200 with thinking on. Silently not doing what was
    # asked is the defect #14 was, and the answer there was the same: name it
    # on the wire rather than discard it.
    chat_template_kwargs: dict | None = None

    @model_validator(mode="after")
    def _resolve_budget(self):
        """Fold the three spellings of the token budget into `max_tokens`.

        Only names the caller actually SENT are considered, so the default
        never competes with an explicit value. Agreeing duplicates are fine,
        since sending `max_tokens` and `max_completion_tokens` together is how
        a client stays compatible with servers that read only one. Disagreeing
        values are a 400 rather than a guess: the caller asked for two
        different things, and silently picking one would misreport what the
        server did.
        """
        sent = {}
        for name in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
            if name in self.model_fields_set:
                v = getattr(self, name)
                if v is not None:
                    sent[name] = v
        distinct = set(sent.values())
        if len(distinct) > 1:
            raise ValueError(
                "conflicting token budgets ("
                + ", ".join(f"{k}={v}" for k, v in sorted(sent.items()))
                + "); they are the same budget under three names, send one")
        if distinct:
            self.max_tokens = distinct.pop()
        return self

    @model_validator(mode="after")
    def _fold_template_kwargs(self):
        """Fold `chat_template_kwargs` into the top-level fields.

        Same discipline as `_resolve_budget` above, for the same reason: two
        spellings of one thing, agreeing duplicates accepted, disagreement a
        400 rather than a guess. An unsupported KEY is also a 400 and not a
        silent drop, because the whole point of this validator is that a
        control the caller sent must never vanish. Naming the supported keys
        in the message means a client discovers the right spelling from the
        error instead of from a 200 that did not do what was asked.
        """
        if self.chat_template_kwargs is None:
            return self
        if not isinstance(self.chat_template_kwargs, dict):
            raise ValueError("chat_template_kwargs must be an object")
        unknown = [k for k in self.chat_template_kwargs
                   if k not in TEMPLATE_KWARGS]
        if unknown:
            raise ValueError(
                "chat_template_kwargs contains unsupported key(s) "
                + ", ".join(repr(k) for k in sorted(unknown))
                + "; this server honours "
                + ", ".join(TEMPLATE_KWARGS)
                + ", which may also be sent as top-level fields")
        for k in TEMPLATE_KWARGS:
            if k not in self.chat_template_kwargs:
                continue
            nested = self.chat_template_kwargs[k]
            if k in self.model_fields_set:
                top = getattr(self, k)
                if top is not None and top != nested:
                    raise ValueError(
                        f"conflicting {k}: {top!r} at the top level and "
                        f"{nested!r} in chat_template_kwargs; they are the "
                        f"same control under two names, send one")
            setattr(self, k, nested)
        return self
    # an earlier change: IMPLEMENTED. temperature 0 (the default) is greedy and keeps
    # the speculative fast path; anything above 0 samples and is routed to
    # SERIAL decode, because the spec commit rule is argmax equality and
    # sampled tokens would both collapse acceptance and emit the wrong
    # distribution (an earlier change trap 1). an earlier change lifts that.
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    # vLLM-compatible extensions; not in the OpenAI schema but widely sent.
    top_k: int | None = None
    min_p: float | None = None
    # an earlier change: IMPLEMENTED, sampler-only (they need temperature > 0).
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    # Still accepted and NOT implemented. Silently dropping this would
    # misrepresent the output, so /health reports it and the server logs.
    n: int | None = None
    # Public issue #14, reported by @hvico: declared so it can be
    # REFUSED. Undeclared, pydantic dropped it and the request returned 200
    # with prose, so a client sending a schema could not tell that nothing
    # enforced it -- and `json.loads(content)` then fails on markdown. The
    # README's stated policy for anything this build cannot honour is a 400
    # naming the field, which is what 0.5.0 already does for an image with no
    # tower. This field was the one exception.
    response_format: dict | None = None


# What remains accepted-but-ignored. temperature/top_p/seed/top_k/min_p came
# off this list at an earlier change -- the point of the list is that it shrinks.
class ResponsesReq(BaseModel):
    """OpenAI Responses API request.

    THE SHAPE HERE WAS CAPTURED FROM THE CODEX CLI, NOT READ OFF THE SPEC.
    A recording proxy sat between `codex exec` and this server and every field
    below is one Codex actually sends: 20 KB of `instructions`, an `input` list
    mixing messages with `function_call` / `function_call_output` items, ten
    tools of three different `type`s, and eight more keys that are metadata.
    Building this from the specification alone is how a client-compatibility
    layer ends up passing its author's tests and failing the client.

    Unknown keys are ACCEPTED AND IGNORED rather than rejected. This endpoint
    exists to be talked to by a specific evolving client, and a 422 on a field
    Codex added last week is a worse failure than quietly not using it. The
    fields that would change the ANSWER if ignored are handled explicitly.
    """
    model: str = MODEL_ID
    # The system prompt, sent separately from the turns.
    instructions: str | None = None
    # Either a bare string (one user turn) or the item list.
    input: list[dict] | str = ""
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    parallel_tool_calls: bool | None = None
    max_output_tokens: int | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    drafter: str | None = None
    # Accepted and ignored, with a reason for each:
    #   store              nothing is persisted here, so there is no id to GET
    #   previous_response_id  same, and the client resends the full history
    #   include            asks for encrypted reasoning we do not produce
    #   reasoning          effort is mapped; summary (any value but none) asks
    #                      for the reasoning text as the item's summary too (#44)
    #   prompt_cache_key   the prefix cache keys on the PREFIX already
    #   metadata / client_metadata / service_tier / user  telemetry
    store: bool | None = None
    previous_response_id: str | None = None
    include: list[str] | None = None
    reasoning: dict | None = None
    prompt_cache_key: str | None = None
    metadata: dict | None = None
    client_metadata: dict | None = None
    service_tier: str | None = None
    user: str | None = None
    text: dict | None = None

    model_config = {"extra": "allow"}


def responses_tools(tools):
    """Responses tool list -> Chat Completions tool list.

    Codex sends three `type`s in one array: `function`, `namespace` (a group
    with nested tools) and `web_search`. Only `function` is something this
    server can offer the model, so the others are dropped. Nested functions
    inside a namespace ARE lifted out, because they are ordinary callable
    functions that merely arrived wrapped.

    Returns (chat_tools, dropped) so the caller can say what it ignored
    instead of silently narrowing the model's options.
    """
    if not tools:
        return None, []
    out, dropped = [], []

    def take(t):
        if t.get("type") != "function" or not t.get("name"):
            dropped.append(t.get("type") or "?")
            return
        out.append({"type": "function", "function": {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("parameters") or {"type": "object",
                                                  "properties": {}}}})

    for t in tools:
        if t.get("type") == "namespace":
            for sub in t.get("tools") or []:
                take(sub)
            continue
        take(t)
    return (out or None), dropped


def _item_parts(item):
    """One Responses content array -> a chat-shaped `content` value.

    Text-only items return the concatenated STRING they always returned, so
    every text request renders exactly the bytes it rendered before. An item
    carrying an image returns the ordered part LIST that
    /v1/chat/completions already renders, so both wires reach one image path.

    IMAGES WERE SILENTLY DROPPED HERE. The previous form kept only parts with
    a `text` key, so an `input_image` vanished and the model answered about a
    picture it never saw, fluently and with no error: the same failure the
    endpoint smoke caught engine-side, live on the wire the Codex CLI uses.
    A part this server cannot serve now REFUSES instead of disappearing.

    The Responses spelling is `{"type": "input_image", "image_url": "data:..."}`
    with `image_url` a STRING; chat nests it one level deeper as
    `{"url": ...}`. Both are accepted, because clients send both and neither
    is ambiguous.

    Raises ValueError, which the caller turns into a 400.
    """
    c = item.get("content")
    if isinstance(c, str):
        return c
    if not isinstance(c, list):
        return ""
    parts, has_img = [], False
    for p in c:
        if isinstance(p, str):
            parts.append({"type": "text", "text": p})
            continue
        if not isinstance(p, dict):
            continue
        kind = p.get("type") or ""
        if not kind:
            kind = ("input_image" if ("image_url" in p or "image" in p)
                    else "input_text")
        if kind in ("input_image", "image_url", "image"):
            u = p.get("image_url", p.get("image"))
            if isinstance(u, dict):
                u = u.get("url")
            if not u:
                if p.get("file_id"):
                    raise ValueError(
                        "input_image.file_id is not supported: this server "
                        "has no file store. Inline the image as a data: URL.")
                raise ValueError("input_image carries no image_url")
            parts.append({"type": "image_url", "image_url": {"url": u}})
            has_img = True
        elif kind in ("input_file", "file"):
            # Refuse rather than drop. A dropped attachment reads to the
            # client as an answer about a document nobody sent, which is the
            # defect this function was fixed for.
            raise ValueError("file inputs are not supported by this server")
        elif isinstance(p.get("text"), str):
            parts.append({"type": "text", "text": p["text"]})
    if not has_img:
        return "".join(p["text"] for p in parts if p["type"] == "text")
    return parts


def responses_messages(instructions, items):
    """Responses `instructions` + `input` -> this template's message list.

    Three item shapes matter and each maps onto something the chat template
    already renders:

      message              -> the same role, except `developer`, which is
                              OpenAI's name for a system turn that follows the
                              real system prompt. This template has no such
                              role and raises on an unknown one, so it becomes
                              `system`.
      function_call        -> an assistant turn carrying `tool_calls`
      function_call_output -> a `tool` turn carrying `tool_call_id`

    CONSECUTIVE function_calls MERGE into one assistant turn. The Responses
    wire lists parallel calls as separate top-level items, while chat carries
    them as one message with several `tool_calls`, and the template renders
    tool results positionally against that list. Emitting one assistant turn
    per call would present two parallel calls as two sequential turns and
    silently reassociate the results.
    """
    # EVERY SYSTEM-ISH TURN BECOMES ONE LEADING SYSTEM MESSAGE.
    #
    # Codex sends BOTH `instructions` (20 KB) and a `developer` message, and
    # this template refuses with "System message must be at the beginning."
    # on the second one. Found by running the real client against this
    # endpoint; the translation looked correct until then, because a request
    # built by hand has only one of the two.
    #
    # So they are concatenated in arrival order into a single system turn.
    # That is a real narrowing -- a developer turn placed AFTER a user turn
    # moves to the front -- and it is the only mapping this template admits.
    # Codex places its developer turn first, so for the client this endpoint
    # exists for, order is preserved exactly.
    sys_parts = []
    if instructions:
        sys_parts.append(instructions)
    msgs = []
    if isinstance(items, str):
        if sys_parts:
            msgs.append({"role": "system", "content": "\n\n".join(sys_parts)})
        if items:
            msgs.append({"role": "user", "content": items})
        return msgs

    for it in items or []:
        if not isinstance(it, dict):
            continue
        kind = it.get("type") or "message"
        if kind == "function_call":
            call = {"id": it.get("call_id") or it.get("id") or "",
                    "type": "function",
                    "function": {"name": it.get("name") or "",
                                 "arguments": it.get("arguments") or "{}"}}
            if msgs and msgs[-1].get("role") == "assistant"                     and "tool_calls" in msgs[-1]                     and not msgs[-1].get("content"):
                msgs[-1]["tool_calls"].append(call)
            else:
                msgs.append({"role": "assistant", "content": "",
                             "tool_calls": [call]})
            continue
        if kind == "function_call_output":
            out = it.get("output")
            if not isinstance(out, str):
                out = json.dumps(out, ensure_ascii=False)
            msgs.append({"role": "tool",
                         "tool_call_id": it.get("call_id") or "",
                         "content": out})
            continue
        if kind in ("reasoning", "web_search_call", "item_reference"):
            # Reasoning items are echoed back by the client from a previous
            # turn. This server does not persist reasoning and the template
            # re-renders the assistant turn without it, so there is nothing
            # to reconstruct and dropping them is the honest move.
            continue
        role = it.get("role") or "user"
        content = _item_parts(it)
        if role in ("developer", "system"):
            # A system turn is CONCATENATED into one leading message, which a
            # part list cannot survive; and the template writes no image
            # placeholder for a system turn anyway. Refusing says which turn
            # to move the image to.
            if isinstance(content, list):
                raise ValueError("images are not supported in a system or "
                                 "developer turn; send the image in a user "
                                 "turn")
            if content:
                sys_parts.append(content)
            continue
        if content or role == "assistant":
            msgs.append({"role": role, "content": content})
    if sys_parts:
        msgs.insert(0, {"role": "system", "content": "\n\n".join(sys_parts)})
    return msgs


SAMPLING_FIELDS = ("n",)



def penalty_spec(req):
    """The PENALTY / BIAS / LOGPROBS suffixes, or "" for none.

    All three are sampler-only and the engine REJECTS them without
    temperature>0 rather than ignoring them: greedy takes the argmax, so an
    additive delta that does not reorder it changes nothing a client could
    see, and a logprob has no sampler to come from. Building them here only
    when sampling keeps that rejection unreachable from a well-formed request.
    """
    if sample_spec(req) is None:
        return ""
    out = ""
    pp = float(server_default(req, "presence_penalty") or 0.0)
    fp = float(server_default(req, "frequency_penalty") or 0.0)
    if pp or fp:
        out += " PENALTY %.9g %.9g" % (pp, fp)
    if req.logit_bias:
        items = []
        for k, v in req.logit_bias.items():
            try:
                tid = int(k)
            except (TypeError, ValueError):
                raise HTTPException(400, f"logit_bias key {k!r} is not a "
                                         f"token id")
            if not 0 <= tid < 248320:
                raise HTTPException(400, f"logit_bias token id {tid} is "
                                         f"outside 0..248319")
            items.append((tid, float(v)))
        if len(items) > 20480:
            raise HTTPException(400, f"logit_bias has {len(items)} entries, "
                                     f"over this server's cap of 20480")
        out += " BIAS %d " % len(items) + " ".join(
            "%d %.9g" % (i, v) for i, v in items)
    if req.logprobs:
        out += " LOGPROBS"
    return out


def sample_spec(req):
    """(temp, top_k, top_p, min_p, seed) or None for greedy.

    None is returned for temperature 0 or absent, and that is not a shortcut:
    temperature 0 IS greedy, and routing it through the sampler would put a
    new kernel under the identity hashes that certify the greedy path.
    """
    # Issue #30: a request that OMITS temperature takes the server's default;
    # one that SENDS 0 chose greedy and takes no sampling default at all.
    t = server_default(req, "temperature")
    if t is None or t <= 0.0:
        return None
    # Whether the ENGINE samples is check_sampling's question (it has the
    # engine's INFO in scope); this only shapes the request.
    # A seed is REQUIRED by the engine (the RNG is counter-based and stateless),
    # but optional in the API. Absent, draw one: the request is then genuinely
    # unreproducible, which is the honest reading of "no seed given" -- rather
    # than a fixed default that would make every unseeded request identical.
    seed = req.seed if req.seed is not None else random.getrandbits(63)
    return (float(t), int(server_default(req, "top_k") or 0),
            float(server_default(req, "top_p") or 0.0),
            float(server_default(req, "min_p") or 0.0), int(seed))
_warned_sampling = set()
# Wire values of src/serve.cpp's GEN drafter field (an earlier change, an earlier change).
DRAFTERS = {"serial": 0, "mtp": 1, "dflash2": 2}
# Which INFO capability makes each one selectable. serial always is.
DRAFTER_CAP = {"mtp": "mtp", "dflash2": "dflash2"}


# an earlier change. The forced-call opener. This template has no tool_choice support of
# its own, so `required` and a named function are implemented as a PROMPT
# PREFILL: the opener is appended after the generation prompt and the model
# resumes inside a call it cannot decline. The cost is stated in /health and
# the prefill occupies the position the reasoning block would
# have used, so a forced call does not think.
FORCE_OPEN = "<tool_call>\n<function="


def tool_choice_mode(req):
    """-> (mode, forced_name); mode is auto | none | required | function."""
    tc = req.tool_choice
    if tc is None or tc == "auto":
        return "auto", None
    if isinstance(tc, str):
        if tc == "none":
            return "none", None
        if tc == "required":
            if not req.tools:
                raise HTTPException(400, "tool_choice 'required' needs tools")
            return "required", None
        raise HTTPException(400, f"tool_choice '{tc}' unknown; use "
                                 f"auto | none | required | "
                                 f"{{'type': 'function', ...}}")
    if isinstance(tc, dict) and tc.get("type") == "function":
        name = (tc.get("function") or {}).get("name")
        known = {(t.get("function") or t).get("name") for t in req.tools or []}
        if not isinstance(name, str):
            raise HTTPException(400, "tool_choice.function.name is required")
        if name not in known:
            raise HTTPException(400, f"tool_choice names '{name}', which is "
                                     f"not in tools")
        return "function", name
    raise HTTPException(400, "tool_choice must be a string or a "
                             "{'type': 'function', ...} object")


def force_prefill(mode, name, tools):
    """The text appended after the generation prompt to force a call.

    MEASURED 2026-08-22, and it is the difference between a guarantee and a
    hope: prefilling only `<tool_call>\n<function=` still lets the model
    decline, by emitting the closing tags with an EMPTY name —

        <tool_call>\n<function=\n</function>\n</tool_call>

    which is what it did when asked "Hi, how are you?" with tool_choice
    'required'. Naming the function in the prefill removes that escape
    entirely (same prompt, same effort: it called get_weather for New York).
    So when there is exactly one tool there is no choice to make and the name
    goes in — 'required' is then exact. With several tools the model must
    pick, the escape is open again, and serve() turns a declined forced call
    into a 502 rather than handing back raw markup as content.
    """
    if mode == "function":
        return FORCE_OPEN + name + ">\n"
    if mode == "required":
        names = [(t.get("function") or t).get("name") for t in tools or []]
        if len(names) == 1 and isinstance(names[0], str):
            return FORCE_OPEN + names[0] + ">\n"
        return FORCE_OPEN
    return ""


def wants_usage(req):
    so = req.stream_options
    return bool(isinstance(so, dict) and so.get("include_usage"))


def chat_kwargs(req, mode="auto"):
    """Template kwargs from an OpenAI-shaped request. Only pass what the
    caller actually set — the template's own defaults (thinking on, effort
    xhigh) are the model's intended behavior. Issue #30: the operator may
    move the effort default (HALOGEN_REASONING_EFFORT); a request that names
    one still wins."""
    kw = {}
    if req.reasoning_effort is not None and not effort_is_none(req):
        e = EFFORT_MAP.get(req.reasoning_effort.lower())
        if e is None:
            raise HTTPException(400, f"reasoning_effort "
                                     f"'{req.reasoning_effort}' unsupported; "
                                     f"use {'|'.join(EFFORT_VALUES)}")
        kw["reasoning_effort"] = e
    elif DEFAULTS["reasoning_effort"] is not None:
        kw["reasoning_effort"] = DEFAULTS["reasoning_effort"]
    if server_default(req, "enable_thinking") is not None:
        kw["enable_thinking"] = server_default(req, "enable_thinking")
    if effort_is_none(req):
        # #50: after the server default, so "none" wins over
        # HALOGEN_ENABLE_THINKING=1; the effort is meaningless without a
        # think block and is not passed.
        kw["enable_thinking"] = False
        kw.pop("reasoning_effort", None)
    if req.preserve_thinking is not None:
        kw["preserve_thinking"] = req.preserve_thinking
    # tool_choice='none' means the model must not call one, and the cleanest
    # guarantee of that is a prompt that never mentions tools. Note the
    # consequence: it is a DIFFERENT prefix from an 'auto' turn, so alternating
    # the two in one session costs a prompt-cache miss each time (an earlier change).
    if req.tools and mode != "none":
        kw["tools"] = req.tools
    return kw


class ThinkSplit:
    """Splits generated text into reasoning_content / content at the first
    '</think>'.

    The template puts the OPENING '<think>' into the prompt itself, so the
    model emits reasoning first and only ever produces the CLOSING tag —
    everything before it is reasoning, everything after is the answer. With
    enable_thinking=false the prompt is pre-closed, no marker appears, and
    all output is content. Tracking full text (rather than per-delta) keeps
    a marker split across two deltas correct.
    """

    MARK = "</think>"

    def __init__(self, thinking=True):
        # thinking=False: the template already emitted '<think>\n\n</think>'
        # into the PROMPT, so no marker appears in the generated text and
        # every byte is answer. Without this flag a no-marker stream is
        # ambiguous and the whole reply misfiles as reasoning.
        self.thinking = thinking
        self.full = ""
        self.sent_r = self.sent_c = 0

    def push(self, delta):
        """-> (reasoning_delta, content_delta)"""
        self.full += delta
        if not self.thinking:
            out = self.full[self.sent_c:]
            self.sent_c = len(self.full)
            return "", out
        i = self.full.find(self.MARK)
        if i < 0:
            # marker may be half-arrived; hold back a possible prefix
            hold = 0
            for k in range(len(self.MARK) - 1, 0, -1):
                if self.full.endswith(self.MARK[:k]):
                    hold = k
                    break
            r = self.full[:len(self.full) - hold]
            out = r[self.sent_r:]
            self.sent_r = len(r)
            return out, ""
        r, c = self.full[:i], self.full[i + len(self.MARK):]
        rd, cd = r[self.sent_r:], c[self.sent_c:]
        self.sent_r, self.sent_c = len(r), len(c)
        return rd, cd

    def parts(self):
        if not self.thinking:
            return "", self.full
        i = self.full.find(self.MARK)
        if i < 0:
            # thinking on but never closed (hit max_tokens mid-reasoning)
            return self.full.strip(), ""
        return (self.full[:i].strip(),
                self.full[i + len(self.MARK):].lstrip("\n"))


def build_app(tok, engine, ctx, max_cap=4096, queue_timeout=600):
    engine.tokenizer = tok   # 0.8.0: the VOCAB line is built from it
    app = FastAPI(title="halogen")

    # Incremental detokenization is sound only if
    # decode splits on token boundaries, i.e.
    #     decode(ids) == decode(ids[:-1]) + decode([ids[-1]])
    # The one setting that breaks that is `clean_up_tokenization_spaces`,
    # which rewrites punctuation spacing across the whole string and so makes
    # a token's rendering depend on what follows it. This tokenizer ships it
    # False and we verified the property directly (9,998 sequences, random ids
    # and every prefix of real byte-fallback-heavy text, 0 failures). A
    # tokenizer that turned it on would silently corrupt output, so the guard
    # is a PROPERTY OF THE TOKENIZER read at load, not a constant we assume:
    # when it does not hold, the older whole-list decode runs unchanged.
    INCREMENTAL_DETOK = not getattr(tok, "clean_up_tokenization_spaces", False)
    # A UTF-8 character is at most 4 bytes, so at most 4 byte-fallback tokens
    # can be mid-character. The cap is a backstop against a tokenizer that
    # holds back further: past it we stop trusting the window and re-decode.
    CARRY_MAX = 8

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request, exc):
        # 503 here is only ever the batch-1 queue timing out, and saying so in
        # `code` lets a client back off on that specifically rather than on
        # every 5xx.
        return oai_error(exc.status_code, str(exc.detail),
                         "engine_busy" if exc.status_code == 503 else None)

    @app.exception_handler(Exception)
    async def _unhandled(request, exc):
        # A bare "Internal Server Error" with nothing in the container log is
        # a dead end for whoever is debugging it. Print the traceback where
        # `podman logs` shows it and name
        # the exception in the body; Starlette re-raises after this, so
        # uvicorn's own logging is unchanged.
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        return oai_error(500, f"internal error: {type(exc).__name__}: "
                              f"{str(exc)[:300]}", "server_error")

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request, exc):
        # Reported as 400, not FastAPI's 422: OpenAI uses 400 for a malformed
        # request and SDK retry logic is written against that.
        parts = []
        for e in exc.errors():
            loc = ".".join(str(x) for x in e.get("loc", ())
                           if x != "body") or "body"
            parts.append(f"{loc}: {e.get('msg', 'invalid')}")
        return oai_error(400, "; ".join(parts) or "invalid request")
    def get_eos_ids(stops=None):
        ids = set()
        raw = getattr(tok, "eos_token_id", None)
        if isinstance(raw, (list, tuple, set)):
            ids.update(raw)
        elif isinstance(raw, int) and raw >= 0:
            ids.add(raw)
        for s in ("<|im_end|>", "<|endoftext|>"):
            tid = tok.convert_tokens_to_ids(s)
            if isinstance(tid, int) and tid >= 0:
                ids.add(tid)
        for s in (stops or []):
            if isinstance(s, str):
                tid = tok.convert_tokens_to_ids(s)
                if isinstance(tid, int) and tid >= 0 and tid != getattr(tok, "unk_token_id", None):
                    ids.add(tid)
        return [i for i in ids if isinstance(i, int) and i >= 0]

    eos_ids = get_eos_ids()

    def stop_list(stop):
        if stop is None:
            return []
        return [stop] if isinstance(stop, str) else list(stop)

    def schema_for(req):
        """The schema text serve() sends, or None. check_sampling() has
        already refused what cannot be enforced."""
        rf = getattr(req, "response_format", None)
        if isinstance(rf, dict) and rf.get("type") in ("json_schema", "json_object"):
            return schema_text(rf)
        return None

    def check_sampling(req):
        """Reject what cannot be honoured, rather than accepting and dropping.

        the same lesson in a different place: a request that is silently not
        served the way it was asked is indistinguishable, from the response,
        from one that was.
        """
        # Reject HERE, where we can say why, rather than letting the engine
        # refuse the GEN downstream: that path surfaces as a generic "engine
        # rejected the request" and once guessed "prompt longer than
        # context?", so a six-token prompt with temperature 0.7 was told its
        # prompt might be too long. Keyed on the engine's own INFO field.
        # Structured output is not implemented. Refuse rather than
        # ignore, and name what would have to exist: constrained decoding.
        # `{"type": "text"}` is the default and IS honoured, so it passes.
        # Refuse EXACTLY the two structured types this build cannot honour,
        # and pass anything else through as before. The wider form (refuse
        # everything that is not "text") was written first and is wrong here:
        # /v1/responses exists so the Codex CLI works, `text` is a field no
        # test we own populates and no record here says what Codex puts in
        # it, and turning an unknown-but-harmless value into a 400 would
        # break the client this endpoint is FOR. Refusing a silent drop is
        # the fix; refusing the unrecognised is a guess, and this line faces
        # a real client.
        # 0.8.0: structured output. The schema is taken here, once,
        # in either spelling, and checked against what this connection's
        # engine can enforce; the engine compiles it at admission and a
        # keyword it refuses comes back as the same 400 shape.
        rf = getattr(req, "response_format", None)
        if isinstance(rf, dict) and rf.get("type") in ("json_schema", "json_object"):
            if not engine.info.get("grammar"):
                why = engine.info.get("grammar_error")
                raise HTTPException(
                    400,
                    f"response_format (`text.format` on /v1/responses) "
                    f"{rf.get('type')!r} cannot be enforced here: "
                    + (f"the engine's vocabulary handshake failed ({why})"
                       if why else
                       f"the engine (version {engine.info.get('version', 'unknown')}) "
                       "has no constrained decoding; 0.8.0 and later do")
                    + ". /health reports `structured_output`. Omit the field "
                    "and parse the reply, or prompt for JSON and validate it.")
            if sample_spec(req) is not None:
                raise HTTPException(
                    400,
                    "response_format json_schema / json_object decodes greedy "
                    "in this release: send temperature 0 (or omit it). "
                    "Sampling under a schema is not implemented yet.")
            schema_text(rf)   # the shape check; the routes take the text via schema_for()
        if sample_spec(req) is not None and not engine.info.get("sampling"):
            raise HTTPException(
                400,
                "this engine build decodes greedy only (a forward pass and an "
                "argmax); it does not sample. Omit `temperature` or send "
                "`temperature: 0`. /health reports what this build supports.")
        # RANGE VALIDATION, and it is the an earlier change rule again rather than
        # schema pedantry. The engine reads top_p >= 1 as "no nucleus filter",
        # so top_p=5 does not fail — it silently serves an UNFILTERED nucleus
        # to a client that asked for a tight one, with a 200 and output that
        # looks entirely reasonable. That is the same defect an earlier change found from
        # the other direction, where "%g" rendered 0.9999999 as "1". A value
        # outside the range a parameter is DEFINED on is a client error and
        # must say so.
        for f, lo, hi in (("temperature", 0.0, 2.0), ("top_p", 0.0, 1.0),
                          ("min_p", 0.0, 1.0)):
            v = getattr(req, f)
            if v is not None and not (lo <= float(v) <= hi):
                raise HTTPException(
                    400, f"{f}={v} is outside {lo}..{hi}. Values past the "
                         f"range are NOT clamped, because a clamped request is "
                         f"indistinguishable, from the response, from one "
                         f"that was served as asked.")
        if req.top_k is not None and req.top_k < 0:
            raise HTTPException(400, f"top_k={req.top_k} must be >= 0 "
                                     f"(0 disables the filter).")
        if req.logprobs and req.stream:
            raise HTTPException(
                400, "logprobs with stream=true is not implemented. The "
                     "logprob column is collected and returned with the "
                     "finished response. Send stream=false, or drop "
                     "logprobs.")
        if req.top_logprobs:
            raise HTTPException(
                400, "top_logprobs is not implemented: this server returns "
                     "the logprob of the CHOSEN token only. Send "
                     "logprobs=true without top_logprobs.")
        if sample_spec(req) is None:
            for f, why in (("presence_penalty", req.presence_penalty),
                           ("frequency_penalty", req.frequency_penalty),
                           ("logit_bias", req.logit_bias),
                           ("logprobs", req.logprobs)):
                if why:
                    raise HTTPException(
                        400, f"{f} applies to the sampler, and this request "
                             f"decodes greedy (temperature absent or 0). "
                             f"Send temperature > 0, or drop {f}: a request "
                             f"silently served without it would be "
                             f"indistinguishable from one that honoured it.")

    def drafter_for(req):
        """Wire drafter value.

        Sampling does not constrain the drafter: temperature>0 on the MTP
        drafter runs the speculative accept/reject rule -- min(1, p/q) with a
        residual correction, which emits exactly p rather than requiring
        argmax equality -- so a sampled request keeps the fast path.

        So sampling no longer constrains the drafter and this is a plain
        lookup again. What it must NOT become is a silent substitution:
        s7's rule is that a quiet downgrade makes any drafter comparison lie,
        which is why the restriction was a 400 rather than a fallback while it
        existed.
        """
        return drafter_id(req.drafter)

    def drafter_id(name):
        """Request drafter name -> wire value.

        Unknown or unavailable is a 400, never a silent downgrade: a request
        that quietly fell back to serial would make an mtp-vs-serial or an
        mtp-vs-dflash2 comparison lie about what actually ran. The benchmark varies
        exactly this field and nothing else, so the field has to be honest.
        """
        if name is None:
            return None                 # engine applies HALOGEN_DRAFTER
        key = name.lower()
        v = DRAFTERS.get(key)
        if v is None:
            raise HTTPException(400, f"drafter '{name}' unknown; use "
                                     f"{' | '.join(DRAFTERS)}")
        cap = DRAFTER_CAP.get(key)
        if cap is not None and not engine.info.get(cap):
            raise HTTPException(400, f"drafter '{key}' unavailable: the "
                                     f"loaded checkpoint cannot serve it")
        return v

    # 0.7.0, #44: `</think>` is ONE token here (id 248069; checked against
    # the tokenizer at startup below), so the number of reasoning tokens a
    # reply spent is its position in the generated ids plus one, exactly,
    # on both the streamed and the collected path. 0 means the marker never
    # came: every token was reasoning if thinking was on, none if it was off,
    # and the caller (which knows the template's state) decides which.
    THINK_END_ID = tok.convert_tokens_to_ids(ThinkSplit.MARK) \
        if hasattr(tok, "convert_tokens_to_ids") else None
    if not isinstance(THINK_END_ID, int) or THINK_END_ID < 0:
        THINK_END_ID = None
    # 0.8.1 (issue #56): Qwen's thinking-budget close, as ids. The sentence
    # is the one Qwen's own budget mechanism appends when a think block runs
    # out of room; the marker and the two newlines are what the template
    # writes after a block. Tokenized once, with the marker as its own token
    # (a plain encode may merge it with its neighbours).
    THINK_CLOSE_IDS = (
        tok("\n\nConsidering the limited time by the user, I have to give the "
            "solution based on the thinking directly now.\n",
            add_special_tokens=False)["input_ids"]
        + ([THINK_END_ID] if THINK_END_ID is not None else [])
        + tok("\n\n", add_special_tokens=False)["input_ids"])
    # 0.8.0: the <tool_call> token, the one escape from a schema.
    TOOL_CALL_ID = tok.convert_tokens_to_ids("<tool_call>") \
        if hasattr(tok, "convert_tokens_to_ids") else None
    if not isinstance(TOOL_CALL_ID, int) or TOOL_CALL_ID < 0:
        TOOL_CALL_ID = None

    async def run(ids, max_tokens, stops, drafter=None, sample=None,
                  penalty="", snap=0, snap2=0, images=None, schema=None,
                  after=None, escape=(), think=None):
        """Drives the engine and incrementally detokenizes.

        Public issue #13, reported with measurements and a patch by
        @rosstang, independently confirmed by @hvico). This loop used to
        decode the WHOLE token list every step and diff against what had
        been emitted, which is O(n^2) in the answer length: ~19 us at 128
        tokens, ~1,088 us at 8,192, and about 28 s of pure detokenization
        spread through a reply that runs to the 16,384 cap. The ledger's own
        comment below named it a SUSPECT and never measured it.

        The whole-list decode was justified here on the grounds that a
        token's text can depend on its successor. For this tokenizer that is
        false, and it is checked rather than assumed (see INCREMENTAL_DETOK):
        decode splits on token boundaries, so the only text that can still
        change is a byte-fallback token holding part of an unfinished UTF-8
        character. So we decode a CARRY WINDOW of at most a few tokens, which
        is O(1) a step, and the window closes the moment the character
        resolves.

        Where the window's assumption fails for a token, this falls back to
        the original whole-list decode FOR THAT STEP and emits exactly what
        the older loop would have. That is the difference between this and
        the reported patch, which appends the differing suffix instead and so
        changes what the fallback branch does.
        """
        # an earlier change: ask the ENGINE, don't trust a CLI default. --context and
        # Model::kMaxCtx are two copies of one number and they drifted the
        # moment kMaxCtx moved: the front-end would have kept rejecting at
        # 32,768 against an engine serving 131,072, and the symptom is a 400
        # that looks like a client bug. INFO already reports it.
        limit = engine.info.get("ctx") or ctx
        if len(ids) >= limit:
            raise HTTPException(400, f"prompt {len(ids)} tokens exceeds "
                                     f"context {limit}")
        out, text, done = [], "", None
        lps = []

        def think_end(dd):
            # the marker's position + 1 = the reasoning tokens (THINK_END_ID)
            if THINK_END_ID is not None and THINK_END_ID in out:
                dd["think_end"] = out.index(THINK_END_ID) + 1
            else:
                dd["think_end"] = 0
            return dd
        # The carry window. `tail_ids` are tokens whose rendering is not
        # yet complete (a byte-fallback token holding part of a character) and
        # `tail_text` is what has already been emitted for them. Both empty in
        # the steady state, which is every token that is not mid-character.
        tail_ids, tail_text = [], ""
        # Sticky: if the window's assumption ever fails for this request, stop
        # using it for the REST of the request rather than reopening it on the
        # next token. A failure can leave a character still unfinished, and the
        # tokens carrying it are not recoverable from a whole-list decode, so
        # reopening empty would drop bytes. Unreachable on this tokenizer.
        use_window = INCREMENTAL_DETOK
        # an earlier change step 9. an earlier change step 8 measured 32.5 ms/step of front-end cost
        # at c=8 and named a SUSPECT (O(n^2) detok) without measuring it.
        # METHOD 45: measure before fixing. These three sum to the wall of
        # this loop, so they apportion it rather than sampling it.
        prof = {"detok": 0.0, "wait": 0.0, "n": 0}
        _t_last = time.perf_counter()
        req_eos = get_eos_ids(stops)
        async for tid, d, lp in engine.generate(ids, max_tokens, req_eos,
                                                drafter, sample, penalty,
                                                snap=snap, snap2=snap2,
                                                images=images, schema=schema,
                                                after=after, escape=escape,
                                                think=think):
            if lp is not None:
                lps.append(lp)
            if d is not None:
                done = d
                if d.get("reason") != "error":
                    engine.metrics.record(timings(d), structured=bool(schema))
                # The serving ledger an earlier change asked for: what real traffic
                # actually commits per round, not what a fixture does.
                # an earlier change: this WAS guarded by `if d.get("rounds")`, i.e. it
                # only logged when a DRAFTER ran. The batched scheduler emits a
                # plain 6-field D line, so batched requests logged NOTHING --
                # and tools/bench-serving.py, the benchmark of record, parses
                # exactly this line, so the whole batched path was invisible to
                # it. The throughput half is now unconditional and keeps that
                # regex's shape; the spec detail stays conditional because only
                # a drafter has rounds to report.
                # A 1-token response has NO decode rate to report: its only
                # token is the one prefill already produced, so decode_ms
                # rounds to 0. This used to floor dt at 1e-9 and print
                # "1000000000.00 t/s" -- a fabricated number sitting in the
                # exact field a real rate goes. `sweep`'s prefill probes
                # (max_tokens=1) hit it on every single call, so the first
                # thing a user benchmarking this image saw in the log was a
                # 1e9 t/s line.
                dt = d["decode_ms"] / 1000
                name = {1: "mtp", 2: "dflash2"}.get(d.get("drafter"),
                                                    "spec" if d.get("rounds")
                                                    else "batch")
                spec = ""
                shared = d.get("shared", 0)
                if d.get("rounds"):
                    # PUBLIC ISSUE #22: `commit` counts EVERY token the
                    # request produced, and `rounds` only the speculative
                    # rounds it ran while alone. A request that got one round
                    # alone and then shared the engine for 189 tokens printed
                    # `1 rounds, commit 190.00/round`, and two reporters on
                    # two machines read that as a decode defect. The ratio is
                    # now over the speculative tokens only, and the shared
                    # tokens are named at the end of the line. Healthy is
                    # 1.6 to 1.8. The `N rounds, commit X/round | prompt`
                    # shape is what bench-serving.py's LEDGER regex parses,
                    # so the new count goes after it, not inside it.
                    spec = (f"{d['rounds']} rounds, commit "
                            f"{(d['commit'] - shared) / d['rounds']:.2f}"
                            f"/round | ")
                pld = ""
                if d.get("pld_rounds"):
                    # how many of those rounds drafted from the request's
                    # own context (prompt lookup), and what they accepted
                    # per round. A TRAILING clause, after detok: the
                    # `rounds, commit X/round | prompt N` shape is what
                    # bench-serving.py's LEDGER regex parses, and the first
                    # cut put this inside it and silently dropped every
                    # prompt-lookup case from the bench's mean.
                    pld = (f" | pld {d['pld_rounds']} rounds, "
                           f"{d['pld_accepted'] / d['pld_rounds']:.2f}"
                           f" acc/round")
                # LEDGER in tools/bench-serving.py requires `([\d.]+) t/s`
                # AND a `rounds, commit` clause. These degenerate lines never
                # carry rounds, so "n/a" cannot break that parser -- but the
                # rate keeps its numeric shape in every case that does.
                rate = (f"{d['n_gen'] / dt:.2f} t/s"
                        if dt > 0 and d["n_gen"] > 1 else "n/a")
                print(f"serve_api: {name} {d['n_gen']} tok in {dt:.2f}s = "
                      f"{rate} | {spec}"
                      f"prompt {d['n_prompt']}"
                      f"{f' ({c} cached)' if (c := d.get('n_cached')) else ''}"
                      f", prefill {d['prefill_ms'] / 1000:.2f}s"
                      # detok us/token is a CANARY, not a stat: this loop
                      # decodes the WHOLE token list every step, so its cost
                      # is O(n^2) and this figure climbs with output length
                      # -- 17 us/tok at 128 tokens, a projected 2000 us/tok at
                      # the 16,384 cap (~6% of the request). It is the only
                      # thing that would show that, and the measurement that
                      # "refuted" the O(n^2) suspect was taken at n=128, where
                      # O(n^2) is invisible by construction. `wait` and `sse`
                      # were dropped: wait duplicates the t/s field and
                      # changes meaning with concurrency, sse is a constant.
                      f" | detok {prof['detok'] / max(prof['n'], 1) * 1e6:.0f}us/tok"
                      # Issue #22, the other half: how many of this request's
                      # tokens were produced beside other streams (one row of
                      # a batched step, or a serial rest stretch), where the
                      # per-stream rate is the README's concurrency table and
                      # not the single-stream figure.
                      + (f" | {shared} tok beside other streams" if shared
                         else "")
                      + pld,
                      flush=True)
                break
            # time spent WAITING on the engine == everything since we last
            # finished a token; it is the term the other two are competing
            # against and it must be in the ledger or the shares are wrong.
            _t0 = time.perf_counter()
            prof["wait"] += _t0 - _t_last
            out.append(tid)
            # Decode the carry window, not the transcript. `base` is
            # whatever `new` is expected to start with, so the diff below is
            # the same diff as before on a bounded string.
            windowed = use_window
            if windowed:
                win_ids = tail_ids + [tid]
                new, base = tok.decode(win_ids, skip_special_tokens=True), tail_text
            else:
                new, base = tok.decode(out, skip_special_tokens=True), text
            prof["detok"] += time.perf_counter() - _t0
            prof["n"] += 1
            # HOLD BACK an incomplete multi-byte character. A byte-fallback
            # token can carry the first byte(s) of a UTF-8 sequence, and
            # tok.decode renders that as U+FFFD until the next token completes
            # it. Emitting the U+FFFD means the NEXT decode no longer starts
            # with what we already sent.
            #
            # This was a live bug, found by hitting the endpoint with a real
            # prompt: the old code answered a mismatch by resetting text to ""
            # — which makes the next delta the ENTIRE accumulated string, so
            # the client receives the whole reasoning block a second time,
            # spliced into the middle of the answer. `5 ÷ 2 = 2.5` came back as
            # 167 characters instead of 12. Its comment called the case a
            # "rare retokenization shuffle"; it is neither rare nor a shuffle,
            # it fires on any character the tokenizer byte-splits (÷ does;
            # café, 😊 and 日本語 are single tokens here and do not).
            holding = new.endswith("�")
            while new.endswith("�"):
                new = new[:-1]
            if windowed and not new.startswith(base):
                # The window did not behave. Rather than guess, re-decode
                # the whole list and run the ORIGINAL logic for this step, so
                # the bytes are exactly the older path's. Then close the
                # window and carry on incrementally.
                new = tok.decode(out, skip_special_tokens=True)
                while new.endswith("�"):
                    new = new[:-1]
                base, tail_ids, tail_text = text, [], ""
                holding, windowed, use_window = False, False, False
            if not new.startswith(base):
                # Genuine retokenization: emit only what actually differs
                # rather than re-sending everything. Never reset to "".
                n = 0
                while n < len(base) and n < len(new) and base[n] == new[n]:
                    n += 1
                delta = new[n:]
                text = text + delta if windowed else new
            else:
                delta = new[len(base):]
                text = text + delta if windowed else new
            if windowed:
                # Hold the window open only while a character is unfinished,
                # and only as long as one can be: past CARRY_MAX this is not a
                # split character any more, so stop trusting the window.
                if holding and len(win_ids) < CARRY_MAX:
                    tail_ids, tail_text = win_ids, new
                else:
                    tail_ids, tail_text = [], ""
            # A stop string can only become newly complete inside `delta`
            # plus the longest stop minus one character of what preceded it.
            # Scanning the whole transcript every step was this loop's second
            # O(n^2) term, invisible beside the decode. The span is derived,
            # not guessed, so this finds exactly what the full scan found.
            if stops and delta:
                span = max(len(t) for t in stops if t) - 1 + len(delta)
                win = text[-span:] if 0 < span < len(text) else text
                hit = next((s for s in stops if s and s in win), None)
            else:
                win, hit = "", None
            if hit:
                cut = len(text) - len(win) + win.index(hit)
                tail = delta[:max(0, cut - (len(text) - len(delta)))]
                if tail:
                    yield tail, None
                await engine.abort()   # drains through D — never bare cancel
                yield None, think_end({"reason": "stop", "n_gen": len(out),
                                       "n_prompt": len(ids), "prefill_ms": 0.0,
                                       "decode_ms": 0.0, "logprobs": lps})
                return
            if delta:
                _t_last = time.perf_counter()
            yield delta, None
        if done is not None:
            done["logprobs"] = lps
        yield None, think_end(done or {"reason": "length", "n_gen": len(out),
                                       "n_prompt": len(ids), "prefill_ms": 0.0,
                                       "decode_ms": 0.0, "logprobs": lps})

    def reasoning_tokens(d, thinking):
        """0.7.0, #44: how many generated tokens were reasoning. The
        collector records the `</think>` token's position (`think_end`);
        with thinking on and no marker, the whole reply was reasoning (it
        hit the budget inside the think block); with thinking off, none."""
        if d.get("think_end"):
            return min(int(d["think_end"]), int(d.get("n_gen", 0)))
        return int(d.get("n_gen", 0)) if thinking else 0

    def usage(d, n_prompt):
        u = {"prompt_tokens": n_prompt, "completion_tokens": d["n_gen"],
             "total_tokens": n_prompt + d["n_gen"]}
        # an earlier change. OpenAI's own field name for this, so a client that already
        # tracks cache hits reads it without a halogen-specific branch. It
        # counts prompt tokens the engine did NOT re-prefill; they are still
        # billed as prompt_tokens because they are still context.
        if d.get("n_cached"):
            u["prompt_tokens_details"] = {"cached_tokens": d["n_cached"]}
        # 0.7.0, #44: OpenAI's own field for the reasoning share of the
        # output, set by the route once it knows the template's state.
        if "n_reasoning" in d:
            u["completion_tokens_details"] = {"reasoning_tokens": d["n_reasoning"]}
        return u

    def responses_usage(d, n_prompt):
        """The Responses wire's usage block from the chat one: OpenAI's
        names, with the cached prompt tokens and (#44) the reasoning share."""
        u = usage(d, n_prompt)
        r = {"input_tokens": u.get("prompt_tokens", 0),
             "output_tokens": u.get("completion_tokens", 0),
             "total_tokens": u.get("total_tokens", 0)}
        if "prompt_tokens_details" in u:
            r["input_tokens_details"] = dict(u["prompt_tokens_details"])
        if "completion_tokens_details" in u:
            r["output_tokens_details"] = dict(u["completion_tokens_details"])
        return r

    def timings(d):
        """0.7.0, public issue #45: llama-server's `timings` object, the one
        llama-swap reads from a body or a stream frame to show prefill and
        decode rates and the draft count (`prompt_n`, `predicted_n`,
        `prompt_ms`, `predicted_ms`, `prompt_per_second`,
        `predicted_per_second`, `cache_n`, `draft_n`, `draft_n_accepted`).
        Every field is the engine's own D line, copied, not measured here:
        `predicted_ms` is the decode wall from the first generated token, so
        `predicted_per_second` is the same figure the engine log prints.
        `prompt_n` is the count the engine PROCESSED, the prompt minus the
        prefix the cache covered, because `prompt_ms` is the engine's time
        on exactly those pieces and nothing else (`work_ms` in flash_serve):
        public issue #48 read a warm 97,052-token turn as 53,000 tok/s
        because the first cut divided the whole prompt by the tail's time.
        That is llama-server's meaning too ("number of prompt tokens being
        processed"; the context is `prompt_n + cache_n + predicted_n`), and
        it is what llama-swap's Input column then shows. `usage` is
        unchanged: `prompt_tokens` stays the whole prompt with
        `prompt_tokens_details.cached_tokens` beside it. `draft_n` counts
        the draft head's proposals (one a round) and the prompt-lookup
        chains' proposed tokens together, since both are speculative rounds
        against the same verify; a serial request reads 0 / 0."""
        pn = int(d.get("n_prompt", 0))
        gn = int(d.get("n_gen", 0))
        cached = int(d.get("n_cached", 0))
        processed = max(0, pn - cached)
        pms = float(d.get("prefill_ms", 0.0))
        dms = float(d.get("decode_ms", 0.0))
        t = {"prompt_n": processed, "predicted_n": gn,
             "prompt_ms": round(pms, 3), "predicted_ms": round(dms, 3),
             "prompt_per_second": (round(processed / (pms / 1000.0), 3)
                                   if pms > 0 and processed > 0 else 0.0),
             "predicted_per_second": round(gn / (dms / 1000.0), 3) if dms > 0 else 0.0}
        if "n_cached" in d:
            t["cache_n"] = cached
        if "rounds" in d:
            pld_r = int(d.get("pld_rounds", 0))
            pld_a = int(d.get("pld_accepted", 0))
            head_r = int(d["rounds"]) - pld_r
            head_a = int(d["commit"]) - int(d["rounds"]) - int(d.get("shared", 0)) - pld_a
            if pld_r == 0 or "pld_drafted" in d:
                t["draft_n"] = max(0, head_r) + int(d.get("pld_drafted", 0))
                t["draft_n_accepted"] = max(0, head_a) + pld_a
        return t

    async def keepalive(frames, every):
        """Yield `frames` through, and a `: keepalive` SSE comment whenever
        `every` seconds pass with nothing to send (issues #3/#36 above).

        The next frame is awaited as a task so the wait can time out
        without cancelling the generator underneath. A cancellation of THIS
        generator (the client hung up) is forwarded to that task and awaited,
        so the inner generators' finally blocks, which stop the engine, run
        before the caller's own cleanup, exactly as they did when the caller
        iterated `frames` directly.
        """
        if every <= 0:
            async for f in frames:
                yield f
            return
        it = frames.__aiter__()
        nxt = None
        try:
            while True:
                if nxt is None:
                    nxt = asyncio.ensure_future(it.__anext__())
                done, _ = await asyncio.wait({nxt}, timeout=every)
                if not done:
                    yield ": keepalive\n\n"
                    continue
                task, nxt = nxt, None
                try:
                    f = task.result()
                except StopAsyncIteration:
                    return
                yield f
        finally:
            if nxt is not None and not nxt.done():
                nxt.cancel()
                try:
                    await nxt
                except BaseException:
                    pass

    async def guarded(gen):
        """Issue #39, the second half: an error raised by the generation
        AFTER the SSE headers went out -- the engine's D-line refusal, a
        first-token or mid-decode timeout, a closed engine socket -- used to
        propagate as an HTTPException into Starlette, which cannot send a
        status it has already sent, and the client got a cut connection and
        the log a `Caught handled exception, but response already started`
        traceback. It is now the last item of the generation, a done record
        with reason `error`, and each serializer writes it in its own wire's
        error shape (OpenAI's `{"error": ...}` data frame, the Responses
        `error` event), then ends the stream. The log gets one line.
        """
        try:
            async for item in gen:
                yield item
        except HTTPException as e:
            print(f"serve_api: error after the stream started: "
                  f"{e.status_code} {e.detail}", flush=True)
            yield None, {"reason": "error", "status": e.status_code,
                         "message": str(e.detail),
                         "code": ("engine_busy" if e.status_code == 503
                                  else None)}

    def error_body(d):
        return {"message": d["message"],
                "type": ERROR_TYPES.get(d["status"], "server_error"),
                "param": None, "code": d.get("code")}

    async def sse(gen, cid, created, chat, split, tstream=None, pre="",
                  parallel=True, include_usage=False):
        def envelope():
            return {"id": cid, "created": created, "model": MODEL_ID,
                    "object": "chat.completion.chunk" if chat
                              else "text_completion"}

        def frame(payload, fin=None):
            f = envelope()
            f["choices"] = [{"index": 0, "finish_reason": fin, **payload}]
            # OpenAI: with include_usage every OTHER chunk carries an explicit
            # null usage, and only the extra final chunk carries the numbers.
            if include_usage:
                f["usage"] = None
            return f

        buf = ""
        # STREAMING MUST RETURN WHAT NON-STREAMING RETURNS.
        #
        # The non-streaming arm finishes through ThinkSplit.parts(), which
        # does `.strip()` on the reasoning and `.lstrip("\n")` on the content;
        # this arm emitted the raw deltas. Measured on the shipped 0.4.0 image
        # across five prompts: the content differed on ALL FIVE (the template
        # closes its thinking block with "</think>\n\n", so every streamed
        # answer carried a two-newline prefix the non-streamed one did not)
        # and the reasoning differed on all five by leading or trailing
        # whitespace. The generated TOKENS were identical every time, so this
        # was never a model difference, only two serializers disagreeing.
        #
        # It mattered beyond cosmetics in one case that changes client logic:
        # on a tool-call turn the non-streamed content is "" and the streamed
        # content was "\n\n", so `if content:` answered differently depending
        # on how the same turn was requested.
        #
        # Content: hold back leading newlines until the first real character.
        # Reasoning: hold back leading whitespace the same way, and hold back
        # a whitespace-only TAIL until something follows it, because in a
        # stream a trailing newline is only trailing once nothing else comes.
        c_started = False
        rd_started = False
        rd_pending = ""
        # an earlier change step 9: front-end CPU per token, excluding the await.
        # PROCESS-WIDE and never reset: 8 concurrent requests share this
        # module-level dict, so a per-request reset would have each stream
        # clobbering the others and reporting noise. The quantity wanted is
        # us/token of front-end CPU, which is concurrency-independent, so a
        # running mean over the process is the right shape.
        async for delta, d in gen:
            _s0 = time.perf_counter()
            if d is not None and d.get("reason") == "error":
                # The shape OpenAI streams an error in: a data frame whose
                # top level is `error`, which the SDKs raise as APIError.
                yield f"data: {json.dumps({'error': error_body(d)})}\n\n"
                yield "data: [DONE]\n\n"
                return
            if d is not None:
                fin = {"stop": "stop", "cancel": "stop",
                       "length": "length"}.get(d["reason"], "stop")
                # A run cut off by max_tokens reports 'length' even if a
                # complete call came first — the client must see that the
                # turn was truncated, not that it ended on a tool call.
                if tstream is not None and tstream.any_calls() \
                        and fin != "length":
                    fin = "tool_calls"
                d["n_reasoning"] = reasoning_tokens(d, chat and split.thinking)
                # #45: `timings` rides the finish chunk whether or not usage
                # was asked for (llama-swap reads the last frame carrying it;
                # an OpenAI client ignores an extra top-level key).
                last = frame({'delta': {}} if chat else {'text': ''}, fin)
                last["timings"] = timings(d)
                yield f"data: {json.dumps(last)}\n\n"
                if include_usage:
                    # Per OpenAI: choices is ALWAYS empty on this chunk, and
                    # it is the last thing before [DONE].
                    u = envelope()
                    u["choices"] = []
                    u["usage"] = usage(d, d.get("n_prompt", 0))
                    u["timings"] = timings(d)
                    yield f"data: {json.dumps(u)}\n\n"
                yield "data: [DONE]\n\n"
                return
            if not chat:
                _ev = f"data: {json.dumps(frame({'text': delta}))}\n\n"
                SSE_PROF["t"] += time.perf_counter() - _s0
                SSE_PROF["n"] += 1
                yield _ev
                continue
            # chat: reasoning goes to its own field until '</think>'
            rd, cd = split.push(delta)
            SSE_PROF["n"] += 1
            SSE_PROF["t"] += time.perf_counter() - _s0
            if rd:
                if not rd_started:
                    rd = rd.lstrip()
                    if rd:
                        rd_started = True
                if rd_started and rd:
                    rd = rd_pending + rd
                    kept = rd.rstrip()
                    rd_pending = rd[len(kept):]
                    rd = kept
                if rd:
                    yield (f"data: "
                           f"{json.dumps(frame({'delta': {'reasoning_content': rd}}))}"
                           f"\n\n")
            if not cd:
                continue
            if not c_started:
                cd = cd.lstrip("\n")
                if not cd:
                    continue
                c_started = True
            if tstream is None:
                yield (f"data: "
                       f"{json.dumps(frame({'delta': {'content': cd}}))}\n\n")
                continue
            # an earlier change: content and tool-call markup share one token stream, so
            # they are separated HERE and never both emitted. Feeding the full
            # accumulated text (not the delta) is what makes a marker split
            # across two tokens safe — same reason ThinkSplit does it.
            buf += cd
            td, evs = tstream.push(pre + buf)
            if td:
                yield (f"data: "
                       f"{json.dumps(frame({'delta': {'content': td}}))}\n\n")
            for e in evs:
                if not parallel and e["index"] > 0:
                    continue
                yield (f"data: "
                       f"{json.dumps(frame({'delta': {'tool_calls': [e]}}))}"
                       f"\n\n")

    async def sse_responses(gen, rid, created, model, split,
                            tstream=None, pre="", parallel=True,
                            max_out=0, want_summary=False):
        """Serialize one generation as a Responses API event stream.

        THE EVENT ORDER WAS VERIFIED AGAINST THE CODEX CLI, not inferred. Text
        goes created -> in_progress -> output_item.added -> content_part.added
        -> output_text.delta* -> output_text.done -> content_part.done ->
        output_item.done -> completed, and a call replaces the two content_part
        events with function_call_arguments.delta* / .done.

        `output_index` is allocated in EMISSION ORDER rather than reserved for
        the message, because a turn that is only a tool call must still put
        that call at index 0. The client keys its own bookkeeping on the index
        and a gap is not a shape it expects.
        """
        seq = itertools.count(0)

        def ev(kind, **body):
            body["type"] = kind
            body["sequence_number"] = next(seq)
            return f"event: {kind}\ndata: {json.dumps(body)}\n\n"

        def envelope(status, output, usage_obj=None):
            return {"id": rid, "object": "response", "created_at": created,
                    "status": status, "model": model, "output": output,
                    "parallel_tool_calls": parallel,
                    "max_output_tokens": max_out or None,
                    "usage": usage_obj}

        base = envelope("in_progress", [])
        yield ev("response.created", response=base)
        yield ev("response.in_progress", response=base)

        nxt = itertools.count(0)          # output_index allocator
        # The non-streaming path finishes with ThinkSplit.parts(), which does
        # `.lstrip("\n")` on the content, and this path emitted the raw
        # deltas: the same prompt came back as "Hello there" from one and
        # "\n\nHello there" from the other. Found by the official SDK, whose
        # typed `output_text` made the difference visible; a client comparing
        # a cached streamed answer against a fresh non-streamed one would see
        # two different strings. Leading newlines are held back until the
        # first real character, so the two paths agree by construction.
        started = False
        msg_id = "msg_" + rid.split("_", 1)[-1]
        msg_open = False                  # the assistant message item
        msg_index = None
        text_out = ""
        buf = ""
        # index in ToolStream's numbering -> our per-call bookkeeping
        calls = {}
        done_items = []

        def open_message():
            nonlocal msg_open, msg_index
            msg_index = next(nxt)
            msg_open = True
            item = {"id": msg_id, "type": "message", "role": "assistant",
                    "status": "in_progress", "content": []}
            return (ev("response.output_item.added", output_index=msg_index,
                       item=item)
                    + ev("response.content_part.added", item_id=msg_id,
                         output_index=msg_index, content_index=0,
                         part={"type": "output_text", "text": "",
                               "annotations": []}))

        def close_message():
            item = {"id": msg_id, "type": "message", "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text_out,
                                 "annotations": []}]}
            done_items.append((msg_index, item))
            return (ev("response.output_text.done", item_id=msg_id,
                       output_index=msg_index, content_index=0, text=text_out)
                    + ev("response.content_part.done", item_id=msg_id,
                         output_index=msg_index, content_index=0,
                         part={"type": "output_text", "text": text_out,
                               "annotations": []})
                    + ev("response.output_item.done", output_index=msg_index,
                         item=item))

        # 0.7.0, public issue #44: THE REASONING IS FORWARDED. It goes out as
        # a `reasoning` output item AHEAD of the message, in the open-model
        # convention (a `reasoning_text` content part, `response.reasoning_
        # text.delta` events, no `encrypted_content`), and, when the request
        # asked for a reasoning summary, the SAME text again as the item's
        # `summary_text` with `response.reasoning_summary_*` events. There is
        # no separate summarizer; the summary is the reasoning. Both are
        # sent because they reach different readers: Codex renders summaries
        # by default and raw content only with `show_raw_agent_reasoning`
        # (its source, codex-rs/protocol legacy_events.rs), and the request
        # recorded from the real CLI asks for `{"summary": "auto"}`; an SDK
        # client reads `content`. The text is trimmed the way the chat route trims
        # `reasoning_content`, so the item's final text equals what the
        # non-streamed body carries.
        rs_id = "rs_" + rid.split("_", 1)[-1]
        rs_open = False
        rs_index = None
        rs_text = ""
        rs_pending = ""
        rs_started = False

        def open_reasoning():
            nonlocal rs_open, rs_index
            rs_index = next(nxt)
            rs_open = True
            item = {"id": rs_id, "type": "reasoning", "status": "in_progress",
                    "summary": [], "content": [], "encrypted_content": None}
            out = ev("response.output_item.added", output_index=rs_index,
                     item=item)
            if want_summary:
                out += ev("response.reasoning_summary_part.added",
                          item_id=rs_id, output_index=rs_index,
                          summary_index=0,
                          part={"type": "summary_text", "text": ""})
            return out

        def reasoning_delta(rd):
            nonlocal rs_text
            rs_text += rd
            out = ev("response.reasoning_text.delta", item_id=rs_id,
                     output_index=rs_index, content_index=0, delta=rd)
            if want_summary:
                out += ev("response.reasoning_summary_text.delta",
                          item_id=rs_id, output_index=rs_index,
                          summary_index=0, delta=rd)
            return out

        def close_reasoning():
            nonlocal rs_open
            rs_open = False
            item = {"id": rs_id, "type": "reasoning", "status": "completed",
                    "summary": ([{"type": "summary_text", "text": rs_text}]
                                if want_summary else []),
                    "content": [{"type": "reasoning_text", "text": rs_text}],
                    "encrypted_content": None}
            done_items.append((rs_index, item))
            out = ev("response.reasoning_text.done", item_id=rs_id,
                     output_index=rs_index, content_index=0, text=rs_text)
            if want_summary:
                out += ev("response.reasoning_summary_text.done",
                          item_id=rs_id, output_index=rs_index,
                          summary_index=0, text=rs_text)
                out += ev("response.reasoning_summary_part.done",
                          item_id=rs_id, output_index=rs_index,
                          summary_index=0,
                          part={"type": "summary_text", "text": rs_text})
            out += ev("response.output_item.done", output_index=rs_index,
                      item=item)
            return out

        async for delta, d in gen:
            if d is not None and d.get("reason") == "error":
                # The Responses wire's own error event; the SDK raises on
                # it and the stream ends without a `completed`.
                yield ev("error", **error_body(d))
                return
            if d is not None:
                if rs_open:
                    yield close_reasoning()
                if msg_open:
                    yield close_message()
                for st in calls.values():
                    if st["closed"]:
                        continue
                    st["closed"] = True
                    item = {"id": st["fc_id"], "type": "function_call",
                            "status": "completed", "name": st["name"],
                            "arguments": st["args"], "call_id": st["call_id"]}
                    done_items.append((st["index"], item))
                    yield ev("response.function_call_arguments.done",
                             item_id=st["fc_id"], output_index=st["index"],
                             arguments=st["args"])
                    yield ev("response.output_item.done",
                             output_index=st["index"], item=item)
                status = "incomplete" if d.get("reason") == "length" \
                    else "completed"
                out = [i for _, i in sorted(done_items, key=lambda x: x[0])]
                d["n_reasoning"] = reasoning_tokens(d, split.thinking)
                final = envelope(status, out, responses_usage(d, d.get("n_prompt", 0)))
                if status == "incomplete":
                    final["incomplete_details"] = {"reason": "max_output_tokens"}
                    # #45: `timings` at the frame's top level, where llama-swap
                    # reads it; not a field of the Responses object
                    yield ev("response.incomplete", response=final, timings=timings(d))
                else:
                    yield ev("response.completed", response=final, timings=timings(d))
                return

            rd, cd = split.push(delta)
            if rd:
                # the chat route's trim: no leading whitespace, trailing
                # whitespace held until more text follows it
                if not rs_started:
                    rd = rd.lstrip()
                    if rd:
                        rs_started = True
                if rs_started and rd:
                    rd = rs_pending + rd
                    kept = rd.rstrip()
                    rs_pending = rd[len(kept):]
                    rd = kept
                if rd:
                    if not rs_open:
                        yield open_reasoning()
                    yield reasoning_delta(rd)
            if not cd:
                continue
            if rs_open:
                yield close_reasoning()
            if not started:
                cd = cd.lstrip("\n")
                if not cd:
                    continue
                started = True
            if tstream is None:
                if not msg_open:
                    yield open_message()
                text_out += cd
                yield ev("response.output_text.delta", item_id=msg_id,
                         output_index=msg_index, content_index=0, delta=cd)
                continue
            buf += cd
            td, evs = tstream.push(pre + buf)
            if td:
                if not msg_open:
                    yield open_message()
                text_out += td
                yield ev("response.output_text.delta", item_id=msg_id,
                         output_index=msg_index, content_index=0, delta=td)
            for e in evs:
                i = e["index"]
                if not parallel and i > 0:
                    continue
                fn = e.get("function") or {}
                if i not in calls:
                    # A call is starting. The message item, if any, is closed
                    # FIRST: output items do not interleave on this wire.
                    if msg_open:
                        yield close_message()
                        msg_open = False
                    idx = next(nxt)
                    calls[i] = {"index": idx, "fc_id": f"fc_{uuid.uuid4().hex[:24]}",
                                "call_id": e.get("id") or "",
                                "name": fn.get("name") or "", "args": "",
                                "closed": False}
                    st = calls[i]
                    yield ev("response.output_item.added", output_index=idx,
                             item={"id": st["fc_id"], "type": "function_call",
                                   "status": "in_progress", "name": st["name"],
                                   "arguments": "", "call_id": st["call_id"]})
                    if not fn.get("arguments"):
                        continue
                st = calls[i]
                frag = fn.get("arguments") or ""
                if not frag:
                    continue
                st["args"] += frag
                yield ev("response.function_call_arguments.delta",
                         item_id=st["fc_id"], output_index=st["index"],
                         delta=frag)

    @app.get("/health")
    async def health():
        rev = {v: k for k, v in DRAFTERS.items()}
        # A HEALTHCHECK THAT ONLY REACHES THIS PROCESS IS NOT A HEALTHCHECK.
        # This endpoint answered "ok" for a server whose engine was stopped
        # dead: the front-end was fine, the engine was not, and nothing here
        # asked it anything. It asks now, and a server that cannot generate
        # returns 503 so an orchestrator can act on it.
        live, took, why = await engine.probe_live()
        body = {"status": "ok" if live else "engine_unresponsive",
                "model": MODEL_ID,
                # THE ROUTES THIS BUILD ACTUALLY SERVES, listed rather than
                # assumed. A client that has to discover by trial which wire a
                # server speaks is the situation /v1/responses was added to
                # end, and a field that names a route it does not serve is the
                # same defect one level up. Generated from the router, so it
                # cannot drift from the code.
                "endpoints": sorted(
                    r.path for r in app.routes
                    if getattr(r, "path", "").startswith("/v1/")),
                # /cache is not under /v1/, so the endpoint list
                # above cannot advertise it and a reporter reading
                # /health has no way to learn it exists. It carries the
                # live prompt-cache counters, including the row copies
                # that a support log needs. Named, not listed, because
                # reading it takes the engine lock and this route
                # deliberately stays cheap.
                "cache_counters": "/cache",
                # 0.8.0: Prometheus, llama-server's metric names, not under
                # /v1 so named here the way /cache is.
                "metrics": "/metrics",
                "context": engine.info.get("ctx") or ctx,
                # Contexts past the native 262,144 run Qwen's static YaRN
                # (HALOGEN_ROPE_YARN); null means the native RoPE.
                "rope_scaling": ({"type": "yarn",
                                  "factor": engine.info.get("rope_factor"),
                                  "original_context": 262144}
                                 if (engine.info.get("rope_factor") or 1.0) > 1.0
                                 else None),
                # Whether this server accepts images, on which routes, and
                # when it does not, WHY. A client cannot otherwise tell an
                # image-capable server from one that will answer its
                # screenshot question from the text alone.
                "vision": vision_status(),
                # The two containers' releases side by side (#26: an api four
                # releases behind its engine answered image questions from
                # nothing, and no surface said which version either one was).
                "version": version_status(),
                # Where the loaded trunk came from, from the engine's INFO:
                # "hgn" is the engine's own checkpoint; "gguf" a llama.cpp
                # file repacked at startup; "gguf-cache" that repack read back
                # from the file HALOGEN_GGUF_CACHE wrote. The MTP head is the
                # engine's own on every one of them.
                "checkpoint_format": engine.info.get("checkpoint_format", "hgn"),
                # Whether the engine's loop is TURNING, asked rather than
                # assumed: PONG on a queued connection is the one signal a
                # listen backlog cannot fake.
                "engine": {"responds": live, "probe_s": took,
                           **({"detail": why} if why else {})},
                "busy": engine.slots.locked(),
                "slots": engine.n_slots,
                "slot_ctx": engine.info.get("slot_ctx") or 0,
                # 0.3: positions resident across all slots; each request
                # reserves its prompt plus max_tokens of them
                "kv_pool_positions": engine.info.get("kv_pool") or 0,
                "in_flight": len(engine.inflight),
                # Seconds the current request has held the batch-1 slot. A
                # number beats a boolean here: a wedge and a long generation
                # are both "busy", and only the duration separates them
                # without going to the GPU counter.
                "busy_for_s": (round(time.monotonic() - engine.busy_since, 1)
                               if engine.busy_since and engine.inflight
                               else 0),
                "queued": engine.waiting,
                "decode": ("greedy at temperature 0 (the default); sampled "
                           "at temperature > 0" if engine.info.get("sampling")
                           else "greedy"),
                # an earlier change: what the LOADED checkpoint carries, straight from
                # the engine's INFO — not a guess, and not config.
                "drafters_available": [
                    n for n in DRAFTERS
                    if n not in DRAFTER_CAP
                    or engine.info.get(DRAFTER_CAP[n])],
                "drafter_default": rev.get(engine.info.get("default", 0),
                                           "serial"),
                # Prompt lookup: chains drafted from the request's own text
                # beside the head, greedy requests only. What the ENGINE
                # runs, not what this process was configured with (the two
                # can be different containers).
                "prompt_lookup": (
                    {"ngram": engine.info.get("pld_n"),
                     "chain": engine.info.get("pld_k"),
                     "applies_to": "greedy requests with the mtp drafter"}
                    if engine.info.get("pld_n") else "off"),
                # False here means every draft step reads the full 248,320-row
                # lm_head instead of the 98,304 shortlist (an earlier change.f).
                "shortlist_draft_head": bool(engine.info.get("draft_head")),
                # The drafter's weights are loaded. In this fork the MTP
                # head IS the drafter, so this is the MTP head's readiness,
                # which is exactly the question the field name asks.
                #
                # It read False on every build ever shipped while the drafter
                # was loaded, default and measurably working, because the
                # engine sent a hardcoded 0: the field is inherited from an
                # engine with a separate draft model and was never populated
                # here. At least one public diagnosis misfired off it, reading
                # it as "the drafter never loads".
                #
                # Kept separate from drafters_available, and from
                # shortlist_draft_head above, which stays False and is a true
                # statement about a feature this fork does not have. The fix
                # for a lying field is not to make an honest neighbour lie the
                # other way.
                "drafter_weights_loaded":
                    bool(engine.info.get("drafter_weights")),
                # The prompt cache. `mode` is reported because it is what
                # decides whether a warm answer is byte-identical to a cold
                # one, and `align` only means anything in mode 1.
                "prompt_cache": {
                    "enabled": bool(engine.info.get("cache_mb")),
                    "cap_mb": engine.info.get("cache_mb", 0),
                    "mode": engine.info.get("cache_mode", 1),
                    # Mode 1 snapshots on a prefill-chunk boundary, so a resume
                    # replays the boundaries a cold run would have used and the
                    # warm answer is bitwise the cold one. Mode 2 snapshots at
                    # every request end and gives that up for a follow-up turn
                    # that costs a second or two at any prompt length. In mode
                    # 2 the alignment is meaningless — the snapshot is wherever
                    # the last request stopped — so it is reported as 0 rather
                    # than as the chunk, which would read as a guarantee.
                    "snapshot_align":
                        engine.info.get("cache_align", 0)
                        if engine.info.get("cache_mode", 1) == 1 else 0,
                    # Read from the mode the engine reports, which is the
                    # only thing that decides it. Deriving this from anything
                    # else (a hardcoded limit, or "is the prefill chunked")
                    # gets it wrong in one direction or the other, and a field
                    # that lies about determinism is worse than no field.
                    "bitwise_identical_to_cold":
                        bool(engine.info.get("cache_mb"))
                        and engine.info.get("cache_mode", 1) == 1,
                },
                # an earlier change. The wire format is Qwen's XML-in-XML, NOT the
                # JSON-in-<tool_call> shape the name usually implies, and the
                # parser needs each tool's JSON Schema to type its arguments
                # so a request that omits `tools` gets
                # best-effort types, which is why that is said here.
                "tool_calls": {
                    "wire_format": "qwen-xml (<function=>/<parameter=>)",
                    "streaming": True,
                    "tool_choice": ["auto", "none", "required",
                                    "{type: function, function: {name}}"],
                    # A forced call is a prompt prefill, not a grammar, so
                    # 'required' is only EXACT when the prefill can name the
                    # function: a named tool_choice, or 'required' with
                    # exactly one tool. With several the model can still
                    # close an empty <function=>, and that returns 502.
                    "forced_call_is_a_prefill": True,
                    "required_is_exact": "named tool_choice, or one tool",
                    "parallel_tool_calls": True,
                    # Nothing constrains the grammar. A malformed call is
                    # returned as content, never repaired into a call the
                    # model did not make.
                    "constrained_decoding": False,
                    # A forced call is a prompt prefill and takes the slot
                    # the reasoning block would have used.
                    "forced_call_disables_thinking": True,
                    "argument_types_need_schema": True,
                },
                "supported": ["reasoning_effort", "enable_thinking",
                              "preserve_thinking", "tools", "tool_choice",
                              "parallel_tool_calls", "stop",
                              "max_tokens", "max_completion_tokens",
                              "max_output_tokens", "stream", "stream_options"],
                # All three name one budget over reasoning + content. Listed
                # separately because a client reads this to find out which
                # spelling it may use, and the modern ones used to be dropped
                # in silence.
                "token_budget_aliases": ["max_tokens", "max_completion_tokens",
                                         "max_output_tokens"],
                "max_tokens_default": ChatReq.model_fields["max_tokens"].default,
                # PUBLIC ISSUES #21 AND #24: WHAT THAT BUDGET IS SPENT
                # ON. The budget above covers reasoning AND content, and with
                # no `reasoning_effort` sent the chat template's own default is
                # `xhigh`, which on a long agentic prompt can consume the whole
                # 8192 before the model closes its thinking block. The caller
                # then gets content "" with everything in reasoning_content and
                # `finish_reason: "length"`, which at least one agent harness
                # reports as "the model returned no assistant message" and
                # retries, deterministically, forever.
                #
                # The default was discoverable nowhere. A client could read
                # `max_tokens_default` here and still have no way to learn that
                # the budget it sets is being spent on reasoning it never asked
                # for. These three fields say it.
                "reasoning_effort_default": DEFAULTS["reasoning_effort"]
                                            or "xhigh",
                "reasoning_effort_values": list(EFFORT_VALUES),
                "token_budget_covers_reasoning": True,
                # 0.8.1 (#56): the thinking budget's server default, None =
                # no budget; a request's max_thinking_tokens wins.
                "max_thinking_tokens_default": DEFAULTS["max_thinking_tokens"],
                # PUBLIC ISSUE #30: what the operator set as the default for
                # a field a request leaves out, keyed by the variable, so a
                # client can see the server's policy rather than infer it
                # from an answer. Empty when nothing is set. The rule is
                # stated beside it because "temperature absent" now means
                # two different things on two differently configured
                # servers, and only this field tells them apart.
                "server_defaults": defaults_summary(),
                "server_defaults_rule": (
                    "a field the request sends always wins; a default fills "
                    "only a field the request omits. A request that sends "
                    "temperature 0 decodes greedy and takes no sampling "
                    "default"),
                # Both spellings reach the same three controls; see
                # TEMPLATE_KWARGS. An unsupported key in the nested form is a
                # 400, never a silent drop.
                "chat_template_kwargs": list(TEMPLATE_KWARGS),
                # an earlier change: errors use OpenAI's {"error": {...}} envelope, not
                # FastAPI's {"detail": ...}, and a malformed body is a 400
                # rather than a 422.
                "error_format": "openai",
                "accepted_but_ignored": list(SAMPLING_FIELDS),
                # Refused with a 400, not ignored. Listed so a client
                # can check before sending rather than discovering it from a
                # 200 that did not do what was asked.
                "not_implemented": (
                    [] if engine.info.get("grammar")
                    else ["response_format", "text.format"]),
                # 0.8.0: STRUCTURED OUTPUT, keyed on the engine's INFO
                # field and on the vocabulary handshake, never assumed. The
                # keyword lists are the engine's contract (src/grammar.h):
                # enforced, accepted-but-not-enforced, refused-by-name.
                "structured_output": {
                    "enabled": bool(engine.info.get("grammar")),
                    "reason": engine.info.get("grammar_error") if not engine.info.get("grammar") else None,
                    "response_format": ["json_schema", "json_object"],
                    "text.format": ["json_schema", "json_object"],
                    "routes": ["/v1/chat/completions", "/v1/completions", "/v1/responses"],
                    "sampling": "greedy only (temperature 0); sampled + schema is a 400",
                    "images": "a schema with images is a 400",
                    "thinking": "the reasoning block is unconstrained; the "
                                "final text is constrained after </think>",
                    "tools": "a request with tools may open a <tool_call> "
                             "instead of the JSON (the schema binds the "
                             "final text, not a call)",
                    "keys": "object keys are emitted in schema order",
                    "enforced": ["type", "properties", "required",
                                 "additionalProperties", "items", "minItems",
                                 "maxItems", "minLength", "maxLength", "enum",
                                 "const", "anyOf", "oneOf (as anyOf)",
                                 "type: [..., \"null\"]", "$ref", "$defs"],
                    "accepted_not_enforced": ["minimum", "maximum",
                                              "exclusiveMinimum",
                                              "exclusiveMaximum", "multipleOf",
                                              "title", "description",
                                              "default", "examples"],
                    "refused": ["pattern", "format", "allOf", "not", "if",
                                "then", "else", "patternProperties",
                                "dependentRequired", "dependentSchemas",
                                "uniqueItems", "contains", "propertyNames",
                                "unevaluatedProperties", "unevaluatedItems",
                                "minProperties", "maxProperties",
                                "prefixItems"],
                },
                # an earlier change. Stated explicitly because "temperature works now"
                # and "temperature works at the same speed" are different
                # claims, and a client cannot tell them apart from a response.
                # WHAT THE ENGINE SAID, not what this file assumes: the
                # block that sat here before listed nine sampling parameters
                # for an engine that had none, so a client could read
                # /health, send temperature 0.7, and get a 400. The engine's
                # INFO line now carries a sampling field and this is keyed
                # on it.
                "sampling": ({
                    "implemented": ["temperature", "top_p", "top_k", "min_p",
                                    "seed", "presence_penalty",
                                    "frequency_penalty", "logit_bias",
                                    "logprobs"],
                    "decode": (
                        ("temperature absent samples at the server's "
                         "default of %g (HALOGEN_TEMPERATURE); temperature "
                         "0 is greedy: " % DEFAULTS["temperature"])
                        if DEFAULTS["temperature"] else
                        "temperature 0 (the default) is greedy: ")
                        + "a forward pass and an argmax, byte-identical to "
                          "serial greedy decode. temperature > 0 samples "
                          "from the filtered target on the same drafter "
                          "the request would otherwise get; with the MTP "
                          "drafter the speculative accept/reject rule "
                          "emits exactly the target distribution",
                    "filters": "top_k, top_p and min_p compose as an "
                               "intersection (the HF/vLLM convention); "
                               "top_p >= 1 and top_k = 0 disable a filter",
                    "seed": "reproduces a request on the same drafter; a "
                            "sampled MTP run and a sampled serial run agree "
                            "in distribution, not token for token",
                    "penalties": "presence and frequency count GENERATED "
                                 "tokens only; applied on both drafters",
                    "not_implemented": ["top_logprobs", "logprobs with "
                                        "stream=true", "n > 1"],
                } if engine.info.get("sampling") else {
                    "implemented": [],
                    "decode": "greedy only: a forward pass and an argmax",
                    "temperature_gt_0": "NOT IMPLEMENTED by this engine "
                                        "build; rejected with 400 rather "
                                        "than silently served greedy",
                }),
                "max_tokens_cap": max_cap,
                # an earlier change: a client cannot distinguish a cap that REJECTS from
                # one that silently truncates by looking at a response — both
                # end with finish_reason "length". Say which this is.
                "max_tokens_over_cap": "400",
                "reasoning_effort_values": sorted(set(EFFORT_VALUES))}
        if live:
            return body
        # A STATUS CODE, because that is what a container healthcheck reads.
        # A field saying the engine is gone inside a 200 is a field nobody
        # acts on, which is how a wedged server stayed green for hours.
        return JSONResponse(status_code=503, content=body)

    @app.get("/metrics")
    async def metrics():
        """0.8.0: Prometheus text exposition in llama-server's names (its
        `--metrics` endpoint), so a llama.cpp dashboard reads this server
        unchanged, plus `halogen:` counters for what llama-server has no
        name for (the cache's share, the drafters' accept count, structured
        requests). Cheap: no engine round trip; the totals are the D lines
        already summed and the gauges are this process's own counters."""
        from fastapi.responses import PlainTextResponse
        kv_pool = int(engine.info.get("kv_pool") or 0) or (
            int(engine.info.get("kv_slots", 1)) * int(engine.info.get("slot_ctx") or 0))
        body = engine.metrics.render(len(engine.inflight), engine.waiting,
                                     sum(engine.reserved.values()), kv_pool)
        return PlainTextResponse(body, media_type="text/plain; version=0.0.4; charset=utf-8")

    @app.get("/cache")
    async def cache():
        """Live prompt-cache counters. Separate from /health so a
        poll for hit rate does not have to take the engine lock on every
        health check — this one does, and /health stays free."""
        st = await engine.cache_stats()
        if st is None:
            raise HTTPException(501, "this engine reports no prompt cache")
        total = st["hits"] + st["misses"]
        st["hit_rate"] = round(st["hits"] / total, 4) if total else None
        return st

    @app.get("/v1/models")
    async def models():
        return {"object": "list",
                "data": [{"id": MODEL_ID, "object": "model",
                          "owned_by": "halogen", "created": 0}]}

    def render_prompt(msgs, kw):
        """-> (ids, snap): the tokenized chat prompt and the STABLE PREFIX
        length to hand the engine's prompt cache as SNAP.

        The prompt ends in the template's assistant opener
        (`<|im_start|>assistant\n<think>\n`). The next turn re-renders that
        assistant message from the client's history, and when the client
        omits `reasoning_content` (OpenAI-shaped clients do) the template
        writes `<think>\n\n</think>` there, so the token after `<think>`
        differs from the one the engine cached and the whole prefix is lost.
        Every turn then re-prefilled cold: 21-25 s on a 20k prompt where a
        hit is ~1 s. The stable prefix is everything BEFORE the opener, i.e.
        the history rendered without the generation prompt, and the engine
        snapshots there instead.

        One tokenization, not two: the rendered text is tokenized with
        offsets, and the split is the token boundary at the end of the
        history text. That boundary is exact because the opener starts with
        a special token, which the tokenizer never merges across. If no
        token ends exactly there (a template that opens differently), snap
        is 0 and the engine keeps its old position rather than guessing.
        """
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True, **kw)
        stable = tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=False, **kw)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(enc["input_ids"])

        def boundary_at(cut):
            """The token count whose text ends exactly at `cut`, else 0."""
            if not (0 < cut < len(text)):
                return 0
            for i, (a, b) in enumerate(enc["offset_mapping"]):
                if b == cut:
                    return i + 1
                if b > cut:
                    break
            return 0
        snap = boundary_at(len(stable)) if text.startswith(stable) else 0
        # The SECOND point: the end of the system block. Requests that share
        # a system prompt and ask different things cannot use the history
        # point (their user turns differ); they resume from this one. The
        # template will not render a system message on its own ("No user
        # query found"), so the boundary is where the FIRST user turn opens
        # in the full render: the opener is a special token, which the
        # tokenizer never merges across, so the cut is a token boundary.
        # Only with a system message, and only when shorter than the
        # history point.
        snap2 = 0
        if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
            cut = text.find("<|im_start|>user")
            if cut > 0:
                snap2 = boundary_at(cut)
            if snap and snap2 >= snap:
                snap2 = 0
        return ids, snap, snap2

    def version_status():
        """This front-end's release beside the engine's, and whether they agree.

        `match` is None when either side is unknown (an engine older than
        the INFO field, or a front-end run outside the image), False when
        both are known and differ. A False here is the whole of public issue
        #26: the front-end that renders the prompt and the engine that runs
        it were built four releases apart, and the image path that one had
        the other did not.
        """
        eng = engine.info.get("version") or "unknown"
        known = API_VERSION != "unknown" and eng != "unknown"
        return {"api": API_VERSION, "engine": eng,
                "match": (API_VERSION == eng) if known else None}

    def vision_status():
        """What this server can do with an image, and why not when it cannot.

        THREE things must all hold and only ONE of them is knowable here, so
        the engine's half is READ from its INFO line rather than inferred: a
        /health that advertises a capability the engine does not have is a
        field that lies, and this project has shipped four of those.
        """
        why = []
        if not engine.info.get("vision"):
            why.append("the engine was started without a vision tower "
                       "(HALOGEN_VISION_TOWER)")
        if IMAGE_TOKEN_ID is None:
            why.append("this tokenizer has no image placeholder token")
        if not pillow_ready():
            why.append("Pillow is not installed for the front-end")
        st = {"enabled": not why,
              # The routes that actually reach the image path. The vision
              # smoke asserts each one accepts an image, so this list is
              # PROVED rather than declared: a route named here and not
              # wired fails the gate.
              "endpoints": ["/v1/chat/completions", "/v1/responses"],
              "input": "data: URL or bare base64 in an image content part; "
                       "http(s) URLs are refused",
              "max_pixels": VIS_MAX_PIXELS,
              "size_multiple": VIS_UNIT}
        if why:
            st["disabled_because"] = why
        return st

    def collect_images(msgs):
        """Every image in the conversation, REFUSED HERE if it cannot be served.

        The engine refuses an image it has no tower for, but its refusal
        reaches the client as a generic rejection that blames the prompt --
        the misdirection this endpoint has produced before. With the
        capability on the INFO line the front-end can say the true reason
        before queueing.
        """
        try:
            imgs = vis_collect(msgs)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if imgs:
            st = vision_status()
            if not st["enabled"]:
                raise HTTPException(400, "this server does not accept images: "
                                    + "; ".join(st["disabled_because"]))
        return imgs

    def render_with_images(msgs, kw, imgs):
        """-> (ids, snap, snap2, wire_imgs). ONE renderer for BOTH wires.

        /v1/responses shipped without this and dropped every image, because
        the expansion lived inline in the chat handler and the second wire
        was written against the message list alone. Two copies of a path is
        one copy that gets forgotten.
        """
        try:
            ids, snap, snap2 = render_prompt(msgs, kw)
        except HTTPException:
            raise
        except Exception as e:                      # template raise_exception
            raise HTTPException(400, f"chat template rejected the request: "
                                     f"{e}")
        # Issue #39: a mentioned placeholder becomes text BEFORE the pads
        # are counted against the images, so a mention beside a real image
        # does not read as an extra image.
        ids, snap, snap2, _ = textify_pad_mentions(ids, snap, snap2, len(imgs))
        if not imgs:
            return ids, snap, snap2, None
        ids, snap, snap2, placed = expand_image_pads(ids, snap, snap2, imgs)
        return ids, snap, snap2, [(f, h, w, imgs[i][2])
                                  for i, (f, h, w) in enumerate(placed)]

    async def serve(ids, max_tokens, stops, stream, chat, prefix,
                    thinking=True, drafter=None, tools=None, parallel=True,
                    sample=None, penalty="",
                    pre="", forced=False, include_usage=False, snap=0,
                    snap2=0, wire="chat", images=None, want_summary=False,
                    schema=None, think_budget=None):
        # 0.8.0: the schema rides the GEN line; the engine constrains
        # only after the thinking block (the </think> token, when the request
        # thinks) and lets a request with tools open a tool call instead of
        # the JSON (the <tool_call> token at the very start), because the
        # schema binds the final text output and a call is not text.
        # (/v1/completions is a raw prompt: nothing says whether it opened a
        # think block, so a schema there constrains from the first token.)
        after = THINK_END_ID if (schema and thinking and chat) else None
        escape = [TOOL_CALL_ID] if (schema and tools and TOOL_CALL_ID is not None) else ()
        # 0.8.1 (issue #56): the thinking budget rides the GEN line only when
        # the prompt opened a think block (the template's state, which the
        # engine cannot see) and the marker is one token. The close is
        # Qwen's own budget sentence: the model is told why the block ends
        # rather than cut mid-word, and `</think>\n\n` is what the template
        # itself writes after the block, so the answer starts where it
        # always does.
        think = None
        if think_budget and thinking and chat and THINK_END_ID is not None:
            think = (int(think_budget), THINK_END_ID, THINK_CLOSE_IDS)
        # batch-1: QUEUE rather than reject. Reasoning defaults to xhigh, so
        # one request routinely runs minutes at ~10 t/s; failing every other
        # caller instantly for that whole window made the endpoint look dead
        # to a second client. Waiting is slow but correct; only a wait past
        # the timeout is a real 503.
        # VALIDATE BEFORE QUEUEING. Everything below the lock can wait
        # queue_timeout seconds (600 by default) for a slot; telling a client
        # its request was malformed only after ten minutes of waiting is
        # strictly worse than telling it now, and none of these checks need
        # the engine.
        #
        # an earlier change: this used to CLAMP — `max_tokens = min(max_tokens, max_cap)`
        # — and a benchmark run lost its results to it. Asking for 16,384 got
        # exactly 4,096 back with no error and no field saying the server had
        # reduced the request, so the output looked like the MODEL stopping
        # rather than the SERVER truncating. `finish_reason: "length"` is the
        # same value in both cases, which is precisely why it could not be
        # diagnosed from the response. Three lines up in run(), an over-length
        # PROMPT already raises a 400 that names the number; there was never a
        # reason for the output limit to behave differently, and an earlier change's
        # rule applies — an error the client cannot read is not an error.
        want = int(max_tokens)
        if want > max_cap:
            raise HTTPException(
                400, f"max_tokens {want} exceeds this server's cap of "
                     f"{max_cap}. The cap is server policy, not a model "
                     f"limit: halogen is batch-1, so one long request holds "
                     f"the GPU for its whole run and every other client "
                     f"queues behind it. Raise it with --max-tokens-cap; "
                     f"/health reports the live value as max_tokens_cap.")
        # The other ceiling, and the one that is physics rather than policy.
        # an earlier change's lesson, applied to the output side: ask the ENGINE for the
        # context, never a CLI default, because --context and Model::kMaxCtx
        # are two copies of one number and they have already drifted once.
        limit = engine.info.get("ctx") or ctx
        room = limit - len(ids)
        if want > room:
            # Default: a refusal that names the numbers — the error the
            # client CAN read. With HALOGEN_FIT_TO_ROOM=1 (the interactive
            # unit only; see FIT_TO_ROOM) the budget is instead clamped to
            # the room the prompt actually left, so a long session runs to
            # the true edge of the window. The clamped value reaches the
            # response as max_tokens / max_output_tokens, and is echoed on
            # stderr so the operator can watch the tail run thin. Below the
            # floor the prompt has taken the window and there is no honest
            # answer left to clamp to, so the refusal stays even when gated on.
            if FIT_TO_ROOM and room >= FIT_TO_ROOM_FLOOR:
                print(f"serve_api: fit-to-room clamped max_tokens {want} -> "
                      f"{room} (prompt={len(ids)} ctx={limit})",
                      file=sys.stderr, flush=True)
                want = room
            else:
                raise HTTPException(
                    400, f"max_tokens {want} does not fit: prompt is {len(ids)} "
                         f"tokens and the context is {limit}, leaving room for "
                         f"{room}."
                         + ("" if not FIT_TO_ROOM else
                            f" (fit-to-room is on, but {room} is below the "
                            f"floor of {FIT_TO_ROOM_FLOOR}: the prompt has "
                            f"taken the window)"))
        max_tokens = max(1, want)

        # an earlier change: a SEMAPHORE of engine slots, not a single lock. At one slot
        # this is the batch-1 behaviour verbatim -- queue rather than reject,
        # because reasoning defaults to xhigh and failing every other caller
        # for that window made the endpoint look dead.
        engine.waiting += 1
        try:
            await asyncio.wait_for(engine.slots.acquire(),
                                   timeout=queue_timeout)
        except asyncio.TimeoutError:
            raise EngineBusy()
        finally:
            engine.waiting -= 1
        # WHEN this slot was taken, keyed by request. /health reports the
        # OLDEST live hold: "busy" alone cannot distinguish a long xhigh
        # request from a wedge, and that ambiguity is what made the
        # 2026-08-24 lockup look like normal load until the GPU counter was
        # checked. With N slots a single scalar could not express it at all.
        slot_key = uuid.uuid4().hex
        engine.inflight[slot_key] = time.monotonic()
        engine.reserved[slot_key] = len(ids) + int(max_tokens)   # what the engine's pool holds for it
        engine.busy_since = min(engine.inflight.values())
        # Chat ids are `chatcmpl-<hex>` and Responses ids are `resp_<hex>`:
        # the separator differs between the two APIs and clients do match on
        # the prefix.
        cid = (f"resp_{uuid.uuid4().hex[:24]}" if wire == "responses"
               else f"{prefix}-{uuid.uuid4().hex[:24]}")
        created = int(time.time())
        split = ThinkSplit(thinking)
        tstream = ToolStream(tools) if tools else None
        if stream:
            n_ev = 0
            n_ka = 0
            finished = False
            t_body = time.perf_counter()
            # Issue #3's instrument: the largest gap between two DATA frames
            # and when the last one went out, so a hangup's line says
            # whether the stream had gone quiet and for how long.
            t_last = t_body
            gap_max = 0.0

            async def body():
                nonlocal n_ev, n_ka, finished, t_last, gap_max
                try:
                    gen = guarded(run(ids, max_tokens, stops, drafter, sample,
                                      penalty, snap, snap2, images, schema,
                                      after, escape, think))
                    # The Responses wire is a different SERIALIZATION of the
                    # same generation. Everything that matters for safety --
                    # the slot semaphore, the abort on hangup, the inflight
                    # bookkeeping -- is shared by construction rather than
                    # duplicated, which is why this branches here and not in
                    # a parallel endpoint of its own.
                    stream_iter = (
                        sse_responses(gen, cid, created, MODEL_ID, split,
                                      tstream, pre, parallel, max_tokens,
                                      want_summary)
                        if wire == "responses" else
                        sse(gen, cid, created, chat, split, tstream, pre,
                            parallel, include_usage))
                    async for ev in keepalive(stream_iter, SSE_KEEPALIVE_S):
                        if ev.startswith(":"):
                            n_ka += 1
                            yield ev
                            continue
                        now = time.perf_counter()
                        gap_max = max(gap_max, now - t_last)
                        t_last = now
                        n_ev += 1
                        yield ev
                    finished = True
                finally:
                    # PUBLIC ISSUE #3: A HANGUP AND A CLEAN FINISH USED
                    # TO LOOK IDENTICAL IN THE LOG.
                    #
                    # uvicorn logs `200 OK` either way, because the status went
                    # out with the headers long before, and the `serve_api:`
                    # summary never prints on a hangup because it is emitted
                    # when the generator yields its completion record and the
                    # generator was cancelled before reaching it. So the whole
                    # server-side trace of a client that gave up mid-stream was
                    # a bare access line indistinguishable from success.
                    #
                    # That cost a public thread several rounds: a reporter
                    # inferred an engine stall from the ABSENCE of our timeout
                    # message, which is inference from silence, and the silence
                    # was ours. A disconnect is a normal event, not an error,
                    # so this is one line rather than a traceback -- but it has
                    # to be a line.
                    if not finished:
                        now = time.perf_counter()
                        print("serve_api: client disconnected mid-stream, %s, "
                              "%d SSE frames and %d chars sent in %.1fs, "
                              "largest gap between frames %.1fs, last frame "
                              "%.1fs before the hangup, %d keepalives "
                              "(the engine is being stopped; this is the "
                              "client hanging up, not a server error)"
                              % (cid, n_ev, len(split.full), now - t_body,
                                 max(gap_max, now - t_last), now - t_last,
                                 n_ka),
                              flush=True)
                    # client hung up (or errored) mid-stream: stop the engine
                    # instead of letting it decode to max_tokens with nobody
                    # listening, then hand the slot to whoever is queued.
                    #
                    # THE RELEASE IS IN ITS OWN finally AND ABORT IS CAPPED.
                    # It used to read `await engine.abort(); lock.release()`,
                    # so anything that made abort() hang or raise skipped the
                    # release entirely and the batch-1 lock was held forever.
                    # Releasing the slot is the part other clients depend on;
                    # tidying the stream is best-effort and must never be able
                    # to prevent it.
                    try:
                        await asyncio.wait_for(engine.abort(),
                                               timeout=ABORT_DRAIN_S + 5)
                    except Exception:
                        await engine.close()
                    finally:
                        engine.inflight.pop(slot_key, None); engine.reserved.pop(slot_key, None)
                        engine.busy_since = (min(engine.inflight.values())
                                             if engine.inflight else None)
                        engine.slots.release()
            return StreamingResponse(body(), media_type="text/event-stream")
        try:
            text, done = "", None
            async for delta, d in run(ids, max_tokens, stops, drafter,
                                      sample, penalty, snap, snap2, images,
                                      schema, after, escape, think):
                if d is not None:
                    done = d
                    break
                text += delta
        finally:
            # Same shape as the streaming arm above, and for the same reason.
            try:
                await asyncio.wait_for(engine.abort(),
                                       timeout=ABORT_DRAIN_S + 5)
            except Exception:
                await engine.close()
            finally:
                engine.inflight.pop(slot_key, None); engine.reserved.pop(slot_key, None)
                engine.busy_since = (min(engine.inflight.values())
                                     if engine.inflight else None)
                engine.slots.release()
        fin = {"stop": "stop", "cancel": "stop",
               "length": "length"}.get(done["reason"], "stop")
        if wire == "responses":
            split.full = text
            reasoning, content = split.parts()
            tool_calls = []
            if tools:
                content, tool_calls, _ = split_tool_calls(pre + content, tools)
                if not parallel:
                    tool_calls = tool_calls[:1]
            out = []
            if reasoning:
                # #44: the reasoning item first, the same shape the stream
                # closes with (see sse_responses)
                out.append({"id": "rs_" + cid.split("_", 1)[-1],
                            "type": "reasoning", "status": "completed",
                            "summary": ([{"type": "summary_text",
                                          "text": reasoning}]
                                        if want_summary else []),
                            "content": [{"type": "reasoning_text",
                                         "text": reasoning}],
                            "encrypted_content": None})
            if content:
                out.append({"id": "msg_" + cid.split("_", 1)[-1],
                            "type": "message", "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text",
                                         "text": content,
                                         "annotations": []}]})
            for tc in tool_calls:
                out.append({"id": f"fc_{uuid.uuid4().hex[:24]}",
                            "type": "function_call", "status": "completed",
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                            "call_id": tc["id"]})
            done["n_reasoning"] = reasoning_tokens(done, split.thinking)
            body = {"id": cid, "object": "response", "created_at": created,
                    "status": "incomplete" if fin == "length" else "completed",
                    "model": MODEL_ID, "output": out,
                    "parallel_tool_calls": parallel,
                    "max_output_tokens": max_tokens,
                    "usage": responses_usage(done, len(ids)),
                    "timings": timings(done)}
            if fin == "length":
                body["incomplete_details"] = {"reason": "max_output_tokens"}
            return body
        if chat:
            split.full = text
            reasoning, content = split.parts()
            tool_calls = []
            if tools:
                content, tool_calls, _ = split_tool_calls(pre + content, tools)
                if not parallel:
                    tool_calls = tool_calls[:1]
            msg = {"role": "assistant", "content": content}
            if reasoning:
                msg["reasoning_content"] = reasoning
            if forced and not tool_calls:
                # tool_choice asked for a call and the model declined by
                # closing an empty <function=>. Returning its markup as
                # content would look like an answer; this is a failure.
                raise HTTPException(502, "tool_choice required a call and the "
                                         "model did not name a function: "
                                         "retry, or name the function in "
                                         "tool_choice")
            if tool_calls:
                msg["tool_calls"] = tool_calls
                if fin != "length":
                    fin = "tool_calls"
            choice = {"index": 0, "finish_reason": fin, "message": msg}
        else:
            choice = {"index": 0, "finish_reason": fin, "text": text}
        # an earlier change. OpenAI's shape, minus top_logprobs (rejected at the door, not
        # returned empty) and minus `bytes`, which would need a per-token
        # detokenisation this loop does not keep -- run() decodes the whole
        # list each step and diffs, so individual token text is not retained.
        # Reporting the fields we do not have as null beats inventing them.
        if done and done.get("logprobs"):
            choice["logprobs"] = {"content": [
                {"token": None, "logprob": v, "bytes": None,
                 "top_logprobs": []} for v in done["logprobs"]]}
        done["n_reasoning"] = reasoning_tokens(done, chat and split.thinking)
        return {"id": cid, "created": created, "model": MODEL_ID,
                "object": "chat.completion" if chat else "text_completion",
                "choices": [choice], "usage": usage(done, len(ids)),
                "timings": timings(done)}

    @app.post("/v1/completions")
    async def completions(req: CompletionReq):
        ids = tok(req.prompt, add_special_tokens=False)["input_ids"]
        # No image path on this route, so every placeholder is a mention.
        ids, _, _, _ = textify_pad_mentions(list(ids))
        check_sampling(req)
        return await serve(ids, req.max_tokens, stop_list(req.stop),
                           req.stream, False, "cmpl",
                           drafter=drafter_for(req),
                           sample=sample_spec(req),
                           penalty=penalty_spec(req),
                           include_usage=wants_usage(req),
                           schema=schema_for(req))

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatReq):
        # FIRST, before the template and the tokenizer. This call was missing
        # entirely: /v1/completions had it and chat did not, so on the endpoint
        # everyone actually uses, presence_penalty / frequency_penalty /
        # logit_bias / logprobs without temperature>0 were SILENTLY DROPPED
        # (penalty_spec returns "" when there is no sampling, which puts the
        # engine's own rejection out of reach), and logprobs+stream and
        # top_logprobs were never refused. /health advertised the opposite.
        # an earlier change's rule, on the primary endpoint: a request not served the way
        # it was asked must say so.
        check_sampling(req)
        for f in SAMPLING_FIELDS:
            if getattr(req, f) is not None and f not in _warned_sampling:
                _warned_sampling.add(f)
                print(f"serve_api: '{f}' accepted but IGNORED "
                      f"by this endpoint", flush=True)
        mode, forced = tool_choice_mode(req)
        pre = force_prefill(mode, forced, req.tools)
        thinking = thinking_on(req)
        kw = chat_kwargs(req, mode)
        if pre:
            # The prefill lands where the reasoning block would start, so the
            # think block is pre-closed instead of left dangling — otherwise
            # '</think>' never arrives and the whole turn misfiles as
            # reasoning. A forced call trades thinking for the guarantee.
            kw["enable_thinking"] = False
            thinking = False
        try:
            # an earlier change F3: OpenAI sends `arguments` as a JSON STRING and this
            # template indexes it as a mapping. Without this the second turn
            # of every tool loop is a 400.
            msgs = normalize_messages(req.messages)
        except ValueError as e:
            raise HTTPException(400, str(e))
        imgs = collect_images(msgs)
        ids, snap, snap2, wire_imgs = render_with_images(msgs, kw, imgs)
        if pre:
            ids += tok(pre, add_special_tokens=False)["input_ids"]
        return await serve(ids, req.max_tokens, stop_list(req.stop),
                           req.stream, True, "chatcmpl",
                           thinking=thinking,
                           think_budget=server_default(req, "max_thinking_tokens"),
                           drafter=drafter_for(req),
                           sample=sample_spec(req),
                           penalty=penalty_spec(req),
                           tools=(req.tools if mode != "none" else None),
                           parallel=req.parallel_tool_calls is not False,
                           pre=pre, forced=bool(pre),
                           include_usage=wants_usage(req), snap=snap,
                           snap2=snap2, images=wire_imgs,
                           schema=schema_for(req))

    @app.post("/v1/responses")
    async def responses(req: ResponsesReq):
        """OpenAI Responses API, for clients that dropped Chat Completions.

        The Codex CLI is the reason this exists: it speaks only this wire, so
        without it halogen is not a backend it can use at all. The request and
        the event stream were both established by recording a real `codex exec`
        session rather than from the specification.

        This is a TRANSLATION, not a second engine path. The turns become the
        same message list `/v1/chat/completions` renders, generation is the
        same call, and only the serialization differs.
        """
        chat_tools, dropped = responses_tools(req.tools)
        if dropped:
            key = ",".join(sorted(set(dropped)))
            if key not in _warned_sampling:
                _warned_sampling.add(key)
                print(f"serve_api: /v1/responses ignored {len(dropped)} "
                      f"non-function tool(s) of type: {key}", flush=True)
        # tool_choice arrives in the Responses spelling ({"type":"function",
        # "name":...}); chat nests the name one level deeper.
        tc = req.tool_choice
        if isinstance(tc, dict) and tc.get("type") == "function" \
                and "function" not in tc:
            tc = {"type": "function", "function": {"name": tc.get("name")}}
        effort = (req.reasoning or {}).get("effort")
        shim = ChatReq(model=req.model, messages=[{"role": "user",
                                                   "content": ""}],
                       max_tokens=req.max_output_tokens or CHAT_MAX_TOKENS,
                       stream=req.stream, tools=chat_tools, tool_choice=tc,
                       parallel_tool_calls=req.parallel_tool_calls,
                       reasoning_effort=effort, drafter=req.drafter,
                       # 0.8.1 (#56): the Responses wire has no field for a
                       # thinking budget; a top-level `max_thinking_tokens`
                       # is accepted (the model allows extra fields) and the
                       # server default applies otherwise.
                       max_thinking_tokens=getattr(req, "max_thinking_tokens", None),
                       temperature=req.temperature, top_p=req.top_p,
                       # Issue #14: the Responses spelling of
                       # response_format is `text.format`. Carried onto the
                       # shim so ONE check refuses both wires: a second inline
                       # copy of a path is how /v1/responses once ended up
                       # answering about images it never saw.
                       response_format=(req.text or {}).get("format"))
        check_sampling(shim)
        mode, forced = tool_choice_mode(shim)
        pre = force_prefill(mode, forced, chat_tools)
        # The Responses wire has no enable_thinking field, so the server
        # default (HALOGEN_ENABLE_THINKING) or `reasoning: {effort: "none"}`
        # (#50, mapped onto the shim's reasoning_effort) turns thinking off.
        thinking = thinking_on(shim)
        kw = chat_kwargs(shim, mode)
        if pre:
            kw["enable_thinking"] = False
            thinking = False
        try:
            msgs = normalize_messages(
                responses_messages(req.instructions, req.input))
        except ValueError as e:
            raise HTTPException(400, str(e))
        if not msgs or all(m.get("role") == "system" for m in msgs):
            raise HTTPException(400, "input is empty: /v1/responses needs at "
                                     "least one non-system turn")
        imgs = collect_images(msgs)
        ids, snap, snap2, wire_imgs = render_with_images(msgs, kw, imgs)
        if pre:
            ids += tok(pre, add_special_tokens=False)["input_ids"]
        return await serve(ids, shim.max_tokens, stop_list(None),
                           req.stream, True, "resp",
                           thinking=thinking,
                           think_budget=server_default(shim, "max_thinking_tokens"),
                           drafter=drafter_for(shim),
                           sample=sample_spec(shim),
                           penalty=penalty_spec(shim),
                           tools=(chat_tools if mode != "none" else None),
                           parallel=req.parallel_tool_calls is not False,
                           pre=pre, forced=bool(pre),
                           snap=snap, snap2=snap2, wire="responses",
                           images=wire_imgs,
                           # #44: a client that asks for a reasoning summary
                           # (Codex sends {"summary": "auto"}, recorded from the
                           # real CLI) gets the reasoning text as the summary too
                           want_summary=bool((req.reasoning or {}).get("summary")
                                             not in (None, "", "none")),
                           schema=schema_for(shim))

    app.state.run = run
    app.state.serve = serve
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True,
                    help="HF snapshot dir or model id (tokenizer only)")
    ap.add_argument("--engine", default="127.0.0.1:8730")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8731)
    ap.add_argument("--context", type=int, default=32768,
                    help="FALLBACK only — the engine's INFO reports the real "
                         "kMaxCtx and that is what is enforced")
    ap.add_argument("--queue-timeout", type=float, default=3600,
                    help="seconds a request waits for the batch-1 slot "
                         "before 503")
    ap.add_argument("--max-tokens-cap", type=int, default=65536,
                    help="server-side ceiling on max_tokens (batch-1: one "
                         "long request blocks every other client)")
    # Issue #30: validate the HALOGEN_* defaults and exit. The entrypoint
    # runs this BEFORE the engine starts, so a bad variable is found in a
    # second rather than after minutes of pinning the checkpoint. The values
    # themselves were checked at import; what remains is the one that needs
    # the cap.
    ap.add_argument("--check-defaults", action="store_true",
                    help="validate the HALOGEN_* request defaults and exit")
    args = ap.parse_args()

    if DEFAULTS["max_tokens"] and DEFAULTS["max_tokens"] > args.max_tokens_cap:
        raise SystemExit(
            f"serve_api: HALOGEN_MAX_TOKENS_DEFAULT={DEFAULTS['max_tokens']} "
            f"exceeds the cap of {args.max_tokens_cap} "
            f"(HALOGEN_MAX_TOKENS_CAP); every request that omitted a budget "
            f"would be refused. Lower the default or raise the cap.")
    if args.check_defaults:
        d = defaults_summary()
        print("serve_api: request defaults "
              + (", ".join(f"{k}={v}" for k, v in d.items()) if d
                 else "none set (built-in)"), flush=True)
        return

    import uvicorn
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if vis_resolve_token(tok) is None:
        print("serve_api: no <|image_pad|> in this tokenizer; image requests "
              "will be refused", flush=True)
    host, _, port = args.engine.partition(":")
    engine = Engine(host, int(port))
    app = build_app(tok, engine, args.context, args.max_tokens_cap,
                    args.queue_timeout)

    @app.on_event("startup")
    async def _connect():
        await engine.connect()
        print(f"serve_api: engine at {args.engine}, "
              f"listening on {args.host}:{args.port}", flush=True)
        # Issue #30: a sampling default on an engine build that cannot
        # sample would turn every unadorned request into a 400. Refuse to
        # serve rather than misrepresent the configuration; the entrypoint
        # takes the container down on an api exit.
        if DEFAULTS["temperature"] and not engine.info.get("sampling"):
            print("serve_api: HALOGEN_TEMPERATURE is set but this engine "
                  "build decodes greedy only; unset it or use a build that "
                  "samples", flush=True)
            os._exit(1)
        d = defaults_summary()
        if d:
            print("serve_api: request defaults "
                  + ", ".join(f"{k}={v}" for k, v in d.items())
                  + " (a field the request sends wins; temperature 0 in a "
                    "request is greedy and takes no sampling default)",
                  flush=True)
        eng = engine.info.get("version") or "unknown"
        print(f"serve_api: version {API_VERSION}, engine version {eng}",
              flush=True)
        if eng != API_VERSION:
            print(f"serve_api: WARNING the api ({API_VERSION}) and the engine "
                  f"({eng}) are different releases. Run both containers from "
                  f"the same image tag: what one of them can do, the other "
                  f"may not know how to ask for (images, for one), and the "
                  f"answer may be wrong without any error saying so.",
                  flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info",
                timeout_keep_alive=KEEPALIVE_S)


if __name__ == "__main__":
    main()
