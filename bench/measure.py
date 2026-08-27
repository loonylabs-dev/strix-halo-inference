#!/usr/bin/env python3
"""measure — the one place where an answer turns into a measurement.

Why separate: the same four lines stood in ten measurement scripts, and all
ten carried the same bug. If `usage` is missing from the answer — a different
backend, a different llama.cpp build, or simply an error instead of a result —
then

    inp = u.get("input_tokens", -1)
    cr  = u.get("cache_read_input_tokens", 0)
    q   = 100.0 * cr / (inp + cr)

computed a rate of **-0.0 %**. That reads like a measurement ("the cache does
not work"), but it is not a measurement at all. That is exactly how a wrong
number ends up in the documentation, and afterwards nobody can tell it apart
from a real one.

Hence: no number is better than an invented one.
"""

class NoMeasurement(Exception):
    """The answer carries no usable accounting."""

def _hint(answer):
    if isinstance(answer, dict) and answer.get("error"):
        return "The server says: %s" % str(answer["error"])[:160]
    return "The answer was: %s" % str(answer)[:120]

def required(u, field="input_tokens"):
    """Take a required value out of usage — or give up.

    Replaces the old `u.get("input_tokens", -1)`: better an abort with a reason
    than a number nobody recognises as an error any more.
    """
    if not isinstance(u, dict) or field not in u:
        raise NoMeasurement("usage.%s is missing — nothing was measured here "
                            "(usage=%s)" % (field, str(u)[:120]))
    return int(u[field])

def evaluate(answer, seconds=None):
    """Derive new, cached and the cache rate from a /v1/messages answer.

    Raises NoMeasurement instead of inventing a rate.
    """
    u = answer.get("usage") if isinstance(answer, dict) else None
    if not isinstance(u, dict) or "input_tokens" not in u:
        raise NoMeasurement("no usage.input_tokens in the answer — nothing was "
                            "measured here. %s" % _hint(answer))
    new = int(u["input_tokens"])
    cached = int(u.get("cache_read_input_tokens", 0))
    total = new + cached
    if total <= 0:
        raise NoMeasurement("usage reports %d tokens in total (new=%d, cached=%d)"
                            % (total, new, cached))
    d = {"new": new, "cached": cached, "rate": round(100.0 * cached / total, 1)}
    if seconds is not None:
        d["seconds"] = round(seconds, 2)
    return d

def gtt_gib():
    """Used GTT memory in GiB, or None.

    Globs card* instead of hard-coding card1: on this machine the GPU is card1,
    on the next one it may be card0. llm-profile corrected exactly this bug for
    itself in rev. 2 — in bench/ it stayed, and there it does not show, because
    the field simply stays empty.
    """
    import glob
    for path in sorted(glob.glob("/sys/class/drm/card*/device/mem_info_gtt_used")):
        try:
            with open(path) as f:
                return round(int(f.read()) / 1073741824.0, 2)
        except Exception:
            continue
    return None


# --------------------------------------------------------- request bodies ---
# Captured Claude Code bodies carry an e-mail address, device_id, account_uuid
# and Anthropic's system prompt. They are excluded from the repo, so six of the
# measurement suites could not run on a fresh checkout at all. They can all be
# built synthetically instead — the measurements need the *structure*, not the
# content.

DROP_FIELDS = ("thinking", "context_management", "output_config")

# The project path a captured body carries. Setting `project` on a capture
# means rewriting these; on a synthetic body it is simply passed through.
CAPTURE_MARKERS = ("-tmp-cc-jagd", "/tmp/cc-jagd")


class BodyProblem(Exception):
    """The request body cannot be built the way it was asked for."""


def request_body(project=None, n_tools=None, question=None, capture=None,
                 max_tokens=1, stream=False):
    """A Claude-Code-shaped request body — from a capture if there is one.

    `capture` is a path, or a bare file name looked up under
    bench/suites/bodies/. Without one, tools/synthetic.py builds the body.

    The three parameters mean the same thing for both sources, but they have to
    be applied differently, and that difference matters. A captured body
    carries its own project path; if `project` did not rewrite it, two
    "different" projects would share one prefix and a multi-project measurement
    would quietly turn into a single-project one. So when the rewrite finds
    nothing to replace, this raises instead of returning a body that measures
    the wrong thing.
    """
    import copy, json, os, sys
    here = os.path.dirname(os.path.abspath(__file__))

    if capture:
        path = capture
        if not os.path.isabs(path):
            local = os.path.join(here, "suites", "bodies", path)
            if os.path.exists(local):
                path = local
        if not os.path.exists(path):
            raise BodyProblem(
                "no capture at %s. Leave `capture` out and the body is built "
                "by tools/synthetic.py instead." % capture)
        with open(path, encoding="utf-8") as f:
            p = json.load(f)
        p = copy.deepcopy(p)
        if project is not None:
            hits = 0
            for b in p.get("system", []):
                if isinstance(b, dict) and b.get("type") == "text":
                    for marker in CAPTURE_MARKERS:
                        if marker in b["text"]:
                            hits += 1
                            b["text"] = b["text"].replace(
                                marker, marker.replace("cc-jagd", project.strip("/")))
            if not hits:
                raise BodyProblem(
                    "the capture carries none of %s, so project=%r changed "
                    "nothing. Every 'project' would share one prefix and the "
                    "measurement would be meaningless."
                    % (list(CAPTURE_MARKERS), project))
        if n_tools is not None:
            p["tools"] = p.get("tools", [])[:n_tools]
        if question is not None:
            _set_question(p, question)
    else:
        sys.path.insert(0, os.path.join(here, "..", "tools"))
        from synthetic import body                       # noqa: E402
        kw = {}
        if project is not None:
            kw["project"] = project
        if n_tools is not None:
            kw["n_tools"] = n_tools
        if question is not None:
            kw["question"] = question
        p = body(**kw)

    for k in DROP_FIELDS:
        p.pop(k, None)
    p["stream"] = stream
    p["max_tokens"] = max_tokens
    return p


def _set_question(p, question):
    """Replace the last text block of the last user message."""
    for m in reversed(p.get("messages", [])):
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, list):
            for b in reversed(c):
                if b.get("type") == "text":
                    b["text"] = question
                    return
        elif isinstance(c, str):
            m["content"] = question
            return
    raise BodyProblem("no user text block to put the question in")
