#!/usr/bin/env python3
"""effort-vocabulary — what each model's chat template does with a reasoning level.

    python3 bench/suites/effort-vocabulary.py                 all profiles
    python3 bench/suites/effort-vocabulary.py --profile qwen38
    python3 bench/suites/effort-vocabulary.py --out bench/reports/…/effort.json

Needs NO GPU, NO llama-server and NO loaded model. The chat template is
metadata inside the GGUF, and llama.cpp ships the engine that renders it
(`test-chat-template`, which is minja — the same one the server uses). So a
180 B model is as cheap to measure here as a 13 GiB one.

Why this exists
---------------
A consumer speaks Anthropic's vocabulary and it is FIXED: low, medium, high,
xhigh, max. Every model's template speaks its own, and it is decided by
whoever exported the GGUF. Nothing in between translates. Measured 28.08.2026,
seven profiles, four distinct behaviours:

    qwen38 · flashnext    low · medium · xhigh (high aliases to xhigh),
                          default xhigh, and an unknown value RAISES —
                          `max` and `none` are an HTTP 500 at the server
    gptoss                any string at all, interpolated straight into the
                          prompt (`{{- "Reasoning: " + reasoning_effort }}`),
                          default medium, no validation whatsoever
    gemma26 · gemma31     reasoning_effort is IGNORED. Only enable_thinking
    · batch · laguna      exists

So the two failure modes are opposite and both bad. Qwen turns an
out-of-vocabulary level into a crash, which is loud and gets fixed. gpt-oss
turns it into `Reasoning: max` inside the prompt — a phrase from no training
run, no error anywhere, just worse answers. That one is found by nobody.

A third thing the measurement decides, and it only became true on 28.08.:
`chat_template_kwargs` are part of the prefix id now (dialects.prefix_text).
Sending a kwarg a template IGNORES therefore costs a second cache key for a
byte-identical rendering — a whole prefill, for nothing. So a profile must not
declare levels its model does not read, and this suite is what says which
those are.

What it reports, and the three answers are not two
--------------------------------------------------
For every level, rendering the template either

    crashes             the template's own raise_exception. The server
                        answers 500. This is the value to CLAMP away.
    renders its own     the level reached the prompt and changed it
    renders as another  the level is an alias (qwen: high == xhigh == the
                        default), or it is ignored entirely (gemma: every
                        level renders exactly like no level at all)

"Accepted" alone would merge the last two, and they need different answers: an
alias is fine to send, an ignored value must not be sent.

A false start worth keeping
---------------------------
The first version of this hashed `test-chat-template`'s STDOUT. That is trace
output, not the rendered prompt, and it varies with things that never reach
the model — it reported `high` and `xhigh` as different renderings, which
contradicts the template's own source. `--output <path>` writes the actual
result. A measurement that cannot be explained by the thing it measures is
wrong even when it looks plausible.
"""
import argparse, hashlib, json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "setup", "lib"))
sys.path.insert(0, os.path.join(REPO, "setup", "gateway"))
import systemdfile                                            # noqa: E402
import modes as MODES                                         # noqa: E402

# The vocabulary a consumer may ask for, taken from modes.py rather than
# retyped — and that import is this file's whole contract. What it SUGGESTS is
# parsed by the same module that reads it back out of the profile, so the two
# cannot drift. They did: until 28.08.2026 this suite emitted `nothink:off` and
# an `EFFORT_MAP=` line, which modes.parse_modes REFUSES and nothing reads,
# while all seven profiles named this suite as the source of lines that had in
# fact been typed by hand. The order matters too — it is ascending, and
# value_for() takes the last member as this template's ceiling.
LEVELS = MODES.VOCABULARY

# One user turn and one tool, because several templates emit the reasoning
# instruction only inside their tool block — without a tool the difference
# would not appear and every level would look identical.
TOOLS = [{"type": "function",
          "function": {"name": "t", "description": "d",
                       "parameters": {"type": "object"}}}]


def profiles():
    """(name, env path) for every setup/env/*.env, sorted."""
    d = os.path.join(REPO, "setup", "env")
    return sorted((f[:-4], os.path.join(d, f))
                  for f in os.listdir(d) if f.endswith(".env"))


def model_path(env_path):
    """The -m argument of a profile, expanded. None if it has none."""
    argv = systemdfile.llama_args(env_path)
    for i, tok in enumerate(argv):
        if tok in ("-m", "--model") and i + 1 < len(argv):
            return argv[i + 1]
    return None


def chat_template(gguf):
    """The template out of the GGUF's metadata, or None.

    Read from the file rather than from a running server on purpose: that is
    what makes this suite answerable for a model that is not loaded, which is
    every model but one.
    """
    sys.path.insert(0, os.path.expanduser("~/llama.cpp/gguf-py"))
    try:
        from gguf import GGUFReader
    except ImportError:
        raise SystemExit("gguf-py not found — expected under ~/llama.cpp/gguf-py")
    r = GGUFReader(gguf)
    for f in r.fields.values():
        if f.name == "tokenizer.chat_template":
            return bytes(f.parts[f.data[0]]).decode("utf-8", "replace")
    return None


def render(binary, tmpl_file, ctx, workdir):
    """sha of the RENDERED prompt, or None when the template raises.

    --output, never stdout. See the module docstring: stdout is trace.
    """
    body = {"messages": [{"role": "user", "content": "hi"}],
            "bos_token": "<s>", "eos_token": "</s>",
            "add_generation_prompt": True, "tools": TOOLS}
    body.update(ctx)
    j = os.path.join(workdir, "in.json")
    o = os.path.join(workdir, "out.txt")
    with open(j, "w") as fh:
        json.dump(body, fh)
    if os.path.exists(o):
        os.remove(o)
    r = subprocess.run([binary, tmpl_file, "--json", j, "--no-common",
                        "--output", o], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(o):
        return None
    with open(o, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:8]


def cli_kwargs(env_path):
    """The chat_template_kwargs the profile's own command line already sets.

    Without it a suggestion cannot tell a mode that CHANGES something from one
    that repeats the default. qwen38.env switches thinking off on the command
    line, so a `none` mode there renders exactly what the bare alias renders —
    the same prompt under a second prefix-cache key, which since 28.08. costs a
    whole prefill.
    """
    argv = systemdfile.llama_args(env_path)
    for i, tok in enumerate(argv):
        if tok == "--chat-template-kwargs" and i + 1 < len(argv):
            try:
                return json.loads(argv[i + 1].rstrip(":"))
            except ValueError:
                return {}
    return {}


def value_for(word, levels, reads_thinking, canon=None):
    """What a mode named `word` should SEND on this template, or None.

    None means the model cannot express it — gpt-oss has no thinking knob, so
    `none` is unavailable there at all.

    `canon` maps a level to the level that RENDERS THE SAME and comes first in
    the vocabulary. Sending the consumer's own word would be the obvious thing
    and is wrong: on the Qwen templates `high` and `xhigh` render identically
    (the template aliases one to the other), so sending both words verbatim
    makes two different chat_template_kwargs for one prompt — and since the
    kwargs entered the prefix id, two cache keys and a wasted prefill.
    Canonicalising collapses them to one.
    """
    if word == "none":
        return MODES.OFF if reads_thinking else None
    if not levels:
        return MODES.ON if reads_thinking else None
    top = [lv for lv in LEVELS if lv in levels][-1]
    lvl = word if word in levels else top
    lvl = (canon or {}).get(lvl, lvl)
    return ("%s+%s" % (MODES.ON, lvl)) if reads_thinking else lvl


def measure(binary, tmpl, workdir, base=None):
    """What this template does with every word, ON TOP OF the profile's own
    command line — see cli_kwargs."""
    base = base or {}
    f = os.path.join(workdir, "t.jinja")
    with open(f, "w") as fh:
        fh.write(tmpl)

    def r(extra):
        kw = dict(base)
        kw.update(extra)
        return render(binary, f, kw, workdir)

    plain = r({})
    think = {v: r({"enable_thinking": v}) for v in (True, False)}
    reads_thinking = think[True] != think[False]

    # LEVELS ARE PROBED WITH THINKING ON. The Qwen templates gate the whole
    # effort block on that knob and qwen38.env switches it off on the command
    # line — probing on top of the raw baseline therefore reports every level
    # as identical and the template as having none, which is how the first
    # version of this function decided that qwen38 reads no levels at all.
    probe = {"enable_thinking": True} if reads_thinking else {}
    probe_base = r(probe)
    raw = {lv: r(dict(probe, reasoning_effort=lv))
           for lv in LEVELS if lv != "none"}

    accepted = {lv: h for lv, h in raw.items() if h}
    ignored = all(h == probe_base for h in accepted.values()) and len(accepted) > 1
    levels = set() if ignored else set(accepted)
    # A template that raises on nothing has told us nothing: it renders any
    # string, so no measurement here distinguishes a trained level from a typo.
    measurable = bool([lv for lv, h in raw.items() if h is None])

    # What each vocabulary word would actually produce ON TOP OF the profile's
    # own command line, so a word that merely repeats the baseline is dropped
    # rather than advertised under its own prefix-cache key.
    # Levels that render alike are ONE level wearing several names; the first
    # in vocabulary order speaks for the group.
    canon = {}
    for lv in LEVELS:
        h = raw.get(lv)
        if h:
            canon[lv] = next(x for x in LEVELS if raw.get(x) == h)

    words = {}
    for w in LEVELS:
        v = value_for(w, levels, reads_thinking, canon)
        words[w] = None if v is None else {"value": v,
                                           "render": r(MODES.kwargs_for(v))}

    return {
        "default_render": plain,
        "levels": raw,
        "crashes": sorted(lv for lv, h in raw.items() if h is None),
        "reads_effort": not ignored,
        "reads_enable_thinking": reads_thinking,
        "probe_render": probe_base,
        "measurable": measurable,
        "distinct_renders": len(set(accepted.values())) if not ignored else 0,
        "words": words,
    }


def suggest(m):
    """The two profile lines this measurement implies.

    Emitted through modes.py's own vocabulary and value grammar, and the module
    docstring says why. A word is left out when it renders exactly what the
    bare alias renders: an unoffered name falls through to the bare alias and
    renders the same thing anyway, so leaving it out costs nothing and saves a
    prefix-cache key.

    TEMPLATE_LEVELS carries one of three answers, never an empty field —
    `no-levels` and `unmeasurable` are measurements, silence is not. See
    modes.parse_levels.
    """
    if m["reads_effort"] and not m["measurable"]:
        # gpt-oss renders any string it is given, so nothing measured here says
        # which levels the MODEL was trained on. Suggesting them anyway is how
        # `max:max` — an untrained word straight into the prompt — got proposed.
        return ("# MODES=…  DECLARE BY HAND: this template validates nothing, so "
                "its levels\n      #   cannot be measured. Take them from the "
                "model card and clamp the rest.",
                "TEMPLATE_LEVELS=" + MODES.UNMEASURABLE)

    base = m["default_render"]
    pairs = []
    for word in LEVELS:
        w = m["words"].get(word)
        if not w or w["render"] == base:
            continue
        pairs.append("%s:%s" % (word, w["value"]))

    levels = (MODES.NO_LEVELS if not m["reads_effort"]
              else "  ".join(lv for lv in LEVELS
                             if lv != "none" and m["levels"].get(lv)))
    return "MODES=" + "  ".join(pairs), "TEMPLATE_LEVELS=" + levels


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", help="only this one (default: all)")
    ap.add_argument("--binary", default=None,
                    help="test-chat-template. Default: beside the llama-server "
                         "the stable build points at")
    ap.add_argument("--out", help="write the result as JSON")
    a = ap.parse_args()

    binary = a.binary or os.path.expanduser(
        "~/llama.cpp/build-rocm-patched/bin/test-chat-template")
    if not os.path.exists(binary):
        raise SystemExit(
            "\ntest-chat-template not found at %s\n"
            "  It is built with llama.cpp; pass --binary <path> if your build "
            "lives elsewhere.\n"
            "  Without it this suite cannot render anything, and guessing what "
            "a\n  template accepts is the thing it exists to stop." % binary)

    todo = [(n, p) for n, p in profiles() if not a.profile or n == a.profile]
    if not todo:
        raise SystemExit("no profile called %r" % a.profile)

    seen, report = {}, {}
    with tempfile.TemporaryDirectory(prefix="effort-") as wd:
        for name, env in todo:
            gguf = model_path(env)
            if not gguf or not os.path.exists(gguf):
                print("%-11s no readable model file — skipped" % name)
                continue
            tmpl = chat_template(gguf)
            if not tmpl:
                print("%-11s no chat template in the GGUF" % name)
                continue
            base = cli_kwargs(env)
            key = hashlib.sha256(
                (tmpl + json.dumps(base, sort_keys=True)).encode()).hexdigest()[:12]
            # Several profiles share one template (qwen38/flashnext,
            # gemma26/gemma31/batch). Measuring it once and saying so is more
            # useful than four identical blocks.
            m = seen.get(key) or measure(binary, tmpl, wd, base)
            shared = key in seen
            seen[key] = m
            report[name] = dict(m, template_sha=key)

            print("=== %-10s %s%s" % (name, key,
                                      "   (same template as an earlier profile)"
                                      if shared else ""))
            if m["reads_effort"]:
                for lv in LEVELS:
                    if lv == "none":
                        continue
                    h = m["levels"][lv]
                    if h is None:
                        note = "RAISES — clamp this one away"
                    elif h == m["default_render"]:
                        note = "renders like the default (alias)"
                    else:
                        note = "own rendering"
                    print("      %-6s %-9s %s" % (lv, h or "—", note))
                print("      -> %d distinct renderings" % m["distinct_renders"])
            else:
                print("      reasoning_effort is IGNORED by this template")
            print("      enable_thinking: %s"
                  % ("works" if m["reads_enable_thinking"] else "IGNORED"))
            for line in suggest(m):
                print("      %s" % line)
            print()

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(report, fh, indent=1, sort_keys=True)
        print("written: %s" % a.out)


if __name__ == "__main__":
    main()
