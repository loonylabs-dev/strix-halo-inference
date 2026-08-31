#!/usr/bin/env python3
"""modes — thinking modes that belong to the model that is actually loaded.

    MODES=low:on+low  medium:on+medium  high:on+high  xhigh:on+high  max:on+high
    TEMPLATE_LEVELS=low medium high xhigh

Two lines in `setup/env/<model>.env`, and the gateway derives everything else
from them plus the alias llama-server reports.

Why the names are derived and not listed
----------------------------------------
Until 28.08.2026 the three names for one model lived in `KWARGS_BY_MODEL`, a
JSON blob in `~/.config/cc-gateway.env`. The decision "which model, in which
thinking mode" was therefore spread over three files that nothing reconciled:

    which model runs      switch-model.sh -> systemd
    which names exist     ~/.config/cc-gateway.env
    which name is asked   setup/claude/local.json (ANTHROPIC_MODEL)

After a `switch-model.sh flashnext` the first had changed and the other two had
not, so `qwen38-think` was still answered — injecting qwen38's thinking mode
into requests bound for another model, over a command line that had set it
otherwise, with no error anywhere. Scoping the blob by the served model fixes
that case but not the shape: the guard is per-table, so the moment the blob
names two profiles it switches itself off again.

Deriving removes the third place. `qwen38-think` cannot exist while flashnext
serves, because the names are BUILT from what serves. The listing and the
injection cannot disagree either — both call names() and resolve() on the same
two inputs.

Two knobs, not one
------------------
`enable_thinking` decides WHETHER, `reasoning_effort` decides HOW MUCH, and a
template may have either, both, or neither. Measured across the seven profiles
here (bench/suites/effort-vocabulary.py, 28.08.2026):

    qwen38 · flashnext   both. Levels low/medium/xhigh, `high` aliases to
                         xhigh, and `max`/`none` RAISE — an HTTP 500
    gptoss               only reasoning_effort, and it validates NOTHING:
                         any string is interpolated into the prompt
    gemma26 · gemma31    only enable_thinking. reasoning_effort is ignored
    · batch · laguna     completely

So a mode's value is either `off`, `on`, or a template level, and this module
keeps them apart. The old blob mixed both knobs into one dict per name.

Why an unmapped level is dropped rather than passed on
-----------------------------------------------------
Since 28.08. the chat template kwargs are part of the prefix id
(dialects.prefix_text). A kwarg a template IGNORES therefore costs a second
cache key for a byte-identical prompt — a whole prefill, for nothing. And a
kwarg a template REJECTS costs a 500. So a profile declares only what its
template reads, and `check_modes()` refuses one that does not.

Pure: no I/O, no configuration, no network. The caller reads the profile.
"""
import collections

# The value that means "turn thinking off" / "on" rather than "this much of
# it". Kept as constants because they are the one place the two knobs meet.
OFF = "off"
ON = "on"

# The names a mode may carry, and the only ones. This is Anthropic's scale —
# what a consumer already speaks — and it is fixed on purpose: three invented
# words (`think`, `deep`, `full`) lived in these profiles until 28.08.2026 and
# were a third register for an idea that already had two. A vocabulary that
# admits one more word is not a vocabulary.
#
# It is the CONSUMER's scale, not any template's. What each word turns into is
# the profile's business, because no template here speaks all six: qwen38
# raises on `max` and on `none`, gemma reads no levels at all. The name exists
# everywhere it makes sense; the value is measured per model.
VOCABULARY = ("none", "low", "medium", "high", "xhigh", "max")


def _pairs(spec, what):
    """`a:1  b:2` -> [(a,1), (b,2)]. A half-written entry is refused.

    Skipping it silently would leave a profile advertising fewer modes than it
    declares, with nothing anywhere saying why — the failure this repository
    keeps finding, so not one to add.
    """
    out = []
    for token in (spec or "").split():
        if ":" not in token:
            raise SystemExit(
                "\n%s: %r is not a pair.\n"
                "  Expected name:value, e.g. think:low" % (what, token))
        name, _, value = token.partition(":")
        if not name or not value:
            raise SystemExit("\n%s: %r is not a pair." % (what, token))
        out.append((name, value))
    return out


def parse_modes(spec):
    """MODES -> {vocabulary word: what to send}, in VOCABULARY order.

    Ordered by the vocabulary rather than by the file, because the order is
    what a picker shows and `none` before `max` is the only arrangement that
    means anything. Declaration order would make the list depend on how
    somebody happened to type the line.

    A name outside the vocabulary is refused. Quietly accepting it is how the
    invented names got in.
    """
    got = dict(_pairs(spec, "MODES"))
    for name in got:
        if name not in VOCABULARY:
            raise SystemExit(
                "\nMODES: %r is not a reasoning level.\n"
                "  The vocabulary is fixed: %s\n"
                "  It is the consumer's scale — what each word SENDS is what "
                "the profile\n  translates, and that part is measured by "
                "bench/suites/effort-vocabulary.py."
                % (name, "  ".join(VOCABULARY)))
    return collections.OrderedDict((w, got[w]) for w in VOCABULARY if w in got)


# The two answers that are not a list of levels. Spelled out, because
# `TEMPLATE_LEVELS=` and a missing line read back identically through
# systemdfile.variable, and they are not the same claim.
NO_LEVELS = "no-levels"        # measured: this template reads none
UNMEASURABLE = "unmeasurable"  # it renders any string, so nothing can be measured
# Returned when the profile says NOTHING. Distinct from UNMEASURABLE, which is
# a stated answer: "I looked, and this template cannot be measured". Silence is
# not an answer, and the first version of this collapsed the two — which is the
# very conflation between an empty line and a missing one that these constants
# exist to end.
NOT_STATED = "not-stated"


def parse_levels(spec):
    """TEMPLATE_LEVELS -> a set of levels, an empty set, or None.

    Three answers, because there are three:

        low medium xhigh   measured: these render
        no-levels          measured: this template reads no levels at all
        unmeasurable       -> None. gpt-oss interpolates whatever it is given
                           (`{{- "Reasoning: " + reasoning_effort }}`), so
                           rendering proves nothing about what the MODEL knows

    An empty set and None used to be the same value, and the difference is the
    whole point: `no-levels` must REFUSE a mode that names a level, while
    `unmeasurable` must let one through. Conflating them turned the guard off
    for the four profiles that need it most.

    A missing line returns None here and is refused by check_modes when there
    are modes to check — the profile has to say which of the three it means.
    """
    spec = (spec or "").strip()
    if not spec:
        return NOT_STATED
    if spec == UNMEASURABLE:
        return None
    if spec == NO_LEVELS:
        return set()
    return set(spec.split())


def kwargs_for(value):
    """The chat template kwargs one mode value implies.

    A value is `off`, `on`, a template level, or a level composed with the
    thinking knob: `on+low`.

    THE COMPOSITION IS NOT OPTIONAL SUGAR, and measuring is what showed it.
    The Qwen template gates the whole effort block on the knob:

        {%- if enable_thinking is undefined or enable_thinking is true %}
            {%- set resolved_reasoning_effort = reasoning_effort|default(…) %}

    and qwen38.env's command line sets `enable_thinking:false`. Request kwargs
    merge over the command line per key, so a mode sending only the level
    leaves the knob at false and the block never runs. Measured 28.08.2026:

        {"reasoning_effort":"low"}                    sha 1ad7792b
        (nothing at all)                              sha 1ad7792b
        {"enable_thinking":true,"reasoning_effort":"low"}  sha 938681af

    Identical rendering, and since the kwargs entered the prefix id it would
    have carried its own cache key — worse than doing nothing.

    Composed EXPLICITLY rather than inferred, because the inference would be
    wrong elsewhere: gpt-oss has no thinking knob at all, and sending one there
    is an ignored kwarg, which is the same wasted cache key from the other
    direction. Only the profile knows, and the profile has measured.

    `None` means the bare alias: ask for nothing and let the command line
    stand. `{}` is returned for it so that nothing is written into the body.
    """
    if value is None:
        return {}
    parts = [p for p in str(value).split("+") if p]
    out = {}
    for part in parts:
        if part == OFF:
            out["enable_thinking"] = False
        elif part == ON:
            out["enable_thinking"] = True
        else:
            out["reasoning_effort"] = part
    if out.get("enable_thinking") is False and "reasoning_effort" in out:
        raise SystemExit(
            "\nMODES: %r asks for no thinking AND an effort level.\n"
            "  Those cannot both be meant — the effort would be rendered into "
            "a prompt\n  whose thinking is switched off, or dropped. Say one." % value)
    return out


def names(alias, modes):
    """The names a PICKER should show: one per distinct behaviour.

    Not the same list as what resolve() accepts, and the difference is the
    point. On qwen38 `high`, `xhigh` and `max` all send xhigh — the template
    aliases high itself and raises on max — so naming all three offers one
    choice three times. A user wants to see what is actually different.

    Every declared word still RESOLVES, because a client configured
    `ANTHROPIC_MODEL=qwen38-max` means "as much as this model has": if the name
    matched nothing it would fall through to the bare alias, and on qwen38 the
    bare alias does not think at all. So the synonyms are accepted and simply
    not advertised.

    The invariant that matters survives, in the safe direction: everything
    OFFERED resolves. The defect this design was written against was a listing
    advertising what the injection then refused — the reverse.

    The representative of a group is its LOWEST vocabulary word. Deterministic,
    and it errs low: `qwen38-max` in a picker would promise something this
    template raises on, while `qwen38-high` promises what it delivers.
    """
    out, seen = [alias], set()
    for name, value in modes.items():
        if value in seen:
            continue
        seen.add(value)
        out.append("%s-%s" % (alias, name))
    return out


def resolve(model, alias, modes):
    """(kwargs, matched) for a request's model name.

    `matched` false means the name is not ours — a stale one from before a
    model switch, or another provider's. The caller forwards it untouched;
    llama-server ignores the field anyway, and the log keeps showing what the
    consumer asked for.
    """
    if model == alias:
        return {}, True
    prefix = alias + "-"
    if isinstance(model, str) and model.startswith(prefix):
        level = modes.get(model[len(prefix):])
        if level is not None:
            return kwargs_for(level), True
    return {}, False


def check_modes(modes, levels):
    """Refuse a mode whose value this template does not render.

    The trap the fixed vocabulary creates: `max` is a legal NAME and, on every
    template measured here, an illegal VALUE. `max:on+max` looks symmetrical
    and is an HTTP 500 for whoever selects it. Catching it at load costs one
    check instead of one failed request per attempt, forever.

    `levels` is what parse_levels returned, and all three of its answers mean
    something different here:

        a set        only those levels, plus the two knob words
        set()        `no-levels` — the knob words and nothing else
        None         `unmeasurable`, or the profile did not say. With modes
                     declared, not saying is refused: an answer that cannot be
                     measured may be stated, but it may not be omitted.
    """
    if levels is None:
        return          # stated: this template cannot be measured
    if levels is NOT_STATED:
        if modes:
            raise SystemExit(
                "\nMODES is declared but TEMPLATE_LEVELS is not.\n"
                "  Say which of the three this template is:\n"
                "    TEMPLATE_LEVELS=low medium xhigh   the levels it renders\n"
                "    TEMPLATE_LEVELS=%s          it reads no levels at all\n"
                "    TEMPLATE_LEVELS=%s       it validates nothing, so its\n"
                "                                 levels cannot be measured\n"
                "  bench/suites/effort-vocabulary.py answers this."
                % (NO_LEVELS, UNMEASURABLE))
        return
    allowed = set(levels) | {ON, OFF}
    for name, value in modes.items():
        for part in str(value).split("+"):
            if part and part not in allowed:
                raise SystemExit(
                    "\nMODES: %s:%s — this template does not render %r.\n"
                    "  It renders: %s\n"
                    "  Measured by bench/suites/effort-vocabulary.py; re-run it "
                    "rather than\n  editing TEMPLATE_LEVELS by hand."
                    % (name, value, part,
                       "  ".join(sorted(levels)) or "no levels at all"))
