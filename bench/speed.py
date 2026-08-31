#!/usr/bin/env python3
"""speed — what a configuration COSTS: time and tokens per second.

    python3 bench/speed.py --url http://127.0.0.1:8080 --label qwen38-live

Split out of bench/quality.py on 26.08. when the model battery was removed.
The two halves were always different questions wearing one file:

    how good is this MODEL      not measured here, and deliberately. Other
                                people benchmark models, at more scale than
                                a home-grown battery survives, and every
                                number one produces ages the week a new
                                model lands.
    what does this SETUP cost   measured here, because nobody else measures
                                this build on this hardware at this window.

So this file has no `tasklib` in it and never will — and on 27.08. the traffic
went the other way for the first time: tasklib's last free-prose task moved
HERE, as a workload. It was carrying a quality checker (is this a good
explanation?) into a repo that stopped judging model quality on 26.08., while
its only surviving caller wanted the one thing a workload provides — a decode
rate on novel text.

`run()` sends one request per (depth x workload) through /v1/chat/completions
and reads the SERVER's clock. `probes()` still exists beside it for the cache
question, where the shape of a real Claude-Code body is the point and the rate
is secondary; it goes through the Anthropic route with the mid-conversation
system remap, because that is what Claude Code actually sends.

Speculation makes decode workload-dependent, so one number is never the whole
story — and until 27.08. this file got that story half wrong. It said counting
is "near the ceiling", full stop. It is near the ceiling for a TRAINED DRAFT
HEAD, which predicts `1\n2\n3` easily. It is near the FLOOR for an N-GRAM
drafter, which drafts from the PROMPT — and the digits of a counting task
appear nowhere in the filler text, so no amount of working drafter can show up
in that number.

That is not a hypothetical. It is how `--spec-type ngram-mod` came to be
measured on this repo's own hardware at 8.5 t/s against 8.6 without it, and
written down as "the ngram drafters give nothing", while upstream reports 79 %
acceptance and a 2.4x speedup for the same drafter on the same architecture —
on a copy-heavy edit. The drafter was not broken. The probe could not see it.

So there are THREE decode workloads here now, measured at every depth, and
together they BRACKET the range rather than pick a point in it:

    prose   novel text that is neither in the prompt nor predictable — the
            FLOOR. No drafter of any kind can help; this is what the hardware
            does with no speculation to hide behind.
    count   output that exists nowhere in the prompt but is trivially
            predictable. Ceiling for a TRAINED DRAFT HEAD, floor for an
            n-gram drafter.
    copy    output that is almost entirely present in the prompt — a block
            reproduced with one substitution, which is the shape of most of
            what an agent emits. Ceiling for an N-GRAM drafter.

No single one of them is "the decode rate", and a configuration that wins one
and loses another is a real result rather than a contradiction. The floor and
the two ceilings are what a reader needs in order to know which of them their
own workload sits near.

Both non-trivial workloads CHECK THEMSELVES, in opposite directions, because
the mistake this file is recovering from was a probe that could not fail
visibly. `copy` measures how much of the answer came from the block and says
so when the model did not copy. `prose` measures the same overlap against the
filler and says so when the model DID — an answer that parrots its prompt is
not novel text, and its rate is a copy rate wearing the floor's label.
"""
import argparse, json, os, sys, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)
from measure import evaluate, gtt_gib, request_body            # noqa: E402


def post(url, path, payload, timeout):
    r = urllib.request.Request(url + path, data=json.dumps(payload).encode(),
                               headers={"content-type": "application/json",
                                        "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(r, timeout=timeout) as x:
        return json.loads(x.read().decode())


def _system_mid_conversation_remap(p):
    """Non-leading system messages become user TEXT BLOCKS.

    The Claude-Code-shaped body carries a system block BEHIND the user
    question (that placement is the whole SWA finding). Laguna's template
    accepts it; Qwen 3.8's raises 'System message must be at the beginning'
    and the server answers 500. The remap keeps position and size of the
    block — the prefix structure the probe measures stays intact.

    Production implication, found 24 Aug: if the real bodies carry this
    structure, a Qwen production server rejects every Claude Code request
    until the gateway applies the same remap (MID_SYSTEM_TO_USER).

    Moved here from bench/quality.py on 26.08. UNCHANGED — the first version
    of the move dropped the string-to-block conversion on the last line and
    tests/test_sweep.py caught it. A function that moves has to arrive the
    same function; "while I am here" is how a refactor becomes a regression.
    """
    msgs = p.get("messages", [])
    for i, m in enumerate(msgs):
        if i > 0 and m.get("role") == "system":
            c = m["content"]
            m["role"] = "user"
            m["content"] = [{"type": "text", "text": c}] if isinstance(c, str) else c
    return p


FILLER = "The quick brown fox jumps over the lazy dog. "

# The copy source. Deterministic, code-shaped, and generated rather than
# quoted: a real file would put someone else's text in a benchmark, and a
# famous one would be in the model's weights, which is the one thing that must
# NOT be true here — a block the model can recite from memory is not a block it
# copied from the prompt, and the whole probe rests on that difference.
_NOUNS = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
          "hotel", "india", "juliet", "kilo", "lima")


def copy_source(n_lines=12):
    """A config block: repetitive in shape, unpredictable in its values.

    Both halves matter. The shape is what an n-gram drafter gets right, and it
    is why this workload is the drafter's ceiling. The values are what keeps a
    model from producing the block WITHOUT reading it — if it could guess the
    next line, the probe would measure prediction, not copying.
    """
    out = []
    for i in range(n_lines):
        noun = _NOUNS[i % len(_NOUNS)]
        out.append('    "%s_%02d": {"enabled": %s, "retries": %d, '
                   '"budget_ms": %d, "path": "/srv/%s/%02d"},'
                   % (noun, i, "true" if i % 3 else "false", i % 5,
                      1000 + (i * 37) % 900, noun, i))
    return "\n".join(out)


def copied_fraction(text, source):
    """Share of the answer's word trigrams that occur in the source block.

    Deliberately crude and deliberately not an equality check: the probe asks
    for one substitution, a model may re-indent or stop mid-line, and none of
    that makes the decode any less copy-heavy. What it separates is copying
    from NOT copying — a refusal, a summary, or an answer written in prose —
    and that is the only distinction the number needs to survive.
    """
    def grams(s):
        w = s.split()
        return {" ".join(w[i:i + 3]) for i in range(len(w) - 2)}
    out, src = grams(text or ""), grams(source)
    if not out:
        return 0.0
    return round(len(out & src) / len(out), 3)


def answer_text(r):
    """The visible answer, and the thinking channel is NOT it.

    A model that answers in `reasoning_content` used to score zero here, which
    is the defect that voided a whole battery run on 26.08. The copy check has
    the same exposure — thinking text does not copy the block — so the two
    channels are separated rather than concatenated.
    """
    msg = ((r.get("choices") or [{}])[0].get("message") or {})
    return msg.get("content") or "", msg.get("reasoning_content") or ""


def payload_for(workload, depth_tokens, gen_tokens, model):
    """The two workloads, built side by side so the difference stays visible.

    The filler is trimmed by what the copy block itself costs, so that both
    workloads land at the SAME depth rather than the same amount of filler.
    Without that the copy cell sits a few hundred tokens deeper than the count
    cell it is compared against, and decode moves with depth — which would put
    a second workload-dependent effect inside a number built to isolate one.
    The trim cannot go below the block, so at the shallowest depth the copy
    cell is as deep as the block is long; `prompt_n` in the result is what the
    server counted, and it is the number to read.
    """
    if workload in ("count", "prose"):
        src, overhead = "", 0
    elif workload == "copy":
        src = copy_source()
        overhead = len(src) // 4          # ~4 chars per token, near enough
    else:
        raise ValueError("unknown workload %r" % workload)

    filler = FILLER * max(1, (depth_tokens - overhead) // 9)
    if workload == "count":
        ask = filler + "\n\nCount from 1 to 60, one per line."
    elif workload == "prose":
        # Novel, unpredictable, and deliberately about nothing in the filler.
        # The subject is chosen to be answerable by any model without domain
        # knowledge — what is being measured is the decode path, not what the
        # model knows — and the length is pinned so that runs stay comparable.
        ask = (filler + "\n\nIgnore the text above. Explain, in about 300 "
               "words of flowing prose and without any lists or headings, why "
               "a bridge builder and a software architect think about "
               "load-bearing structure in different ways. Do not repeat the "
               "question.")
    else:
        ask = (filler + "\n\nHere is a configuration block:\n\n" + src +
               "\n\nReproduce the block above exactly, line for line, changing "
               "only every \"retries\": 0 into \"retries\": 1. Output the block "
               "and nothing else — no explanation, no code fence.")
    return {"model": model, "stream": False, "max_tokens": gen_tokens,
            "messages": [{"role": "user", "content": ask}]}, src


# Below this share of copied trigrams, the model did not do what the copy
# probe asked, and its decode rate is a rate for something else. Not an
# assertion and not a failure — the cell is recorded WITH the warning, because
# "the model would not copy" is itself a finding about the model.
COPIED_MIN = 0.5

# Floor first, then the two ceilings. The order is not cosmetic: `prose` runs
# at each depth before the others and establishes the filler prefix, so the
# cells that follow it are measuring decode rather than a cache miss.
WORKLOADS = ("prose", "count", "copy")

# Above this share of copied trigrams, a PROSE answer is not novel text — the
# model has echoed its own prompt back, and the rate belongs in the copy
# column rather than the floor. The mirror image of COPIED_MIN, and it exists
# for the same reason: a probe that cannot fail visibly is how this file came
# to report a working drafter as worthless.
PARROT_MAX = 0.25


def rates(url, depth_tokens, gen_tokens=128, timeout=1800, model="probe",
          extra=None, workload="count"):
    """Prefill and decode rates at a given KV depth, from the SERVER's clock.

    Through /v1/chat/completions, and that is not a preference. The Anthropic
    route returns no `timings` object at all — checked on 26.08. against a live
    server — so a probe sent there can only divide tokens by wall time, and
    wall time contains the prefill.

    That is not a theoretical objection. Upstream issue #27623 reported decode
    on this very model family "collapsing 25x" past 80K context and chased it
    as far as a suspected NVIDIA driver regression, before the reporter
    withdrew the whole thing:

        "My measurement metric was flawed. The numbers were completion_tokens
         / total request time, which includes prompt processing (~43 s for a
         68K fill). That's why every configuration collapsed to ~1.4 t/s: it
         was prefill time, not decode."

    Our own probes had exactly that shape until this function replaced them,
    and they were producing 8.6 t/s for a configuration whose server-side
    decode is materially higher. A wrong number is worse than none: it gets
    written down.

    `timings` also carries what speculation is actually doing — draft_n and
    draft_n_accepted — which is the number that decides whether a draft model
    is worth its memory, and which wall time cannot see at all.
    """
    payload, src = payload_for(workload, depth_tokens, gen_tokens, model)
    if extra:
        payload.update(extra)
    t0 = time.time()
    r = post(url, "/v1/chat/completions", payload, timeout)
    wall = time.time() - t0
    tm = r.get("timings") or {}
    u = r.get("usage") or {}
    if not tm:
        # Recorded as a gap, never silently replaced by a wall-clock figure —
        # that substitution is the whole reason this function exists.
        return {"error": "server returned no timings; wall %.1fs" % wall,
                "workload": workload, "wall_s": round(wall, 2)}
    da, dn = tm.get("draft_n_accepted"), tm.get("draft_n")
    # `prompt_n` is what the server PROCESSED, not how deep the request sat:
    # a reused prefix is in `cache_n` and in neither is it in the other. The
    # depth is the sum, and reporting prompt_n as the depth — which this file
    # did until 27.08. — understates it by exactly the cache hit. It showed up
    # as two cells both labelled "2280 tokens" that were in fact 9,358 and
    # 36,658 deep, which is the one thing a per-depth measurement must not get
    # wrong.
    pn, cn = tm.get("prompt_n") or u.get("prompt_tokens") or 0, tm.get("cache_n") or 0
    out = {
        "workload": workload,
        "depth_n": pn + cn,
        "prompt_n": pn,
        "cache_n": cn,
        "cached_pct": round(100.0 * cn / (pn + cn), 1) if (pn + cn) else None,
        "pp_tps": round(tm["prompt_per_second"], 1) if tm.get("prompt_per_second") else None,
        "predicted_n": tm.get("predicted_n") or u.get("completion_tokens"),
        "tg_tps": round(tm["predicted_per_second"], 1) if tm.get("predicted_per_second") else None,
        "draft_n": dn, "draft_accepted": da,
        "draft_accept_pct": round(100.0 * da / dn, 1) if dn else None,
        "wall_s": round(wall, 2),
    }
    if workload in ("copy", "prose"):
        content, thinking = answer_text(r)
        against = src if workload == "copy" else payload["messages"][0]["content"]
        out["copied_pct"] = round(100.0 * copied_fraction(content, against), 1)
        empty = "; the model answered in the thinking channel" if (
            thinking and not content.strip()) else ""
        if workload == "copy" and out["copied_pct"] < COPIED_MIN * 100:
            out["warning"] = (
                "only %.1f %% of the answer was copied from the block%s — this "
                "is a decode rate, but not a COPY-HEAVY one. Read it with the "
                "count cell, not instead of it." % (out["copied_pct"], empty))
        elif workload == "prose" and out["copied_pct"] > PARROT_MAX * 100:
            out["warning"] = (
                "%.1f %% of the answer already appears in the prompt%s — the "
                "model echoed its filler instead of writing novel text, so "
                "this is not the speculation FLOOR it is labelled as."
                % (out["copied_pct"], empty))
    return out


def probes(url, timeout=1800, ctx_project=None):
    """Cold prefill and warm decode, Claude-Code-shaped, WALL CLOCK.

    Kept for the cache question — how much of a real agent prefix survives —
    where the shape of the body is the point and the rate is secondary. Do not
    read its `tps` as a decode rate: it is tokens over TOTAL request time and
    therefore includes whatever prefill the cache missed. `rates()` is the
    function that answers "how fast".
    """
    out = {}
    project = ctx_project or "/tmp/bench-probe-%d" % int(time.time())

    p = _system_mid_conversation_remap(request_body(
        project=project, n_tools=24, question="Say alpha.", max_tokens=1))
    t0 = time.time()
    r = post(url, "/v1/messages", p, timeout)
    m = evaluate(r, time.time() - t0)
    out["prefill_cold"] = dict(m, tps=round(m["new"] / m["seconds"], 1)
                               if m.get("seconds") else None)

    p = _system_mid_conversation_remap(request_body(
        project=project, n_tools=24, max_tokens=256,
        question="Count from 1 to 80, one number per line."))
    t0 = time.time()
    r = post(url, "/v1/messages", p, timeout)
    secs = time.time() - t0
    outtok = (r.get("usage") or {}).get("output_tokens", 0)
    m = evaluate(r, secs)
    out["decode_warm"] = {"out": outtok, "seconds": round(secs, 2),
                          "cached_pct": m["rate"],
                          "tps": round(outtok / secs, 1) if secs > 0 else None}
    return out


def ctx_of(argv):
    """The window this server was started with, so a comparison can be read.

    A t/s number without its context size is not comparable to another one:
    prefill and decode both change as the window grows, which is the whole
    point of measuring per window.
    """
    for short, long in (("-c", "--ctx-size"),):
        for n in (short, long):
            if n in argv:
                i = argv.index(n)
                if i + 1 < len(argv):
                    try:
                        return int(argv[i + 1])
                    except ValueError:
                        return None
    return None


DEPTHS = (512, 8192, 32768)

# Three, not one, and the default is three because ONE IS MEASURABLY NOT
# ENOUGH. Measured 27.08. on the live qwen38 server, the same count cell at
# depth 586, two runs six minutes apart: 44.6 t/s and 135.1 t/s. Draft
# acceptance went 79.9 % -> 100 % between them, which is the registered defect
# `spec-decoding-unrepeatable` showing up inside the instrument that is
# supposed to measure it. The deep cells reproduced to 0.1 %; the shallow ones
# did not reproduce at all.
#
# So a single run of this file decides nothing at shallow depth, and a median
# that hides that spread would be the same class of lie as a wall-clock decode
# rate. Both are recorded: the median is the number, tg_min/tg_max is whether
# to believe it.
REPS = 3


def median(xs):
    s = sorted(x for x in xs if x is not None)
    if not s:
        return None
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else round((s[mid - 1] + s[mid]) / 2.0, 1)


def merge_reps(cells):
    """One cell out of several runs: medians, plus the spread that earned them.

    The first successful run supplies everything that is not a rate — depth,
    cache share, copied share — because those do not vary in a way a median
    would describe. Errors are kept only if EVERY run failed; one bad request
    in three is not a dead cell.
    """
    good = [c for c in cells if not c.get("error")]
    if not good:
        return dict(cells[0], reps=len(cells))
    out = dict(good[0])
    out["reps"] = len(cells)
    out["reps_ok"] = len(good)
    # DECODE is medianed; PREFILL is not, and the difference is not a
    # preference. Repeating a prefill does not measure the prefill again — the
    # prefix is in the cache by then, so runs 2..n process almost nothing and
    # time almost nothing. A median over those is a cache measurement wearing
    # a prefill label. It showed up immediately: the first version of this
    # function printed a row reading "0 % cached, pp 19.8 t/s", which cannot
    # both be true. So pp and the cache share it qualifies BOTH come from the
    # first successful run, and they stay consistent with each other.
    for key in ("tg_tps", "draft_accept_pct", "copied_pct"):
        out[key] = median([c.get(key) for c in good])
    tgs = [c.get("tg_tps") for c in good if c.get("tg_tps") is not None]
    if tgs:
        out["tg_min"], out["tg_max"] = min(tgs), max(tgs)
        # A cell whose runs disagree by more than half is not one measurement
        # with noise on it; it is two different things sharing a label.
        if out["tg_min"] and out["tg_max"] / out["tg_min"] > 1.5:
            out["spread_warning"] = (
                "decode ranged %.1f-%.1f t/s over %d runs — the median is not "
                "a number to compare against anything"
                % (out["tg_min"], out["tg_max"], len(tgs)))
    return out


def run(url, label, out_dir, argv=None, timeout=1800, depths=DEPTHS,
        workloads=WORKLOADS, reps=REPS, echo=print):
    """Measure one configuration at several DEPTHS and both WORKLOADS.

    Several depths and not one, because prefill and decode both move as the
    window fills, and a single rate quoted without its position is how "25x
    decode collapse" gets reported and then withdrawn. The depth that comes
    back is the one the SERVER counted (`prompt_n`), not the one we asked for.

    Both workloads and not one, because a drafter that cannot help with the
    probe's output is indistinguishable from a drafter that does not work —
    see the module docstring. The count cell is run first at each depth so the
    copy cell finds the filler prefix warm, which is the case a real agent is
    in and which keeps the two cells' prefill out of the comparison.
    """
    os.makedirs(out_dir, exist_ok=True)
    result = {"label": label, "url": url,
              "started": time.strftime("%Y-%m-%d %H:%M:%S"),
              "gtt_gib": gtt_gib(), "ctx": ctx_of(argv or []),
              "argv": list(argv or []), "depths": []}
    for d in depths:
        echo("  depth ~%d ..." % d)
        for w in workloads:
            runs = []
            for _ in range(max(1, reps)):
                try:
                    runs.append(rates(url, d, timeout=timeout, workload=w))
                except Exception as e:
                    # Recorded, not raised: one dead run must not cost the rest.
                    runs.append({"error": str(e)[:200], "workload": w})
            r = merge_reps(runs)
            r["asked"] = d
            result["depths"].append(r)
            if r.get("error"):
                echo("    %-5s FAILED: %s" % (w, str(r["error"])[:100]))
                continue
            echo("    %-5s at %s tokens (%s%% cached): pp %s t/s · tg %s t/s%s%s"
                 % (w, r.get("depth_n"), r.get("cached_pct"),
                    r.get("pp_tps"), r.get("tg_tps"),
                    "" if r.get("draft_accept_pct") is None
                    else " · draft accepted %s %%" % r["draft_accept_pct"],
                    "" if r.get("copied_pct") is None
                    else " · copied %s %%" % r["copied_pct"]))
            for key in ("warning", "spread_warning"):
                if r.get(key):
                    echo("          ! %s" % r[key])
    echo("  GTT %s GiB" % result["gtt_gib"])
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--label", default="live")
    ap.add_argument("--out", default=None, help="directory for summary.json")
    ap.add_argument("--depths", default=",".join(str(d) for d in DEPTHS),
                    help="comma-separated KV depths to measure at")
    ap.add_argument("--reps", type=int, default=REPS,
                    help="runs per cell; the median is reported with its "
                         "spread. 1 is quick and decides nothing at shallow "
                         "depth — see REPS in this file")
    ap.add_argument("--workloads", default=",".join(WORKLOADS),
                    help="count (output the prompt does not contain) and/or "
                         "copy (output it does). Both, unless you know why not")
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, "reports",
                                time.strftime("%Y-%m-%d_%H%M_speed_") + a.label)
    wl = [w.strip() for w in a.workloads.split(",") if w.strip()]
    bad = [w for w in wl if w not in WORKLOADS]
    if bad:
        raise SystemExit("unknown workload(s): %s — known: %s"
                         % (", ".join(bad), ", ".join(WORKLOADS)))
    r = run(a.url, a.label, out, depths=[int(x) for x in a.depths.split(",")],
            workloads=wl, reps=a.reps)
    print("report: %s" % out)
    return 0 if any(not d.get("error") for d in r["depths"]) else 1


if __name__ == "__main__":
    sys.exit(main())
