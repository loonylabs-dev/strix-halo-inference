#!/usr/bin/env python3
"""compare — turn the summaries of a sweep (or several report dirs) into one
markdown table a decision can be read from.

    python3 bench/compare.py bench/reports/2026-08-24_2300_sweep_qwen38
    python3 bench/compare.py <dir-a> <dir-b>          # cross-report

Columns, and why exactly these:
    ctx           the window the server was started with. Two rates measured
                  at different -c are not comparable and the table says so
    GTT GiB       what the configuration costs in pinned memory
    depth         KV depth the server COUNTED — `prompt_n + cache_n`, not the
                  one we asked for and not `prompt_n` alone, which is only
                  what was PROCESSED and understates the depth by the cache hit
    cached %      how much of that depth came from the prefix cache. It is
                  here because it qualifies the column beside it: a pp rate
                  measured on a 94 %-cached prompt is not a cold prefill rate
    pp t/s        prefill, from the server's clock
    tg prose      decode on novel text — the FLOOR, no drafter can help
    tg count      predictable output — ceiling for a trained draft head
    tg copy       output already in the prompt — ceiling for an n-gram
                  drafter, and the shape of most agent work
    draft acc. %  drafted tokens accepted, prose/count/copy

**One row per depth AND the three decode columns are the point.** A single tg
number is not a property of a configuration: it is a property of the
configuration and the workload together, and the columns routinely disagree
by a factor. A drafter that looks worthless in `count` and doubles `copy` is
the normal case for an n-gram drafter, and reading only one column is how this
repo concluded "the ngram drafters give nothing" from a probe that could not
have shown them working. `prose` is the floor the two ceilings stand on: the
spread between prose and copy is what speculation is worth here.

Rewritten 27.08. Until then this docstring described the columns of the model
battery, removed on 26.08. — pass rate, seconds-to-correct, output tokens —
and `render()` read `summary["probes"]`, which `speed.run()` had stopped
writing. Every sweep since produced a table of dashes, and the only reports it
could still render were the discredited wall-clock ones. A comparison tool
that silently renders nothing is worse than one that fails.
"""
import glob, json, os, sys


def _fmt(x, digits=1):
    if x is None:
        return "—"
    if isinstance(x, float):
        return ("%%.%df" % digits) % x
    return str(x)



def load_summaries(paths):
    """Every summary.json directly in, or one level under, the given dirs.
    Order: context.json's variant order when present, discovery order else."""
    out = []
    for p in paths:
        candidates = sorted(glob.glob(os.path.join(p, "*", "summary.json")))
        direct = os.path.join(p, "summary.json")
        if os.path.exists(direct):
            candidates.append(direct)
        order = None
        ctx = os.path.join(p, "context.json")
        if os.path.exists(ctx):
            with open(ctx, encoding="utf-8") as f:
                c = json.load(f)
            order = c.get("order", []) + [c.get("reference")]
        found = {}
        for c in candidates:
            with open(c, encoding="utf-8") as f:
                s = json.load(f)
            found[s.get("label") or os.path.basename(os.path.dirname(c))] = s
        if order:
            out += [found.pop(n) for n in order if n in found]
        out += list(found.values())
    return out


def by_depth(summary):
    """The `depths` list regrouped as {asked: {workload: cell}}, order kept."""
    grouped = {}
    for cell in summary.get("depths") or []:
        grouped.setdefault(cell.get("asked"), {})[cell.get("workload") or "count"] = cell
    return grouped


def conditions_note(paths):
    """The line above the table: under what the numbers below were measured.

    context.json has carried `platform_profile` since the first sweep, and
    nothing ever rendered it — so the one fact that decides whether two
    reports may be compared at all lived in a file nobody opens. A sweep run
    at 'quiet' produces a table that looks exactly like one run at
    'performance', a third of the power budget apart.

    Three states, and the third is not the second. A run whose profile HELD
    says so; one where it CHANGED is not comparable with anything, itself
    included; and an older report that never recorded the answer is UNKNOWN —
    which must not be rendered as "fine", because that would turn a gap in
    the record into a claim about it.
    """
    lines = []
    for p in paths:
        ctx = os.path.join(p, "context.json")
        if not os.path.exists(ctx):
            continue
        try:
            with open(ctx, encoding="utf-8") as f:
                c = json.load(f)
        except Exception:
            continue
        started = c.get("platform_profile")
        if started is None:
            continue                    # a machine without the interface
        if c.get("conditions_held") is False:
            lines.append("> **WARNING — conditions did not hold.** "
                         "`platform_profile` changed during this run: "
                         "`%s` -> `%s`. The cells are not comparable with "
                         "each other, and the table is not comparable with "
                         "any other report."
                         % (started, c.get("platform_profile_end")))
        elif c.get("conditions_held") is True:
            lines.append("> Measured at `platform_profile=%s`, verified "
                         "unchanged at the end of the run." % started)
        else:
            lines.append("> Measured at `platform_profile=%s`. Whether it "
                         "held is NOT recorded — this report predates the "
                         "check." % started)
    return "\n>\n".join(lines)


def render(*paths):
    summaries = load_summaries(paths)
    if not summaries:
        return "(no summary.json found under %s)" % ", ".join(paths)
    note = conditions_note(paths)
    return (note + "\n\n" + render_summaries(summaries)) if note \
        else render_summaries(summaries)


def render_summaries(summaries):
    """The table, from summaries already in memory.

    Split from render() on 27.08. so the shape of the output can be tested
    without a filesystem — the reason the previous shape mismatch went
    unnoticed for a day is that nothing exercised this function at all.
    """
    head = ("| variant | ctx | GTT GiB | depth | cached % | pp t/s | tg prose "
            "| tg count | tg copy | draft acc. % |\n|---|---|---|---|---|---|---|---|---|---|")
    lines = [head]
    fails, notes, legacy = [], [], []

    for s in summaries:
        label = s.get("label", "?")
        ctx, gtt = _fmt(s.get("ctx"), 0), _fmt(s.get("gtt_gib"))
        grouped = by_depth(s)

        if not grouped:
            # An older report, from before speed.py measured per depth. Its
            # rate is tokens over TOTAL request time and therefore contains
            # whatever prefill the cache missed — the exact mistake upstream
            # #27623 made and withdrew. It is rendered, because refusing to
            # show an old report helps nobody, and it is MARKED, because
            # putting it in the same column as a server-clock number without
            # a mark is how the two get compared.
            pr = s.get("probes") or {}
            if pr.get("error"):
                fails.append("- %s: %s" % (label, str(pr["error"])[:140]))
                lines.append("| %s | %s | %s | — | — | — | — | — | — | — |"
                             % (label, ctx, gtt))
                continue
            cold, warm = pr.get("prefill_cold") or {}, pr.get("decode_warm") or {}
            if not cold and not warm:
                lines.append("| %s | %s | %s | — | — | — | — | — | — | — |"
                             % (label, ctx, gtt))
                continue
            legacy.append(label)
            lines.append("| %s * | %s | %s | — | — | %s | — | %s | — | — |"
                         % (label, ctx, gtt, _fmt(cold.get("tps")),
                            _fmt(warm.get("tps"))))
            continue

        first = True
        for asked in grouped:
            cells = grouped[asked]
            pro = cells.get("prose") or {}
            cnt, cpy = cells.get("count") or {}, cells.get("copy") or {}
            for w, c in (("prose", pro), ("count", cnt), ("copy", cpy)):
                if c.get("error"):
                    fails.append("- %s, depth %s, %s: %s"
                                 % (label, asked, w, str(c["error"])[:120]))
                for key in ("warning", "spread_warning"):
                    if c.get(key):
                        notes.append("- %s, depth %s, %s: %s"
                                     % (label, asked, w, c[key]))
            shown = pro or cnt or cpy
            drafted = "/".join(_fmt(c.get("draft_accept_pct"))
                               for c in (pro, cnt, cpy))
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                label if first else "", ctx if first else "", gtt if first else "",
                _fmt(shown.get("depth_n") or shown.get("prompt_n"), 0),
                _fmt(shown.get("cached_pct"), 0), _fmt(shown.get("pp_tps")),
                _fmt(pro.get("tg_tps")), _fmt(cnt.get("tg_tps")),
                _fmt(cpy.get("tg_tps")),
                "—" if drafted.strip("—/") == "" else drafted))
            first = False

    md = "\n".join(lines)
    if legacy:
        md += ("\n\n\\* wall clock, not the server's — it contains the prefill "
               "and is NOT comparable with the rows above (%s). Re-measure "
               "with bench/speed.py." % ", ".join(legacy))
    if notes:
        md += "\n\nNumbers that do not mean what the column says:\n" + "\n".join(notes)
    if fails:
        md += "\n\nCells with no measurement (a gap is not a zero):\n" + "\n".join(fails)
    return md + "\n"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    print(render(*sys.argv[1:]))
