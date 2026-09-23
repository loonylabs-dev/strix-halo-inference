#!/usr/bin/env python3
"""sessionload — the load a served log cannot show: several deep sessions at once.

    python3 bench/sessionload.py --label 0.12.3
    python3 bench/sessionload.py --label 0.8.1 --sessions 3 --turns 6 --doc-tokens 15000

WHY THIS EXISTS. bench/logstats.py reads what the operator actually ran, and on
21.09.2026 that turned out to be 90% turns under 2,000 tokens: the flat band was
already as tight as it can get (p50 1.45 s, p99 1.54 s on 0.8.1) while p99 in the
bands above it read 17.8 s, 44.3 s and 240.5 s. No upgrade can be judged on the
flat band, and the deep concurrent case — the one every KV-pool and prompt-cache
entry in setup/defects.json is about — had n=11 to n=250 of incidental traffic.
That case has to be RUN to be measured. This runs it.

WHAT IT DOES. Each session owns a document that stays byte-identical across the
whole run, and asks a new question about it every turn while its history grows.
That is the exact shape the cache fixes of 0.11.0 / 0.11.5 / 0.11.7 / 0.11.8 and
0.12.1's third save point address:

  * the document is a long prefix that SHOULD be served from the cache from turn
    two onwards. Re-prefilling it is the defect, and it is visible as prefill
    seconds and as the `cached` share on the server's own finish line.
  * several sessions taking turns is what made them evict each other (issue #74,
    and this repo's "ping pong" that bought the 768k pool on 12.09.2026).
  * a growing history is what makes a region need to grow, which is what 0.11.5
    and 0.11.7 rebuilt.

HOW TO COMPARE TWO RELEASES. Run it, restart the server onto the other image,
run it again with the same --salt (the default is fixed, so two runs build
identical prompts) and the same --sessions/--turns/--doc-tokens. Then read both
reports. The prompts are identical, the question order is identical, and the
only difference is the server.

    A restart before each run is part of the method, not hygiene: a warm prompt
    cache from a previous run turns turn one into a cache hit and the first
    number of the run is then not a cold prefill. The runner refuses to start
    when /cache reports entries, unless --allow-warm says the operator means it.

WHAT IT REFUSES TO CONCLUDE. It reports its own client-side latencies AND the
server's finish lines, because the two answer different questions and disagreeing
is informative (a client-side wait that the server does not account for is queue
time). It reads platform_profile at both ends, like bench/sweep.py, and says so
when it moved — this machine's profile went from performance to quiet unobserved
on 03.09.2026 and the GPU served eight hours at half power (global
env-machine.md). A run whose conditions did not hold is reported as invalid
rather than quietly averaged in.

It is NOT a throughput benchmark. It does not try to saturate the server or find
a maximum rate; it reproduces one shape and reports what that shape cost.
"""

import argparse
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

URL = "http://127.0.0.1:8080"

# 24 words, as in the long-context sweep on the model's Hugging Face thread:
# enough vocabulary that lines do not repeat, few enough that tokenization stays
# dense. Real prose was rejected there for a reason worth repeating — natural
# repetition compresses in the KV cache and fakes a faster prefill.
WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
         "nu xi omicron pi rho sigma tau upsilon phi chi psi omega").split()

# Questions asked in order, one per turn. Each is short and each asks for a
# different part of the document, so a turn cannot be answered from the previous
# answer — but none of them changes the document, which is the prefix under test.
QUESTIONS = [
    "How many record lines are in the text above? Answer with a number only.",
    "What is the value on the line whose id ends in 7? Answer with the number only.",
    "Name the three words that appear on the first record line, in order.",
    "Which record index is the largest in the text? Number only.",
    "What is the first word of the last record line?",
    "How many distinct words did you see across the first ten lines? Number only.",
    "Quote the shortest record line verbatim.",
    "What is the sum of the first two record indices? Number only.",
]


def platform_profile():
    """Same reader as bench/sweep.py, same reason. See close_conditions()."""
    try:
        with open("/sys/firmware/acpi/platform_profile") as f:
            return f.read().strip()
    except Exception:
        return None


def build_document(salt, session, target_tokens):
    """A document of unique, non-compressible record lines.

    Deterministic in (salt, session): two runs with the same salt build the
    same bytes, which is what makes an A/B between two images an A/B of the
    images. `target_tokens` is approximate — 2.05 chars/token is the constant
    the HF sweep measured for this shape — and the REAL size is read back from
    the server's prompt_tokens, never assumed from here.
    """
    rnd = random.Random(f"{salt}/{session}")
    lines, chars, target_chars = [], 0, int(target_tokens * 2.05)
    i = 0
    while chars < target_chars:
        n = rnd.randint(8, 14)
        words = " ".join(rnd.choice(WORDS) for _ in range(n))
        line = (f"[record {i:07d} run {salt} id {rnd.randint(10**7, 10**8 - 1)}] "
                f"{words} value={rnd.random():.9f}")
        lines.append(line)
        chars += len(line) + 1
        i += 1
    return "\n".join(lines)


def post(url, payload, timeout):
    """One chat completion, non-streaming, with the server's own timings.

    Non-streaming on purpose: TTFT is not what this measures (the server's
    finish line carries prefill separately and more accurately), and a
    streaming read adds a second failure mode — a stalled stream — to a run
    whose subject is latency under contention. 0.10.2 is also the release that
    made a non-streaming request cancellable, so this exercises that path.
    """
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    wall = time.perf_counter() - t0
    return json.loads(raw), wall


def cache_state(timeout=5):
    try:
        with urllib.request.urlopen(URL + "/cache", timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def health(timeout=5):
    try:
        with urllib.request.urlopen(URL + "/health", timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"the server at {URL} does not answer /health: {e}",
              file=sys.stderr)
        return None


def run_session(session, doc, turns, max_tokens, timeout, results, lock,
                barrier=None):
    """One conversation: the document in the first user message, a new question
    each turn, the assistant's answers kept in the history.

    The document stays in message 1 and the question goes in the LAST message,
    which is the shape 0.12.1's third save point was added for — before it, a
    new question behind an unchanged document matched no save point and every
    turn re-read the document (the maintainer's own explanation of the 7/10 on
    the HF thread). So this shape is also the one that separates 0.12.1 from
    everything before it.
    """
    history = [{"role": "user", "content": doc + "\n\n" + QUESTIONS[0]}]
    for t in range(turns):
        if t:
            history.append({"role": "user",
                            "content": QUESTIONS[t % len(QUESTIONS)]})
        if barrier is not None:
            barrier.wait()          # all sessions fire their turn together
        try:
            resp, wall = post(URL, {
                "model": "halogen-qwen3.8-flash-next",
                "messages": history,
                "max_tokens": max_tokens,
                "temperature": 0,           # greedy: no sampling variance
                "stream": False,
                # THINKING OFF, and this is load-bearing for an A/B across
                # releases. 0.11.0 added the answer room: at max_tokens 256 it
                # keeps 1024 for the answer and cuts the think budget to one
                # token, where 0.8.1 would reason until the cap and return an
                # empty content with finish_reason length. Leaving thinking on
                # would make the two releases do different WORK, and this run
                # is about prefill and cache placement.
                "enable_thinking": False,
            }, timeout)
        except Exception as e:
            with lock:
                results.append({"session": session, "turn": t, "error": repr(e)})
            return
        choice = (resp.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        usage = resp.get("usage") or {}
        tim = resp.get("timings") or {}
        history.append({"role": "assistant",
                        "content": msg.get("content") or ""})
        with lock:
            results.append({
                "session": session, "turn": t, "wall_s": round(wall, 3),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "finish_reason": choice.get("finish_reason"),
                # timings is llama.cpp-shaped and release-dependent; whatever
                # is absent stays absent rather than being defaulted to 0.
                "prompt_ms": tim.get("prompt_ms"),
                "predicted_ms": tim.get("predicted_ms"),
                "cached_tokens": (usage.get("prompt_tokens_details") or {}).get(
                    "cached_tokens"),
            })


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", required=True,
                    help="what is being measured (e.g. the image tag)")
    ap.add_argument("--sessions", type=int, default=3,
                    help="concurrent conversations (default 3: what the 768k "
                         "pool was bought for on 12.09.2026)")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--doc-tokens", default="15000",
                    help="approximate document size per session. ONE value for "
                         "every session, or a comma-separated list, one per "
                         "session — the sessions that hurt are not the same "
                         "depth. The operator's own problem shape, recalled "
                         "21.09.2026, was two sessions at 180k-230k beside one "
                         "at 80k: `--doc-tokens 200000,190000,80000`, which "
                         "reserves 566k of a 524288 pool (108%%) and is "
                         "therefore the eviction case. A single value is the "
                         "easy case and will not reproduce it.")
    ap.add_argument("--long-answer", action="store_true",
                    help="replace the questions with one that forces a LONG "
                         "answer, so the run measures DECODE at depth instead "
                         "of prefill and cache placement. Turn 1 is still the "
                         "cold prefill; from turn 2 the prompt is cached and "
                         "what is left is the token rate at that KV size, "
                         "which is what 0.12.0's indexer work claims to "
                         "improve (upstream: 30.0 -> 35.2 tok/s serial at "
                         "262k, 43.5 -> 49.4 with the draft head). Use with a "
                         "max_tokens large enough to let it run.")
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="short answers on purpose: this measures PREFILL and "
                         "cache behaviour, and a long generation would bury "
                         "them in decode time")
    ap.add_argument("--mode", choices=("parallel", "pingpong"),
                    default="parallel",
                    help="parallel: every session fires each turn together "
                         "(pool contention). pingpong: strictly one at a time, "
                         "round robin (cache eviction between turns)")
    ap.add_argument("--salt", default="ab2609",
                    help="fixed by default so two runs build IDENTICAL prompts; "
                         "change it only to force a cold run on the same server")
    ap.add_argument("--timeout", type=int, default=900,
                    help="per-request timeout; the baseline's worst prefill was "
                         "240 s, so this is deliberately generous")
    ap.add_argument("--allow-warm", action="store_true",
                    help="run even when the prompt cache already holds entries")
    ap.add_argument("--out", help="report directory (default: bench/reports/...)")
    a = ap.parse_args()

    h = health()
    if not h:
        return 2
    if h.get("busy") or h.get("in_flight"):
        print("the server is serving something else; this run would measure "
              "contention with it rather than with itself", file=sys.stderr)
        return 2

    c0 = cache_state()
    entries = (c0 or {}).get("entries")
    if entries and not a.allow_warm:
        print(f"the prompt cache already holds {entries} entries: turn one "
              f"would be a cache hit and not a cold prefill. Restart the "
              f"server, or pass --allow-warm if that is what you mean.",
              file=sys.stderr)
        return 2

    context = {
        "label": a.label,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform_profile": platform_profile(),
        "mode": a.mode, "sessions": a.sessions, "turns": a.turns,
        "doc_tokens_requested": str(a.doc_tokens), "max_tokens": a.max_tokens,
        "salt": a.salt,
        "server": {k: h.get(k) for k in (
            "version", "context", "slot_ctx", "slots", "kv_pool_positions",
            "indexer_budget", "rope_scaling", "prompt_cache", "drafter_default",
            "thinking_answer_room", "chat_template")},
        "cache_before": c0,
    }
    print(f"=== sessionload: {a.label} ===")
    print(f"server {json.dumps(h.get('version'))}, pool "
          f"{h.get('kv_pool_positions')}, slots {h.get('slots')}, "
          f"slot_ctx {h.get('slot_ctx')}")
    print(f"{a.sessions} sessions x {a.turns} turns, mode {a.mode}, "
          f"max_tokens {a.max_tokens}, platform_profile "
          f"{context['platform_profile']}")

    if a.long_answer:
        # One question, repeated: every turn asks for the same long piece of
        # writing, so every turn generates about the same number of tokens and
        # the per-turn rates are comparable to each other. Asking about the
        # DOCUMENT rather than for free invention keeps the answer grounded in
        # the context that is under test — a model writing from nothing would
        # decode at the same rate whatever the KV size, which would measure
        # nothing.
        QUESTIONS[:] = [
            "Describe in detail, in at least 800 words of flowing prose, what "
            "kind of data the text above contains: the structure of a record "
            "line, which parts of it vary and which repeat, what the numeric "
            "fields look like, and what someone would need to know to parse "
            "it. Do not list, write continuous prose."
        ]
    sizes = [int(x) for x in str(a.doc_tokens).split(",")]
    if len(sizes) == 1:
        sizes = sizes * a.sessions
    if len(sizes) != a.sessions:
        print(f"--doc-tokens has {len(sizes)} values for {a.sessions} "
              f"sessions: give one value, or one per session",
              file=sys.stderr)
        return 2
    docs = [build_document(a.salt, s, sizes[s]) for s in range(a.sessions)]
    print(f"documents built: {[len(d) for d in docs]} chars "
          f"(~{sizes} tokens requested)")
    # The number that decides whether this run reproduces anything. A KV
    # reservation is prompt PLUS max_tokens, so a shallow run with a large
    # budget can still overbook the pool and a deep one with a small budget
    # may not. Printed rather than assumed, because the first version of this
    # tool defaulted to 8.7% of the pool and would have measured nothing.
    reserved = sum(s + a.max_tokens for s in sizes)
    pool = h.get("kv_pool_positions") or 0
    if pool:
        share = 100.0 * reserved / pool
        print(f"reservation ~{reserved} positions of {pool} = {share:.1f}% "
              f"of the pool" +
              ("  <-- OVERBOOKED, this is the eviction case"
               if share > 100 else
               "  (under 100%: the pool holds every session at once, so "
               "eviction is NOT under test)"))
        context["reservation_positions"] = reserved
        context["reservation_pool_share_pct"] = round(share, 1)
    ctx_limit = h.get("slot_ctx") or 0
    for s, size in enumerate(sizes):
        if ctx_limit and size + a.max_tokens > ctx_limit:
            print(f"  note: session {s} reserves {size + a.max_tokens} > "
                  f"slot_ctx {ctx_limit}: fit-to-room will clamp it if the "
                  f"gate is on, and the server refuses it if not")
    context["doc_tokens"] = sizes

    results, lock = [], threading.Lock()
    t_start = time.perf_counter()

    if a.mode == "parallel":
        barrier = threading.Barrier(a.sessions)
        threads = [threading.Thread(
            target=run_session,
            args=(s, docs[s], a.turns, a.max_tokens, a.timeout, results, lock,
                  barrier)) for s in range(a.sessions)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:
        # Strictly round robin: session 0 turn 0, session 1 turn 0, ... Each
        # session's next turn therefore lands after every other session has
        # taken a region, which is the eviction shape of issue #74.
        histories = [[{"role": "user",
                       "content": docs[s] + "\n\n" + QUESTIONS[0]}]
                     for s in range(a.sessions)]
        for t in range(a.turns):
            for s in range(a.sessions):
                if t:
                    histories[s].append(
                        {"role": "user",
                         "content": QUESTIONS[t % len(QUESTIONS)]})
                try:
                    resp, wall = post(URL, {
                        "model": "halogen-qwen3.8-flash-next",
                        "messages": histories[s],
                        "max_tokens": a.max_tokens,
                        "temperature": 0,
                        "stream": False,
                        "enable_thinking": False,   # see the note in run_session
                    }, a.timeout)
                except Exception as e:
                    results.append({"session": s, "turn": t, "error": repr(e)})
                    continue
                choice = (resp.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                usage = resp.get("usage") or {}
                tim = resp.get("timings") or {}
                histories[s].append({"role": "assistant",
                                     "content": msg.get("content") or ""})
                results.append({
                    "session": s, "turn": t, "wall_s": round(wall, 3),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "finish_reason": choice.get("finish_reason"),
                    "prompt_ms": tim.get("prompt_ms"),
                    "predicted_ms": tim.get("predicted_ms"),
                    "cached_tokens": (usage.get("prompt_tokens_details") or {}
                                      ).get("cached_tokens"),
                })
                print(f"  s{s} t{t}: {wall:6.2f}s  prompt "
                      f"{usage.get('prompt_tokens')}", flush=True)

    context["wall_total_s"] = round(time.perf_counter() - t_start, 1)
    context["platform_profile_end"] = platform_profile()
    context["conditions_held"] = (
        context["platform_profile"] == context["platform_profile_end"])
    context["cache_after"] = cache_state()

    ok = [r for r in results if "error" not in r]
    errs = [r for r in results if "error" in r]
    summary = {"n": len(ok), "errors": len(errs)}
    if ok:
        walls = sorted(r["wall_s"] for r in ok)
        summary["wall_s"] = {
            "p50": statistics.median(walls),
            "max": walls[-1],
            "min": walls[0],
        }
        # Turn one is the cold one by construction; every later turn SHOULD be
        # served from the cache. Reporting them together would average the
        # thing under test away.
        first = [r["wall_s"] for r in ok if r["turn"] == 0]
        later = sorted(r["wall_s"] for r in ok if r["turn"] > 0)
        summary["turn1_wall_s"] = first
        # Decode rate per turn, from the server's own timings, for the runs
        # that generate enough tokens to have one. Computed here rather than
        # read from a single field because predicted_ms covers the generation
        # and completion_tokens covers what it produced; a turn of two tokens
        # has no meaningful rate and is left out.
        rates = [1000.0 * r["completion_tokens"] / r["predicted_ms"]
                 for r in ok
                 if r.get("predicted_ms") and (r.get("completion_tokens") or 0) >= 64]
        if rates:
            rates.sort()
            summary["decode_tps"] = {
                "n": len(rates), "min": round(rates[0], 2),
                "p50": round(rates[len(rates) // 2], 2),
                "max": round(rates[-1], 2)}
            summary["completion_tokens"] = sorted(
                r["completion_tokens"] for r in ok if r.get("completion_tokens"))
        if later:
            summary["later_turns_wall_s"] = {
                "n": len(later), "p50": statistics.median(later),
                "p90": later[min(int(len(later) * 0.9), len(later) - 1)],
                "max": later[-1]}
        pts = [r["prompt_tokens"] for r in ok if r["prompt_tokens"]]
        if pts:
            summary["prompt_tokens"] = {"min": min(pts), "max": max(pts)}

    print(f"\nwall {context['wall_total_s']}s, {len(ok)} turns, "
          f"{len(errs)} errors")
    if ok and "later_turns_wall_s" in summary:
        l = summary["later_turns_wall_s"]
        print(f"turn 1 (cold): {summary['turn1_wall_s']}")
        print(f"turns 2+: n={l['n']} p50={l['p50']:.2f}s p90={l['p90']:.2f}s "
              f"max={l['max']:.2f}s")
    if not context["conditions_held"]:
        print("\n" + "!" * 78)
        print(f"WARNING: platform_profile changed DURING this run: "
              f"{context['platform_profile']} -> "
              f"{context['platform_profile_end']}")
        print("This run is NOT comparable with any other. Re-run it.")
        print("!" * 78)
    for r in errs[:5]:
        print(f"  ERROR s{r['session']} t{r['turn']}: {r['error'][:140]}")

    out = a.out or os.path.join(
        HERE, "reports", time.strftime("%Y-%m-%d_%H%M") +
        f"_sessionload_{a.label.replace('/', '-')}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "context.json"), "w") as f:
        json.dump(context, f, indent=2)
    with open(os.path.join(out, "turns.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nreport: {out}")
    print("Read the server's own finish lines for prefill and cached share:")
    print("  podman logs halogen 2>&1 | python3 bench/logstats.py --log - "
          f"--label '{a.label}'")
    return 0 if not errs and context["conditions_held"] else 1


if __name__ == "__main__":
    sys.exit(main())
