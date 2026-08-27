#!/usr/bin/env python3
"""depth-curve — what prefill and decode really cost as the window fills.

Every estimate in the docs about "slow at depth" was extrapolated from two
points. This measures the curve: at a series of fill levels, how fast are
new prompt tokens processed, and how fast does the model then generate?

Method, and why it is built this way:

  * The conversation GROWS append-only. Each step adds filler, so the cached
    prefix is reused and the measured prompt rate is the rate for the NEW
    tokens at that depth — which is exactly what a session pays per turn.
    A cold prefill is processed in ub-sized chunks anyway, so the same curve
    integrates to the cold cost of a prompt of that size.
  * Decode is measured TWICE per depth, because speculation makes the answer
    workload-dependent: once on free prose (spec cannot help — the floor)
    and once on counting (spec's best case — the ceiling). Real agent work
    lives between the two.
  * Numbers come from llama-server's own `timings`, not from the wall clock.
  * It talks to the server DIRECTLY (8080), bypassing the gateway, so no
    admission control or prefix bookkeeping distorts the picture.

The GPU must be otherwise idle — a second session sharing the slots halves
the rates and the curve becomes meaningless.

    python3 bench/suites/depth-curve.py
    python3 bench/suites/depth-curve.py --depths 2000,16000,64000 --tokens 64
"""
import argparse, json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
REPO = os.path.dirname(BENCH)
sys.path.insert(0, BENCH)
sys.path.insert(0, os.path.join(REPO, "tools"))
import sweep                                                  # noqa: E402

URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")
# Capped at 160k on purpose (operator decision 25.08.): beyond it the
# machine is measurably out of productive range — 35 t/s prefill and 4.6 t/s
# prose decode mean a single turn costs minutes. Measuring further would
# describe a mode nobody would work in. Raise it with --depths if a future
# model changes that.
DEPTHS = [2000, 8000, 16000, 32000, 64000, 96000, 128000, 160000]

SYSTEM = ("You are a careful engineering assistant. Answer exactly what is "
          "asked, without preamble.")

PROSE_Q = ("Write a short paragraph about why measuring a system beats "
           "guessing about it. Do not repeat yourself.")
COUNT_Q = "Count from 1 upwards, one number per line, nothing else."


def post(payload, timeout=3600):
    r = urllib.request.Request(URL + "/v1/chat/completions",
                               data=json.dumps(payload).encode(),
                               headers={"content-type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as x:
        return json.loads(x.read().decode())


def filler(index, words):
    """Deterministic, non-repetitive-ish filler. Seeded by index so the same
    depth always produces the same text — reruns stay comparable."""
    out = []
    for i in range(words):
        n = (index * 7919 + i * 104729) % 1000
        out.append("item%03d-%d" % (n, (i * 31 + index) % 97))
    return " ".join(out)


def ask(messages, question, max_tokens, timeout=3600):
    body = {"model": "depth-curve", "stream": False, "max_tokens": max_tokens,
            "seed": 7, "messages": messages + [{"role": "user",
                                                "content": question}]}
    t0 = time.time()
    resp = post(body, timeout)
    tm = resp.get("timings") or {}
    return {"wall": round(time.time() - t0, 2),
            "prompt_n": tm.get("prompt_n"),
            "pp_tps": round(tm["prompt_per_second"], 1)
                      if tm.get("prompt_per_second") else None,
            "predicted_n": tm.get("predicted_n"),
            "tg_tps": round(tm["predicted_per_second"], 1)
                      if tm.get("predicted_per_second") else None,
            "cache_n": (resp.get("usage") or {}).get("prompt_tokens")}


def tokens_of(text):
    r = urllib.request.Request(URL + "/tokenize",
                               data=json.dumps({"content": text}).encode(),
                               headers={"content-type": "application/json"})
    with urllib.request.urlopen(r, timeout=300) as x:
        return len(json.loads(x.read().decode())["tokens"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", help="comma-separated target depths")
    ap.add_argument("--tokens", type=int, default=128,
                    help="tokens to generate per decode measurement")
    ap.add_argument("--out", help="report directory")
    a = ap.parse_args()
    depths = ([int(x) for x in a.depths.split(",")] if a.depths else DEPTHS)
    sweep.reexec_with_inhibit()

    stamp = time.strftime("%Y-%m-%d_%H%M")
    dest = a.out or os.path.join(BENCH, "reports", "%s_depth-curve" % stamp)
    os.makedirs(dest, exist_ok=True)

    # One filler word is ~2 tokens with this tokenizer; measure instead of
    # assuming, so the depths are actually hit.
    probe = filler(0, 100)
    per_word = tokens_of(probe) / 100.0

    messages = [{"role": "system", "content": SYSTEM}]
    rows, reached = [], 0
    print("depth-curve · %d levels · %d tokens per decode · ~%.1f tok/word"
          % (len(depths), a.tokens, per_word))
    print("%9s %9s %10s %12s %12s" % ("depth", "new tok", "prefill", "decode prose", "decode count"))

    try:
      for i, target in enumerate(depths):
        need = target - reached
        if need > 200:
            words = max(1, int(need / per_word))
            messages.append({"role": "user", "content": filler(i, words)})
            messages.append({"role": "assistant", "content": "Noted."})

        pre = ask(messages, "Say alpha.", 1)
        depth = pre["cache_n"] or 0
        reached = depth
        prose = ask(messages, PROSE_Q, a.tokens)
        count = ask(messages, COUNT_Q, a.tokens)
        row = {"target": target, "depth": depth,
               "new_tokens": pre["prompt_n"], "prefill_tps": pre["pp_tps"],
               "decode_prose_tps": prose["tg_tps"],
               "decode_count_tps": count["tg_tps"],
               "prefill_seconds": pre["wall"]}
        rows.append(row)
        print("%9d %9s %10s %12s %12s"
              % (depth, row["new_tokens"], row["prefill_tps"],
                 row["decode_prose_tps"], row["decode_count_tps"]))
        with open(os.path.join(dest, "rows.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
    finally:
        write_report(dest, rows, a.tokens, stamp)


def write_report(dest, rows, tokens, stamp):
    """Render table and result from the levels that finished.

    Called in a finally, so a run stopped halfway keeps the levels it paid
    for — the expensive ones are the deep ones, and losing them to a Ctrl-C
    would mean measuring them twice.
    """
    md = ["| fill depth | new tokens | prefill t/s | decode prose t/s | decode repetitive t/s |",
          "|---|---|---|---|---|"]
    for r in rows:
        md.append("| %s | %s | %s | %s | %s |"
                  % ("{:,}".format(r["depth"]).replace(",", "."),
                     r["new_tokens"], r["prefill_tps"],
                     r["decode_prose_tps"], r["decode_count_tps"]))
    table = "\n".join(md)
    with open(os.path.join(dest, "tabelle.md"), "w") as f:
        f.write(table + "\n")
    with open(os.path.join(dest, "result.json"), "w") as f:
        json.dump({"stamp": stamp, "url": URL, "tokens": tokens,
                   "rows": rows}, f, indent=1)
    print("\n" + table)
    print("\nreport: %s" % dest)


if __name__ == "__main__":
    main()
