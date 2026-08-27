#!/usr/bin/env python3
"""determinism — does the same question give the same answer twice?

It does not, and that was found the expensive way: on 26.08. the task
`hard-count-numbers` failed once in a battery run and then passed three times,
at 161 / 243 / 284 seconds, writing 4386 to 6578 output tokens, with `seed: 7`
in every request. Three further runs at `temperature: 0` — where greedy
decoding is supposed to give byte-identical output for identical input —
produced 5700 / 8192 / 7212 tokens. So the variance is not in the sampler.

That matters beyond curiosity. A single run is one sample, and a
model decision was made on a one-task margin. Before a new model is compared
against qwen38 on that battery, it is worth knowing what a single run can
carry.

This suite asks the question directly instead of through a 250-second task:
one fixed prompt, temperature 0, a few hundred tokens, asked N times, compared
byte for byte. Three conditions, because each isolates one suspect:

    warm      N times in a row against the same warm slot
    cold      the slots are ERASED before each request, so every one is a
              fresh prefill — if warm and cold disagree, the KV/batch state
              is what changes the answer
    (spec)    run the whole thing again against a server started WITHOUT
              --spec-type, on a side port. That is the suspect the sampler
              elimination leaves: accepted drafts change the batch shape,
              a different batch shape changes the reduction order in the
              matmuls, that changes the logits in their last bits, and at
              greedy decoding one flipped tie is enough.

    python3 bench/suites/determinism.py --label prod-with-spec
    LLAMA_URL=http://127.0.0.1:8081 \\
        python3 bench/suites/determinism.py --label nospec --no-erase

`--no-erase` skips the cold condition, which is what you want on a side server
you started by hand for one purpose.

Talks to the server DIRECTLY, never through the gateway: the gateway's own
restore and admission logic would be a second variable.
"""
import argparse, hashlib, json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
REPO = os.path.dirname(BENCH)
sys.path.insert(0, BENCH)
sys.path.insert(0, os.path.join(REPO, "tools"))
import sweep                                                  # noqa: E402

URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")

SYSTEM = ("You are a careful engineering assistant. Answer precisely and "
          "completely.")
# Long enough that a prefill really happens and speculation has something to
# draft from, short enough that ten repetitions cost minutes rather than an
# hour. Deterministic content, so two runs on two servers see the same bytes.
CONTEXT = "\n".join(
    "Record %04d: component=%s status=%s latency_ms=%d"
    % (i, ["alpha", "beta", "gamma", "delta"][i % 4],
       ["ok", "degraded", "failed"][i % 3], (i * 37) % 900 + 12)
    for i in range(400))
# TWO questions, because the first version of this suite only asked the short
# one and concluded "reproducible" from ten identical 140-token answers. That
# conclusion was too small: the battery task that started all this writes 4386
# to 8192 tokens, and a divergence is one flipped tie that then CASCADES. A
# probe that cannot generate long cannot see the thing it is looking for.
QUESTIONS = {
    "short": ("From the records above: how many are 'failed', and what is the "
              "sum of their latency_ms? Then explain in three sentences how "
              "you counted, and list the first five failed record numbers."),
    "long": ("Go through the records above in order. For every record whose "
             "status is 'failed', write one line of the form "
             "'<number>: <component> <latency_ms>'. Do not summarise, do not "
             "stop early, and write nothing else."),
    # ~1449 tokens was still identical five times over, so the length has to
    # go up to where the battery task lives: 4386 to 8192 tokens. All 400
    # records, not one in three.
    "verylong": ("Go through ALL the records above in order, without "
                 "exception. For each one write a line of the form "
                 "'<number>: <component> <status> <latency_ms>'. All 400 of "
                 "them. Do not summarise, do not stop early, do not comment."),
}


def post(path, payload, timeout=1800):
    r = urllib.request.Request(URL + path, data=json.dumps(payload).encode(),
                               headers={"content-type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as x:
        return json.loads(x.read().decode())


def erase_slots():
    """Every slot back to nothing, so the next request is a real cold one."""
    try:
        slots = json.loads(urllib.request.urlopen(URL + "/slots", timeout=30)
                           .read().decode())
    except Exception:
        return 0
    n = 0
    for s in slots:
        try:
            post("/slots/%d?action=erase" % s["id"], {})
            n += 1
        except Exception:
            pass
    return n


def ask(max_tokens, question):
    body = {"model": "determinism", "stream": False, "max_tokens": max_tokens,
            "seed": 7, "temperature": 0,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": CONTEXT},
                         {"role": "user", "content": question}]}
    t0 = time.time()
    resp = post("/v1/chat/completions", body)
    msg = (resp.get("choices") or [{}])[0].get("message") or {}
    tm = resp.get("timings") or {}
    text = (msg.get("content") or "")
    return {"text": text,
            "sha": hashlib.sha256(text.encode()).hexdigest()[:12],
            "chars": len(text),
            "tokens": tm.get("predicted_n"),
            "prompt_n": tm.get("prompt_n"),
            "wall": round(time.time() - t0, 1)}


def first_divergence(a, b):
    """Where do two answers stop being the same? The position says which kind
    of difference it is: near zero means the very first token flipped, deep in
    means the two runs agreed for a long time and then split."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else -1


def condition(name, reps, max_tokens, erase, question):
    print("\n%s (%d reps%s)" % (name, reps, ", slots erased before each" if erase else ""))
    runs = []
    for i in range(reps):
        if erase:
            erase_slots()
        r = ask(max_tokens, question)
        runs.append(r)
        print("   %d  sha=%s  %5d chars  %4s tokens  prompt_n=%-6s %5.1f s"
              % (i + 1, r["sha"], r["chars"], r["tokens"], r["prompt_n"], r["wall"]))
    shas = {r["sha"] for r in runs}
    identical = len(shas) == 1
    div = None
    if not identical:
        div = min(first_divergence(runs[0]["text"], r["text"])
                  for r in runs[1:] if r["sha"] != runs[0]["sha"])
    print("   -> %s" % ("IDENTICAL" if identical else
                        "%d distinct answers, first divergence at character %d"
                        % (len(shas), div)))
    return {"name": name, "reps": reps, "identical": identical,
            "distinct": len(shas), "first_divergence": div,
            "runs": [{k: v for k, v in r.items() if k != "text"} for r in runs]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--question", choices=sorted(QUESTIONS), default="short",
                    help="short: a 140-token answer. long: thousands of "
                         "tokens, which is where the divergence lives.")
    ap.add_argument("--label", default="unlabelled")
    ap.add_argument("--no-erase", action="store_true",
                    help="skip the cold condition (leaves other sessions alone)")
    ap.add_argument("--out")
    a = ap.parse_args()
    sweep.reexec_with_inhibit()

    stamp = time.strftime("%Y-%m-%d_%H%M")
    dest = a.out or os.path.join(BENCH, "reports",
                                 "%s_determinism_%s" % (stamp, a.label))
    os.makedirs(dest, exist_ok=True)
    print("determinism · %s · %s · question=%s, max_tokens=%d"
          % (a.label, URL, a.question, a.tokens))

    q = QUESTIONS[a.question]
    result = {"label": a.label, "url": URL, "stamp": stamp, "question": a.question,
              "reps": a.reps, "tokens": a.tokens, "conditions": []}
    result["conditions"].append(
        condition("warm", a.reps, a.tokens, erase=False, question=q))
    if not a.no_erase:
        result["conditions"].append(
            condition("cold", a.reps, a.tokens, erase=True, question=q))

    with open(os.path.join(dest, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print("\nreport: %s" % dest)
    for c in result["conditions"]:
        print("  %-6s %s" % (c["name"],
                             "reproducible" if c["identical"]
                             else "NOT reproducible (%d distinct)" % c["distinct"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
