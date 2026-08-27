#!/usr/bin/env python3
"""depth-correctness — is the backend still RIGHT as the window fills?

bench/suites/depth-curve.py measures what depth costs. It says nothing about
whether the answer is still correct, and on this hardware that is not a
theoretical worry: llama.cpp #27579 reports gfx1151/ROCm producing corrupted
output where Vulkan with identical weights and flags does not, and describes
Qwen3.8 specifically as "correct at shallow depths but fails past ~29k tokens,
confabulating tool definitions".

qwen38 runs here with -c 204800, and Claude Code's tool head alone is ~43k. If
that report holds for this machine, the production backend is quietly wrong in
every deep session. That is worth measuring rather than believing either way.

Method
------
* The conversation grows APPEND-ONLY, the way depth-curve.py does it, so the
  cached prefix is reused and reaching 96k costs one prefill rather than six.
* Every `--every` tokens a block of filler is appended that contains exactly
  one ANCHOR line:  `ANCHOR-0007: 41926`. The number is derived from the
  index, so a rerun asks for the same values.
* A block of fake TOOL DEFINITIONS is planted early, because the upstream
  report names confabulated tool definitions as the symptom.
* At each depth the anchors are queried at three RELATIVE positions — the
  first one planted, one in the middle, the most recent — plus the tool
  question. An exact string match decides.
* Each question is asked `--repeat` times. Two different answers to the same
  question at temperature 0 over the same prefix is a stronger signal than a
  wrong one: a model that cannot find a needle is wrong CONSISTENTLY.

What a result means
-------------------
A failure here is not by itself proof of backend corruption — a needle in 96k
of filler is genuinely hard, and a model may simply lose it. The decisive
comparison is the SAME run against a different backend:

    # ROCm (production)
    python3 bench/suites/depth-correctness.py --label rocm
    # then start the server from build-vulkan with identical flags
    python3 bench/suites/depth-correctness.py --label vulkan

Identical prompts, identical seed. If ROCm degrades where Vulkan does not,
that is the upstream defect on this machine and the backend choice has to be
revisited. If both degrade the same way, it is the model.

Talks to the server DIRECTLY (8080), bypassing the gateway, so no admission
control or prefix bookkeeping distorts anything. The GPU must be otherwise
idle.
"""
import argparse, json, os, re, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
REPO = os.path.dirname(BENCH)
sys.path.insert(0, BENCH)
sys.path.insert(0, os.path.join(REPO, "tools"))
import sweep                                                  # noqa: E402

URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")
DEPTHS = [4000, 16000, 32000, 48000, 64000, 96000]

SYSTEM = ("You are a careful engineering assistant. Answer with the exact "
          "value asked for and nothing else — no preamble, no explanation.")

# Deliberately shaped like a tool block, because #27579 names confabulated
# tool definitions as the symptom. The names are nonsense on purpose: a model
# that answers from its training data instead of from the context gets them
# wrong, and that is a different failure worth telling apart.
TOOLS = """Available tools for this session:
  - grozzle_fetch(path, mode)      reads a grozzle record from the vault
  - kwintle_apply(id, delta)       applies a delta to a kwintle entry
  - vermble_check(token)           validates a vermble token
End of tool list."""


def post(payload, timeout=3600):
    r = urllib.request.Request(URL + "/v1/chat/completions",
                               data=json.dumps(payload).encode(),
                               headers={"content-type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as x:
        return json.loads(x.read().decode())


def tokens_of(text):
    r = urllib.request.Request(URL + "/tokenize",
                               data=json.dumps({"content": text}).encode(),
                               headers={"content-type": "application/json"})
    with urllib.request.urlopen(r, timeout=300) as x:
        return len(json.loads(x.read().decode())["tokens"])


def secret(index):
    """Deterministic five digits per anchor — a rerun asks the same thing."""
    return 10000 + (index * 7919) % 90000


def filler(index, words):
    """Same generator as depth-curve.py: seeded by index, so the same depth
    always produces the same text and two backends see identical prompts."""
    out = []
    for i in range(words):
        n = (index * 7919 + i * 104729) % 1000
        out.append("item%03d-%d" % (n, (i * 31 + index) % 97))
    return " ".join(out)


def block(index, words):
    """Filler with exactly one anchor buried in the middle of it."""
    half = words // 2
    return ("%s\nANCHOR-%04d: %d\n%s"
            % (filler(index, half), index, secret(index),
               filler(index + 500, words - half)))


def ask(messages, question, max_tokens=32, timeout=3600):
    body = {"model": "depth-correctness", "stream": False,
            "max_tokens": max_tokens, "seed": 7, "temperature": 0,
            "messages": messages + [{"role": "user", "content": question}]}
    t0 = time.time()
    resp = post(body, timeout)
    msg = (resp.get("choices") or [{}])[0].get("message") or {}
    tm = resp.get("timings") or {}
    return {"text": (msg.get("content") or "").strip(),
            "wall": round(time.time() - t0, 2),
            "prompt_n": tm.get("prompt_n"),
            "cached": (resp.get("usage") or {}).get("prompt_tokens")}


def check_anchor(answer, want):
    """The value must appear as a standalone number. Substring matching would
    accept '141926' for 41926 and call a wrong answer right."""
    return bool(re.search(r"(?<!\d)%d(?!\d)" % want, answer))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", help="comma-separated target depths")
    ap.add_argument("--every", type=int, default=4000,
                    help="plant an anchor every N tokens (default 4000)")
    ap.add_argument("--repeat", type=int, default=2,
                    help="ask each question N times (default 2)")
    ap.add_argument("--label", default="unlabelled",
                    help="backend or configuration under test")
    ap.add_argument("--out", help="report directory")
    a = ap.parse_args()
    depths = [int(x) for x in a.depths.split(",")] if a.depths else DEPTHS
    sweep.reexec_with_inhibit()

    stamp = time.strftime("%Y-%m-%d_%H%M")
    dest = a.out or os.path.join(BENCH, "reports",
                                 "%s_depth-correctness_%s" % (stamp, a.label))
    os.makedirs(dest, exist_ok=True)
    rows_path = os.path.join(dest, "rows.jsonl")
    rows = open(rows_path, "w", encoding="utf-8")

    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": TOOLS},
                {"role": "assistant", "content": "Noted."}]
    planted = []                       # anchor indexes, in order of planting
    depth = tokens_of(SYSTEM + TOOLS)
    index = 0

    # Calibrate words -> tokens instead of assuming. The filler words look like
    # "item003-42" and the tokenizer splits each into four or five pieces, so
    # the 1.6-tokens-per-word rule of thumb was out by a factor of four: a
    # block meant to be 2000 tokens came out at 8600 and the first two depths
    # collapsed into one measurement.
    probe_words = 200
    per_word = tokens_of(block(9999, probe_words)) / probe_words
    print("calibration: %.2f tokens per filler word" % per_word)
    print("depth-correctness · %s · %s" % (a.label, URL))
    print("planting an anchor every %d tokens, each question asked %dx"
          % (a.every, a.repeat))

    result = {"label": a.label, "url": URL, "stamp": stamp,
              "every": a.every, "repeat": a.repeat, "depths": []}

    for target in depths:
        # --- grow to the target, planting anchors on the way --------------
        while depth < target:
            words = max(40, int(a.every / per_word))
            text = block(index, words)
            messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": "Recorded."})
            planted.append(index)
            depth += tokens_of(text) + 4
            index += 1

        # --- three relative positions plus the tool question --------------
        picks = []
        if planted:
            picks = [("first", planted[0]),
                     ("middle", planted[len(planted) // 2]),
                     ("last", planted[-1])]
        questions = [("anchor:%s" % where,
                      "What is the value on the line ANCHOR-%04d? "
                      "Answer with the number only." % idx,
                      lambda ans, i=idx: check_anchor(ans, secret(i)),
                      str(secret(idx)))
                     for where, idx in picks]
        questions.append(
            ("tools",
             "List the names of the tools available in this session, "
             "comma separated, nothing else.",
             lambda ans: all(t in ans for t in
                             ("grozzle_fetch", "kwintle_apply", "vermble_check")),
             "grozzle_fetch, kwintle_apply, vermble_check"))

        entry = {"target": target, "depth": depth, "anchors": len(planted),
                 "questions": []}
        print("depth %6d (%d anchors)" % (depth, len(planted)))
        for name, q, ok_fn, want in questions:
            answers, oks = [], []
            for _ in range(a.repeat):
                r = ask(messages, q)
                answers.append(r["text"][:120])
                oks.append(bool(ok_fn(r["text"])))
                rows.write(json.dumps({"target": target, "depth": depth,
                                       "question": name, "want": want,
                                       "got": r["text"][:300], "ok": oks[-1],
                                       "wall": r["wall"],
                                       "prompt_n": r["prompt_n"]}) + "\n")
                rows.flush()
            stable = len(set(answers)) == 1
            entry["questions"].append({"name": name, "want": want,
                                       "ok": oks, "answers": answers,
                                       "stable": stable})
            mark = "ok  " if all(oks) else ("MIXED" if any(oks) else "WRONG")
            print("   %-14s %-5s %s%s"
                  % (name, mark, answers[0][:60],
                     "" if stable else "   <-- the repeats DISAGREE"))
        result["depths"].append(entry)

    rows.close()
    with open(os.path.join(dest, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)

    # --- the summary that decides something ------------------------------
    print("\n%-8s %-8s %-8s %-8s %-8s" % ("depth", "first", "middle", "last", "tools"))
    for e in result["depths"]:
        cells = {q["name"].split(":")[-1]: q for q in e["questions"]}
        def cell(k):
            q = cells.get(k)
            if not q:
                return "-"
            if not q["stable"]:
                return "UNSTABLE"
            return "ok" if all(q["ok"]) else "WRONG"
        print("%-8d %-8s %-8s %-8s %-8s"
              % (e["depth"], cell("first"), cell("middle"), cell("last"),
                 cell("tools")))
    print("\nreport: %s" % dest)
    print("A WRONG here is not yet proof of a backend defect — run the same "
          "thing\nagainst the other backend and compare. See the docstring.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
