#!/usr/bin/env python3
"""stopstring_ab — do the gateway's end-token stop STRINGS undo Halogen's
end-of-turn guard?

    python3 bench/stopstring_ab.py --url http://127.0.0.1:8080 --n 5 \
        --out bench/reports/<date>_stop-strings/runs.jsonl

The suspicion (docs of 25.09.2026, the empty turn at 09:16:37): Halogen 0.13.4
keeps an `<|im_end|>` the model writes INSIDE its think block as literal text
(upstream #84's fix). The gateway adds `<|im_end|>` and `<|endoftext|>` to
every request as stop strings (gateway.py, CONTAINER_STOPS), and Halogen's
text matcher runs over reasoning and answer alike — so the text the guard
kept would end the turn after all: finish "stop", no content, and timings 0
(upstream #106, the stop-string path's signature).

Two arms, sent straight to the Halogen server so nothing but the `stop` field
differs:

    gateway-stops   stop = ["<|im_end|>", "<|endoftext|>"], as the gateway sends
    no-stops        no stop field

The prompt asks the model to reason about a ChatML chat template, which makes
it write the end token in its reasoning (#84's quoting shape). Each run gets
its own nonce, because the server answers an exact repeat with its first
answer (bench/reports/2026-09-24_classifier/README.md). Arms alternate
ABBA so a drift in the machine does not land on one arm.

What decides it, per run: whether the reasoning carries the literal end token
(no-stops arm: the guard kept it) and whether a turn ended with no content and
zeroed timings (gateway-stops arm: the stop string fired on it).
"""
import argparse, json, os, sys, time, urllib.request, uuid

END_TOKENS = ["<|im_end|>", "<|endoftext|>"]
ARMS = {"gateway-stops": END_TOKENS, "no-stops": None}

PROMPT = (
    "Run {nonce}. Write the Jinja chat template for a ChatML model in the "
    "Qwen style, including tool calls. Before answering, reason step by step "
    "through exactly which special tokens open and close each turn, and write "
    "each of those tokens out literally while you reason.")


def post(url, body, timeout):
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def served_model(url):
    with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=30) as r:
        data = json.loads(r.read().decode("utf-8")).get("data") or []
    if len(data) != 1:
        sys.exit(f"expected exactly one served model at {url}, got {len(data)}")
    return data[0]["id"]


def one(url, model, arm, max_tokens, timeout):
    nonce = uuid.uuid4().hex[:12]
    body = {"model": model, "max_tokens": max_tokens, "stream": False,
            "messages": [{"role": "user", "content": PROMPT.format(nonce=nonce)}],
            # both spellings, as the gateway sends them (trace 25.09. 09:16:37)
            "enable_thinking": True, "reasoning_effort": "low",
            "chat_template_kwargs": {"enable_thinking": True,
                                     "reasoning_effort": "low"}}
    if ARMS[arm] is not None:
        body["stop"] = ARMS[arm]
    t0 = time.time()
    try:
        resp = post(url, body, timeout)
    except Exception as e:                      # a failed run is a row, not a crash
        return {"arm": arm, "nonce": nonce, "error": repr(e),
                "took_s": round(time.time() - t0, 2)}
    took = round(time.time() - t0, 2)
    ch = (resp.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    tm = resp.get("timings") or {}
    return {
        "arm": arm, "nonce": nonce, "took_s": took,
        "finish_reason": ch.get("finish_reason"),
        "content_chars": len(content),
        "tool_calls": len(msg.get("tool_calls") or []),
        "reasoning_chars": len(reasoning),
        "reasoning_has_end_token": any(t in reasoning for t in END_TOKENS),
        "content_has_end_token": any(t in content for t in END_TOKENS),
        # #106: a stop-string finish reports prompt_ms and predicted_ms as 0.
        "timings_zeroed": tm.get("prompt_ms") == 0 and tm.get("predicted_ms") == 0,
        "usage": resp.get("usage"), "timings": tm,
        "reasoning_tail": reasoning[-300:], "content_head": content[:300],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", required=True,
                    help="the Halogen server itself, not the gateway")
    ap.add_argument("--n", type=int, default=5, help="runs per arm")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", required=True, help="jsonl, appended")
    a = ap.parse_args()

    model = served_model(a.url)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    order = []
    for i in range(a.n):                        # ABBA
        pair = list(ARMS)
        order += pair if i % 2 == 0 else pair[::-1]
    rows = []
    with open(a.out, "a") as f:
        f.write(json.dumps({"start": time.strftime("%F %T"), "url": a.url,
                            "model": model, "n": a.n,
                            "max_tokens": a.max_tokens}) + "\n")
        for k, arm in enumerate(order, 1):
            row = one(a.url, model, arm, a.max_tokens, a.timeout)
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"[{k}/{len(order)}] {arm:14s} {row.get('finish_reason')!s:10s} "
                  f"content={row.get('content_chars')} "
                  f"reasoning_end_token={row.get('reasoning_has_end_token')} "
                  f"zeroed={row.get('timings_zeroed')} took={row['took_s']}s"
                  + (f" ERROR {row['error']}" if "error" in row else ""),
                  flush=True)

    s = summarize(rows)
    print("\narm             runs  empty  of-them-zeroed  end-token-in-reasoning  errors")
    for arm in ARMS:
        a_ = s[arm]
        print(f"{arm:15s} {a_['runs']:4d}  {a_['empty']:5d}  {a_['zeroed']:14d}  "
              f"{a_['kept']:22d}  {a_['errors']:6d}")


def summarize(rows):
    """Per arm: runs, EMPTY turns (no content, no tool call), how many of
    those carried zeroed timings, reasonings that kept the end token, errors.

    Empty and zeroed are counted apart. Up to Halogen 0.13.8 a stop-string
    finish zeroed the timings (#106) and this summary required both; 0.14.0
    reports the real figures, and on 26.09.2026 the combined column read 0
    while 6 of 6 stop-string turns were empty. The zeroed count stays as a
    column because it says which release produced the rows."""
    out = {}
    for arm in ARMS:
        rs = [r for r in rows if r["arm"] == arm]
        ok = [r for r in rs if "error" not in r]
        empty = [r for r in ok if r["content_chars"] == 0 and not r["tool_calls"]]
        out[arm] = {"runs": len(rs), "errors": len(rs) - len(ok),
                    "empty": len(empty),
                    "zeroed": sum(1 for r in empty if r["timings_zeroed"]),
                    "kept": sum(1 for r in ok if r["reasoning_has_end_token"])}
    return out


if __name__ == "__main__":
    main()
