#!/usr/bin/env python3
"""Decode rate, same prompt and same length, with the noise made visible.

    python3 bench/suites/decode-rate.py <label>

SEVEN ROUNDS, NOT THREE, AND THAT IS THE POINT. Decode on this machine has a
19 % coefficient of variation -- seven identical rounds of one prompt on one
binary gave 15.4, 18.1, 19.7, 19.9, 21.4, 24.1 and 27.5 t/s. Two standard
errors is 4.6 t/s at n=3, so a three-round mean cannot resolve anything under
roughly 22 %.

Measured 30.08.2026, and it cost a wrong conclusion first: a three-round run
read 18.46 t/s against a 20.89 baseline and looked like a 12 % regression from
a llama.cpp rebuild. Seven rounds on the same binary read 20.89 -- the
baseline figure exactly. The regression was the instrument.

So this prints every rate, not just the mean. If the spread of one condition
overlaps the other, there is no finding, however far apart the means look.
A 2x difference here is real; a 10 % one is not measured by this tool.
"""
import json, sys, urllib.request

URL = "http://127.0.0.1:8080"
# Two shapes, because speculation helps code far more than prose and one
# number would hide that.
PROMPTS = {
    "prose": "Schreibe einen kurzen Absatz ueber den Herbst.",
    "code": "Schreibe eine Python-Funktion, die eine Liste von Zahlen sortiert "
            "und die Duplikate entfernt. Mit Docstring.",
}


def post(path, body, timeout=900):
    r = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())


label = sys.argv[1]
for kind, ask in PROMPTS.items():
    base = post("/apply-template", {"messages": [{"role": "user", "content": ask}],
                                    "add_generation_prompt": True})["prompt"]
    rates = []
    for i in range(7):
        d = post("/completion", {"prompt": base, "n_predict": 200,
                                 "cache_prompt": True, "seed": 1000 + i})
        t = d.get("timings") or {}
        rates.append((t.get("predicted_per_second"), t.get("predicted_n")))
    ok = [r for r, n in rates if r]
    print("%-14s %-6s n=%d  rates %s  ->  mean %.2f t/s"
          % (label, kind, len(ok), " ".join("%.1f" % r for r in ok),
             sum(ok) / len(ok) if ok else 0.0))
