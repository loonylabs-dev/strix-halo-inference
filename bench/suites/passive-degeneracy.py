#!/usr/bin/env python3
"""Can the degeneracy check move out of the probe and onto real traffic?

    python3 bench/suites/passive-degeneracy.py

No GPU, no server. The ground truth is already in this repo: every
slot-corruption run recorded its answers with a per-answer verdict, which is
316 genuinely corrupted and 383 healthy responses from this machine.

Why it matters: llama-probe costs the operator up to five minutes of waiting
per incident because it takes the one slot (see
bench/reports/2026-08-29_probe-cost). A check that reads answers the gateway
already has needs no slot at all — but only if it does not cry wolf, because
a watchdog that cries wolf is one nobody reads.

The second corpus is therefore hand-built: the shapes a character histogram is
most likely to trip over. They are not model output, they are the shapes model
output CONTAINS — a markdown rule, a table, base64, a progress bar, a list of
paths.
"""
import json, glob, collections
def load():
    dirty, clean = [], []
    for f in sorted(glob.glob("bench/reports/*slot-corruption*/*.json")):
        for r in json.load(open(f)).get("runs", []):
            for a in r.get("answers") or []:
                d = a if isinstance(a, dict) else (json.loads(a) if isinstance(a, str) else None)
                if not isinstance(d, dict): continue
                (dirty if str(d.get("verdict")).upper()=="CORRUPT" else clean).append(str(d.get("text","")))
    return dirty, clean

def variant(text, min_len=24, share=0.9, symbol_only=False):
    body = "".join(text.split())
    if len(body) < min_len: return False
    char, n = collections.Counter(body).most_common(1)[0]
    if n/len(body) < share: return False
    if symbol_only and char.isalnum(): return False
    return True

dirty, clean = load()
HARD = {
  "Trennlinie in Prosa": "Hier eine Antwort.\n\n" + "-"*80 + "\n\nUnd weiter im Text.",
  "reine Trennlinie":    "-"*90,
  "ASCII-Tabelle":       "| a | b |\n" + "|---|---|\n"*12,
  "base64":              "A"*64,
  "Fortschrittsbalken":  "[" + "="*70 + "] 100%",
  "Pfadliste":           "\n".join("/usr/lib/x%d/y/z" % i for i in range(12)),
  "lange Prosa":         "Der Cache hält den Prefix im Slot. " * 12,
  "JSON mit Nullen":     json.dumps({"v": [0]*60}),
}
print("  %-34s %9s %9s  %s" % ("Variante","gefunden","Fehlalarm","fällt herein auf"))
for name, kw in (("heute:      min 24, 60 %", dict(share=0.6)),
                 ("strenger:   min 24, 90 %", dict(share=0.9)),
                 ("+ nur Symbole (kein a-z0-9)", dict(share=0.9, symbol_only=True))):
    hit = sum(1 for t in dirty if variant(t, **kw))
    fp  = sum(1 for t in clean if variant(t, **kw))
    hard= [k for k,v in HARD.items() if variant(v, **kw)]
    print("  %-34s %8.1f%% %8.1f%%  %s" % (name, 100*hit/len(dirty), 100*fp/len(clean),
                                           ", ".join(hard) or "nichts"))

# wie viele verdorbene Antworten kommen am Stück? Entscheidet, ob "N in Folge" trägt.
runs=[]
for f in sorted(glob.glob("bench/reports/*slot-corruption*/*.json")):
    for r in json.load(open(f)).get("runs", []):
        seq=[]
        for a in r.get("answers") or []:
            d = a if isinstance(a, dict) else (json.loads(a) if isinstance(a,str) else None)
            if isinstance(d, dict): seq.append(str(d.get("verdict")).upper()=="CORRUPT")
        if any(seq): runs.append(seq)
streaks=[max((sum(1 for _ in g) for k,g in __import__("itertools").groupby(s) if k), default=0) for s in runs]
print("\n  Läufe mit mindestens einer verdorbenen Antwort: %d" % len(runs))
print("  längste Kette verdorbener Antworten je Lauf: Median %d, Minimum %d"
      % (sorted(streaks)[len(streaks)//2], min(streaks)))
