"""Fängt der RAM-Cache eine ÜBERNAHME auf — so wie die Probe sie macht?

Der Quelltext sagt ja: bei Auswahl per LRU rettet llama.cpp den weichenden
Prompt (server-context.cpp:1631) und lädt danach den besten passenden. Löschen
tut das nicht — das war der vorige Versuch. Hier also die echte Form:

    A1, A2      ein Gespräch aufbauen
    B           ein FREMDER Prefix nimmt den Slot   (was die Probe tut)
    A3          das Gespräch unverändert fortsetzen

Bleibt die Wiederverwendung hoch, ist die Verdrängung durch die Probe
folgenlos, solange der Cache den Zustand hält — und der Defekt ist kleiner,
als er im Register steht. Bricht sie ein, war meine Korrektur falsch.
"""
import json, sys, time, urllib.request
sys.path.insert(0, "tools")
import synthetic as SYN
A = "/tmp/takeover-a-%d" % int(time.time())
B = "/tmp/takeover-b-%d" % int(time.time())

def send(project, msgs, tag, tools=6):
    b = SYN.body(project=project, n_tools=tools, question="egal")
    b["messages"] = msgs
    b["model"]="qwen38-low"; b["max_tokens"]=8; b["stream"]=False
    r = urllib.request.Request("http://127.0.0.1:8090/v1/messages", data=json.dumps(b).encode(),
        headers={"Content-Type":"application/json","anthropic-version":"2023-06-01"})
    t0=time.time()
    with urllib.request.urlopen(r, timeout=900) as x: d=json.loads(x.read())
    u=d.get("usage",{})
    print("  %-30s %5.1f s  reused=%-6s computed=%-6s" % (tag, time.time()-t0,
          u.get("cache_read_input_tokens"), u.get("input_tokens")), flush=True)

m1=[{"role":"user","content":"Sag nur: eins."}]
m2=m1+[{"role":"assistant","content":"eins"},{"role":"user","content":"Sag nur: zwei."}]
m3=m2+[{"role":"assistant","content":"zwei"},{"role":"user","content":"Sag nur: drei."}]
send(A,m1,"A1 (kalt)")
send(A,m2,"A2 (angehängt, warm)")
send(B,[{"role":"user","content":"Sag nur: fremd."}],"B  (fremder Prefix nimmt den Slot)")
send(A,m3,"A3 (Gespräch geht weiter)")
