"""Die andere Hälfte des Beweises: Historie steht, Zustand ist weg.

Turn 1 baut ein Gespräch. Turn 2 hängt an — warm, Historie unverändert.
Dann wird der Slot GELÖSCHT (nicht übernommen, also rettet llama.cpp den
Zustand auch nicht in den RAM-Cache). Turn 3 hängt wieder nur an:

    msgs_kept == msgs_prev   die Historie steht
    reused    <  prev_in     der Zustand ist trotzdem weg

Das muss in der Spalte ROT werden. Steht dort etwas anderes, ist die Logik falsch.
"""
import json, sys, time, urllib.request
sys.path.insert(0, "tools")
import synthetic as SYN
P = "/tmp/lostcheck-%d" % int(time.time())

def send(msgs, tag):
    b = SYN.body(project=P, n_tools=6, question="egal")
    b["messages"] = msgs
    b["model"]="qwen38-low"; b["max_tokens"]=8; b["stream"]=False
    r = urllib.request.Request("http://127.0.0.1:8090/v1/messages", data=json.dumps(b).encode(),
        headers={"Content-Type":"application/json","anthropic-version":"2023-06-01"})
    t0=time.time()
    with urllib.request.urlopen(r, timeout=900) as x: d=json.loads(x.read())
    u=d.get("usage",{})
    print("  %-26s %5.1f s  reused=%-6s computed=%-5s" % (tag, time.time()-t0,
          u.get("cache_read_input_tokens"), u.get("input_tokens")), flush=True)

m1=[{"role":"user","content":"Sag nur: eins."}]
m2=m1+[{"role":"assistant","content":"eins"},{"role":"user","content":"Sag nur: zwei."}]
m3=m2+[{"role":"assistant","content":"zwei"},{"role":"user","content":"Sag nur: drei."}]
send(m1,"Turn 1 (kalt)")
send(m2,"Turn 2 (angehängt)")
r=urllib.request.Request("http://127.0.0.1:8080/slots/0?action=erase", data=b"{}",
                         headers={"Content-Type":"application/json"})
urllib.request.urlopen(r, timeout=60).read()
print("  --- Slot gelöscht, Historie unverändert ---", flush=True)
send(m3,"Turn 3 (Zustand weg)")
