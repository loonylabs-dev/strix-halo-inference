"""Zwei Formen der Verdrängung, gegeneinander gemessen.

  ÄHNLICH   ein zweites Projekt (teilt den halben Prefix) -> Slot per LCP
            gewählt, llama.cpp sichert den alten Zustand NICHT
  FREMD     eine winzige, völlig andere Anfrage (die Form der Probe)
            -> Slot per LRU gewählt, der alte Zustand WIRD gesichert

Wenn der Unterschied so groß ist wie der Quelltext nahelegt, ist die Probe
harmlos und der Projektwechsel teuer — also genau andersherum als ich es
heute Abend in den Defekteintrag geschrieben habe.
"""
import json, sys, time, urllib.request
sys.path.insert(0, "tools")
import synthetic as SYN
GW="http://127.0.0.1:8090"; LL="http://127.0.0.1:8080"

def send(project, msgs, tag):
    b = SYN.body(project=project, n_tools=6, question="egal")
    b["messages"]=msgs; b["model"]="qwen38-low"; b["max_tokens"]=8; b["stream"]=False
    r=urllib.request.Request(GW+"/v1/messages", data=json.dumps(b).encode(),
      headers={"Content-Type":"application/json","anthropic-version":"2023-06-01"})
    t0=time.time()
    with urllib.request.urlopen(r,timeout=900) as x: d=json.loads(x.read())
    u=d.get("usage",{})
    print("  %-34s %5.1f s  reused=%-6s computed=%-6s" % (tag,time.time()-t0,
          u.get("cache_read_input_tokens"),u.get("input_tokens")), flush=True)
    return u

def probe_shaped():
    """Genau das, was setup/scripts/probe.py schickt: winzig und ohne Bezug."""
    body={"model":"probe","max_tokens":16,"stream":False,
          "messages":[{"role":"user","content":"Was ist 17 mal 23? Nur die Zahl."}]}
    r=urllib.request.Request(LL+"/v1/chat/completions", data=json.dumps(body).encode(),
      headers={"Content-Type":"application/json"})
    t0=time.time()
    with urllib.request.urlopen(r,timeout=300) as x: d=json.loads(x.read())
    tm=d.get("timings",{})
    print("  %-34s %5.1f s  cache_n=%-6s prompt_n=%-6s" % ("PROBE (winzig, am Gateway vorbei)",
          time.time()-t0, tm.get("cache_n"), tm.get("prompt_n")), flush=True)

A="/tmp/tk2-a-%d" % int(time.time())
m1=[{"role":"user","content":"Sag nur: eins."}]
m2=m1+[{"role":"assistant","content":"eins"},{"role":"user","content":"Sag nur: zwei."}]
m3=m2+[{"role":"assistant","content":"zwei"},{"role":"user","content":"Sag nur: drei."}]
send(A,m1,"A1 (kalt)")
send(A,m2,"A2 (warm)")
probe_shaped()
send(A,m3,"A3 nach der PROBE")
