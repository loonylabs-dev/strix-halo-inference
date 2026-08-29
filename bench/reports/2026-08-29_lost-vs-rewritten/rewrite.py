"""Drei Turns: anhängen, anhängen, dann die Historie ändern."""
import json, sys, time, urllib.request
sys.path.insert(0, "tools")
import synthetic as SYN
P="/tmp/rewritecheck"
def send(msgs, tag):
    b = SYN.body(project=P, n_tools=6, question="egal")
    b["messages"] = msgs
    b["model"]="qwen38-low"; b["max_tokens"]=8; b["stream"]=False
    r = urllib.request.Request("http://127.0.0.1:8090/v1/messages", data=json.dumps(b).encode(),
        headers={"Content-Type":"application/json","anthropic-version":"2023-06-01"})
    t0=time.time()
    with urllib.request.urlopen(r, timeout=900) as x: d=json.loads(x.read())
    print("  %-22s %5.1f s  usage=%s" % (tag, time.time()-t0, d.get("usage")), flush=True)

m1=[{"role":"user","content":"Sag nur: eins."}]
m2=m1+[{"role":"assistant","content":"eins"},{"role":"user","content":"Sag nur: zwei."}]
m3=[{"role":"user","content":"Sag nur: EINS GEÄNDERT."},
    {"role":"assistant","content":"eins"},{"role":"user","content":"Sag nur: drei."}]
send(m1,"Turn 1")
send(m2,"Turn 2 (angehängt)")
send(m3,"Turn 3 (umgeschrieben)")
