#!/usr/bin/env python3
"""gateway-concurrency — what the admission control actually does under load.

Every measurement so far went straight to llama-server on 8080. Nothing had
ever measured the gateway itself: the priority gate, the per-access throttle,
and whether a waiting caller is starved by a busy local user.

Four questions:

  1. What does the gateway cost when nothing is contended?
  2. Does the per-access throttle fire, and on the right request?
  3. Does priority really put a local request ahead of a LAN one?
  4. Case D: how long does a LAN caller wait while a local user keeps working?

The LAN zone stands in for "remote" here — the same priority mechanics, one
class lower, without Cloudflare's latency in the numbers.

    python3 bench/suites/gateway-concurrency.py
"""
import json, os, subprocess, sys, threading, time, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tools"))
from synthetic import body                                # noqa: E402

LOCAL = "http://127.0.0.1:8090"
DIRECT = "http://127.0.0.1:8080"
LAN = ""
TOKEN = ""


def read_access():
    """The LAN address and the first configured token.

    NOT at import time, and that is the whole point of it being a function.
    tests/common.py states the precondition for loading a script by path:
    "an import without consequences: no network, no TOKEN FILE, no
    web.run_app". This file read ~/.config/cc-gateway-tokens and shelled out
    for the interface address while being imported — two of the three, in the
    one directory nothing imports today. Which is the entire defence, and it
    is the kind that stops being true without anybody deciding it should.
    """
    global LAN, TOKEN
    LAN = os.environ.get("LAN") or subprocess.run(
        "ip -4 -o addr show scope global | grep -v docker "
        "| awk '{print $4}' | cut -d/ -f1 | head -1",
        shell=True, capture_output=True, text=True).stdout.strip()
    with open(os.path.expanduser("~/.config/llm-gateway-tokens"),
              encoding="utf-8") as fh:
        for line in fh:
            if line.strip() and not line.startswith("#") \
                    and len(line.split(None, 1)) == 2:
                TOKEN = line.split(None, 1)[1].strip()
                break

LOAD = "Count slowly from one to sixty, one number per line."

def call(base, project, max_tokens, token=None, n_tools=4, question="Say alpha."):
    p = body(project=project, n_tools=n_tools, question=question)
    p["model"], p["max_tokens"], p["stream"] = "laguna", max_tokens, False
    h = {"content-type": "application/json", "anthropic-version": "2023-06-01"}
    if token:
        h["Authorization"] = "Bearer " + token
    r = urllib.request.Request(base + "/v1/messages", data=json.dumps(p).encode(), headers=h)
    t0 = time.time()
    try:
        with urllib.request.urlopen(r, timeout=600) as x:
            a = json.loads(x.read().decode())
        u = a.get("usage", {})
        return {"code": 200, "s": time.time() - t0, "out": u.get("output_tokens", 0),
                "cached": u.get("cache_read_input_tokens", 0), "new": u.get("input_tokens", 0)}
    except urllib.error.HTTPError as e:
        return {"code": e.code, "s": time.time() - t0, "out": 0, "cached": 0, "new": 0}

def pct(m):
    t = m["cached"] + m["new"]
    return 100.0 * m["cached"] / t if t else 0.0

# The gateway saves the warm-up prefix automatically. That is correct
# behaviour, but /tmp/gw-warm is not a project — clean the file up at the end
# so the store does not fill with measurement debris.
import atexit, glob
def _cleanup():
    ident = None
    try:
        sys.path.insert(0, os.path.join(REPO, "setup", "gateway"))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "gw", os.path.join(REPO, "setup", "gateway", "gateway.py"))
        gw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gw)
        p = body(project=WARM, n_tools=4, question="Say alpha.")
        ident = gw.prefix_id(p)[0]
    except Exception:
        return
    for f in glob.glob(os.path.expanduser("~/.cache/llama-slots/%s.*" % ident)):
        try:
            os.remove(f)
            print("cleaned up %s" % os.path.basename(f))
        except OSError:
            pass
# Everything below RUNS. It is guarded because this file used to do all of
# it while being imported — including reading ~/.config/llm-gateway-tokens,
# which tests/common.py names outright as the thing an import must not do.
# The functions above stay importable, which is what makes the rule keepable
# rather than a rule about this directory being left alone.
if __name__ == "__main__":
    atexit.register(_cleanup)

    read_access()
    print("=" * 78)
    print("llm-gateway under load   LAN=%s  token=%s" % (LAN, "yes" if TOKEN else "MISSING"))
    print("=" * 78)

    WARM = "/tmp/gw-warm"
    print("\n0 · warm the prefix up (through the gateway)")
    for i in (1, 2):
        m = call(LOCAL, WARM, 8)
        print("   call %d   %5.1f s   cache %5.1f %%" % (i, m["s"], pct(m)))

    print("\n1 · what the gateway costs when nothing is contended")
    # Five calls each, the first thrown away and the median taken: a single
    # re-prefill in the middle would otherwise drown the difference.
    def median(xs):
        xs = sorted(xs); return xs[len(xs)//2]
    direct = [call(DIRECT, WARM, 8)["s"] for _ in range(5)][1:]
    viagw = [call(LOCAL, WARM, 8)["s"] for _ in range(5)][1:]
    print("   straight to llama-server  %s -> median %.3f s" % (["%.2f" % x for x in direct], median(direct)))
    print("   through the gateway       %s -> median %.3f s" % (["%.2f" % x for x in viagw], median(viagw)))
    print("   overhead                  %.0f ms" % ((median(viagw) - median(direct)) * 1000))

    print("\n2 · per-access throttle: three at once from the same access")
    res = {}
    def fire(k, base, tok, mt=8):
        res[k] = call(base, WARM, mt, token=tok)
    ts = [threading.Thread(target=fire, args=(i, "http://%s:8090" % LAN, TOKEN)) for i in range(3)]
    for t in ts: t.start()
    for t in ts: t.join()
    codes = sorted(res[i]["code"] for i in range(3))
    print("   answers: %s" % codes)
    print("   -> %dx 200, %dx 429" % (codes.count(200), codes.count(429)))

    print("\n3 · priority: does a local request overtake a LAN one?")
    res2 = {}
    # The load has to come from the LOCAL zone: PER_TOKEN_MAX does not apply there,
    # so two long requests really can hold both slots. Coming from the LAN access
    # they would have used up its own quota and the request under test would have
    # hit the throttle instead of the queue.
    def fire_load(k):
        res2[k] = call(LOCAL, WARM, 400, question=LOAD)
    long_ts = [threading.Thread(target=fire_load, args=("load%d" % i,)) for i in range(2)]
    for t in long_ts: t.start()
    time.sleep(4)
    def fire2(k, base, tok):
        res2[k] = call(base, WARM, 8, token=tok)
    pair = [threading.Thread(target=fire2, args=("lan", "http://%s:8090" % LAN, TOKEN)),
            threading.Thread(target=fire2, args=("local", LOCAL, None))]
    for t in pair: t.start()
    for t in pair: t.join()
    for t in long_ts: t.join()
    print("   local finished after  %5.1f s (code %d)" % (res2["local"]["s"], res2["local"]["code"]))
    print("   lan   finished after  %5.1f s (code %d)" % (res2["lan"]["s"], res2["lan"]["code"]))
    if res2["lan"]["code"] == 429:
        print("   -> LAN hit the per-access throttle, not the queue")
    elif res2["local"]["s"] < res2["lan"]["s"]:
        print("   -> local was served first, as intended")
    else:
        print("   -> LAN came first; priority did NOT bite")

    print("\n4 · case D: a LAN caller waits while the local user keeps working")
    stop = threading.Event()
    lan_result = {}
    def keep_local_busy():
        while not stop.is_set():
            call(LOCAL, WARM, 400, question=LOAD)
    def lan_waiter():
        lan_result["m"] = call("http://%s:8090" % LAN, WARM, 8, token=TOKEN)
    busy = [threading.Thread(target=keep_local_busy) for _ in range(2)]
    for t in busy: t.start()
    time.sleep(1)
    w = threading.Thread(target=lan_waiter)
    w.start()
    w.join(timeout=180)
    stop.set()
    for t in busy: t.join(timeout=60)
    if "m" in lan_result:
        m = lan_result["m"]
        print("   LAN caller served after %5.1f s (code %d)" % (m["s"], m["code"]))
        print("   -> %s" % ("beyond Cloudflare's 125 s: a remote caller would have got a 524"
                            if m["s"] > 125 else "within Cloudflare's 125 s"))
    else:
        print("   LAN caller still unserved after 180 s -> starved")

    print("\n5 · the same again, but with FOUR local streams")
    # With two local streams and two slots there is always a moment when the queue
    # is empty and the waiting LAN request takes the freed slot. Starvation needs
    # sustained oversubscription: more waiting requests than slots, at every
    # moment. Two Claude Code sessions are exactly four streams.
    stop2 = threading.Event()
    lan2 = {}
    def keep_busy2():
        while not stop2.is_set():
            call(LOCAL, WARM, 400, question=LOAD)
    def lan_waiter2():
        lan2["m"] = call("http://%s:8090" % LAN, WARM, 8, token=TOKEN)
    busy2 = [threading.Thread(target=keep_busy2) for _ in range(4)]
    for t in busy2: t.start()
    time.sleep(2)
    GRACE = 200
    w2 = threading.Thread(target=lan_waiter2)
    w2.start()
    w2.join(timeout=GRACE)
    still_waiting = w2.is_alive()
    stop2.set()                       # stop the load
    w2.join(timeout=120)
    for t in busy2: t.join(timeout=60)
    if "m" in lan2:
        m = lan2["m"]
        if still_waiting:
            # It was only served once the load stopped — so it did not get in on
            # its own. Reporting the raw number would understate the starvation.
            print("   LAN caller was STILL waiting after %d s and only got in once"
                  % GRACE)
            print("   the local load stopped (total %.1f s, code %d)" % (m["s"], m["code"]))
            print("   -> starved. Beyond Cloudflare's 125 s a remote caller is gone.")
        else:
            print("   LAN caller served after %5.1f s (code %d)" % (m["s"], m["code"]))
            print("   -> %s" % ("BEYOND Cloudflare's 125 s: a remote caller gets a 524"
                                if m["s"] > 125 else "still within Cloudflare's 125 s"))
    else:
        print("   LAN caller still unserved after 200 s -> starved, and a remote")
        print("      caller would have been dropped by Cloudflare long before")
    print("=" * 78)
