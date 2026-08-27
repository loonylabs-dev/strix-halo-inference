#!/usr/bin/env python3
"""sweep — one command, one answer: what each server configuration COSTS.

    python3 bench/sweep.py --variants bench/variants/qwen38.json --reference laguna
    python3 bench/sweep.py --variants bench/variants/qwen38.json --only rocm-medium-spec

What it does, in order:
  1. stops the active llama-user@ service (the GPU belongs to the sweep),
  2. starts each variant's llama-server, measures cold prefill and warm
     decode (bench/speed.py), stops it again — a failing variant is recorded
     and skipped, not fatal,
  3. ALWAYS restores the service it stopped, even on crash or Ctrl-C,
  4. measures the restored service too when --reference is given, so the
     production configuration appears in the same table as the candidates,
  5. writes bench/reports/<stamp>_sweep_<model>/comparison.md via compare.py.

Every variant's full argv is written into its report — a number without its
flags is worthless three weeks later, and speed.py records the WINDOW next to
every rate for the same reason: prefill and decode both move as the context
grows, so a t/s figure without its -c is not comparable to another one.

It measured the task battery until 26.08. That battery is gone: how good a
model IS at the work is not this repo's question, and other people answer it
with more scale. What a configuration COSTS on this hardware is nobody else's
question, so it is this file's.
"""
import argparse, json, os, shutil, signal, subprocess, sys, threading, time
import urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)
import run as runlib                                # noqa: E402
import speed                                         # noqa: E402
import compare                                       # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "setup", "lib"))
import systemdfile                                    # noqa: E402

URL = "http://127.0.0.1:8080"


def slots_ready(url, timeout_s):
    """/slots answers only when the model really serves — /health lies during
    loading — see setup/README.md, "Abolishing the cold start"."""
    end = time.time() + timeout_s
    while time.time() < end:
        try:
            with urllib.request.urlopen(url + "/slots", timeout=5) as x:
                if x.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def port_free(url):
    try:
        urllib.request.urlopen(url + "/health", timeout=3)
        return False
    except urllib.error.HTTPError:
        return False
    except Exception:
        return True


def active_llama_unit():
    """The running llama-user@ instance, or None."""
    r = subprocess.run(["systemctl", "--user", "list-units", "--plain",
                        "--no-legend", "--state=active,activating",
                        "llama-user@*"], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        unit = line.split()[0] if line.split() else ""
        if unit.startswith("llama-user@"):
            return unit
    return None


def systemctl_user(*args):
    return subprocess.run(["systemctl", "--user", *args],
                          capture_output=True, text=True).returncode == 0


# The watchdog that asks production one question every ten minutes and
# reports a failure when nothing answers. It has to go down WITH production,
# or every measurement longer than ten minutes leaves a failed unit behind and
# a red line in check.sh that has nothing to do with anything.
PROBE_TIMER = "llama-probe.timer"


def stop_production(unit):
    """Stop the model unit and take the watchdog down with it.

    bench/sideserver.py has done this since 27.08.2026 and said why at length.
    The two OTHER paths that stop production — this file, and
    bench/suites/restore-safety.py — did not, so the fix covered one of three.

    It bit the same evening: the first restore-safety run after the timer was
    armed left `the last probe FAILED (status 1)` in check.sh, from a probe
    that ran at 19:17:54 against a server this harness had deliberately
    stopped. Exactly the false alarm sideserver's comment describes, from the
    stop path the fix never reached.
    """
    if not unit:
        return
    systemctl_user("stop", PROBE_TIMER)
    systemctl_user("stop", unit)


def start_production(unit):
    """Put both back, in the order that cannot leave the watchdog off.

    The model first: a probe firing against a server that is still coming up
    is the same false alarm one restart later.
    """
    if not unit:
        return
    systemctl_user("start", unit)
    systemctl_user("start", PROBE_TIMER)


def stop_server(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        return
    for _ in range(30):
        if proc.poll() is not None:
            break
        time.sleep(1)
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    # The port has to be actually free before the next variant may start.
    for _ in range(30):
        if port_free(URL):
            return
        time.sleep(1)


def reexec_with_inhibit():
    """Re-exec this process under a suspend/idle inhibitor.

    GNOME suspends this machine after 15 minutes WITHOUT keyboard or mouse
    input, even on AC — GPU load does not count as activity. Found in the
    journal on 25.08.: suspend 00:09→06:46 froze the night sweep (which was
    first misdiagnosed as a ROCm hang), suspend 07:34→17:01 tore down the
    thinking-mode run (that suite is gone) and left the laguna service stopped. An unattended
    measurement therefore inhibits sleep for its own lifetime — the
    inhibitor dies with the process, nothing stays blocked afterwards.
    """
    if os.environ.get("BENCH_INHIBITED") == "1":
        return
    os.environ["BENCH_INHIBITED"] = "1"
    for tool, args in (
            ("gnome-session-inhibit", ["--inhibit", "suspend:idle"]),
            ("systemd-inhibit", ["--what=sleep:idle", "--who=bench",
                                 "--why=measurement run", "--mode=block"])):
        path = shutil.which(tool)
        if path:
            os.execv(path, [tool] + args + [sys.executable, "-u"] + sys.argv)
    # No inhibitor tool on this machine: run anyway — the wall caps still
    # bound the damage, and a server without GNOME may not suspend at all.


def platform_profile():
    try:
        with open("/sys/firmware/acpi/platform_profile") as f:
            return f.read().strip()
    except Exception:
        return None


def main():
    reexec_with_inhibit()
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True, help="bench/variants/<x>.json")
    ap.add_argument("--reference", help="llama-user@ instance to measure after "
                                        "restoring it, e.g. laguna")
    ap.add_argument("--restore", help="llama-user@ instance to bring up at the "
                                      "end even when nothing is active at "
                                      "start — for resuming after an aborted "
                                      "sweep that left the service down")
    ap.add_argument("--only", help="comma-separated variant names")
    ap.add_argument("--skip", help="comma-separated variant names")
    ap.add_argument("--timeout", type=int, default=1200,
                    help="per-request timeout in seconds")
    ap.add_argument("--variant-max-seconds", type=int, default=2700,
                    help="wall cap per variant; on breach its server is "
                         "killed and the sweep moves on")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    with open(a.variants, encoding="utf-8") as f:
        spec = json.load(f)
    variants = spec["variants"]
    if a.only:
        keep = {x.strip() for x in a.only.split(",")}
        variants = [v for v in variants if v["name"] in keep]
    if a.skip:
        drop = {x.strip() for x in a.skip.split(",")}
        variants = [v for v in variants if v["name"] not in drop]
    if not variants:
        raise SystemExit("no variants left after --only/--skip")

    stamp = time.strftime("%Y-%m-%d_%H%M")
    dest = os.path.join(HERE, "reports", "%s_sweep_%s" % (stamp, spec["model"]))
    os.makedirs(dest, exist_ok=True)

    context = {"stamp": stamp, "model": spec["model"], "note": a.note,
               "platform_profile": platform_profile(),
               "variants_file": a.variants, "spec": spec,
               "order": [v["name"] for v in variants],
               "reference": a.reference, "results": {}}

    was_active = active_llama_unit()
    if a.restore and not was_active:
        # An aborted sweep leaves the service down and nothing active — the
        # next run would then neither restore it nor measure the reference.
        was_active = "llama-user@%s.service" % a.restore
    print("=" * 92)
    print("sweep %s · %d variants · profile=%s · active service: %s"
          % (spec["model"], len(variants), context["platform_profile"],
             was_active or "none"))
    print("=" * 92)

    try:
        if was_active:
            print("stopping %s (and %s with it) for the duration of the "
                  "sweep ..." % (was_active, PROBE_TIMER))
            stop_production(was_active)
            for _ in range(60):
                if port_free(URL):
                    break
                time.sleep(1)

        for v in variants:
            name = v["name"]
            # @MODELS@ and @HOME@ through the ONE expander, so a variants file
            # reads the same way a profile and the unit do. Absolute paths
            # stood in bench/variants/*.json until 27.08. and tied every sweep
            # to one computer — including the binary, which is why v["binary"]
            # goes through it too.
            raw_argv = list(spec.get("base_args", [])) + list(v.get("args", []))
            argv = [systemdfile.expand(a) for a in raw_argv]
            # Expanded to RUN, unexpanded to RECORD. A report is read on other
            # machines and lives in the repository; `/home/<someone>/llama.cpp/
            # build-rocm/bin/llama-server` says nothing a reader can use and
            # names a person. What identifies the binary is the build stamp,
            # which is recorded beside it.
            raw_binary = v["binary"]
            v = dict(v, binary=systemdfile.expand(raw_binary))
            vdir = os.path.join(dest, name)
            os.makedirs(vdir, exist_ok=True)
            build = runlib.build_id(v["binary"])
            print("\n--- %s · build %s\n    %s %s"
                  % (name, build, os.path.basename(os.path.dirname(
                      os.path.dirname(v["binary"]))), " ".join(argv)))
            meta = {"variant": name, "binary": raw_binary, "argv": raw_argv,
                    "build": build, "note": v.get("note", "")}
            with open(os.path.join(vdir, "variant.json"), "w") as f:
                json.dump(meta, f, indent=1)
            proc = None
            try:
                proc = runlib.start_server(argv, os.path.join(vdir, "server.log"),
                                           v["binary"])
                if not slots_ready(URL, 180):
                    raise RuntimeError("model loaded, but /slots never answered")
                # The suite runs in a thread under a wall cap. Why: on 25.08.
                # a ROCm soft-hang froze one generation for 6.6 hours with no
                # error anywhere — the socket stayed open, so the client's
                # read timeout never fired (it guards inactivity errors, not
                # a server that holds the connection and sends nothing).
                # Killing the server is the one reliable way to unblock.
                box = {}
                def _suite():
                    try:
                        speed.run(URL, name, vdir, argv=argv, timeout=a.timeout)
                        box["done"] = True
                    except BaseException as e:    # noqa: BLE001 - re-raised
                        box["err"] = e
                th = threading.Thread(target=_suite, daemon=True)
                th.start()
                th.join(a.variant_max_seconds)
                if th.is_alive():
                    print("    WALL CAP after %ds — killing the variant's "
                          "server" % a.variant_max_seconds)
                    stop_server(proc)
                    th.join(120)
                    raise RuntimeError("wall cap: variant still running "
                                       "after %ds" % a.variant_max_seconds)
                if "err" in box:
                    raise box["err"]
                context["results"][name] = "ok"
            except BaseException as e:
                context["results"][name] = "failed: %s" % str(e)[:200]
                print("    VARIANT FAILED: %s" % str(e)[:200])
                if isinstance(e, KeyboardInterrupt):
                    raise
            finally:
                if proc is not None:
                    stop_server(proc)

    finally:
        # The machine must come back the way it was found — whatever happened.
        if was_active:
            print("\nrestoring %s ..." % was_active)
            start_production(was_active)
            if not slots_ready(URL, 900):
                print("WARNING: %s did not become ready within 900 s"
                      % was_active)
        with open(os.path.join(dest, "context.json"), "w") as f:
            json.dump(context, f, indent=1, ensure_ascii=False)

    if a.reference and was_active and context["results"] and slots_ready(URL, 60):
        print("\n--- reference: %s (production flags, via the restored service)"
              % a.reference)
        try:
            speed.run(URL, a.reference, os.path.join(dest, a.reference),
                      argv=[], timeout=a.timeout)
            context["results"][a.reference] = "ok"
        except Exception as e:
            context["results"][a.reference] = "failed: %s" % str(e)[:200]
            print("    REFERENCE FAILED: %s" % str(e)[:200])
        with open(os.path.join(dest, "context.json"), "w") as f:
            json.dump(context, f, indent=1, ensure_ascii=False)

    md = compare.render(dest)
    with open(os.path.join(dest, "comparison.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print("\n" + md)
    print("report: %s" % dest)


if __name__ == "__main__":
    main()
