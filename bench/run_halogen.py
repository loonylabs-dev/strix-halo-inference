#!/usr/bin/env python3
"""run_halogen.py — safe fenced benchmark runner for Halogen Flash Server.

Follows the sideserver protocol:
1. Arms transient dead-man switch on production unit.
2. Stops production unit and settles GTT.
3. Runs Halogen Flash Server container sweep (512, 8192, 32768 tokens).
4. Records full log and output.
5. In finally: tears down container, waits for GTT release, restores production, disarms deadman switch.
"""

import datetime
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run as runlib
import sideserver

REPO = os.path.dirname(HERE)


def ask_models(verb):
    """One reader for what the container backend is: setup/lib/models.sh."""
    r = subprocess.run(["bash", os.path.join(REPO, "setup", "lib", "models.sh"),
                        verb], capture_output=True, text=True)
    return (r.stdout or "").strip()


def main():
    # ASKED, never written down. A constant here names a unit this run will
    # stop and then arm a dead man's switch on, and on 04.09.2026 exactly
    # that constant one file over stopped one model and started another.
    # runlib.serving_unit() reads it off the process holding the port.
    stop_unit = runlib.serving_unit()
    if not stop_unit:
        print("Nothing is serving, or two things are — refusing to guess "
              "which unit to stop and put back.", flush=True)
        return 1
    if stop_unit.startswith("halogen"):
        print("halogen is already serving. This runner stops production and "
              "starts its own container; run it against a llama backend, or "
              "measure the running one directly.", flush=True)
        return 1
    deadman = "sideworkload-deadman-halogen"
    deadline = 25  # 25 minutes deadline
    report_dir = os.path.join(HERE, "reports", datetime.datetime.now().strftime("%Y-%m-%d_%H%M") + "_speed_flashnext-halogen")
    os.makedirs(report_dir, exist_ok=True)
    log_path = os.path.join(report_dir, "halogen-sweep.log")

    print("=== Halogen Flash Server Benchmark Runner ===", flush=True)
    print(f"Report directory: {report_dir}", flush=True)
    print(f"Arming dead-man switch on {stop_unit} (deadline: {deadline}m)...", flush=True)

    baseline = sideserver.gtt_used()
    settled = None
    proc = None
    container_name = "halogen-bench-runner"
    try:
        if not sideserver.arm_deadman(deadman, stop_unit, deadline):
            print("Failed to arm dead-man switch! Aborting.", flush=True)
            return 1

        print(f"Stopping production unit {stop_unit} and settling GTT...", flush=True)
        settled = sideserver.stop_production_and_settle(stop_unit)
        if settled is None:
            print("Failed to settle production unit GTT! Aborting.", flush=True)
            return 1

        print(f"GTT baseline: {sideserver.gtt_used():.2f} GiB", flush=True)

        # Both from setup/lib/models.sh: the directory AND the image tag. A
        # tag written down here would let a sweep measure one version while
        # halogenexec serves another, and the report would name the wrong one.
        models_dir = ask_models("halogen-models")
        if not models_dir:
            print("No halogen model bundle found — run "
                  "bash setup/scripts/fetch-halogen.sh", flush=True)
            return 1
        image = ask_models("halogen-image")

        sweep_args = sys.argv[1:] if len(sys.argv) > 1 else ["-p", "512,8192,32768", "-n", "128", "-d", "serial,mtp", "-r", "2"]
        cmd = [
            "podman", "run", "--rm",
            "--name", container_name,
            "--device", "/dev/kfd",
            "--device", "/dev/dri",
            "--security-opt", "label=disable",
            "-v", f"{models_dir}:/models:ro",
            image,
            "sweep"
        ] + sweep_args

        print(f"Running command: {' '.join(cmd)}", flush=True)
        with open(log_path, "w") as log_fh:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in iter(proc.stdout.readline, ""):
                print(line, end="", flush=True)
                log_fh.write(line)
                log_fh.flush()
            proc.wait()

        rc = proc.returncode
        print(f"\nHalogen container exited with rc={rc}", flush=True)
        return rc
    finally:
        subprocess.run(["podman", "rm", "-f", container_name], check=False, capture_output=True)
        print("Waiting for GTT release...", flush=True)
        runlib.wait_for_gtt_release(sideserver.release_baseline(baseline, settled),
                                    timeout=sideserver.GTT_RELEASE_TIMEOUT_S)
        print(f"GTT now {sideserver.gtt_used():.2f} GiB", flush=True)
        print(f"Restoring production unit {stop_unit}...", flush=True)
        sideserver.restore_production(stop_unit, deadman)
        print("Production restored cleanly.", flush=True)

if __name__ == "__main__":
    sys.exit(main())
