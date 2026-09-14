#!/usr/bin/env python3
"""vision_eval.py — Benchmark Qwen3-VL 2B, 4B, and 8B on Zen 5 CPU across 3 images."""

import argparse
import base64
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "setup", "lib"))
try:
    import systemdfile as SDF
    MODELS_DIR = SDF.models_dir()
except Exception:
    MODELS_DIR = os.environ.get("LLAMA_MODELS", os.path.expanduser("~/models"))

LLAMA_SERVER = os.environ.get("LLAMA_BIN", os.path.expanduser("~/llama.cpp/build-rocm-patched/bin/llama-server"))
IMAGES_DIR = os.path.join(REPO, "tests", "fixtures", "vision")
PORT = 8089

MODELS = {
    "2B": {
        "model": os.path.join(MODELS_DIR, "Qwen3VL-2B-Instruct-Q4_K_M.gguf"),
        "mmproj": os.path.join(MODELS_DIR, "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"),
    },
    "4B": {
        "model": os.path.join(MODELS_DIR, "Qwen3VL-4B-Instruct-Q4_K_M.gguf"),
        "mmproj": os.path.join(MODELS_DIR, "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf"),
    },
    "8B": {
        "model": os.path.join(MODELS_DIR, "Qwen3VL-8B-Instruct-Q4_K_M.gguf"),
        "mmproj": os.path.join(MODELS_DIR, "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"),
    },
}

IMAGES = [
    ("2D Pixelart", os.path.join(IMAGES_DIR, "pixelart_mining_2d.png")),
    ("3D Low-Poly", os.path.join(IMAGES_DIR, "lowpoly_duck_3d.png")),
    ("3D Realistic", os.path.join(IMAGES_DIR, "realistic_classroom_3d.png")),
]

PROMPT = (
    "Analysiere dieses Game-Asset / diesen Screenshot für ein Asset Inventory:\n"
    "1. Stimmung (Atmosphäre)\n"
    "2. Primäre Farbpalette\n"
    "3. Stil (z.B. Pixelart, Low-Poly, Stylized, Realistisch, Hand-Painted)\n"
    "4. Erkannte Objekte / Assets\n"
    "Antworte stichpunktartig, präzise und knapp auf Deutsch."
)

def wait_for_server(port, timeout=60):
    url = f"http://127.0.0.1:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False

def get_process_rss_gib(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    kb = int(parts[1])
                    return round(kb / (1024 * 1024), 2)
    except Exception:
        return 0.0
    return 0.0

def eval_model(name, config):
    print(f"\n==========================================")
    print(f"Testing Qwen3-VL {name} on Zen 5 CPU")
    print(f"==========================================")

    cmd = [
        LLAMA_SERVER,
        "-m", config["model"],
        "--mmproj", config["mmproj"],
        "--device", "none",
        "--mmproj-device", "none",
        "-t", "16",
        "-c", "8192",
        "--image-min-tokens", "1024",
        "--port", str(PORT),
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    results = []
    rss = 0.0
    load_time = 0.0

    try:
        print(f"Starting llama-server (pid {proc.pid})...")
        t_load_start = time.time()
        if not wait_for_server(PORT, timeout=60):
            print("ERROR: Server failed to start within 60s")
            return None
        load_time = time.time() - t_load_start
        rss = get_process_rss_gib(proc.pid)
        print(f"Server ready in {load_time:.2f}s! Memory RSS: {rss} GiB")

        for img_label, img_path in IMAGES:
            print(f"\n--- Evaluating: {img_label} ---")
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")

            payload = {
                "model": f"qwen3vl-{name.lower()}",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPT},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        ],
                    }
                ],
                "temperature": 0.2,
                "presence_penalty": 0.3,
                "max_tokens": 256,
            }

            req = urllib.request.Request(
                f"http://127.0.0.1:{PORT}/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )

            t0 = time.time()
            with urllib.request.urlopen(req, timeout=120) as resp:
                res = json.loads(resp.read().decode("utf-8"))
            t_total = time.time() - t0

            usage = res.get("usage", {})
            out_tokens = usage.get("completion_tokens", 0)
            in_tokens = usage.get("prompt_tokens", 0)
            tps = round(out_tokens / t_total, 1) if t_total > 0 else 0
            content = res["choices"][0]["message"]["content"].strip()

            print(f"Time: {t_total:.2f}s | Output: {out_tokens} tokens ({tps} t/s) | Input: {in_tokens} tokens")
            print(f"Response:\n{content}\n")

            results.append({
                "image": img_label,
                "wall_s": round(t_total, 2),
                "tokens_out": out_tokens,
                "tokens_in": in_tokens,
                "tps": tps,
                "content": content,
            })

    finally:
        print(f"Shutting down server (pid {proc.pid})...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(1)

    return {
        "model": name,
        "rss_gib": rss,
        "load_s": round(load_time, 2),
        "benchmarks": results,
    }

def main():
    target_models = sys.argv[1:] if len(sys.argv) > 1 else ["2B", "4B", "8B"]
    all_results = []

    for m in target_models:
        if m in MODELS:
            r = eval_model(m, MODELS[m])
            if r:
                all_results.append(r)

    report_path = os.path.join(REPO, "bench", "reports", "vision_qwen3_cpu_eval.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {report_path}")

if __name__ == "__main__":
    main()
