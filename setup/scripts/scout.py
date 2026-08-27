#!/usr/bin/env python3
"""scout — look at a model before downloading it.

    python3 setup/scripts/scout.py Qwen/Qwen3.8-Flash-Next
    python3 setup/scripts/scout.py unsloth/Qwen3.8-Flash-Next-GGUF
    python3 setup/scripts/scout.py --config ./config.json        offline

Answers, in this order, the three questions that decide whether a download is
worth starting:

  1. **Can our build even load it?** Not "does llama.cpp support it" in
     general — whether THIS checkout does, checked against
     ~/llama.cpp: the converter registers HF architecture names with
     `@ModelBase.register(...)` in `conversion/*.py`, and the runtime knows
     GGUF architecture strings from `LLM_ARCH_NAMES` in `src/llama-arch.cpp`.
     A model can be convertible and not loadable, or neither.
  2. **What is in the repo and how big is it?** Per file and grouped by quant,
     so a 60 GB download is a decision and not a surprise.
  3. **Does it fit here?** Against the GTT cap this machine currently boots
     with and against total RAM — two different limits, and conflating them
     cost this machine a hang. setup/lib/budget.py owns that arithmetic for a
     model that is already on disk; this makes the cruder version of the same
     judgement from a file listing alone.

It downloads NOTHING but metadata: the file listing and, if present,
`config.json`. Both are a few kilobytes.

Written on 26 August 2026 for the Qwen3.8-Flash-Next release, but there is
nothing model-specific in it.
"""
import argparse, glob, json, os, re, subprocess, sys, urllib.error, urllib.request

HF_API = "https://huggingface.co/api/models/%s/tree/main?recursive=true"
HF_INFO = "https://huggingface.co/api/models/%s"
HF_RAW = "https://huggingface.co/%s/resolve/main/%s"
LLAMA_SRC = os.environ.get("LLAMA_SRC", os.path.expanduser("~/llama.cpp"))

GIB = 1024 ** 3


# --- pure helpers, so the reasoning can be tested without a network -------

# A .gguf in a model repo is not necessarily a model. These are companions —
# they are downloaded ALONGSIDE a quant, so listing them as candidates and
# computing a fit for each is noise in exactly the moment there is no time for
# noise. Unsloth's Qwen3.8-27B repo has four of them among sixteen quants.
COMPANION = re.compile(r"(mmproj|imatrix|/?mtp[-/]|-mtp-|draft|eagle)", re.I)


def group_files(entries):
    """Files by quant-ish prefix, with total size and whether it is a model.

    `entries` is [{path,size}]. A sharded GGUF (`…-00001-of-00003.gguf`) is
    ONE logical model and only its total matters — Laguna is three parts and
    its profile points at part one, which finds the rest.
    """
    groups = {}
    for e in entries:
        path = e["path"]
        if not path.lower().endswith(".gguf"):
            continue
        key = re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", path)
        key = re.sub(r"\.gguf$", "", key)
        g = groups.setdefault(key, {"bytes": 0, "parts": 0,
                                    "companion": bool(COMPANION.search(path))})
        g["bytes"] += e.get("size") or 0
        g["parts"] += 1
    return groups


def fit_verdict(size_bytes, gtt_gib, ram_gib, kv_gib=10.0, host_gib=12.0):
    """Does a model of this size fit, and against WHICH limit does it fail?

    Two different limits, and conflating them is how "it does not fit" turns
    into the wrong fix:
      * the GTT cap bounds what amdgpu may hold — raise it with
        setup/scripts/gtt.sh;
      * total RAM bounds everything together — no parameter helps there.
    """
    weights = size_bytes / GIB
    need_gpu = weights + kv_gib
    need_all = need_gpu + host_gib
    if need_all > ram_gib:
        return ("no", "needs ~%.1f GiB with KV and a host share of %.0f, "
                      "the machine has %.1f — no kernel parameter fixes this"
                % (need_all, host_gib, ram_gib))
    if need_gpu > gtt_gib:
        # Deliberately no longer "raise it to <number>". That advice cost this
        # machine two hangs on 26.08.: the cap is a CEILING as well as a
        # budget, and above it a model that does not fit stops failing to
        # allocate and starts taking the box instead. Raising it is sometimes
        # right — but it is a decision about protection, not a fix for a
        # number, and it should not be suggested in one line by a tool that
        # cannot see what else the machine is doing.
        return ("cap", "needs ~%.1f GiB in GTT (weights %.1f + KV %.0f) and the "
                       "cap is %.1f. Raising the cap also removes the ceiling "
                       "that turns an over-large model into a clean allocation "
                       "failure — see setup/README.md, 'The four ceilings'. A "
                       "smaller quant is usually the better answer."
                % (need_gpu, weights, kv_gib, gtt_gib))
    return ("yes", "~%.1f GiB in GTT of %.1f, %.1f GiB left for KV growth"
            % (need_gpu, gtt_gib, gtt_gib - need_gpu))


def converter_supports(arch, src=LLAMA_SRC):
    """Does the local checkout's converter register this HF architecture?"""
    hits = []
    for path in glob.glob(os.path.join(src, "conversion", "*.py")):
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except OSError:
            continue
        for m in re.finditer(r"ModelBase\.register\(([^)]*)\)", text):
            names = re.findall(r'"([^"]+)"', m.group(1))
            if arch in names:
                hits.append(os.path.basename(path))
    return hits


def architectures_in_text(text):
    """The GGUF architecture strings named in a llama-arch.cpp.

    Split out of runtime_architectures so that ONE parser answers both "what
    can this checkout load" and "what does upstream master know" —
    A caller may ask the second question of a file it fetched, and a
    second parser would be a second thing to get wrong.
    """
    block = text.split("LLM_ARCH_NAMES", 1)
    if len(block) < 2:
        return []
    return sorted(set(re.findall(r'\{\s*LLM_ARCH_\w+,\s*"([^"]+)"', block[1])))


def runtime_architectures(src=LLAMA_SRC):
    """The GGUF architecture strings this checkout's llama.cpp can load."""
    path = os.path.join(src, "src", "llama-arch.cpp")
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return []
    return architectures_in_text(text)


def machine_limits():
    """(GTT cap GiB, total RAM GiB) as this machine currently stands."""
    gtt = 0.0
    for p in glob.glob("/sys/class/drm/card*/device/mem_info_gtt_total"):
        try:
            with open(p) as fh:
                gtt = int(fh.read()) / GIB
            break
        except (OSError, ValueError):
            pass
    ram = 0.0
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    ram = int(line.split()[1]) / 1048576
                    break
    except OSError:
        pass
    return gtt, ram


# --- network -------------------------------------------------------------

def fetch(url, timeout=20):
    req = urllib.request.Request(url, headers={"user-agent": "inference-stack-scout"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def fetch_json(url, timeout=20):
    return json.loads(fetch(url, timeout))


# --- reporting -----------------------------------------------------------

def say(*a):
    print(*a)


def head(title):
    print("\n%s\n%s" % (title, "-" * len(title)))


def report_config(cfg):
    head("config.json")
    arch = (cfg.get("architectures") or ["?"])[0]
    say("  architecture        %s" % arch)
    for key, label in [
        ("model_type", "model_type"),
        ("num_hidden_layers", "layers"),
        ("hidden_size", "hidden size"),
        ("max_position_embeddings", "context"),
        ("torch_dtype", "dtype"),
        ("num_experts", "experts"),
        ("num_experts_per_tok", "experts/token"),
        ("n_routed_experts", "routed experts"),
        ("sliding_window", "sliding window"),
        ("nextn_predict_layers", "MTP head"),
        ("num_nextn_predict_layers", "MTP head"),
    ]:
        if cfg.get(key) is not None:
            say("  %-19s %s" % (label, cfg[key]))

    # The two things this release turns on, by name and by pattern.
    interesting = {k: v for k, v in cfg.items()
                   if re.search(r"ngram|n_gram|hash|sparse|qsa|gdn|linear_attn",
                                k, re.I)}
    if interesting:
        say("\n  the new machinery:")
        for k, v in sorted(interesting.items()):
            say("    %-30s %s" % (k, json.dumps(v)[:80]))
    else:
        say("\n  no ngram/sparse/gdn keys in the config — either the names differ")
        say("  or the mechanism is not configurable. Read the model card.")

    # Sliding window decides whether the profile needs --swa-full, which is
    # the single switch that made the difference between 100.2 s and 10.4 s.
    sw = cfg.get("sliding_window")
    say("\n  MODEL_SWA for the profile:  %s"
        % ("yes  -> the profile MUST carry --swa-full" if sw else
           "no   (no sliding_window in the config)" if sw is None else "no"))
    return arch


def report_support(arch):
    head("Can THIS checkout handle it?  (%s)" % LLAMA_SRC)
    if not os.path.isdir(LLAMA_SRC):
        say("  no llama.cpp checkout at %s — set LLAMA_SRC" % LLAMA_SRC)
        return
    hits = converter_supports(arch)
    if hits:
        say("  converter   YES — registered in conversion/%s" % ", ".join(hits))
    else:
        say("  converter   NO  — no @ModelBase.register(\"%s\") in conversion/" % arch)
        say("              A GGUF from someone else may still exist and load;")
        say("              what this rules out is converting it ourselves.")
    archs = runtime_architectures()
    guess = re.sub(r"(ForCausalLM|ForConditionalGeneration|Model)$", "", arch).lower()
    near = [a for a in archs if a.startswith(guess[:5])] if guess else []
    if guess in archs:
        say("  runtime     YES — llama.cpp knows the GGUF architecture %r" % guess)
    else:
        say("  runtime     UNKNOWN — %r is not literally in LLM_ARCH_NAMES." % guess)
        say("              The GGUF's own general.architecture is what counts;")
        say("              read it with gguf_dump once a file exists.")
        if near:
            say("              nearest known: %s" % ", ".join(near))
    try:
        v = subprocess.run(
            [os.path.join(LLAMA_SRC, "build-rocm-patched", "bin", "llama-server"),
             "--version"], capture_output=True, text=True, timeout=30)
        say("  build       %s" % (v.stdout + v.stderr).strip().splitlines()[0])
    except Exception:
        pass


def report_files(entries):
    head("What is in the repo")
    groups = group_files(entries)
    if not groups:
        say("  no .gguf files — this is a safetensors repo, so it needs")
        say("  converting first (or wait for someone else's GGUF).")
        big = sorted((e for e in entries if (e.get("size") or 0) > 100 * 1024**2),
                     key=lambda e: -(e.get("size") or 0))[:8]
        for e in big:
            say("    %8.1f GiB  %s" % ((e.get("size") or 0) / GIB, e["path"]))
        return groups
    gtt, ram = machine_limits()
    models = {k: v for k, v in groups.items() if not v["companion"]}
    comps = {k: v for k, v in groups.items() if v["companion"]}
    # Biggest first: the largest quant that still fits is the one worth having,
    # so it should be at the top rather than scrolled to.
    for name, g in sorted(models.items(), key=lambda kv: -kv[1]["bytes"]):
        verdict, why = fit_verdict(g["bytes"], gtt, ram)
        mark = {"yes": "  ok ", "cap": "  cap", "no": "  NO "}[verdict]
        say("%s %7.1f GiB  %-44s%s" % (mark, g["bytes"] / GIB, name[:44],
                                       "  (%d parts)" % g["parts"] if g["parts"] > 1 else ""))
        say("       %s" % why)
    if comps:
        say("\n  companions (downloaded alongside, not candidates):")
        for name, g in sorted(comps.items(), key=lambda kv: -kv[1]["bytes"]):
            say("    %7.1f GiB  %s" % (g["bytes"] / GIB, name))
    mm = [e["path"] for e in entries if "mmproj" in e["path"].lower()]
    if not mm:
        say("\n  no mmproj — if the model is multimodal, the vision tower is")
        say("  somewhere else or not converted yet.")
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("repo", nargs="?", help="Hugging Face repo id, e.g. Qwen/Qwen3.8-Flash-Next")
    ap.add_argument("--config", help="a local config.json instead of fetching one")
    ap.add_argument("--arch", help="check support for this HF architecture name and exit")
    a = ap.parse_args()

    gtt, ram = machine_limits()
    say("this machine: GTT cap %.1f GiB, RAM %.1f GiB" % (gtt, ram))

    if a.arch:
        report_support(a.arch)
        return 0
    if a.config:
        arch = report_config(json.load(open(a.config, encoding="utf-8")))
        report_support(arch)
        return 0
    if not a.repo:
        ap.error("give a repo id, --config, or --arch")

    try:
        info = fetch_json(HF_INFO % a.repo)
        say("  %s · last modified %s · %s" %
            (a.repo, info.get("lastModified", "?"),
             ", ".join(info.get("tags", [])[:6])))
    except urllib.error.HTTPError as e:
        # 404 and 401 mean different things and the difference is the whole
        # point of running this before a release: 404 is "no such repo",
        # 401/403 is "the page exists and is not public yet" — which is what
        # an announced-but-unreleased model looks like from here.
        if e.code == 404:
            say("\n%s does not exist. Check the spelling." % a.repo)
            return 1
        if e.code in (401, 403):
            say("\n%s exists but is not readable yet (HTTP %s)." % (a.repo, e.code))
            say("For an announced model that is the normal state before the")
            say("release — the page is up, the files are not. Nothing to scout")
            say("until they are; run this again afterwards.")
            return 1
        say("\nHTTP %s from the model info endpoint" % e.code)
    except Exception as e:
        say("\ncould not reach Hugging Face: %s" % e)
        return 1

    try:
        entries = fetch_json(HF_API % a.repo)
    except Exception as e:
        say("could not list the files: %s" % e)
        entries = []
    if entries:
        report_files(entries)

    cfg = None
    if any(e["path"] == "config.json" for e in entries):
        try:
            cfg = json.loads(fetch(HF_RAW % (a.repo, "config.json")))
        except Exception as e:
            say("\ncould not read config.json: %s" % e)
    if cfg:
        arch = report_config(cfg)
        report_support(arch)
    else:
        say("\nno config.json in this repo — for a GGUF-only repo that is normal.")
        say("Read the architecture out of the file itself once it is here:")
        say("    python3 ~/llama.cpp/gguf-py/gguf/scripts/gguf_dump.py --no-tensors <file.gguf>")

    head("Next")
    say("  setup/env/flashnext.env carries the checklist,")
    say("  setup/lib/budget.py --profile <name>   the full memory arithmetic")
    return 0


if __name__ == "__main__":
    sys.exit(main())
