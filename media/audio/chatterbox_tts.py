#!/usr/bin/env python3
"""chatterbox_tts — text in, WAV out, for Chatterbox-Multilingual.

    chatterbox-tts-cli -p "Es war einmal ein Leuchtturm." -o out.wav
    chatterbox-tts-cli -p "..." -o out.wav --lang en --ref my_voice.wav

Deployed by media/audio/setup-venv.sh as a COPY into the venv's bin (the
repo's location is machine state, a workload profile may not name it — see
that script). The flag surface is the workload contract: -p TEXT, -o OUT,
exactly what bench/sideserver.py's smoke run appends to WORKLOAD_CMD.

Runs on the venv's CPU torch. First call downloads the model from
Hugging Face into the HF cache (~severalGB); after that it is offline.

Determinism is SEEDED, not assumed: torch.manual_seed(--seed) before
generation. Whether that actually yields byte-identical WAVs on this stack
is a measurement (bench/audiobench.py records hashes), not a promise —
Chatterbox samples autoregressively and upstream documents no seed
contract.
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-p", "--prompt", required=True, help="text to speak")
    ap.add_argument("-o", "--out", required=True, help="output .wav path")
    ap.add_argument("--lang", default="de",
                    help="language_id for the multilingual model (default de)")
    ap.add_argument("--ref", default=None,
                    help="reference clip (.wav) for voice cloning; without "
                         "it the model's built-in default voice speaks")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    # Imports live here, not at module top: --help must work in a bare
    # venv, and the error for a missing torch should name the fix.
    try:
        import torch
        import torchaudio
        from huggingface_hub import snapshot_download
        from pathlib import Path
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    except ImportError as e:
        print("missing dependency (%s) — run: bash media/audio/setup-venv.sh"
              % e, file=sys.stderr)
        return 2

    torch.manual_seed(a.seed)
    # PINNED revision, not `main`: from_pretrained() resolves main at call
    # time, so upstream pushing new weights would silently swap the model
    # under the measured figures in setup/workloads/chatterbox.env — the
    # repointed-symlink failure in Hugging Face clothes (review,
    # 01.09.2026). This is the snapshot the 2026-09-01_0530 bench ran on;
    # moving it is a deliberate edit here plus a re-measurement, never a
    # side effect. After the first download this is a cache hit — offline.
    revision = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
    ckpt = Path(snapshot_download(
        "ResembleAI/chatterbox", revision=revision,
        allow_patterns=["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt",
                        "grapheme_mtl_merged_expanded_v1.json", "conds.pt",
                        "Cangjie5_TC.json"]))
    model = ChatterboxMultilingualTTS.from_local(ckpt, "cpu")
    kwargs = {"language_id": a.lang}
    if a.ref:
        kwargs["audio_prompt_path"] = a.ref
    wav = model.generate(a.prompt, **kwargs)
    # PCM_S 16-bit EXPLICITLY: torchaudio's default for a float tensor is a
    # float32 WAV (format tag 3), which bench/audiocheck.py refuses by
    # design — measured 01.09.2026, first fenced run, one traceback.
    torchaudio.save(a.out, wav, model.sr, encoding="PCM_S",
                    bits_per_sample=16)
    print("wrote %s (%.1f s of audio at %d Hz)"
          % (a.out, wav.shape[-1] / model.sr, model.sr))
    return 0


if __name__ == "__main__":
    sys.exit(main())
