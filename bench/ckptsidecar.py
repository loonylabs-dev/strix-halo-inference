#!/usr/bin/env python3
"""ckptsidecar — read the context checkpoints beside a saved slot state.

    python3 bench/ckptsidecar.py ~/.cache/llama-slots/session-abc.bin.ckpt

Why this is an instrument and not a debugging one-liner. A hybrid slot cannot
be trimmed to an arbitrary prefix, so the only rollback llama.cpp has is a
context checkpoint (setup/defects.json,
hybrid-checkpoints-round-partial-reuse-down). Which checkpoints exist, and how
far apart they sit, therefore DECIDES how much a warm request re-prefills —
and until now nothing in this repo could see them. The trace shows the
consequence (`reused` falling back to a fixed value) and never the cause.

Measured 07.09.2026 on a live session: the seven `reused` plateaus in that
day's trace were the seven checkpoint positions in this file, exactly. That is
what makes the sidecar the right place to measure a checkpoint-grid change.

The file is written by setup/patches/restore-carries-checkpoints.patch. Format,
little-endian and densely packed:

    u32 magic 0x504B434C ("LCKP"), u32 version 1, u32 count
    per checkpoint: i64 n_tokens, i32 pos_min, i32 pos_max,
                    then three blobs (u64 length + bytes):
                    data_tgt (target state), data_dft (draft state),
                    data_spec (speculative sampler state)

Blobs are seeked over, never read, so a multi-GiB sidecar costs one stat and a
few dozen small reads. That matters: this runs while a measurement is going on
and must not itself become a load.
"""
import argparse
import json
import struct
import sys

MAGIC = 0x504B434C
VERSION = 1
MIB = 1048576.0


class SidecarError(Exception):
    """The file is not a sidecar this reader understands."""


def _blob_len(f):
    raw = f.read(8)
    if len(raw) != 8:
        raise SidecarError("truncated blob header")
    (n,) = struct.unpack("<Q", raw)
    f.seek(n, 1)
    return n


def read(path):
    """[{index, n_tokens, pos_min, pos_max, tgt, dft, spec, size}], sizes in bytes.

    Raises SidecarError rather than returning a half-read list: a partial
    answer here would be read as "these are the checkpoints", and a missing
    one is the whole finding in a checkpoint-grid comparison.
    """
    out = []
    with open(path, "rb") as f:
        head = f.read(12)
        if len(head) != 12:
            raise SidecarError("file shorter than its header")
        magic, version, count = struct.unpack("<III", head)
        if magic != MAGIC:
            raise SidecarError("bad magic 0x%08X (expected 0x%08X)" % (magic, MAGIC))
        if version != VERSION:
            raise SidecarError("version %d, this reader knows %d" % (version, VERSION))
        for i in range(count):
            raw = f.read(16)
            if len(raw) != 16:
                raise SidecarError("truncated entry %d of %d" % (i, count))
            n_tokens, pos_min, pos_max = struct.unpack("<qii", raw)
            tgt, dft, spec = _blob_len(f), _blob_len(f), _blob_len(f)
            out.append({"index": i, "n_tokens": n_tokens,
                        "pos_min": pos_min, "pos_max": pos_max,
                        "tgt": tgt, "dft": dft, "spec": spec,
                        "size": tgt + dft + spec})
    return out


def summarise(cps):
    """The three numbers a grid comparison actually turns on."""
    if not cps:
        return {"count": 0, "bytes": 0, "tail_gap": None, "max_gap": None}
    gaps = [b["n_tokens"] - a["n_tokens"] for a, b in zip(cps, cps[1:])]
    return {"count": len(cps),
            "bytes": sum(c["size"] for c in cps),
            # How far the NEWEST checkpoint sits behind the end of what the
            # slot holds — the distance a warm request falls back by when its
            # own end checkpoint has been pruned.
            "tail_gap": cps[-1]["n_tokens"] - cps[-2]["n_tokens"] if len(cps) > 1 else None,
            "max_gap": max(gaps) if gaps else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path", help="a *.bin.ckpt sidecar")
    ap.add_argument("--json", action="store_true",
                    help="machine-readable, for a report to carry verbatim")
    a = ap.parse_args(argv)

    try:
        cps = read(a.path)
    except (SidecarError, OSError) as e:
        print("cannot read %s: %s" % (a.path, e), file=sys.stderr)
        return 1

    s = summarise(cps)
    if a.json:
        print(json.dumps({"path": a.path, "checkpoints": cps, "summary": s},
                         indent=2))
        return 0

    print("%d checkpoint(s), %.2f GiB total" % (s["count"], s["bytes"] / 1024 / MIB))
    prev = None
    for c in cps:
        gap = "" if prev is None else "  (+%d)" % (c["n_tokens"] - prev)
        prev = c["n_tokens"]
        print("  #%-2d n_tokens=%-8d pos=[%d,%d]  %7.1f MiB  "
              "[tgt %.1f / dft %.1f / spec %.1f]%s"
              % (c["index"], c["n_tokens"], c["pos_min"], c["pos_max"],
                 c["size"] / MIB, c["tgt"] / MIB, c["dft"] / MIB,
                 c["spec"] / MIB, gap))
    if s["count"]:
        print("mean %.1f MiB, largest gap %s tokens, newest gap %s tokens"
              % (s["bytes"] / s["count"] / MIB, s["max_gap"], s["tail_gap"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
