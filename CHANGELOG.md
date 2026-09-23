# Changelog

What changed in this stack, for someone who does not want to read 228 commits.

**Versioning.** These numbers describe THIS REPOSITORY — the configuration, the
instruments and the guards — not any component it serves. A new capability
(a backend, a model, an instrument) bumps the minor (`0.x.0`); a fix, a
measurement or a documentation change bumps the patch (`0.0.x`). The component
versions that were current for each entry are named in its header, because
"faster" is meaningless without them.

**Numbers here are measured on one machine** (Ryzen AI MAX+ 395, 128 GiB UMA,
gfx1151, Fedora), and each one says when. A figure without a date and a method
is a claim; this file tries not to carry any. Where a measurement is weak, the
weakness is written next to it rather than left out.

**Entries before 0.5.0 were condensed from the git history in retrospect**, on
21.09.2026. They are a reader's summary, not a record written as it happened —
`git log` is the authority, and the commit bodies carry the measurements. Note
that the commits pushed on 14.09.2026 were re-dated in a history regroup, so
their dates are not the days the work was done; the dates in the text are.

---

## 0.5.0 — 2026-09-21

*Halogen Flash Server `0.12.3` (from `0.8.1`), llama.cpp `master-2patches`.*

### Added

*   **`CLAMP_DEEP_PROMPT_CHARS` + `CLAMP_MAX_TOKENS`: an opt-in ceiling on the
    output budget of a DEEP request** — off by default, and for most operators
    it should stay off.

    A Halogen KV reservation is prompt **plus** `max_tokens`, reserved whether
    or not it is used. Measured this day against a 524288-position pool: three
    sessions of 193k / 183k / 77k prompt tokens with 32000 output each reserve
    129% of it and re-prefill **every** follow-up turn — p50 184 s, cache hit
    rate 0.0, 32 entries evicted, and the server naming each one in the log. The
    same prompts at 8000 reserve 94% and every follow-up turn is a cache hit:
    p50 **0.95 s**, nothing evicted. 194x a turn, from one number.

    **But two deep sessions with 32000 output each fit** — 86% to 99.9% of the
    pool at 193k to 230k prompt tokens — so the common shape (a main session and
    one subagent) needs no clamp at all, and a large output budget is not the
    problem there. It is a third deep session that breaks it.

    Hence the gate is the PROMPT SIZE, in characters, not `max_tokens`. The
    first version of this clamped on the budget alone and was retired within the
    hour: a shallow request asking for 32k reserves 34k of the pool, which is
    harmless, and clamping it would cut a real answer to buy room that was never
    short. Characters rather than estimated tokens because a chars-per-token
    constant is content-dependent (~2.1 for synthetic records, 3.5-4 for prose
    and code), and a threshold that silently means something different per
    content type is worse than none.

    Each clamp logs the prompt size and puts `max_tokens_clamped_from` into the
    request, so a shortened answer is never silent — a silent clamp cost this
    project a benchmark run once, and `finish_reason: length` reads the same
    whether the model stopped or the server truncated.

    The rule is the arithmetic, not the constant:
    `sum over live sessions of (prompt + max_tokens) <= HALOGEN_KV_POOL_POSITIONS`.
    Raising the pool instead is the wrong direction — 786432 left 5.6 GiB of host
    RAM and six watchdog shutdowns.

*   **`bench/logstats.py` reads what the server actually served**, from the
    container's own finish lines, in percentiles per prompt-depth band. It
    refuses to print a max/min spread (the flat band reads 98.7x that way and
    1.54 s at p99 — the spread was one outlier against a 0.06 s minimum) and it
    prints what the log does NOT cover, because a log cannot say what was never
    run.

*   **`bench/sessionload.py` runs the load a served log cannot show**: several
    deep sessions taking turns, each with its own document, deterministic in
    (salt, session) so two runs against two images are an A/B of the images. It
    prints the reservation as a share of the pool before starting, refuses a
    warm prompt cache unless told otherwise, and reads `platform_profile` at
    both ends.

*   **The offline suite asserts the vendored front-end's four changes are
    present** instead of describing them in prose. The prose count had drifted
    in three separate places.

### Changed

*   **Halogen Flash Server `0.8.1` → `0.12.3`.** What it is worth here, measured
    at 94% pool pressure where capacity is not the binding constraint: on 0.8.1
    **5 of 15** follow-up turns lost their cache (p90 187.5 s), on 0.12.3
    **none** (p90 1.20 s); 1191 s against 466 s for the same work. What 0.12.3
    does instead is visible in the log three times as `moved … (no loss)` in 20
    to 67 ms. Cold prefill is unchanged between the releases within 2%, and the
    flat band (90% of served turns) was already at p50 1.45 s / p99 1.54 s — the
    upgrade buys reliability under contention, not speed.

    On 0.8.1 the loss was **invisible**: no log line, and `/cache` has no pool
    object before 0.11.5. A turn that cost 187 s instead of 1.2 s looked like a
    slow model.

*   **The second vendored file is retired.** `tool_parse.py` had carried nothing
    but its own header since upstream 0.6.1 adopted the streaming it once
    patched, so the mount froze a file it did not patch — and 0.10.2 added the
    merge of several leading system messages to it, which the stale copy would
    have reverted silently while `podman ps` reported the new tag. One vendored
    file remains, and an unmodified copy of the other lives under
    `tests/fixtures/` for the offline suite, never mounted.

*   **The thinking budget now depends on `max_tokens`.** Upstream 0.11.0 keeps
    `max(1024, 15%)` of `max_tokens` for the answer and cuts the think budget to
    what is left. At Claude Code's 32000 this repo's 26000 passes untouched with
    1200 to spare; below a `max_tokens` of about 30,600 it starts being cut.
    Before this, such a request reasoned to the cap and returned an empty
    `content` with `finish_reason: length`.

### Fixed

*   **The page-cache reserve was derived, not measured.** Two places claimed
    "~55 GiB left for the Linux page cache"; the figure came from a 32.5 GiB
    weights estimate, while the server measures 68.0 GiB locked in RAM. At
    `HALOGEN_KV_POOL_POSITIONS=786432` the real number on a host that also runs
    a desktop is **5.6 GiB**, so the model's 47.7 GiB lookup table does not stay
    resident and every uncached prompt reads it from disk. Measured 17.09.2026:
    lowering the pool to 524288 left 12.7 GiB, moved the decode median from 24.8
    to 39.4 t/s and the prefill median from 6.69 s to 1.51 s at an identical
    prompt shape. The sample sizes were uneven (n=5 before, n=33 after) and the
    best case was already equal — what the smaller pool removes is the tail.

### Documentation

*   **1M context is documented, and why it stays off.** It has been opt-in since
    upstream 0.2.0 (`HALOGEN_ROPE_YARN=4` with `HALOGEN_CTX=1048576`) and this
    repo had never said so. The reasons it stays off are written with their
    arithmetic: YaRN is global and costs every short turn about 0.5% perplexity,
    the pool never sizes below `HALOGEN_CTX` (~28 KiB a position, so ~28.8 GiB
    for ONE conversation where 14.4 GiB holds two 262k ones), that leaves the
    lookup table nothing in the page cache, and a user's sweep on the same
    hardware class measured 1344 s to first token at 937k tokens.

*   **The second end token now has documentary evidence.** The model's own
    `generation_config.json` declares `eos_token_id` as the list
    `[248046, 248044]` while `tokenizer_config.json` declares only `<|im_end|>`,
    and the server builds its EOS set from the latter — so `<|endoftext|>` is
    never registered and the file that declares it is never read. That the token
    is unregistered is now documented from the project's own files; that it
    caused the original runaway remains reported, not reproduced, and the defect
    entry keeps the two apart.

---

## 0.4.0 — up to 2026-09-17

*Halogen Flash Server `0.8.1`, llama.cpp `master-2patches`.*

### Added

*   **The engine bounds its own thinking.** Upstream 0.8.1's wire protocol
    (`THINK <budget> <end_id> <len> <close_ids>`) injects Qwen's closing
    sentence and continues the answer in the same stream, which retired a
    Python-level abort-and-continue workaround in the vendored front end.
*   **`HALOGEN_FIT_TO_ROOM`**: a `max_tokens` that does not fit the window the
    prompt left is clamped to that room instead of refused with a 400 the client
    cannot tell from a stall. Off by default, per unit, and never on a bench side
    server. Claude Code sends 32000 against a 262144 window, so a session past
    ~230k tokens hits this.
*   **The sampling point from the model card** (`T=1.0 / top_p 0.95 / top_k 20`),
    after greedy decoding at depth produced deterministic reasoning loops past
    32k tokens.

### Changed

*   The KV pool went 524288 → 786432 and back to 524288 on 17.09.2026. 524288
    was exactly two full 262144 windows, so a third deep session evicted the
    others on alternating turns; 786432 held three but left 5.6 GiB of host RAM
    and produced six engine-watchdog container shutdowns between 12.09. and
    17.09. See 0.5.0's *Fixed*, and the `CLAMP_MAX_TOKENS` entry for what the
    right lever turned out to be.

---

## 0.3.0 — up to 2026-09-14

*Halogen Flash Server `0.6.3`.*

### Added

*   **Vision.** A multimodal tower on the container backend, plus a dedicated
    CPU vision sidecar (Qwen3-VL 4B) with dual-gate routing, CCD isolation and
    an on-demand lifecycle, so an image turn does not take the GPU from a text
    session.
*   **Tool-call streaming that does not hold its breath.** Upstream 0.6.1
    adopted the incremental parameter streaming this repo had vendored; the
    gateway translates upstream SSE comments into Anthropic pings, which is what
    stops a long prefill from being killed as a 524 at the edge.

---

## 0.2.0 — up to 2026-09-14

*Halogen Flash Server as the second backend.*

### Added

*   **A second backend that speaks OpenAI while the consumer speaks
    Anthropic**: the bridge in the gateway, a real prefix store instead of the
    phantom one, and model modes that survive a `switch-model`.

---

## 0.1.0 — 2026-08-27

*llama.cpp backend, first public release.*

### Added

*   The llama.cpp stack this repository started as: per-model profiles whose
    every figure carries its measurement, the gateway with its prefix store and
    trace, the memory guard that refuses a start the machine cannot hold, the
    build family mechanics, and `setup/defects.json` as the place a measured
    defect goes instead of a comment.
