# Tests

Five entry points, answering different questions. They do not replace one
another.

| With | What it answers | Needs | Duration |
|---|---|---|---|
| `bash tests/run.sh` | Is the logic right? (1046 tests) | nothing | ~15 s |
| `bash tests/live_prefix.sh` | Do saving **and** restoring really bite? | GPU, running stack | ~4 min |
| `bash tests/live_answer_freshness.sh` | Does a RESTORED prefix still answer the CURRENT question? | GPU, running stack | ~3 min |
| `bash tests/live_concurrency.sh` | Does the admission control hold under load? | GPU, running stack, a token | ~2 min |
| `bash setup/smoketest.sh` | Does the protection hold? | running stack, tunnel | ~20 s |
| `bash setup/check.sh` | Does what the repo says actually run? | running system | ~2 s |

## Why the unit tests

Until August 2026 the stack was only ever checked through the running service.
That let a whole class of bugs through: the ones that **break nothing but
simply let an effect fail to appear**. The most expensive find was exactly one
of those — the automatic prefix saving ran, reported success, wrote its file,
and the prefix was still never restored, because it sat under a key that no
incoming request produces. No error code, no warning, just 110 seconds of cold
start again.

You do not find bugs like that by touching the service. You find them by
nailing down the **contracts** between the parts. Five of them are written down
here:

- The stable prefix has to stay equal character for character across turns
  (`TestCorrection`). If `VOLATILE` one day no longer matches the
  `<total_tokens>` counter, *every* request runs cold, without a message.
- The gateway's id and the key in the store must be the same number
  (`TestIdContract`, `TestAutoSave`, `TestSave`).
- The router must not let a credential header through to the foreign server
  (`TestLocalBranch`). The damage would happen at the consumer, not at the
  operator — who would therefore never notice.
- An answer without `usage` must not produce a rate (`TestEvaluate`). It used
  to yield "-0.0 %", which reads like a finding ("the cache does not work") and
  would have travelled into the documentation that way.
- Nobody waits in the queue forever and nobody waits in silence
  (`TestPriorityGate`, `TestZoneRemote`). Strict priority had starved a LAN
  caller past 200 s under four local streams, and a queued caller got no sign
  of life at all — Cloudflare drops those after 125 s.
- The two dialects must never share a prefix id (`TestPrefixId`). Anthropic
  and OpenAI render the same conversation differently; one id for both would
  make them evict each other from the slot every turn.
- A rewritten message keeps the content shape its endpoint expects
  (`TestMidSystemToUser`). Anthropic blocks handed to an OpenAI endpoint are
  a 400 at best and a silently truncated prompt at worst.
- A caller that vanishes frees its slot at once (`TestClientAbort`). The
  accounting was built on aiohttp cancelling the handler — true when written,
  opt-in since 3.9. Without it the gateway held a slot for the full
  ten-minute generation of a client that had long gone, and answered everyone
  else with 429.
- A restore only ever touches an idle server (`TestRestoreGuard`), and
  nothing restores at boot (`TestUserUnit`). A restore landing during another
  slot's prompt processing corrupts the KV state — every answer after it is
  garbage until the next restart.
- Every model the repo knows must be named in `Conflicts=` (`TestConflicts`).
  systemd has no wildcard for template instances, so that one list cannot be
  derived from `setup/env/*.env` the way everything else now is. A model
  missing from it lets TWO llama-servers start; one loses the race for port
  8080, systemd still reports `active`, and the gateway answers from whichever
  won. Nothing fails — the wrong model answers.
- One reader for systemd syntax, and every caller gets the same answer
  (`TestArgsReader`). Three parsers for `LLAMA_ARGS` existed on 26.08. and
  they disagreed: the one in `bench/suites/slot-scaling.sh` appended the words
  of the comment lines after the assignment to the server's command line, and
  an earlier one stripped the quotes out of `--chat-template-kwargs`. A bench
  harness that reads a profile differently from the service is not measuring
  the service.
- `switch-model.sh` decides everything before it touches anything
  (`TestSwitchPreflight`). Six aborts are driven against a scratch repo, and
  each asserts afterwards that nothing was written.
- Nothing on the kernel command line may be lost when one parameter changes
  (`TestSetting`). `root=UUID=…` sits next to `ttm.pages_limit=`; a regex one
  character off does not give a wrong GTT size, it gives a machine that stops
  at the initramfs prompt. `set_params` checks its own output and refuses.
- A measurement may not start a model that does not fit beside what is
  already running (`test_memory_guard.py`). GTT comes out of system RAM and
  cannot be swapped, so the failure is not a slow machine, it is a dead one —
  and the estimate must count every shard of a sharded GGUF, because
  underestimating is the dangerous direction.
- The prefix save must not depend on how the CLIENT hangs up
  (`TestSaveSurvivesTheClientLeaving`). handler_cancellation exists so a
  vanished caller frees its slot at once; under aiohttp 3.13 it also fires
  when the client closes normally, and it landed before the save was
  scheduled. Every curl-shaped consumer therefore paid a full cold start,
  every time, and nothing said so.
- A restored prefix must still answer the CURRENT question
  (`tests/live_answer_freshness.sh`). live_prefix proves a prefix comes
  back; it never checked what the model says afterwards. A consumer
  received another session's answer six times through exactly that gap,
  with `SAVED` and `RESTORED` looking healthy the whole time.

## What is checked where

| File | Subject |
|---|---|
| `test_gateway.py` | cache correction, id contract, zones, access, throttle, cancellations |
| `test_router.py` | the model switch and — above all — which headers leave the machine |
| `test_prewarm.py` | saving, cleanup rules, restoring, migration |
| `test_dialects.py` | how a request body is read — the shared truth behind the prefix id, held for BOTH dialects |
| `test_bench.py` | the place where an answer turns into a measurement |
| `test_tasks.py` | the task battery and its checkers: every checker driven with a known-good AND a known-bad answer |
| `test_sweep.py` | the measurement chain: env parsing, server start, variant files, comparison |
| `test_scripts.py` | `waitformodel`, token reading, the abort in `switch.sh`, that no file hides a standard library module, and the three properties of `llama-user@.service` (profile from `$HOME`, no boot restore, binary fallback) |
| `test_gtt.py` | editing the kernel command line, driven against this machine's real one — the one place here where a wrong string is an unbootable machine rather than a wrong number |
| `test_docs.py` | that nothing PUBLISHED points at something that is not — five files are the maintainer's own and gitignored, and making them so broke 61 references. Prose may name what is excluded; a link or a command may not |
| `test_hardware.py` | what machine this is — the GPU identified twice so the question survives a missing ROCm, the defect registry finally reading its own `applies_to.gpu`, the preflight changing nothing and refusing to scale, and the rule that a script printing a decimal must pin `LC_ALL=C` |
| `test_systemunit.py` | that the system unit is DERIVED and cannot disagree with the one that runs — the mapping is complete (no `%h` survives into a unit where it means `/root`), the ceilings and the exec chain are the user unit's, and the prose is not copied |
| `test_localenv.py` | what belongs to the MACHINE and not to the repo: the two readers of `~/.config/llm-stack.env` agree, the conventions hold no absolute path, and no line that RUNS anywhere in the repo names one computer |
| `test_budget.py` | **the one memory budget**: the arithmetic, where the KV figure comes from, that an estimate announces itself, that it fails OPEN on facts it cannot read and CLOSED on a real refusal — and the three machine hangs it exists for, with their real numbers |
| `test_memory_guard.py` | that a measurement cannot start a server which does not fit. GTT is not swappable, so starting anyway does not page — it hangs the machine, which is what happened on 26.08. |
| `test_scout.py` | judging a model before downloading it: shards are one model, companions are not candidates, and "over the cap" is never confused with "larger than the machine" |
| `test_vacuity.py` | the suite read as a subject: which tests assert ONLY inside a loop that might not run, and therefore pass when nothing was checked. One was live — the guard against the boot restore had stopped reading anything when its directive was removed |
| `test_models.py` | the model registry: that `setup/env/*.env` IS the list, that `Conflicts=` names all of it, that one reader for systemd syntax exists and every caller agrees with it, that every profile fits the machine it is on, and that no way of starting a model goes around the budget |

## In CI

    .github/workflows/tests.yml     Python 3.10, 3.12, 3.13 on every push

The whole suite runs there because it needs nothing this repo is about: no
GPU, no llama-server, no model, no service. Verified before the workflow was
VERIFIED IN CI RATHER THAN CLAIMED, since 27.08.2026: 523 tests, 6 skipped,
~10 s, on Python 3.10, 3.12 and 3.13. The repository itself runs 536; the 13
that do not travel belong to the Flash-Next watcher, which is not published.

The first run was RED, and the reason is the argument for running it at all.
Nine tests failed on a runner with 7.8 GiB of RAM — gtt.sh refuses a cap
larger than the machine, sideserver derives its ceiling from the RAM present,
and switch-model.sh runs the real memory guard, whose 6 GiB buffer floor plus
12 GiB host reserve cannot be met under about 18 GiB. All three scripts were
right; the tests were asserting the size of the machine they ran on. They
state it or scale to it now.

The same suite had been "verified" that morning against a simulated clone in
an empty environment, and that run shared the one thing that decided the
outcome: this machine. An independent check that is not independent is the
failure this repository keeps finding.

3.10 is the floor and it is a measured one: the syntax parses back to 3.8, and
`test_scripts.py` uses `sys.stdlib_module_names`, which arrived in 3.10.
Development happens on 3.14.

The shellcheck job BLOCKS since 27.08.2026. It was advisory before that, for
an honest reason that turned out to be no reason at all: shellcheck is not
installed on this machine and installing it needs root — but a container over a
`git ls-files` copy needs neither. Reading it once found 73 findings, four of
them real, and three of those were the same silent-no-op shape this repo keeps
meeting: a `printf` that swallowed two lines of the preflight's UMA branch, a
`$( [[ -r $f ]] && <"$f" || … )` that returned EMPTY exactly when the value was
known, and `cat $(ls …)` unquoted, where an empty glob makes cat read stdin and
the watch loop hang.

The gate is `--severity=warning`, currently zero. Style notes are printed in
full and do not gate: this repo uses `A && B || C` and single-quoted `$VAR` in
text meant to be read, and a gate that enforces a house style is a different
instrument from one that catches a check with no effect.

## Structure

`tests/common.py` loads the scripts by file path; they carry a hyphen in their
name and are not on the search path. The precondition is that an import stays
without consequences — which is why `cc-gateway.py`, `cc-router.py` and
`prewarm.py` each have a `main()` guard.

llama-server is replaced by a small aiohttp application that records what
arrives at it. That makes it possible to check things deterministically that
would be a gamble against the real GPU: the per-access throttle, cancelled
callers, a reload whose file was just deleted.

Standard library plus `aiohttp`, which is needed anyway — no pytest. A setup
repo whose tests need an installation first does not get run.

## What the unit tests cannot do

No test double can tell you whether llama.cpp actually accepts a restored slot
as a prefix. That is what `live_prefix.sh` is for: cold request, wait for the
automatic save, clear the slots, send the same request again — and then the
question of whether `RESTORED` appears in the log and the second request takes
seconds instead of minutes.
