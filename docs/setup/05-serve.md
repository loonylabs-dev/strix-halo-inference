# 05 · Serving a model

Twenty minutes, and at the end a coding agent is answering. Everything before
this chapter was the machine; this is the stack.

## Where the models live

They are large — the production model is 16.7 GiB and the library runs past
200 GB — so they usually get a volume of their own. On a dual-boot machine
that is often a partition shared with Windows, which brings two traps worth
knowing before you hit them:

* **`nofail` in `/etc/fstab`.** Without it a missing volume stops the boot;
  with it, a missing volume passes silently and the model service fails three
  times in fifteen seconds and then stays down for good. That is why
  `setup/waitformodel` runs as `ExecStartPre` and waits rather than failing.
* **NTFS and Windows fast startup.** If you skipped `powercfg /h off` in
  [01](01-before-you-start.md), `ntfs3` refuses the volume outright.

You do not have to tell the repository where they are. `install.sh` looks for
a directory that actually holds `.gguf` files and writes the answer down once:

    bash setup/install.sh

      -> ~/.config/llm-stack.env (from the template; stays local)
         models found at /mnt/shared/LLM

That file is the only place this machine's own answers live — model directory,
tunnel hostname if you have one. It is gitignored and never travels.

## Fetching a model

    bash setup/get-model.sh --list       what exists, and what is already here
    bash setup/get-model.sh qwen38       fetch it — resumable, sha256-checked
    bash setup/switch-model.sh qwen38    serve it

There is no `pull` command here and that is deliberate. The list is not a
generic catalogue: **every entry has been measured on this hardware**, and the
name you fetch is the name you serve.

    NAME         FILE   TITLE
    qwen38       here   Coding agent + vision + judge · 16.7 GiB · production
    gemma26      here   Fast sidekick · 13.5 GiB · MoE 128/8 active · QAT-q4_0
    gptoss       here   Judge/evals · 59.0 GiB · MXFP4 native · KV very cheap

"Coding agent + vision + judge · 16.7 GiB · production since 25.08." is a
different kind of statement from a version tag.

A model **is** its profile in `setup/env/`. Nothing else holds a list of them,
and each profile says where its weights come from, what one token of context
costs in KV, and why every flag on its command line is there.

## Starting it

    systemctl --user --now enable llama-user@qwen38     # the model server
    systemctl --user --now enable cc-gateway            # the gateway
    sudo loginctl enable-linger $USER                   # so both come up at boot
    systemctl --user --now enable prefix-cleanup.timer  # weekly cache cleanup

A **user** service, not a system one. On Fedora with SELinux a system service
may not execute a binary out of a home directory — `AVC denied`, `203/EXEC` —
and the llama.cpp build lives in `~/llama.cpp`. The user service runs in the
user context and may. `loginctl enable-linger` covers the "only starts on
login" limitation. If you want a system service anyway (a box with no user
session at all), `bash setup/install.sh --system-unit` derives one.

**Before it starts, the profile is weighed:**

    llm-check-room: qwen38 fits: 38.1 GiB in GTT, 70.1 resident of 124.9

If it does not fit, the unit refuses to start and prints the arithmetic. That
is the good outcome — GTT is pinned, so starting anyway does not fail, it
freezes the machine.

## Checking it

    bash tests/run.sh          logic, no GPU and no service needed (~8 s)
    bash setup/check.sh        configuration and state
    bash setup/smoketest.sh    function and protection, all three zones

`check.sh` compares the repository against what the running system actually
reads, and prints the memory budget of the running profile against what it
really pinned:

    = GTT predicted 38.1 GiB, observed 35.6 (-6 %) — not under-predicting

Only one direction is a defect. Observed comfortably below predicted means the
guard is conservative, which is its job.

## Pointing a client at it

    bash setup/consumer-info.sh --local

      Endpoint    http://127.0.0.1:8090
      Access      none — the local zone on 127.0.0.1 needs no token
      Window      204800 per slot — set CLAUDE_CODE_MAX_CONTEXT_TOKENS=200000
      Models      qwen38 [completion,multimodal]
                  qwen38-think [completion,multimodal]
                  qwen38-deep [completion,multimodal]

[`docs/CONSUMERS.md`](../CONSUMERS.md) has the four ways to wire Claude Code up
and the OpenAI-dialect route for other agents.

## The part that decides whether this is usable

Read the prompt-cache section of `CONSUMERS.md` before you form an opinion
about speed. The founding measurement of this whole repository is that Claude
Code against a local model cost about **140 seconds per request** — not
because the model was slow, but because the prompt cache never bit. The same
work with the cache surviving from turn to turn is **1.3 seconds**.

Four client settings decide it, and getting one wrong costs you a cold prefill
on every single turn. They are listed in `CONSUMERS.md` under *The four
mandatory settings*, with a way to check that the cache actually bites rather
than assuming it.

That gap — a hundredfold, from configuration alone — is the reason this
repository is not just a list of flags.

---

Previous: [04 · Building llama.cpp](04-build-llama.md) ·
Next: [06 · When it does not work](06-when-it-does-not-work.md)
