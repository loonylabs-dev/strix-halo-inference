# Reporting a vulnerability

**info@loonylabs.dev**, or GitHub's private vulnerability reporting if it is
enabled on this repository.

Please do not open a public issue for something exploitable. There is no
bounty and no SLA — this is one person's machine documentation, not a product
with a security team — but a report will be read and answered.

## What is worth reporting

This repository is mostly configuration and measurement, and most of it cannot
be attacked. One part can: `setup/claude/cc-gateway.py` sits in front of a
local inference server and is deliberately reachable from the internet through
a Cloudflare tunnel. It decides three things, and a mistake in any of them
matters:

* **which zone a request is in** — local, LAN, or remote. The port decides
  before the IP does, because behind a tunnel the source address is
  `cloudflared`'s and looks private.
* **which paths a remote caller may reach.** An allow list, checked BEFORE the
  token, so a 404 does not reveal whether a path exists. `/completion` is off
  it on purpose: it takes a raw prompt and bypasses the chat template.
* **whether a token is valid**, and which consumer it belongs to.

`docs/SECURITY.md` describes all three in detail, along with what has actually
been measured about them — including a case where `/gateway/status` classified
by source IP alone and would have been readable from the internet had
`cloudflared` run natively instead of in a container. That document is the
security MODEL. This one is where you send a finding about it.

## What is not a vulnerability here

* **The local zone needs no token.** `127.0.0.1` on the gateway port is
  trusted by design; if somebody is already on that loopback they have the
  model anyway.
* **A model producing wrong or harmful output.** This repository serves
  whatever GGUF it is pointed at and makes no claim about what the weights do.
* **Memory exhaustion by a large profile.** It is guarded
  (`setup/lib/budget.py`) and the guard's limits are documented; a
  configuration that takes the machine down when the guard is switched off
  with `LLM_NO_MEMORY_GUARD=1` is that switch working as described.

## Please do not send

Captured request bodies. They contain an e-mail address, a `device_id`, an
`account_uuid` and Anthropic's system prompt — yours, not ours.
`tools/synthetic.py` produces equivalent bodies without any of that, and a
reproduction built on it is more useful anyway because anyone can run it.
