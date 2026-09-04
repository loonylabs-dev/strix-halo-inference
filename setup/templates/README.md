# setup/templates — chat templates a GGUF does not carry

A model's chat template normally lives inside its `.gguf`, as
`tokenizer.chat_template`, and llama-server reads it from there. This
directory exists for the case where a published quant simply does not have
one, and the first such case arrived on 04.09.2026.

**A missing template does not fail — it substitutes.** llama-server falls back
to a built-in default and serves. What comes out is a model prompted in a
framing it was never trained on, and the symptoms look like a bad quant rather
than like missing metadata. Measured on `GLM-4.7-Flash-MTP-Q4_K_M.gguf`
(meshllm), against the same model's Q8_0 (jacek2024) which does carry one:

| | with the file's own fallback | with this template |
|---|---|---|
| `copy`, share actually copied | 3.1 / 7.9 / 0.0 % | 88.9 / 96.2 / 88.9 % |
| `prose`, n-gram draft acceptance | 72.0 % at d512 | 0.0 % |
| prompt length, identical prompts | +16 tokens | exactly the Q8's |

The n-gram figure is the one worth keeping. That drafter drafts FROM THE
PROMPT, so on novel prose its acceptance is near zero when the model is
writing and high when the model is repeating. 72 % is a degeneracy meter that
happened to be running.

Files here are extracted from a sibling GGUF of the SAME model — never
written by hand and never borrowed from a different model. A profile points at
one with `--chat-template-file @REPO@/setup/templates/<name>.jinja`.
