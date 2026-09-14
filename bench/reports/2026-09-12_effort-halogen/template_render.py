import sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/models/tokenizer")
msgs = [{"role": "user", "content": "Add 2 and 2."}]
def render(**kw):
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    except Exception as e:
        return "RAISE: %s" % e
for name, kw in [("(nichts)", {}),
                 ("enable_thinking=False", {"enable_thinking": False}),
                 ("on+low", {"enable_thinking": True, "reasoning_effort": "low"}),
                 ("on+medium", {"enable_thinking": True, "reasoning_effort": "medium"}),
                 ("on+xhigh", {"enable_thinking": True, "reasoning_effort": "xhigh"}),
                 ("nur medium", {"reasoning_effort": "medium"}),
                 ("on+high", {"enable_thinking": True, "reasoning_effort": "high"})]:
    print("#"*70); print("## %s" % name); print(repr(render(**kw)))
