# AEON-7 debug: per-module activation capture shared by ref/vllm dumps.
import json, re
MODEL = "/home/center/inkling/models/Inkling-2-Mini"
KL = "/home/center/inkling/phase3/stock_kl"

def prompt_ids(tok, row=0, extra=24):
    idx = json.load(open(f"{KL}/index.json"))
    rows = idx["rows"] if isinstance(idx, dict) and "rows" in idx else idx
    r = rows[row]
    gens = {}
    for line in open(f"{KL}/stock_kl_gens.jsonl"):
        g = json.loads(line); gens[g["id"]] = g
    g = gens[r["id"]]
    text = tok.apply_chat_template([{"role": "user", "content": g["prompt"]}], tokenize=False,
                                   add_generation_prompt=True, reasoning_effort="none")
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return ids + list(g["gen_ids"][:extra])

def layer_of(name):
    m = re.search(r"layers\.(\d+)", name)
    return int(m.group(1)) if m else -1
