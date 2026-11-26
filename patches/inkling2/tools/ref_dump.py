# AEON-7 debug: transformers reference activations on CPU (bf16).
import sys, torch, collections
sys.path.insert(0, "/home/center/inkling/engines/aeon-cleanroom/aeon")
from dump_common import MODEL, prompt_ids, layer_of
from transformers import AutoModelForMultimodalLM, AutoTokenizer
torch.set_num_threads(24)
tok = AutoTokenizer.from_pretrained(MODEL)
ids = prompt_ids(tok)
print("ntok", len(ids), flush=True)
model = AutoModelForMultimodalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cpu",
        low_cpu_mem_usage=True, attn_implementation="eager", experts_implementation="eager")
model.eval()
cap = collections.defaultdict(list)
def mk(key, which):
    def hook(mod, args, out):
        o = out[0] if isinstance(out, tuple) else out
        if which == "in":
            cap[key + ".in"].append(args[0].detach().float().reshape(-1, args[0].shape[-1]).clone())
        cap[key].append(o.detach().float().reshape(-1, o.shape[-1]).clone())
    return hook
for name, m in model.named_modules():
    cn = type(m).__name__
    L = layer_of(name)
    if L < 0 or "vision" in name: continue
    if name.endswith("self_attn"): m.register_forward_hook(mk(f"L{L}.attn", "out"))
    elif name.endswith(".mlp") and "SparseMoe" in cn: m.register_forward_hook(mk(f"L{L}.mlp", "in"))
    elif name.endswith("post_attention_convolution"): m.register_forward_hook(mk(f"L{L}.attn_sconv", "out"))
print([n for n, _ in model.named_modules() if n.endswith("layers.6") or n.endswith("embed_tokens")], flush=True)
with torch.no_grad():
    out = model(input_ids=torch.tensor([ids]), use_cache=False)
cap["logits"] = [out.logits[0].float()]
torch.save({"ids": ids, "cap": dict(cap)}, "/home/center/inkling/engines/aeon-cleanroom/runs/ref_dump.pt")
print({k: len(v) for k, v in cap.items()}, flush=True)
