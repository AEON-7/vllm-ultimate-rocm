# AEON-7 debug: vLLM activations via in-process forward hooks.
import os, sys, torch, collections
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
sys.path.insert(0, "/home/center/inkling/engines/aeon-cleanroom/aeon")
from dump_common import MODEL, prompt_ids, layer_of
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
tok = AutoTokenizer.from_pretrained(MODEL)
ids = prompt_ids(tok)
llm = LLM(model=MODEL, max_model_len=4096, max_num_seqs=8, max_num_batched_tokens=2048,
          gpu_memory_utilization=0.80, enforce_eager=True, limit_mm_per_prompt={"image": 0},
          enable_prefix_caching=False, max_logprobs=20)
CAP = collections.defaultdict(list)
def install(model):
    def mk(key, which):
        def hook(mod, args, out):
            o = out[0] if isinstance(out, tuple) else out
            if which == "in":
                CAP[key + ".in"].append(args[0].detach().float().reshape(-1, args[0].shape[-1]).cpu())
            CAP[key].append(o.detach().float().reshape(-1, o.shape[-1]).cpu())
        return hook
    names = []
    for name, m in model.named_modules():
        L = layer_of(name)
        if L < 0: continue
        if name.endswith(f"layers.{L}.attn"): m.register_forward_hook(mk(f"L{L}.attn", "out")); names.append(name)
        elif name.endswith(f"layers.{L}.mlp"): m.register_forward_hook(mk(f"L{L}.mlp", "in")); names.append(name)
        elif name.endswith(f"layers.{L}.attn_sconv"): m.register_forward_hook(mk(f"L{L}.attn_sconv", "out")); names.append(name)
    return names
print(llm.apply_model(install)[0][:6], flush=True)
out = llm.generate([TokensPrompt(prompt_token_ids=ids)], SamplingParams(max_tokens=1, prompt_logprobs=20))
plp = out[0].prompt_logprobs
def get(model): return dict(CAP)
cap = llm.apply_model(get)[0]
torch.save({"ids": ids, "cap": cap, "plp": [None if p is None else {k: v.logprob for k, v in p.items()} for p in plp]},
           "/home/center/inkling/engines/aeon-cleanroom/runs/vllm_dump.pt")
print({k: (len(v), tuple(v[0].shape)) for k, v in cap.items()}, flush=True)
