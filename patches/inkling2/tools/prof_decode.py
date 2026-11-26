# AEON-7: rough per-module decode time split (synchronizing hooks; ratios only)
import os, time, collections, torch
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
llm = LLM(model="/home/center/inkling/models/Inkling-2-Mini", max_model_len=1024, max_num_seqs=8,
          gpu_memory_utilization=0.80, enforce_eager=True, limit_mm_per_prompt={"image": 0}, enable_prefix_caching=False)
T = collections.defaultdict(float); N = collections.defaultdict(int)
def install(model):
    def pre(key):
        def f(m, a): torch.cuda.synchronize(); m._t0 = time.perf_counter()
        return f
    def post(key):
        def f(m, a, o): torch.cuda.synchronize(); T[key] += time.perf_counter() - m._t0; N[key] += 1
        return f
    for name, m in model.named_modules():
        for suf, key in ((".mlp", "moe"), (".attn", "attn"), (".attn_sconv", "sconv"), (".mlp_sconv", "sconv")):
            if name.endswith(suf) and "layers." in name and name.count(".") == 3:
                m.register_forward_pre_hook(pre(key)); m.register_forward_hook(post(key))
        if name == "lm_head" or name.endswith("logits_processor"):
            pass
    tm = model
    orig = tm.forward
    def fwd(*a, **k):
        torch.cuda.synchronize(); t = time.perf_counter(); o = orig(*a, **k); torch.cuda.synchronize()
        T["forward_total"] += time.perf_counter() - t; N["forward_total"] += 1; return o
    tm.forward = fwd
    return True
llm.apply_model(install)
for bs in (1, 8):
    for k in list(T): T[k] = 0; N[k] = 0
    t = time.perf_counter()
    llm.generate([TokensPrompt(prompt_token_ids=list(range(1000 + i, 1032 + i))) for i in range(bs)],
                 SamplingParams(max_tokens=64, ignore_eos=True))
    wall = time.perf_counter() - t
    print(f"bs={bs} wall={wall:.2f}s fwd_total={T['forward_total']:.2f}s steps={N['forward_total']}",
          {k: f"{v:.2f}s/{N[k]}" for k, v in T.items() if k != 'forward_total'})
