# AEON-7: greedy determinism check (same prompt, repeated, bs1 and bs8 rows)
import os, sys, hashlib
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
from vllm import LLM, SamplingParams
pc = os.environ.get("PC", "1") == "1"
llm = LLM(model=os.path.expanduser("~/inkling/models/Inkling-2-Mini"), max_model_len=2048, max_num_seqs=8,
          gpu_memory_utilization=0.80, limit_mm_per_prompt={"image": 0}, enable_prefix_caching=pc)
kw = dict(chat_template_kwargs={"reasoning_effort": "none"})
sp = SamplingParams(temperature=0.0, max_tokens=300, ignore_eos=True)
allout = {}
qs = ["What is the capital of Australia?", "Write a long story about a lighthouse keeper."]
for q in qs:
    hs = []
    for _ in range(3):
        o = llm.chat([[{"role": "user", "content": q}]], sp, **kw)[0].outputs[0]
        hs.append(o.token_ids)
    first_div = [next((i for i, (a, b) in enumerate(zip(hs[0], h)) if a != b), None) for h in hs[1:]]
    allout[(q, 1)] = list(hs[0])
    print("BS1", q[:30], "div_at", first_div, flush=True)
    outs = llm.chat([[{"role": "user", "content": q}]] * 8, sp, **kw)
    ids = [o.outputs[0].token_ids for o in outs]
    allout[(q, 8)] = [list(r) for r in ids]
    print("BS8", q[:30], "row_div_at", [next((i for i, (a, b) in enumerate(zip(hs[0], r)) if a != b), None) for r in ids], flush=True)

import torch
if os.environ.get("AEON_DET_OUT"): torch.save(allout, os.environ["AEON_DET_OUT"])
