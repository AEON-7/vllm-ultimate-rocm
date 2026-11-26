# AEON-7: race detector: 8 identical rows x 4 repeats must give 1 distinct sequence
import os
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
from vllm import LLM, SamplingParams
llm = LLM(model=os.path.expanduser("~/inkling/models/Inkling-2-Mini"), max_model_len=2048, max_num_seqs=8,
          gpu_memory_utilization=0.80, limit_mm_per_prompt={"image": 0}, enable_prefix_caching=False)
kw = dict(chat_template_kwargs={"reasoning_effort": "none"})
sp = SamplingParams(temperature=0.0, max_tokens=300, ignore_eos=True)
for q in ["What is the capital of Australia?", "Write a long story about a lighthouse keeper."]:
    seqs = []
    for _ in range(4):
        seqs += [tuple(o.outputs[0].token_ids) for o in llm.chat([[{"role": "user", "content": q}]] * 8, sp, **kw)]
    print("DISTINCT", q[:20], len(set(seqs)), flush=True)
