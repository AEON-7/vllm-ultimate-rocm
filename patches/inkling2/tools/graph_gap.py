# AEON-7: decode run for rocprofv3 kernel tracing (graph mode, in-process)
import os, sys, time
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
bs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
llm = LLM(model="/home/center/inkling/models/Inkling-2-Mini", max_model_len=1024, max_num_seqs=8,
          gpu_memory_utilization=0.80, enforce_eager=os.environ.get("AEON_EAGER") == "1",
          limit_mm_per_prompt={"image": 0}, enable_prefix_caching=False)
P = [TokensPrompt(prompt_token_ids=list(range(1000 + (0 if os.environ.get("AEON_SAME")=="1" else i), 1032 + (0 if os.environ.get("AEON_SAME")=="1" else i)))) for i in range(bs)]
llm.generate(P, SamplingParams(max_tokens=8, ignore_eos=True))
t = time.perf_counter(); llm.generate(P, SamplingParams(max_tokens=64, ignore_eos=True)); dt = time.perf_counter() - t
print(f"bs={bs} 64 tok: {dt/64*1000:.2f} ms/step", flush=True)
