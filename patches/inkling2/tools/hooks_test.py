# AEON-7: functional test of AEON_WEIGHT_OVERLAY, AEON_ABLATE and AEON_CAPTURE_*.
import os, sys, glob, torch
R = "/home/center/inkling/engines/aeon-cleanroom/runs/hooks_test"
os.makedirs(R, exist_ok=True)
for f in glob.glob(f"{R}/cap/*"): os.remove(f)
from safetensors.torch import save_file
D = 2304
save_file({"model.llm.layers.12.attn.wo_ud.weight": torch.zeros(D, 3072, dtype=torch.bfloat16)}, f"{R}/overlay.safetensors")
g = torch.Generator().manual_seed(0)
dirs = torch.randn(48, D, generator=g); scales = torch.zeros(48); scales[13] = 1.0; scales[25] = 0.6
save_file({"dirs": dirs, "scales": scales}, f"{R}/ablate.safetensors")
os.environ.update(AEON_WEIGHT_OVERLAY=f"{R}/overlay.safetensors", AEON_ABLATE=f"{R}/ablate.safetensors",
                  AEON_CAPTURE_DIR=f"{R}/cap", AEON_CAPTURE_APPS="0-47")
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
llm = LLM(model="/home/center/inkling/models/Inkling-2-Mini", max_model_len=1024, max_num_seqs=4,
          gpu_memory_utilization=0.80, enforce_eager=True, limit_mm_per_prompt={"image": 0}, enable_prefix_caching=False)
prompts = [list(range(1000, 1037)), list(range(2000, 2061))]
llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], SamplingParams(max_tokens=1))
files = sorted(glob.glob(f"{R}/cap/*.pt"))
print("capture files", len(files))
cap = torch.load(files[-1])
print("positions", cap["positions"].tolist(), "last_ids", cap["last_ids"].tolist())
for a in (12, 24, 36):
    print(f"app{a} oproj_out absmax={cap['oproj_out'][a].abs().max().item():.3g} (overlay zero -> expect 0)")
u = dirs / dirs.norm(dim=-1, keepdim=True)
for a, s in ((13, 1.0), (25, 0.6), (14, 0.0)):
    y = cap["oproj_out"][a]; proj = (y @ u[a]).abs().max().item(); nrm = y.norm(dim=-1).max().item()
    print(f"app{a} scale={s} |y.r| max={proj:.4g} |y| max={nrm:.4g}")
print("kinds", {k: len(v) for k, v in cap.items() if isinstance(v, dict)}, "resid shape", tuple(cap["resid"][47].shape))
