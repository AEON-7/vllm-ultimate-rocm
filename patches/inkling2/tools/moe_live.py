# AEON-7 debug: run vLLM-loaded MoE layers on reference inputs
import os, sys, torch
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
from vllm import LLM
M = "/home/center/inkling/models/Inkling-2-Mini"
R = torch.load("/home/center/inkling/engines/aeon-cleanroom/runs/ref_dump.pt")["cap"]
llm = LLM(model=M, max_model_len=512, max_num_seqs=2, gpu_memory_utilization=0.75, enforce_eager=True,
          limit_mm_per_prompt={"image": 0}, enable_prefix_caching=False)
from vllm.models.inkling2.amd.ops import moe_dequant as md
def run(model):
    out = []
    for L in (0, 1, 20):
        mlp = model.model.layers[L].mlp
        x = R[f"L{L}.mlp.in"][0].cuda().to(torch.bfloat16)
        ref = R[f"L{L}.mlp"][0].cuda()
        y = mlp(x).float()
        ids, w = mlp.gate(x)
        E = mlp.experts
        g13 = E.global_scales.get("w13"); g2 = E.global_scales.get("w2")
        def dq(name, gs, e):
            W = getattr(E, name + "_weight").data[e:e+1]; S = getattr(E, name + "_weight_scale").data[e:e+1]
            return md.dequant_reference(W.view(torch.uint8) if E.nvfp4 else W, S, None if gs is None else gs[e:e+1])[0]
        acc = torch.zeros_like(x, dtype=torch.float32)
        for e in ids.unique().tolist():
            tok, slot = torch.where(ids == e)
            h = torch.nn.functional.linear(x[tok], dq('w13', g13, e)); g, u = h.chunk(2, -1); g, u = g.float(), u.float()
            a = (torch.tanh(g/4)*4*torch.sigmoid(g)*torch.tanh(u/25)*25).to(torch.bfloat16)
            acc.index_add_(0, tok, torch.nn.functional.linear(a, dq('w2', g2, e)).float() * w[tok, slot, None])
        tr = acc.to(torch.bfloat16).float()
        f = lambda a, b: ((a-b).norm()/b.norm()).item()
        out.append(f"L{L} kernel_vs_ref={f(y,ref):.4f} torch_vs_ref={f(tr,ref):.4f} kernel_vs_torch={f(y,tr):.4f} nvfp4={E.nvfp4} g13={None if g13 is None else g13[:3].tolist()}")
    return out
print("\n".join(llm.apply_model(run)[0]))
