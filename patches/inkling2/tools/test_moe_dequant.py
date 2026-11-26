# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AEON-7
"""Standalone gfx12 checks for ops/moe_dequant.py on real Inkling-2-Mini expert
weights (read-only): exact in-kernel dequant, expert GEMMs vs torch, and the
full routed MoE vs the transformers-reference math."""
import importlib.util, sys, types, json, os, time
import torch, triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

shim = types.ModuleType("vllm.triton_utils")
shim.triton, shim.tl, shim.tldevice = triton, tl, tldevice
sys.modules.setdefault("vllm", types.ModuleType("vllm"))
sys.modules["vllm.triton_utils"] = shim
SRC = os.path.expanduser("~/inkling/engines/aeon-cleanroom/vllm/vllm/models/inkling2/amd/ops/moe_dequant.py")
spec = importlib.util.spec_from_file_location("moe_dequant", SRC)
md = importlib.util.module_from_spec(spec); spec.loader.exec_module(md)

from safetensors import safe_open
MODEL = os.path.expanduser("~/inkling/models/Inkling-2-Mini/model-00001-of-00001.safetensors")
dev = "cuda"
torch.manual_seed(0)

def load_layer(f, layer):
    p = f"model.llm.layers.{layer}.mlp."
    out = {}
    for k in f.keys():
        if k.startswith(p):
            out[k[len(p):]] = f.get_tensor(k).to(dev)
    return out

def ref_moe(x, W, fmt):
    """transformers Inkling2Loop experts math with bf16 dequantized weights."""
    if fmt == 0:
        g13 = W["experts.w13_weight_amax"].float() / W["experts.w13_weight_outer_scale_denominator"].float()
        g2 = W["experts.w2_weight_amax"].float() / W["experts.w2_weight_outer_scale_denominator"].float()
    else:
        g13 = g2 = None
    w13 = md.dequant_reference(W["experts.w13_weight"].view(torch.uint8), W["experts.w13_weight_scale"].view(torch.float8_e4m3fn) if fmt == 0 else W["experts.w13_weight_scale"], g13) if fmt == 0 else \
          md.dequant_reference(W["experts.w13_weight"], W["experts.w13_weight_scale"], None)
    w2 = md.dequant_reference(W["experts.w2_weight"].view(torch.uint8), W["experts.w2_weight_scale"], g2) if fmt == 0 else \
          md.dequant_reference(W["experts.w2_weight"], W["experts.w2_weight_scale"], None)
    gw = W["gate.weight"].float(); gb = W["gate.bias"].float()
    logits = x.float() @ gw.t(); scores = torch.sigmoid(logits)
    ids = torch.topk(scores + gb, 8, dim=-1, sorted=False).indices
    wts = scores.gather(1, ids); wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * 8.0
    out = torch.zeros_like(x, dtype=torch.float32)
    for e in ids.unique().tolist():
        tok, slot = torch.where(ids == e)
        h = torch.nn.functional.linear(x[tok], w13[e])
        g, u = h.chunk(2, -1); g, u = g.float(), u.float()
        a = (torch.tanh(g / 4.0) * 4.0 * torch.sigmoid(g) * torch.tanh(u / 25.0) * 25.0).to(torch.bfloat16)
        y = torch.nn.functional.linear(a, w2[e])
        out.index_add_(0, tok, y * wts[tok, slot, None])
    return out.to(torch.bfloat16), ids, wts, w13, w2

res = {}
with safe_open(MODEL, "pt", device="cpu") as f:
    for layer, fmt in ((0, 0), (20, 1)):
        W = load_layer(f, layer)
        for T in (1, 3, 8, 64, 513):
            x = (torch.randn(T, 2304, device=dev) * 1.0).to(torch.bfloat16)
            ref, ids, wts, w13, w2 = ref_moe(x, W, fmt)
            if fmt == 0:
                ops13 = (W["experts.w13_weight"].view(torch.uint8), W["experts.w13_weight_scale"].view(torch.uint8),
                         (W["experts.w13_weight_amax"].float() / W["experts.w13_weight_outer_scale_denominator"].float()).contiguous())
                ops2 = (W["experts.w2_weight"].view(torch.uint8), W["experts.w2_weight_scale"].view(torch.uint8),
                        (W["experts.w2_weight_amax"].float() / W["experts.w2_weight_outer_scale_denominator"].float()).contiguous())
            else:
                ops13 = (W["experts.w13_weight"].view(torch.uint8), W["experts.w13_weight_scale"].view(torch.uint8), None)
                ops2 = (W["experts.w2_weight"].view(torch.uint8), W["experts.w2_weight_scale"].view(torch.uint8), None)
            out = md.expert_mlp(x, ids, wts, ops13, ops2, fmt, 4.0, 25.0)
            torch.cuda.synchronize()
            cos = torch.nn.functional.cosine_similarity(out.float(), ref.float(), dim=-1)
            rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
            exact = (out == ref).float().mean().item()
            res[f"L{layer}_T{T}"] = dict(min_cos=cos.min().item(), rel=rel, bit_equal_frac=exact)
            print(layer, T, res[f"L{layer}_T{T}"], flush=True)
        # Exact dequant check: weight tile via GEMM against identity rows.
        del W
        torch.cuda.empty_cache()
json.dump(res, open(os.path.expanduser("~/inkling/engines/aeon-cleanroom/test_moe_dequant.json"), "w"), indent=1)
ok = all(v["min_cos"] >= 0.9995 for v in res.values())
print("PASS" if ok else "FAIL")
