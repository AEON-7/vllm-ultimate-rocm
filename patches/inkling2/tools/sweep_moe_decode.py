# AEON-7: config sweep for decode MoE kernels (layer 0 NVFP4, layer 20 MXFP8)
import importlib.util, sys, types, os, time, itertools
import torch, triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice
shim = types.ModuleType("vllm.triton_utils"); shim.triton, shim.tl, shim.tldevice = triton, tl, tldevice
sys.modules.setdefault("vllm", types.ModuleType("vllm")); sys.modules["vllm.triton_utils"] = shim
D0 = os.path.expanduser("~/inkling/engines/aeon-cleanroom/vllm/vllm/models/inkling2/amd/ops")
pkg = types.ModuleType("aeonops"); pkg.__path__ = [D0]; sys.modules["aeonops"] = pkg
def load(name):
    spec = importlib.util.spec_from_file_location(f"aeonops.{name}", f"{D0}/{name}.py")
    m = importlib.util.module_from_spec(spec); sys.modules[f"aeonops.{name}"] = m; spec.loader.exec_module(m); return m
md = load("moe_dequant"); mdec = load("moe_decode")
from safetensors import safe_open
F = os.path.expanduser("~/inkling/models/Inkling-2-Mini/model-00001-of-00001.safetensors")
dev = "cuda"; torch.manual_seed(0)
layer = int(sys.argv[1]); fmt = 0 if layer < 20 else 1
with safe_open(F, "pt", device="cpu") as f:
    p = f"model.llm.layers.{layer}.mlp."
    W = {k[len(p):]: f.get_tensor(k).to(dev) for k in f.keys() if k.startswith(p)}
w13 = W["experts.w13_weight"].view(torch.uint8); s13 = W["experts.w13_weight_scale"].view(torch.uint8)
w2 = W["experts.w2_weight"].view(torch.uint8); s2 = W["experts.w2_weight_scale"].view(torch.uint8)
g13 = g2 = None
if fmt == 0:
    g13 = (W["experts.w13_weight_amax"].float() / W["experts.w13_weight_outer_scale_denominator"].float()).contiguous()
    g2 = (W["experts.w2_weight_amax"].float() / W["experts.w2_weight_outer_scale_denominator"].float()).contiguous()
def bench(fn, n=40):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e6
for T in (1, 8):
    ids = torch.stack([torch.randperm(128, device=dev)[:8] for _ in range(T)]).to(torch.int32)
    wts = torch.rand(T, 8, device=dev)
    x = torch.randn(T, 2304, device=dev).to(torch.bfloat16)
    inter = 1152; act = torch.randn(T * 8, inter, device=dev).to(torch.bfloat16); out = torch.empty_like(x)
    res13, res2 = [], []
    for bn, bk, nw in itertools.product((8, 16, 32), (64, 128, 256, 384), (2, 4, 8)):
        if 2304 % bk == 0:
          try:
            t = bench(lambda: mdec._w13_gemv_kernel[(inter // bn, T * 8)](x, ids, w13, s13, s13 if g13 is None else g13, act,
                TOPK=8, N=2304, K=2304, INTER=inter, FMT=fmt, GATE_CAP=4.0, UP_CAP=25.0, BN=bn, BK=bk, num_warps=nw))
            res13.append((t, bn, bk, nw))
          except Exception as e:
            pass
        if inter % bk == 0:
            try:
                t = bench(lambda: mdec._w2_gemv_kernel[(2304 // bn, T)](act, ids, wts, w2, s2, s2 if g2 is None else g2, out,
                    TOPK=8, N=2304, K=inter, FMT=fmt, BN=bn, BK=bk, num_warps=nw))
                res2.append((t, bn, bk, nw))
            except Exception as e:
                pass
    nb13 = len(ids.unique()) * (w13[0].numel() + s13[0].numel()); nb2 = len(ids.unique()) * (w2[0].numel() + s2[0].numel())
    for name, res, nb in (("w13", res13, nb13), ("w2", res2, nb2)):
        res.sort()
        print(f"L{layer} T={T} {name} best:", [(f"{t:.0f}us", bn, bk, nw, f"{nb/t/1e3:.0f}GB/s") for t, bn, bk, nw in res[:4]], flush=True)
