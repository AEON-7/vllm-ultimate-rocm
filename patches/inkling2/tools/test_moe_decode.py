# AEON-7: grouped decode MoE vs reference path (real weights), plus config sweep
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
SWEEP = os.environ.get("SWEEP") == "1"
def deint(t):
    e, n, k = t.shape; return t.view(e, n // 2, 2, k).transpose(1, 2).contiguous().view(e, n, k)
def bench(fn, n=40):
    # GPU time via CUDA graph replay (eager launch overhead excluded).
    for _ in range(2): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10): fn()
    g.replay(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n // 10 + 1): g.replay()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / ((n // 10 + 1) * 10) * 1e6
with safe_open(F, "pt", device="cpu") as f:
    for layer, fmt in ((0, 0), (20, 1)):
        p = f"model.llm.layers.{layer}.mlp."
        W = {k[len(p):]: f.get_tensor(k).to(dev) for k in f.keys() if k.startswith(p)}
        w13 = deint(W["experts.w13_weight"].view(torch.uint8)); s13 = deint(W["experts.w13_weight_scale"].view(torch.uint8))
        w2 = W["experts.w2_weight"].view(torch.uint8); s2 = W["experts.w2_weight_scale"].view(torch.uint8)
        g13 = g2 = None
        if fmt == 0:
            g13 = (W["experts.w13_weight_amax"].float() / W["experts.w13_weight_outer_scale_denominator"].float()).contiguous()
            g2 = (W["experts.w2_weight_amax"].float() / W["experts.w2_weight_outer_scale_denominator"].float()).contiguous()
        gw = W["gate.weight"].contiguous(); gb = W["gate.bias"].float().contiguous()
        A13, A2 = (w13, s13, g13), (w2, s2, g2)
        for T in (1, 2, 3, 4, 8, 16):
            x = torch.randn(T, 2304, device=dev).to(torch.bfloat16)
            sc = torch.sigmoid(x.float() @ gw.float().t()); ids_r = torch.topk(sc + gb, 8, dim=-1, sorted=False).indices
            wr = sc.gather(1, ids_r); wr = wr / (wr.sum(-1, keepdim=True) + 1e-20) * 8.0
            ref = md.expert_mlp(x, ids_r, wr, A13, A2, fmt, 4.0, 25.0)
            out, ids, wts = mdec.moe_decode(x, gw, gb, 8, 8.0, A13, A2, fmt, 4.0, 25.0)
            torch.cuda.synchronize()
            same = (torch.sort(ids.long(), 1).values == torch.sort(ids_r, 1).values).all().item()
            rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
            beq = (out == ref).float().mean().item()
            t_new = bench(lambda: mdec.moe_decode(x, gw, gb, 8, 8.0, A13, A2, fmt, 4.0, 25.0))
            nb = len(ids.unique()) * (w13[0].numel() + s13[0].numel() + w2[0].numel() + s2[0].numel())
            t_old = bench(lambda: md.expert_mlp(x, ids_r, wr, A13, A2, fmt, 4.0, 25.0))
            print(f"old={t_old:.0f}us ", end="")
            print(f"L{layer} T={T} ids_same={same} rel={rel:.2e} biteq={beq:.3f} t={t_new:.0f}us eff_bw={nb/t_new/1e3:.0f}GB/s", flush=True)
            if SWEEP and T in (1, 8):
                for proj in (13, 2):
                    best = []
                    K = 2304 if proj == 13 else 1152
                    for bn, bk, nw in itertools.product((8, 16, 32), (64, 128, 256), (2, 4, 8)):
                        if K % bk: continue
                        cfg = dict(mdec._CFG); cfg[(fmt, proj)] = (bn, bk, nw)
                        try:
                            best.append((bench(lambda: mdec.moe_decode(x, gw, gb, 8, 8.0, A13, A2, fmt, 4.0, 25.0, cfg=cfg), 20), (bn, bk, nw)))
                        except Exception as e:
                            print("fail", bn, bk, nw, str(e)[:80])
                    best.sort(); print(f"   sweep L{layer} T={T} proj{proj}:", [(f"{t:.0f}", c) for t, c in best[:4]], flush=True)
        del W; torch.cuda.empty_cache()

# ---- component timing (T=1 and 8, layer 0) ----
if os.environ.get("PARTS") == "1":
    with safe_open(F, "pt", device="cpu") as f:
        p = "model.llm.layers.0.mlp."
        W = {k[len(p):]: f.get_tensor(k).to(dev) for k in f.keys() if k.startswith(p)}
    w13 = deint(W["experts.w13_weight"].view(torch.uint8)); s13 = deint(W["experts.w13_weight_scale"].view(torch.uint8))
    w2 = W["experts.w2_weight"].view(torch.uint8); s2 = W["experts.w2_weight_scale"].view(torch.uint8)
    g13 = (W["experts.w13_weight_amax"].float() / W["experts.w13_weight_outer_scale_denominator"].float()).contiguous()
    g2 = (W["experts.w2_weight_amax"].float() / W["experts.w2_weight_outer_scale_denominator"].float()).contiguous()
    gw = W["gate.weight"].contiguous(); gb = W["gate.bias"].float().contiguous()
    for T in (1, 8):
        x = torch.randn(T, 2304, device=dev).to(torch.bfloat16)
        E = 128; rows = T * 8; maxr = triton.next_power_of_2(T); n_act = min(E, triton.next_power_of_2(rows))
        scratch = torch.zeros(E + 1 + n_act, dtype=torch.int32, device=dev)
        count, nactive, active = scratch[:E], scratch[E:E + 1], scratch[E + 1:]
        bucket = torch.empty((E, maxr), dtype=torch.int32, device=dev)
        ids = torch.empty((T, 8), dtype=torch.int32, device=dev); wts = torch.empty((T, 8), dtype=torch.float32, device=dev)
        def init():
            scratch[:E + 1].zero_(); scratch[E + 1:].fill_(-1)
        logits = torch.empty((T, E), dtype=torch.float32, device=dev)
        def route():
            be, bd, nwr = mdec._ROUTER_CFG[maxr]
            mdec._router_logits_kernel[(E // be,)](x, gw, logits, scratch, active, T, D=2304, E=E, N_ACT=n_act, BE=be, BD=bd, BT=maxr, num_warps=nwr)
            mdec._route_kernel[(T,)](logits, gb, ids, wts, count, bucket, active, nactive, E=E, TOPK=8, SCALE=8.0, MAXR=maxr, num_warps=mdec.TOPK_WARPS)
        if os.environ.get("RSWEEP") == "1":
            for be in (1, 2, 4, 8):
                for bd in (32, 64, 128, 256, 512):
                    for nwr in (1, 2, 4, 8):
                        if be * bd * maxr > 16384: continue
                        fn = lambda: mdec._router_logits_kernel[(E // be,)](x, gw, logits, scratch, active, T, D=2304, E=E, N_ACT=n_act, BE=be, BD=bd, BT=maxr, num_warps=nwr)
                        try: print(f"  rs T={T} be={be} bd={bd} nw={nwr}: {bench(fn):.1f}us")
                        except Exception as ex: print("  rs fail", be, bd, nwr, str(ex)[:60])
        route(); torch.cuda.synchronize()
        act = torch.empty((rows, 1152), dtype=torch.bfloat16, device=dev); down = torch.empty((rows, 2304), dtype=torch.bfloat16, device=dev)
        out = torch.empty_like(x)
        f13 = lambda: mdec._expert_gemv_kernel[(1152 // 16, n_act, maxr)](x, w13, s13, g13, act, count, bucket, active, TOPK=8, N=2304, K=2304, OUT_N=1152, MAXR=maxr, FMT=0, GATED=True, GATHER=True, HW_FP8=True, GATE_CAP=4.0, UP_CAP=25.0, BN=16, BK=256, num_warps=4, ROUND=False)
        f2 = lambda: mdec._expert_gemv_kernel[(2304 // 16, n_act, maxr)](act, w2, s2, g2, down, count, bucket, active, TOPK=8, N=2304, K=1152, OUT_N=2304, MAXR=maxr, FMT=0, GATED=False, GATHER=False, HW_FP8=True, GATE_CAP=4.0, UP_CAP=25.0, BN=16, BK=128, num_warps=2, ROUND=False)
        fc = lambda: mdec._combine_kernel[(T, 2304 // 256)](down, wts, out, TOPK=8, D=2304, BD=256, num_warps=4)
        f13n = lambda: mdec._expert_gemv_kernel[(1152 // 16, n_act, maxr)](x, w13, s13, g13, act, count, bucket, active, TOPK=8, N=2304, K=2304, OUT_N=1152, MAXR=maxr, FMT=0, GATED=True, GATHER=True, HW_FP8=True, GATE_CAP=4.0, UP_CAP=25.0, BN=16, BK=128, num_warps=4, ROUND=False)
        print(f"w13 noround={bench(f13n):.1f}us")
        print(f"T={T} route={bench(route):.1f}us w13={bench(f13):.1f}us w2={bench(f2):.1f}us combine={bench(fc):.1f}us", flush=True)
