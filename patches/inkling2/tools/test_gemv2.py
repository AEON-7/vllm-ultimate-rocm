# AEON-7: bf16 small-M GEMV vs torch linear (hipBLASLt), CUDA graph timing
import importlib.util, sys, types, os, time, itertools
import torch, triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice
shim = types.ModuleType("vllm.triton_utils"); shim.triton, shim.tl, shim.tldevice = triton, tl, tldevice
sys.modules.setdefault("vllm", types.ModuleType("vllm")); sys.modules["vllm.triton_utils"] = shim
D0 = os.path.expanduser("~/inkling/engines/aeon-cleanroom/vllm/vllm/models/inkling2/amd/ops")
spec = importlib.util.spec_from_file_location("gemv", f"{D0}/gemv.py"); G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)
def bench(fn, n=3):
    # fn(w) over all weight copies (> cache size): uncached bandwidth.
    for c in ws[:2]: fn(c)
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for c in ws: fn(c)
    g.replay(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): g.replay()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / (n * len(ws)) * 1e6
SWEEP = os.environ.get("SWEEP") == "1"
shapes = [(8600, 2304), (1024, 2304), (2304, 3072)]
for N, K in shapes:
    ncopy = max(2, (512 << 20) // (N * K * 2))
    ws = [(torch.randn(N, K, device="cuda") * 0.02).to(torch.bfloat16) for _ in range(ncopy)]
    w = ws[0]
    for M in (1,):
        x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
        ref = torch.nn.functional.linear(x, w)
        y = G.gemv(x, w)
        err = ((y.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
        t0 = bench(lambda c: torch.nn.functional.linear(x, c)); t1 = bench(lambda c: G.gemv(x, c))
        best = ""
        if SWEEP:
            res = []
            bm = triton.next_power_of_2(M)
            for bn, bk, nw in itertools.product((1, 2, 4, 8, 16), (128, 256, 512, 1024), (1, 2, 4, 8)):
                if K % bk or bm * bn * bk > 16384 or bm * bn * bk < 1024: continue
                try: res.append((bench(lambda c: G.gemv(x, c, cfg=(bm, bn, bk, nw)), 2), (bn, bk, nw)))
                except Exception as e: pass
            res.sort(); best = " best=" + str([(round(t,1), c, round(N*K*2/t/1e3)) for t, c in res[:4]])
        print(f"N={N} K={K} M={M}: torch {t0:.1f}us gemv {t1:.1f}us ({N*K*2/t1/1e3:.0f}GB/s) err={err:.1e}{best}", flush=True)
    del ws, w; torch.cuda.empty_cache()
