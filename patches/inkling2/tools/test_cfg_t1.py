# AEON-7: T=1 decode MoE per-row config sweep, rotating 4 layers (uncached), NVFP4 + MXFP8
import itertools, sys
FMT = int(os.environ.get("FMT", "0")) if "os" in dir() else 0
src = open(__file__.replace("test_cfg_t1.py", "test_moe_rows.py")).read()
src = src.replace('__file__.replace("test_moe_rows.py", "test_moe_decode.py")', '__file__.replace("test_cfg_t1.py", "test_moe_decode.py")')
import os
FMT = int(os.environ.get("FMT", "0"))
if FMT == 1:
    src = src.replace("LAYERS = [0, 1, 2, 3]", "LAYERS = [20, 21, 22, 23]")
    src = src.replace('g13 = (W["experts.w13_weight_amax"].float() / W["experts.w13_weight_outer_scale_denominator"].float()).contiguous()', "g13 = None")
    src = src.replace('g2 = (W["experts.w2_weight_amax"].float() / W["experts.w2_weight_outer_scale_denominator"].float()).contiguous()', "g2 = None")
    src = src.replace("mdec.moe_decode(x, gw, gb, 8, 8.0, A13, A2, 0, 4.0, 25.0)", "mdec.moe_decode(x, gw, gb, 8, 8.0, A13, A2, 1, 4.0, 25.0)")
exec(src.split("import itertools\nbase")[0])
T = int(os.environ.get("T", "1"))
xs = [torch.randn(T, 2304, device=dev).to(torch.bfloat16) for _ in Ls]
base = dict(mdec._CFG)
print("default", round(rbench(xs, 5)), base[(FMT, 13)], base[(FMT, 2)], flush=True)
for proj in (2, 13):
    K = 1152 if proj == 2 else 2304
    res = []
    for bn, bk, nw in itertools.product((4, 8, 16, 32, 64), (64, 128, 256, 512), (1, 2, 4, 8)):
        if K % bk or bn * bk > 16384 or bn * bk < 512: continue
        mdec._CFG[(FMT, proj)] = (bn, bk, nw)
        try: res.append((round(rbench(xs, 5)), (bn, bk, nw)))
        except Exception as e: pass
    mdec._CFG[(FMT, proj)] = base[(FMT, proj)]
    res.sort(); print(f"proj{proj}", res[:6], flush=True)
