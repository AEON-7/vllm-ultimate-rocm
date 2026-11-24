# AEON-7: gemv3x (qkvr + K/V tap projections in one launch) == gemv3 + 2x gemv, bitwise
import importlib.util, sys, types, os
import torch, triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice
shim = types.ModuleType("vllm.triton_utils"); shim.triton, shim.tl, shim.tldevice = triton, tl, tldevice
sys.modules.setdefault("vllm", types.ModuleType("vllm")); sys.modules["vllm.triton_utils"] = shim
D0 = os.path.expanduser("~/inkling/engines/aeon-cleanroom/vllm/vllm/models/inkling2/amd/ops")
spec = importlib.util.spec_from_file_location("gemv", f"{D0}/gemv.py"); G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)
for M in (1, 2, 4, 8):
    xs = [torch.randn(M, 2304, device="cuda").to(torch.bfloat16) for _ in range(3)]
    w = (torch.randn(8600, 2304, device="cuda") * 0.02).to(torch.bfloat16)
    for na in (32, 24, 64):
        wa = (torch.randn(na, 2304, device="cuda") * 0.02).to(torch.bfloat16)
        wb = (torch.randn(na, 2304, device="cuda") * 0.02).to(torch.bfloat16)
        o, ya, yb = G.gemv3x(*xs, w, 6552, 7576, wa, wb)
        r = G.gemv3(*xs, w, 6552, 7576)
        print(M, na, torch.equal(o, r), torch.equal(ya, G.gemv(xs[1], wa)), torch.equal(yb, G.gemv(xs[2], wb)))
