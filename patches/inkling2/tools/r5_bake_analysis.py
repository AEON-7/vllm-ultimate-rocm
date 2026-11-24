"""R5-1 bakeability numbers (CPU only, stock checkpoint read-only).

For a residual direction v (stock DoM, visit a, "none_last") measure:
  1. NVFP4 / MXFP8 re-quantisation of edited expert down projections
     W' = (I - v v^T) W  (s = 1): leakage of v back into W'q, error vs W'.
  2. bf16 rounding of edited bf16 writers / consumers (wo_ud, embed rows,
     consumer columns W P).
  3. Short-conv (attn_sconv / mlp_sconv, 4 static per-channel taps) effect:
     what writer-side folding before the conv costs.
  4. RMSNorm gain mismatch: v in normed-input space vs stream space.
"""
import json, math, os, sys
import torch
from safetensors import safe_open

torch.set_num_threads(16)
CK = "/home/center/inkling/models/Inkling-2-Mini/model-00001-of-00001.safetensors"
f = safe_open(CK, "pt")
T = lambda k: f.get_tensor(k)
DIRS = torch.load("/home/center/inkling/directions/inkling2mini_stock_dom_directions.pt",
                  map_location="cpu", weights_only=False)["directions"]
def vdir(app):
    v = DIRS[f"none_last_v{app:02d}"].float(); return v / v.norm()
P = "model.llm.layers"
out = {}
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])

def nvfp4_deq(L, name, e0, e1):
    w = T(f"{P}.{L}.mlp.experts.{name}_weight")[e0:e1]
    sc = T(f"{P}.{L}.mlp.experts.{name}_weight_scale")[e0:e1].float()
    g = (T(f"{P}.{L}.mlp.experts.{name}_weight_amax")[e0:e1].float()
         / T(f"{P}.{L}.mlp.experts.{name}_weight_outer_scale_denominator").float())
    lo, hi = E2M1[(w & 15).long()], E2M1[(w >> 4).long()]
    q = torch.stack([lo, hi], -1).flatten(-2)          # [e, D, F]
    return q * sc.repeat_interleave(16, -1) * g[:, None, None]

def nvfp4_quant(W):
    """nearest e2m1, block 16 along F, e4m3 block scales, fp32 global
    scale = amax/(6*448) per expert (amax_denominator 2688)."""
    amax = W.abs().amax(dim=(1, 2))
    g = amax / 2688.0
    Wb = W.view(*W.shape[:-1], -1, 16)
    s = (Wb.abs().amax(-1) / 6.0 / g[:, None, None]).to(torch.float8_e4m3fn).float()
    x = Wb / (s * g[:, None, None]).clamp_min(1e-30)[..., None]
    grid = E2M1[:8]
    idx = (x.abs()[..., None] - grid).abs().argmin(-1)
    q = grid[idx] * torch.sign(x)
    return (q * s[..., None] * g[:, None, None, None]).view_as(W)

def mxfp8_deq(L, name, e0, e1):
    w = T(f"{P}.{L}.mlp.experts.{name}_weight")[e0:e1].float()
    sc = T(f"{P}.{L}.mlp.experts.{name}_weight_scale")[e0:e1].float()
    return w * torch.exp2(sc - 127).repeat_interleave(32, -1), sc

def mxfp8_quant(W, rows=32):
    """e4m3 nearest, ue8m0 block scales over [rows x 32] blocks, scale =
    2^ceil(log2(amax/448))."""
    e, D, F = W.shape
    Wb = W.view(e, D // rows, rows, F // 32, 32)
    amax = Wb.abs().amax(dim=(2, 4), keepdim=True).clamp_min(1e-30)
    s = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    q = (Wb / s).to(torch.float8_e4m3fn).float() * s
    return q.view_as(W)

def edit_stats(W, v, quant, tag):
    """W [e, D, F] (D = residual/out dim). Edit W' = W - v (v^T W)."""
    vW = torch.einsum("d,edf->ef", v, W)
    Wp = W - v[None, :, None] * vW[:, None, :]
    Wq = quant(Wp)
    W0 = quant(W)
    res = {
        "requant_unedited_exact_frac": float((W0 == W).float().mean()),
        "edit_rel_size": float((Wp - W).norm() / W.norm()),
        "requant_rel_err": float((Wq - Wp).norm() / Wp.norm()),
        "requant_err_over_edit": float((Wq - Wp).norm() / (Wp - W).norm()),
        "leak_v_energy_frac": float(torch.einsum("d,edf->ef", v, Wq).norm() / vW.norm()),
        "orig_v_energy_frac_of_W": float(vW.norm() / W.norm()),
    }
    return res

for L, app, fmt in ((10, 34, "nvfp4"), (19, 41, "nvfp4"), (21, 43, "mxfp8"), (23, 46, "mxfp8")):
    v = vdir(app)
    acc = []
    for e0 in range(0, 128, 16):
        if fmt == "nvfp4":
            W = nvfp4_deq(L, "w2", e0, e0 + 16); r = edit_stats(W, v, nvfp4_quant, fmt)
        else:
            W, sc = mxfp8_deq(L, "w2", e0, e0 + 16)
            if e0 == 0:
                rows_const = bool((sc.view(16, 72, 32, 36) == sc.view(16, 72, 32, 36)[:, :, :1]).all())
                out[f"L{L}_mxfp8_scale_const_over_32_rows"] = rows_const
            r = edit_stats(W, v, lambda X: mxfp8_quant(X, 32 if rows_const else 1), fmt)
        acc.append(r)
    out[f"L{L}_{fmt}_w2_v{app}"] = {k: sum(a[k] for a in acc) / len(acc) for k in acc[0]}
    print(L, out[f"L{L}_{fmt}_w2_v{app}"], flush=True)
# random direction control
g = torch.Generator().manual_seed(0); vr = torch.randn(2304, generator=g); vr /= vr.norm()
W = nvfp4_deq(10, "w2", 0, 16); out["L10_nvfp4_w2_random_dir"] = edit_stats(W, vr, nvfp4_quant, "nvfp4")
W, _ = mxfp8_deq(21, "w2", 0, 16); out["L21_mxfp8_w2_random_dir"] = edit_stats(W, vr, mxfp8_quant, "mxfp8")

# bf16 writers / consumers
def bf16_stats(W, v, side):
    W = W.float()
    if side == "out":   # writer: rows live in residual space
        Wp = W - torch.outer(v, v @ W)
        leak = (v @ Wp.bfloat16().float()).norm() / (v @ W).norm()
    else:               # consumer: columns read residual space, W P
        Wp = W - torch.outer(W @ v, v)
        leak = (Wp.bfloat16().float() @ v).norm() / (W @ v).norm()
    return {"edit_rel_size": float((Wp - W).norm() / W.norm()),
            "bf16_rel_err": float((Wp.bfloat16().float() - Wp).norm() / Wp.norm()),
            "leak_v_energy_frac": float(leak)}
v34 = vdir(34)
out["L10_wo_ud_writer_bf16"] = bf16_stats(T(f"{P}.10.attn.wo_ud.weight"), v34, "out")
out["L11_wq_du_consumer_bf16"] = bf16_stats(T(f"{P}.11.attn.wq_du.weight"), v34, "in")
E = T("model.llm.embed.weight")[:20000].float()
v0 = vdir(0)
Ep = E - torch.outer(E @ v0, v0)
out["embed_rows_bf16_first20k"] = {"leak_v_energy_frac": float((Ep.bfloat16().float() @ v0).norm() / (E @ v0).norm())}

# short conv: writer-side fold through 4 per-channel taps
def conv_stats(L, which, v, n=512):
    c = T(f"{P}.{L}.{which}.weight").float()[:, 0, :]      # [D, 4]
    M = (c.T * v[None])                                    # [4, D] rows c_j*v
    sv = torch.linalg.svdvals(M); sv = sv / sv[0]
    cosj = [float(torch.nn.functional.cosine_similarity(M[j], v, 0)) for j in range(4)]
    g = torch.Generator().manual_seed(1)
    o = torch.randn(n + 3, v.numel(), generator=g)
    def conv(x):  # causal depthwise, tap 3 = current token
        return sum(x[3 - j: 3 - j + n] * c[:, 3 - j] for j in range(4))
    w = conv(o); Pw = w - torch.outer(w @ v, v)
    Pv = lambda x: x - torch.outer(x @ v, v)
    Q, _ = torch.linalg.qr(M.T)                              # [D, 4]
    b1 = conv(Pv(o)); b4 = conv(o - (o @ Q) @ Q.T)
    lk = lambda y: float((y @ v).norm() / (w @ v).norm())
    return {"tap_svals_rel": [round(float(s), 4) for s in sv],
            "cos(c_j*v, v)": [round(x, 4) for x in cosj],
            "fold_P_before_conv": {"leak": lk(b1), "rel_err_vs_hook": float((b1 - Pw).norm() / Pw.norm())},
            "fold_rank4_before_conv": {"leak": lk(b4), "rel_err_vs_hook": float((b4 - Pw).norm() / Pw.norm())}}
for L, app in ((10, 34), (21, 43)):
    for which in ("attn_sconv", "mlp_sconv"):
        out[f"L{L}_{which}_v{app}"] = conv_stats(L, which, vdir(app))

# RMSNorm gains: normed-input space vs stream space
def gcos(gname, v):
    gw = T(gname).float(); u = gw * v
    return float(torch.nn.functional.cosine_similarity(u, v, 0))
out["cos(g*v, v)"] = {
    "L11_q_input_norm_v34": gcos(f"{P}.11.q_input_norm.weight", v34),
    "L11_k_input_norm_v34": gcos(f"{P}.11.k_input_norm.weight", v34),
    "L11_v_input_norm_v34": gcos(f"{P}.11.v_input_norm.weight", v34),
    "L12_attn_norm_v35": gcos(f"{P}.12.attn_norm.weight", vdir(35)),
    "L11_mlp_norm_v34": gcos(f"{P}.11.mlp_norm.weight", v34),
    "final_norm_v47": gcos("model.llm.norm.weight", vdir(47)),
    "embed_norm_v0": gcos("model.llm.embed_norm.weight", v0),
}
gA = T(f"{P}.11.q_input_norm.weight").float(); gB = T(f"{P}.12.attn_norm.weight").float()
out["cos(gL11q*v, gL12*v)_v34"] = float(torch.nn.functional.cosine_similarity(gA * v34, gB * v34, 0))
# loop: same layer, different passes want different directions
out["cos(v_app, v_app+12)_layer10"] = [float(vdir(10) @ vdir(22)), float(vdir(22) @ vdir(34))]
out["cos(v_app, v_app+12)_layer5_17_29"] = [float(vdir(5) @ vdir(17)), float(vdir(17) @ vdir(29))]
json.dump(out, open(os.path.expanduser("~/inkling/engines/aeon-cleanroom/runs/r5/bake_analysis.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
