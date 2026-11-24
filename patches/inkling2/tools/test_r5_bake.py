#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AEON-7
"""CPU unit test for r5_bake.py.
 1. synthetic checkpoint (stock names, small dims): baked tensors vs the
    analytic W (I - s v v^T) / (I - s v v^T) W in fp64 (rank 1 and rank 2),
    names/dtypes/shapes, refusals, full-checkpoint byte patching, stock
    files unchanged;
 2. AEON_WEIGHT_OVERLAY (aeon_hooks.apply_weight_overlay, loaded by path from
    AEON_HOOKS_DIR if given) replaces unembed / attention-input / wo_ud;
 3. optional real stock checkpoint (STOCK=dir, read-only): overlay bake of
    real variants, rounding report.
Usage: python test_r5_bake.py [--stock DIR] [--variants DIR] [--hooks-dir .../inkling2/amd]"""
import argparse, hashlib, importlib.util, json, os, sys, tempfile, types
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r5_bake as rb  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stock"); ap.add_argument("--variants"); ap.add_argument("--hooks-dir")
args = ap.parse_args()
R = {}
g = torch.Generator().manual_seed(0)
D, Q, KV = 64, 96, 32
GLOBAL = {5, 11, 17, 23}


def bf(*s):
    return torch.randn(*s, generator=g).bfloat16()


def synth(dirpath):
    t = {"model.llm.unembed.weight": bf(500, D), "model.llm.embed.weight": bf(500, D),
         "model.llm.norm.weight": bf(D)}
    for L in range(24):
        p = f"model.llm.layers.{L}."
        t[p + "attn.wo_ud.weight"] = bf(D, Q)
        for n, rows in (("wq_du", Q), ("wr_du", 12), ("gate_attn", Q), ("attn_sink_proj", 3), ("wk_dv", KV), ("wv_dv", KV)):
            t[p + f"attn.{n}.weight"] = bf(rows, D)
        if L in GLOBAL:
            t[p + "attn.k_sconv.weight_proj"] = bf(8, D); t[p + "attn.v_sconv.weight_proj"] = bf(8, D)
        t[p + "mlp.experts.w2_weight"] = torch.randint(0, 255, (4, D, 8), generator=g, dtype=torch.uint8)
    os.makedirs(dirpath)
    save_file(t, f"{dirpath}/model-00001-of-00001.safetensors", metadata={"format": "pt"})
    json.dump({"weight_map": {k: "model-00001-of-00001.safetensors" for k in t}},
              open(f"{dirpath}/model.safetensors.index.json", "w"))
    open(f"{dirpath}/config.json", "w").write("{}")
    return t


def sha(d):
    h = hashlib.sha256()
    for fn in sorted(os.listdir(d)):
        h.update(open(os.path.join(d, fn), "rb").read())
    return h.hexdigest()


def ulps(a, b):
    a, b = a.float(), b.float()
    u = 2.0 ** (torch.floor(torch.log2(b.abs().clamp_min(1e-30))) - 7)
    return float(((a - b).abs() / u).max())


tmp = tempfile.mkdtemp()
stock = f"{tmp}/stock"; T = synth(stock); h0 = sha(stock)
v1, v2, v3 = (torch.randn(D, generator=g) for _ in range(3))
u = lambda v: (v / v.norm()).double()

# variant A: rank-1 residual at apps 0, 4 (global consumer), 41, 46 (global), 47; o_proj per-layer, same every pass
s_r = torch.zeros(48); s_r[[0, 4, 41, 46, 47]] = torch.tensor([1.0, 0.7, 1.0, 0.5, 1.0])
dirs_o = torch.randn(24, D, generator=g); s_o = torch.zeros(48); s_o[4:] = 0.8
save_file({"resid_dirs": v1.expand(48, D).contiguous(), "resid_scales": s_r,
           "dirs": dirs_o, "scales": s_o}, f"{tmp}/A.safetensors")
# variant B: rank-2 residual at app 2
rd = torch.zeros(48, 2, D); rd[:, 0] = v2; rd[:, 1] = v3; rs = torch.zeros(48, 2); rs[2] = torch.tensor([1.0, 0.6])
save_file({"resid_dirs": rd, "resid_scales": rs}, f"{tmp}/B.safetensors")
baked, rep = rb.bake(stock, [f"{tmp}/A.safetensors", f"{tmp}/B.safetensors"])

worst, nexact, ntot = 0.0, 0, 0
def check(name, ref):
    global worst, nexact, ntot
    b = baked[name]
    assert b.dtype == T[name].dtype and b.shape == T[name].shape, name
    rq = ref.to(T[name].dtype)
    worst = max(worst, ulps(b, ref)); nexact += int((b == rq).sum()); ntot += b.numel()
I = torch.eye(D, dtype=torch.float64)
P = lambda v, s: I - s * torch.outer(u(v), u(v))
for a, L in ((0, 1), (4, 5), (41, 18), (46, 23)):
    names = [f"layers.{L}.attn.{n}.weight" for n in ("wq_du", "wr_du", "gate_attn", "attn_sink_proj", "wk_dv", "wv_dv")]
    if L in GLOBAL:
        names += [f"layers.{L}.attn.k_sconv.weight_proj", f"layers.{L}.attn.v_sconv.weight_proj"]
    for n in names:
        check("model.llm." + n, T["model.llm." + n].double() @ P(v1, float(s_r[a])))
check("model.llm.unembed.weight", T["model.llm.unembed.weight"].double() @ P(v1, 1.0))
for n in ("wq_du", "wk_dv"):   # rank 2 at app 2 -> layer 3: y' = P1 P0 y -> W' = W P1 P0
    k = f"model.llm.layers.3.attn.{n}.weight"
    check(k, T[k].double() @ P(v3, 0.6) @ P(v2, 1.0))
for L in (4, 10, 20):          # o_proj: (I - s r r^T) W
    k = f"model.llm.layers.{L}.attn.wo_ud.weight"
    check(k, P(dirs_o[L], 0.8) @ T[k].double())
assert "model.llm.layers.2.attn.wo_ud.weight" not in baked and "model.llm.layers.6.attn.k_sconv.weight_proj" not in baked
assert "model.llm.embed.weight" not in baked
R["analytic"] = dict(max_ulps=worst, frac_bitwise_equal=nexact / ntot, tensors=len(baked))
assert worst <= 1.0 and nexact / ntot > 0.99, R["analytic"]

# refusals
def refused(kw, msg):
    save_file(kw, f"{tmp}/bad.safetensors")
    try:
        rb.bake(stock, [f"{tmp}/bad.safetensors"])
    except rb.NotBakeable as e:
        assert msg in str(e), str(e); return str(e).splitlines()[1].strip()
    raise AssertionError(f"not refused: {msg}")
s = torch.zeros(48); s[5] = 1.0
R["refuse_app5"] = refused({"resid_dirs": v1, "resid_scales": s}, "apps [5]")
s = torch.zeros(48); s[[20, 40, 47]] = 1.0
R["refuse_app20_40"] = refused({"resid_dirs": v1, "resid_scales": s}, "apps [20, 40]")
s = torch.zeros(48); s[6:42] = 1.0; s[18:30] = 0.5
R["refuse_perpass_scale"] = refused({"dirs": v1, "scales": s}, "shared loop layer 6")
pv = torch.randn(48, D, generator=g); s = torch.ones(48)
R["refuse_perpass_dirs"] = refused({"dirs": pv, "scales": s}, "shared loop layer")
s = torch.zeros(48); s[42:] = 1.0                          # per-app dirs only on unshared layers: fine
save_file({"dirs": pv, "scales": s}, f"{tmp}/ok.safetensors"); rb.bake(stock, [f"{tmp}/ok.safetensors"])
# quantized consumer refused
q = dict(T); q["model.llm.unembed.weight"] = torch.zeros(500, D, dtype=torch.uint8)
os.makedirs(f"{tmp}/qstock"); save_file(q, f"{tmp}/qstock/model-00001-of-00001.safetensors")
s = torch.zeros(48); s[47] = 1.0; save_file({"resid_dirs": v1, "resid_scales": s}, f"{tmp}/r47.safetensors")
try:
    rb.bake(f"{tmp}/qstock", [f"{tmp}/r47.safetensors"]); raise AssertionError("quantized consumer not refused")
except rb.NotBakeable as e:
    R["refuse_quantized_consumer"] = str(e)

# writers: overlay + full checkpoint
rc = rb.main(["--stock", stock, "--variant", f"{tmp}/A.safetensors", "--variant", f"{tmp}/B.safetensors",
              "--out-overlay", f"{tmp}/ov.safetensors"])
assert rc == 0
ov = load_file(f"{tmp}/ov.safetensors")
assert set(ov) == set(baked) and all(torch.equal(ov[k], baked[k]) for k in ov)
rc = rb.main(["--stock", stock, "--variant", f"{tmp}/A.safetensors", "--variant", f"{tmp}/B.safetensors",
              "--out-checkpoint", f"{tmp}/full"])
assert rc == 0
full = load_file(f"{tmp}/full/model-00001-of-00001.safetensors")
assert set(full) == set(T)
assert all(torch.equal(full[k], baked[k] if k in baked else T[k]) for k in T)
assert rb._header(f"{tmp}/full/model-00001-of-00001.safetensors") == rb._header(f"{stock}/model-00001-of-00001.safetensors")
assert os.path.exists(f"{tmp}/full/config.json") and os.path.exists(f"{tmp}/full/model.safetensors.index.json")
assert rb.main(["--stock", stock, "--variant", f"{tmp}/bad.safetensors", "--out-overlay", f"{tmp}/x.safetensors"]) == 2
try:
    rb.main(["--stock", stock, "--variant", f"{tmp}/A.safetensors", "--out-overlay", f"{stock}/x.safetensors"])
    raise AssertionError("wrote into stock")
except SystemExit:
    pass
assert sha(stock) == h0, "stock files changed"
R["writers"] = "overlay == baked; full checkpoint: header identical, edited tensors patched, others bitwise; stock unchanged"

# AEON_WEIGHT_OVERLAY loader
if args.hooks_dir:
    pkg = types.ModuleType("r5amd"); pkg.__path__ = [args.hooks_dir]; sys.modules["r5amd"] = pkg
    os.environ["AEON_VARIANT_DIR"] = ""
    spec = importlib.util.spec_from_file_location("r5amd.aeon_hooks", f"{args.hooks_dir}/aeon_hooks.py")
    hk = importlib.util.module_from_spec(spec); sys.modules[spec.name] = hk; spec.loader.exec_module(hk)
    os.environ["AEON_WEIGHT_OVERLAY"] = f"{tmp}/ov.safetensors"
    out = dict(hk.apply_weight_overlay(iter(T.items())))
    os.environ.pop("AEON_WEIGHT_OVERLAY")
    assert all(torch.equal(out[k], baked[k] if k in baked else T[k]) for k in T)
    R["aeon_weight_overlay"] = dict(replaced=len(baked), includes=sorted({k.split(".")[-2] if "layers" in k else k for k in baked}))

# real stock checkpoint (read-only): overlay bake of real variants
if args.stock:
    vd = args.variants
    real = {}
    for name in ("r_cons", "r_47", "op_only"):
        b, rp = rb.bake(args.stock, [f"{vd}/{name}.safetensors"])
        leaks = [l["left"] - l["ideal"] for t in rp["tensors"].values() for l in t["leak"]]
        real[name] = dict(tensors=len(b), worst_leak=max(leaks), mean_leak=sum(leaks) / len(leaks),
                          max_rounding_rel_err=max(t["rounding_rel_err"] for t in rp["tensors"].values()),
                          dtypes=sorted({t["dtype"] for t in rp["tensors"].values()}))
    try:
        rb.bake(args.stock, [f"{vd}/r_res.safetensors"]); raise AssertionError("r_res not refused")
    except rb.NotBakeable as e:
        real["r_res"] = str(e).splitlines()[1].strip()[:200]
    R["real_stock"] = real
print(json.dumps(R, indent=1))
print("R5 BAKE TEST: ALL PASS")
