#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AEON-7
"""CPU unit test for r5_bake.py --readers all (R6-1).
 1. synthetic checkpoint (stock names incl. norms, routers, AttnRes queries):
    every reader of the stream after each edited app vs an independent
    stream-space model: the hook at app a is the normed-stream edit
    x -> x - s (v/g)((g*v).x) (g = own consumer gain); a reader behind gain
    h sees W' (h*x) == W (h * M x), M = product of the stream edits of all
    earlier apps in ascending order; the own consumer (every stream) gets
    the plain W (I - s v v^T). Covers rank 2, overlapping dirs, several apps,
    two variant files, queries (h = 1), plain gain mode.
 2. reader set exactly as documented; experts / router bias / loop layers untouched.
 3. refusals (apps 0-40 in all mode, per-pass o_proj, quantized reader).
 4. --readers consumer is byte-identical to the R5 tool (--r5-tool path).
 5. writers + AEON_WEIGHT_OVERLAY loader (--hooks-dir) replace routers/queries/unembed.
 6. optional real stock (--stock, read-only) + variant dir: bake, timings, rounding.
Usage: python test_r6_bake.py [--r5-tool r5_bake_r5.py] [--hooks-dir DIR] [--stock DIR --variants DIR]"""
import argparse, hashlib, importlib.util, json, os, sys, tempfile, time, types
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r5_bake as rb  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--r5-tool"); ap.add_argument("--hooks-dir"); ap.add_argument("--stock"); ap.add_argument("--variants")
args = ap.parse_args()
R = {}
g = torch.Generator().manual_seed(1)
D, Q, KV = 64, 96, 32
GLOBAL = {5, 11, 17, 23}
ATTN = ("wq_du", "wr_du", "gate_attn", "attn_sink_proj", "wk_dv", "wv_dv")


def bf(*s):
    return torch.randn(*s, generator=g).bfloat16()


def gain():
    return (1.0 + 0.5 * torch.randn(D, generator=g)).bfloat16()


def synth(dirpath):
    t = {"model.llm.unembed.weight": bf(500, D), "model.llm.embed.weight": bf(500, D),
         "model.llm.norm.weight": gain(), "model.llm.attn_res.output_query": torch.randn(D, generator=g)}
    for L in range(24):
        p = f"model.llm.layers.{L}."
        t[p + "attn.wo_ud.weight"] = bf(D, Q)
        for n, rows in zip(ATTN, (Q, 12, Q, 3, KV, KV)):
            t[p + f"attn.{n}.weight"] = bf(rows, D)
        if L in GLOBAL:
            t[p + "attn.k_sconv.weight_proj"] = bf(8, D); t[p + "attn.v_sconv.weight_proj"] = bf(8, D)
            for s in "qkv":
                t[p + f"{s}_input_norm.weight"] = gain()
        else:
            t[p + "attn_norm.weight"] = gain()
        t[p + "mlp_norm.weight"] = gain()
        t[p + "mlp.gate.weight"] = bf(16, D)
        t[p + "mlp.gate.bias"] = torch.randn(16, generator=g)
        t[p + "mlp.experts.w13_weight"] = torch.randint(0, 255, (4, 16, D // 2), generator=g, dtype=torch.uint8)
        t[f"model.llm.attn_res.mlp_queries.{L}"] = torch.randn(D, generator=g)
        if L < 23:
            t[f"model.llm.attn_res.attn_queries.{L}"] = torch.randn((3, D) if (L + 1) in GLOBAL else (D,), generator=g)
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


tmp = tempfile.mkdtemp()
stock = f"{tmp}/stock"; T = synth(stock); h0 = sha(stock)
G = lambda n: T["model.llm." + n].double()
u = lambda v: (v / v.norm()).double()
v1, v2, v3, v4 = (u(torch.randn(D, generator=g)) for _ in range(4))
v2 = u(v2 + 0.8 * v1)                         # overlapping directions across apps

# variant A: rank 1 at 41 (s 1.0), 44 (0.7, dir overlaps 41's), 47 (1.0); o_proj on layers 18..23
rdA = torch.zeros(48, 1, D); rsA = torch.zeros(48, 1)
rdA[41, 0], rsA[41, 0] = v1.float(), 1.0
rdA[44, 0], rsA[44, 0] = v2.float(), 0.7
rdA[47, 0], rsA[47, 0] = v4.float(), 1.0
do = torch.randn(48, D, generator=g); so = torch.zeros(48); so[42:] = 0.9
save_file({"resid_dirs": rdA, "resid_scales": rsA, "dirs": do, "scales": so}, f"{tmp}/A.safetensors")
# variant B: rank 2 at 43 (v3, v1 overlapping 41) and rank 1 at 46 (global own consumer, layer 23)
rdB = torch.zeros(48, 2, D); rsB = torch.zeros(48, 2)
rdB[43, 0], rdB[43, 1] = v3.float(), v1.float(); rsB[43] = torch.tensor([1.0, 0.5])
rdB[46, 0], rsB[46, 0] = v3.float(), 0.8
save_file({"resid_dirs": rdB, "resid_scales": rsB}, f"{tmp}/B.safetensors")
EDITS = {41: [(v1, 1.0)], 43: [(v3, 1.0), (v1, 0.5)], 44: [(v2, 0.7)], 46: [(v3, 0.8)], 47: [(v4, 1.0)]}


def layer_of(b):
    return rb.app_to_layer(b)


def own_gain(a):
    if a == 47:
        return G("norm.weight")
    L = layer_of(a + 1)
    return G(f"layers.{L}.q_input_norm.weight" if L in GLOBAL else f"layers.{L}.attn_norm.weight")


def stream_edit(a, mode):
    """normed-stream matrix of app a's hook (all k, ascending)."""
    M = torch.eye(D, dtype=torch.float64)
    gg = own_gain(a) if mode == "oblique" else torch.ones(D, dtype=torch.float64)
    for v, s in EDITS[a]:
        M = (torch.eye(D, dtype=torch.float64) - s * torch.outer(v / gg, gg * v)) @ M
    return M


def expected(name, visit, h, own_app, mode):
    """W_ref for a reader at `visit` behind gain h (None = 1). own_app: app whose own consumer this is."""
    W = G(name)
    shp = W.shape
    W = W.reshape(-1, D)
    hh = torch.ones(D, dtype=torch.float64) if h is None else h
    M = torch.eye(D, dtype=torch.float64)
    for a in sorted(EDITS):
        if a >= visit:
            continue
        if a == own_app or mode == "plain":
            E = torch.eye(D, dtype=torch.float64)
            for v, s in EDITS[a]:
                E = (torch.eye(D, dtype=torch.float64) - s * torch.outer(v, v)) @ E
            E = torch.diag(1 / hh) @ E @ torch.diag(hh)        # plain in the reader's post-gain space
        else:
            E = stream_edit(a, mode)
        M = E @ M
    # reader input y = h*x  ->  W' y = W (h * M x) = W diag(h) M diag(1/h) y
    return (W @ torch.diag(hh) @ M @ torch.diag(1 / hh)).reshape(shp)


def check_all(baked, mode, queries):
    worst, names = 0.0, set()
    for visit in range(42, 49):
        if visit == 48:
            rd = [("unembed.weight", G("norm.weight"), 47 if 47 < visit else None)]
            if queries:
                rd.append(("attn_res.output_query", None, None))
        else:
            L = layer_of(visit)
            own = visit - 1
            rd = []
            for n in ATTN + (("k_sconv", "v_sconv") if L in GLOBAL else ()):
                t = f"layers.{L}.attn.{n}.weight" if n in ATTN else f"layers.{L}.attn.{n}.weight_proj"
                if L in GLOBAL:
                    s = "k" if n in ("wk_dv", "k_sconv") else "v" if n in ("wv_dv", "v_sconv") else "q"
                    h = G(f"layers.{L}.{s}_input_norm.weight")
                else:
                    h = G(f"layers.{L}.attn_norm.weight")
                rd.append((t, h, own))
            rd.append((f"layers.{L}.mlp.gate.weight", G(f"layers.{L}.mlp_norm.weight"), None))
            if queries:
                rd += [(f"attn_res.attn_queries.{L - 1}", None, None), (f"attn_res.mlp_queries.{L}", None, None)]
        for name, h, own in rd:
            ref = expected(name, visit, h, own, mode)
            full = "model.llm." + name
            assert full in baked, f"{full} not baked"
            b = baked[full]
            assert b.dtype == T[full].dtype and b.shape == T[full].shape, full
            # compare in fp64 against the reference rounded once to the stored dtype
            err = float((b.double() - ref).abs().max() / ref.abs().max())
            worst = max(worst, err); names.add(full)
            tol = 1e-6 if b.dtype == torch.float32 else 1e-2
            assert err < tol, (full, err)
            # bf16: also require nearly every element to equal the once-rounded reference
            if b.dtype == torch.bfloat16:
                frac = float((b == ref.to(b.dtype)).float().mean())
                assert frac > 0.98, (full, frac)
    return worst, names


# 1. oblique, no queries
baked, rep = rb.bake(stock, [f"{tmp}/A.safetensors", f"{tmp}/B.safetensors"], readers="all")
w, names = check_all(baked, "oblique", False)
wo = {f"model.llm.layers.{L}.attn.wo_ud.weight" for L in range(18, 24)}
for k in wo:
    L = int(k.split(".")[3]); r = u(do[L + 24])
    ref = (torch.eye(D, dtype=torch.float64) - 0.9 * torch.outer(r, r)) @ T[k].double()
    assert torch.equal(baked[k], ref.to(torch.bfloat16)), k
assert set(baked) == names | wo, sorted(set(baked) ^ (names | wo))
assert not any("experts" in k or k.endswith("gate.bias") or "embed.weight" == k.split("llm.")[-1] for k in baked)
assert not any(f"layers.{L}." in k for k in baked for L in range(0, 18))
R["oblique"] = dict(tensors=len(baked), max_rel_err=w)

# stream equivalence spot check: reader behind gain h sees exactly h * (M x)
x = torch.randn(D, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
name = "model.llm.layers.22.mlp.gate.weight"; h = G("layers.22.mlp_norm.weight")
Mx = x.clone()
for a in (41, 43, 44):
    Mx = stream_edit(a, "oblique") @ Mx
lhs = baked[name].double() @ (h * x); rhs = T[name].double() @ (h * Mx)
R["stream_equiv_rel"] = float((lhs - rhs).norm() / rhs.norm()); assert R["stream_equiv_rel"] < 1e-2
# s = 1 annihilation through a different gain: own-consumer v1 removed from the router's view
E41 = stream_edit(41, "oblique")
R["s1_idempotent"] = float((E41 @ E41 - E41).abs().max() / E41.abs().max()); assert R["s1_idempotent"] < 1e-6, R["s1_idempotent"]

# 2. plain gains + queries
baked_p, _ = rb.bake(stock, [f"{tmp}/A.safetensors", f"{tmp}/B.safetensors"], readers="all", gain_mode="plain",
                     fold_queries=True)
w, names_p = check_all(baked_p, "plain", True)
R["plain_queries"] = dict(tensors=len(baked_p), max_rel_err=w)
baked_q, _ = rb.bake(stock, [f"{tmp}/A.safetensors", f"{tmp}/B.safetensors"], readers="all", fold_queries=True)
w, _ = check_all(baked_q, "oblique", True)
assert baked_q["model.llm.attn_res.attn_queries.22"].shape == (3, D)
R["oblique_queries"] = dict(tensors=len(baked_q), max_rel_err=w)
# variant order: A+B == B+A (merged by app; disjoint apps)
baked_ba, _ = rb.bake(stock, [f"{tmp}/B.safetensors", f"{tmp}/A.safetensors"], readers="all")
assert all(torch.equal(baked_ba[k], baked[k]) for k in baked)
R["variant_order_merge"] = "A+B == B+A bitwise"

# 3. refusals
def refused(kw, msg, **opt):
    save_file(kw, f"{tmp}/bad.safetensors")
    try:
        rb.bake(stock, [f"{tmp}/bad.safetensors"], **opt)
    except rb.NotBakeable as e:
        assert msg in str(e), str(e); return str(e).splitlines()[1].strip()
    raise AssertionError(f"not refused: {msg}")
s = torch.zeros(48); s[[3, 40, 41]] = 1.0
R["refuse_all_app3_40"] = refused({"resid_dirs": v1.float(), "resid_scales": s}, "apps [3, 40]", readers="all")
s = torch.zeros(48); s[6:42] = 1.0; s[18:30] = 0.5
R["refuse_perpass"] = refused({"dirs": v1.float(), "scales": s}, "shared loop layer 6", readers="all")
q = dict(T); q["model.llm.layers.20.mlp.gate.weight"] = torch.zeros(16, D, dtype=torch.uint8)
os.makedirs(f"{tmp}/qstock"); save_file(q, f"{tmp}/qstock/model-00001-of-00001.safetensors")
try:
    rb.bake(f"{tmp}/qstock", [f"{tmp}/A.safetensors"], readers="all"); raise AssertionError("quantized reader")
except rb.NotBakeable as e:
    R["refuse_quantized_reader"] = str(e)

# 4. consumer mode unchanged vs the R5 tool
if args.r5_tool:
    spec = importlib.util.spec_from_file_location("r5old", args.r5_tool)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    s = torch.zeros(48); s[[0, 4, 41, 46, 47]] = 1.0
    save_file({"resid_dirs": v1.float(), "resid_scales": s, "dirs": do, "scales": so}, f"{tmp}/C.safetensors")
    b_new, r_new = rb.bake(stock, [f"{tmp}/C.safetensors"])
    b_old, r_old = old.bake(stock, [f"{tmp}/C.safetensors"])
    assert set(b_new) == set(b_old) and all(torch.equal(b_new[k], b_old[k]) for k in b_new)
    assert r_new == r_old
    R["consumer_mode_vs_r5"] = f"identical ({len(b_new)} tensors, report equal)"

# 5. writers + overlay loader
assert rb.main(["--stock", stock, "--variant", f"{tmp}/A.safetensors", "--variant", f"{tmp}/B.safetensors",
                "--readers", "all", "--fold-queries", "--out-overlay", f"{tmp}/ov.safetensors",
                "--report", f"{tmp}/rep.json"]) == 0
ov = load_file(f"{tmp}/ov.safetensors")
assert set(ov) == set(baked_q) and all(torch.equal(ov[k], baked_q[k]) for k in ov)
assert rb.main(["--stock", stock, "--variant", f"{tmp}/A.safetensors", "--readers", "all",
                "--out-checkpoint", f"{tmp}/full"]) == 0
full = load_file(f"{tmp}/full/model-00001-of-00001.safetensors")
bA, _ = rb.bake(stock, [f"{tmp}/A.safetensors"], readers="all")
assert all(torch.equal(full[k], bA[k] if k in bA else T[k]) for k in T)
assert sha(stock) == h0, "stock files changed"
R["writers"] = "overlay == baked; checkpoint patched; stock unchanged"
if args.hooks_dir:
    pkg = types.ModuleType("r5amd"); pkg.__path__ = [args.hooks_dir]; sys.modules["r5amd"] = pkg
    os.environ["AEON_VARIANT_DIR"] = ""
    spec = importlib.util.spec_from_file_location("r5amd.aeon_hooks", f"{args.hooks_dir}/aeon_hooks.py")
    hk = importlib.util.module_from_spec(spec); sys.modules[spec.name] = hk; spec.loader.exec_module(hk)
    os.environ["AEON_WEIGHT_OVERLAY"] = f"{tmp}/ov.safetensors"
    out = dict(hk.apply_weight_overlay(iter(T.items())))
    os.environ.pop("AEON_WEIGHT_OVERLAY")
    assert all(torch.equal(out[k], ov[k] if k in ov else T[k]) for k in T)
    kinds = sorted({k.split("llm.")[1].split(".")[0] if "layers" not in k else k.split(".")[-2] for k in ov})
    R["aeon_weight_overlay"] = dict(replaced=len(ov), kinds=kinds)

# 6. real stock
if args.stock:
    real = {}
    for name in sorted(os.listdir(args.variants)):
        if not name.endswith(".safetensors"):
            continue
        for opt in (dict(readers="consumer"), dict(readers="all"), dict(readers="all", gain_mode="plain")):
            t0 = time.time()
            try:
                b, rp = rb.bake(args.stock, [f"{args.variants}/{name}"], **opt)
            except rb.NotBakeable as e:
                real[f"{name}:{opt}"] = "refused: " + str(e).splitlines()[1][:160]; continue
            real[f"{name}:{opt}"] = dict(tensors=len(b), secs=round(time.time() - t0, 1),
                                         max_rounding_rel_err=max(t["rounding_rel_err"] for t in rp["tensors"].values()))
    R["real_stock"] = real
print(json.dumps(R, indent=1))
print("R6 BAKE TEST: ALL PASS")
