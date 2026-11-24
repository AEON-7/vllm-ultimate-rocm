"""R5-1 unit test: per-request residual-stream edits (small shapes, one GPU,
< 100 MB). Loads the aeon-r5 worktree modules by path (no install, no
effect on running servers). Run with env.sh sourced and one GPU visible."""
import importlib.util, json, os, sys, tempfile, types
from types import SimpleNamespace
import numpy as np
import torch

WT = os.environ.get("R5_WT", os.path.expanduser("~/inkling/engines/aeon-cleanroom/wt-r5"))
AMD = os.path.join(WT, "vllm/models/inkling2/amd")
tmp = tempfile.mkdtemp(); os.environ["AEON_VARIANT_DIR"] = tmp
pkg = types.ModuleType("r5amd"); pkg.__path__ = [AMD]; sys.modules["r5amd"] = pkg
def load(name):
    spec = importlib.util.spec_from_file_location(f"r5amd.{name}", os.path.join(AMD, f"{name}.py"))
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m); return m
av = load("aeon_variants")
hooks = load("aeon_hooks")
assert av.__file__.startswith(WT), av.__file__
res = {}
D = 2304; T = 24; S = 3
g = torch.Generator().manual_seed(0)
dev = torch.device("cuda")
# variants: o_proj-only, resid-only, both (separate scales), rank-3 resid
av.save_variant("op", torch.randn(48, D, generator=g), torch.linspace(0, 1, 48))
s_r = torch.zeros(48); s_r[[3, 5, 17, 29, 40, 47]] = torch.tensor([1.0, 0.5, 1.0, 0.7, 1.0, 1.0])
info_r = av.save_variant("rs", None, None, resid_dirs=torch.randn(48, D, generator=g), resid_scales=s_r)
info_b = av.save_variant("both", torch.randn(24, D, generator=g), torch.ones(48) * 0.8,
                         resid_dirs=torch.randn(48, 3, D, generator=g), resid_scales=torch.rand(48, 3, generator=g))
res["summaries"] = {"rs": info_r, "both": {k: v for k, v in info_b.items() if k != "resid_nonzero_apps"}}
o, r = av.read_variant_full(os.path.join(tmp, "rs.safetensors")); assert o is None and r[0].shape == (48, 1, D)
d0, s0 = av.read_variant(os.path.join(tmp, "rs.safetensors")); assert not s0.any()
for bad in (dict(dirs=None, scales=None), dict(dirs=torch.randn(48, D), scales=None)):
    try:
        av.save_variant("bad", bad["dirs"], bad["scales"]); raise SystemExit("NO ERROR")
    except ValueError:
        pass
bank = av.VariantBank()
names = [None, "op", "rs", "both", "rs", None]
counts = [5, 1, 6, 4, 3, 2]
for i, n in enumerate(names):
    bank.add_request(i, SimpleNamespace(extra_args={"aeon_variant": n} if n else None))
qsl = np.concatenate([[0], np.cumsum(counts)])  # 21 real rows + 3 padding
ib = SimpleNamespace(num_reqs=len(names), idx_mapping_np=np.arange(len(names)), query_start_loc_np=qsl,
                     num_tokens_after_padding=T, positions=torch.zeros(T, device=dev))
bank.prepare(ib, None)
vid = bank.vid[:T].clone()
stock_rows = (vid == 0)
res["slots"] = bank.slots
res["resid_apps_on"] = np.nonzero(bank.resid_on_apps)[0].tolist()
res["oproj_apps_on"] = int(bank.apps_on.sum())
assert bank.slots["rs"] not in [i for i, m in bank.app_mask.items() if m.any()]
worst = {"bf16_2d": 0.0, "fp32_3d": 0.0, "oproj": 0.0}
x2 = torch.randn(T, D, device=dev).bfloat16()
x3 = torch.randn(S, T, D, device=dev, dtype=torch.float32)
for app in range(48):
    for key, x in (("bf16_2d", x2), ("fp32_3d", x3)):
        ref = av.apply_rows_reference(x, vid, bank.rdirs, bank.rscales, app)
        out = bank.apply_resid(x.clone(), app)
        worst[key] = max(worst[key], (out.float() - ref.float()).abs().max().item())
        assert torch.equal(out[..., stock_rows, :], x[..., stock_rows, :]), ("stock rows changed", app)
        if not bank.resid_on(app):
            assert torch.equal(out, x)
        # o_proj-only rows untouched by residual edits
        opr = vid == bank.slots["op"]
        assert torch.equal(out[..., opr, :], x[..., opr, :]), ("o_proj-only row touched", app)
    out = bank.apply(x2.clone(), app)
    ref = av.apply_rows_reference(x2, vid, bank.dirs, bank.scales, app)
    worst["oproj"] = max(worst["oproj"], (out.float() - ref.float()).abs().max().item())
    assert torch.equal(out[stock_rows], x2[stock_rows])
    rsr = vid == bank.slots["rs"]   # residual-only rows untouched by o_proj edits
    assert torch.equal(out[rsr], x2[rsr]), ("resid-only row touched by o_proj", app)
res["max_abs_kernel_minus_fp32_ref"] = worst
# fp32 stream: full removal (s=1) leaves ~0 along r, exact fp32 formula
a = 3; row = int((vid == bank.slots["rs"]).nonzero()[0])
rv = bank.rdirs[bank.slots["rs"], a, 0]
out = bank.apply_resid(x3.clone(), a)
res["resid_rs_app3_dot_after"] = float((out[:, row] @ rv).abs().max())
res["resid_rs_app3_dot_before"] = float((x3[:, row] @ rv).abs().max())
exp = x3[:, row] - (x3[:, row] @ rv)[:, None] * rv[None]
res["resid_vs_formula_fp32"] = float((out[:, row] - exp).abs().max())
# mixed launch bitwise == per-variant launches (batch-invariant contract)
for app in (3, 5, 17, 29, 40, 47):
    for x in (x2, x3):
        mixed = bank.apply_resid(x.clone(), app).clone()
        for s_ in torch.unique(vid).tolist():
            bank.vid[:T] = torch.where(vid == s_, vid, torch.zeros_like(vid))
            single = bank.apply_resid(x.clone(), app)
            rows = vid == s_
            assert torch.equal(single[..., rows, :], mixed[..., rows, :]), (app, s_)
        bank.vid[:T] = vid
res["mixed_vs_single_bitwise"] = "OK"
# a row's result does not depend on its position or neighbours (same row
# content moved to another index under another batch composition)
perm = torch.randperm(T, generator=g).to(dev)
bank.vid[:T] = vid[perm]
xp = bank.apply_resid(x3[:, perm].clone(), 47)
bank.vid[:T] = vid
xm = bank.apply_resid(x3.clone(), 47)
assert torch.equal(xp, xm[:, perm]); res["row_position_invariant"] = "OK"
# hooks: resid_edit copy semantics and no-op for stock-only steps
H = hooks.AeonHooks(); H.variants = bank
shared = x3.clone(); e = H.resid_edit(shared, 17, copy=True)
assert torch.equal(shared, x3) and not torch.equal(e, x3); res["resid_edit_copy"] = "OK"
bank.end_step(); assert torch.equal(bank.apply_resid(x3.clone(), 3), x3) and not bank.resid_on(3)
# stock-only batch -> inactive
b2 = av.VariantBank()
for i in range(3):
    b2.add_request(i, SimpleNamespace(extra_args=None))
b2.prepare(SimpleNamespace(num_reqs=3, idx_mapping_np=np.arange(3), query_start_loc_np=np.array([0, 2, 4, 6]),
                           num_tokens_after_padding=8, positions=torch.zeros(8, device=dev)), None)
assert not b2.active
# hot add without restart: a new residual variant registered while the bank
# is live is picked up on the next step that names it
av.save_variant("late", None, None, resid_dirs=torch.randn(D, generator=g), resid_scales=torch.ones(48))
bank.add_request(6, SimpleNamespace(extra_args={"aeon_variant": "late"}))
ib2 = SimpleNamespace(num_reqs=2, idx_mapping_np=np.array([0, 6]), query_start_loc_np=np.array([0, 3, 7]),
                      num_tokens_after_padding=8, positions=torch.zeros(8, device=dev))
bank.prepare(ib2, None)
assert "late" in bank.slots and bank.resid_on(30)
y = torch.randn(2, 8, D, device=dev); out = bank.apply_resid(y.clone(), 30)
assert torch.equal(out[:, :3], y[:, :3]) and not torch.equal(out[:, 3:7], y[:, 3:7]) and torch.equal(out[:, 7:], y[:, 7:])
res["hot_add"] = "OK"
# loop placement: which edit lands on which input (mirrors loop.py)
visits = [(i, 0) for i in range(6)] + [(i, p) for p in range(3) for i in range(6, 18)] + [(i, 0) for i in range(18, 24)]
landing = {}
for v, (i, p) in enumerate(visits):
    if i == 6 and p > 0:
        landing[v - 1] = f"start of pass {p} (prelude_dests copy)"
    restart = v + 1 < len(visits) and visits[v + 1][1] > p
    if not restart:
        landing.setdefault(v, "next visit input" if v + 1 < len(visits) else "final hidden (LM head)")
assert sorted(landing) == list(range(48))
res["landing_special"] = {k: landing[k] for k in (5, 17, 29, 41, 47)}
print(json.dumps(res, indent=1, default=str))
print("R5 UNIT: ALL PASS")
