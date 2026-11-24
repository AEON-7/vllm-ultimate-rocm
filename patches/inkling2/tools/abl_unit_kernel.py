import os, tempfile, torch, numpy as np
from types import SimpleNamespace
d = tempfile.mkdtemp(); os.environ["AEON_VARIANT_DIR"] = d
from vllm.models.inkling2.amd import aeon_variants as av
D = 2304
g = torch.Generator().manual_seed(0)
av.save_variant("a", torch.randn(48, D, generator=g), torch.linspace(0, 1, 48))
av.save_variant("b", torch.randn(48, 3, D, generator=g), torch.rand(48, 3, generator=g))
av.save_variant("c", torch.randn(24, D, generator=g), torch.ones(48))
bank = av.VariantBank()
for i, n in enumerate([None, "a", "b", "c", "a"]):
    bank.add_request(i, SimpleNamespace(extra_args={"aeon_variant": n} if n else None))
counts = [5, 1, 7, 1, 3]
qsl = np.concatenate([[0], np.cumsum(counts)])
dev = torch.device("cuda")
ib = SimpleNamespace(num_reqs=5, idx_mapping_np=np.array([0, 1, 2, 3, 4]), query_start_loc_np=qsl,
                     num_tokens_after_padding=24, positions=torch.zeros(24, device=dev))
bank.prepare(ib, None)
print("active", bank.active, "slots", bank.slots, "apps_on", bank.apps_on.sum())
y0 = torch.randn(24, D, device=dev).bfloat16()
worst = 0
for app in range(48):
    ref = av.apply_rows_reference(y0, bank.vid[:24], bank.dirs, bank.scales, app)
    out = bank.apply(y0.clone(), app)
    worst = max(worst, (out.float() - ref.float()).abs().max().item())
    assert torch.equal(out[:5], y0[:5]) and torch.equal(out[17:], y0[17:]), "stock/padding rows changed"
print("max |kernel-ref|", worst)
# rank-1 per-row check against the AEON_ABLATE formula in fp32
r = bank.dirs[bank.slots["a"], 30, 0]; s = bank.scales[bank.slots["a"], 30, 0]
yy = y0[5].float(); exp = yy - s * (yy @ r) * r
print("row5 vs formula", (bank.apply(y0.clone(), 30)[5].float() - exp).abs().max().item())
# app->layer map for [24, D]
dc = bank.dirs[bank.slots["c"], :, 0]
print("layer map ok", torch.allclose(dc[6], dc[18]) and torch.allclose(dc[42], dc[18 + 0]) is False, torch.allclose(dc[42], dc[42]))
print("app_to_layer", [av.app_to_layer(a) for a in (0, 5, 6, 17, 18, 29, 30, 41, 42, 47)])
# mixed-vid launch == per-variant launches, bitwise, row by row
for app in (4, 20, 33, 47):
    y = y0.clone(); mixed = bank.apply(y, app).clone()
    vid_all = bank.vid[:24].clone()
    for s_ in torch.unique(vid_all).tolist():
        bank.vid[:24] = torch.where(vid_all == s_, vid_all, torch.zeros_like(vid_all))
        single = bank.apply(y0.clone(), app)
        rows = (vid_all == s_)
        assert torch.equal(single[rows], mixed[rows]), (app, s_)
    bank.vid[:24] = vid_all
print("mixed launch bitwise == single-variant launches: OK")
bank.end_step(); print("after end_step active", bank.active, "-> apply is a no-op:", torch.equal(bank.apply(y0.clone(), 10), y0))
# request hook (API side)
d = av.request_hook({"model": "m", "aeon_variant": "a", "cache_salt": "user1"})
print("hook", d["vllm_xargs"], d["cache_salt"])
print("hook stock", av.request_hook({"aeon_variant": "stock"}))
for bad in ({"aeon_variant": "nope"}, {"aeon_variant": "../x"}):
    try:
        av.request_hook(bad); print("NO ERROR", bad)
    except ValueError as e:
        print("rejected:", e)
