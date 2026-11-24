"""o_proj variant path: aeon-r5 kernel output bitwise == the 21add0ba8
(installed) kernel on the same inputs."""
import importlib.util, os, sys, tempfile, types
from types import SimpleNamespace
import numpy as np, torch
tmp = tempfile.mkdtemp(); os.environ["AEON_VARIANT_DIR"] = tmp
from vllm.models.inkling2.amd import aeon_variants as old
WT = os.path.expanduser("~/inkling/engines/aeon-cleanroom/wt-r5/vllm/models/inkling2/amd")
pkg = types.ModuleType("r5amd"); pkg.__path__ = [WT]; sys.modules["r5amd"] = pkg
spec = importlib.util.spec_from_file_location("r5amd.aeon_variants", f"{WT}/aeon_variants.py")
new = importlib.util.module_from_spec(spec); spec.loader.exec_module(new)
assert old.__file__ != new.__file__
D, T = 2304, 24; g = torch.Generator().manual_seed(1)
old.save_variant("a", torch.randn(48, D, generator=g), torch.linspace(0, 1, 48))
old.save_variant("b", torch.randn(48, 3, D, generator=g), torch.rand(48, 3, generator=g))
banks = []
for m in (old, new):
    b = m.VariantBank()
    for i, n in enumerate([None, "a", "b", "a"]):
        b.add_request(i, SimpleNamespace(extra_args={"aeon_variant": n} if n else None))
    b.prepare(SimpleNamespace(num_reqs=4, idx_mapping_np=np.arange(4), query_start_loc_np=np.array([0, 5, 9, 17, 21]),
                              num_tokens_after_padding=T, positions=torch.zeros(T, device="cuda")), None)
    banks.append(b)
y = torch.randn(T, D, device="cuda").bfloat16()
for app in range(48):
    assert torch.equal(banks[0].apply(y.clone(), app), banks[1].apply(y.clone(), app)), app
print("o_proj path bitwise identical to 21add0ba8 for 48 apps: OK")
