"""CPU test of the HTTP register path with resid keys (aeon-r5 via PYTHONPATH)."""
import asyncio, os, tempfile
os.environ["AEON_VARIANT_DIR"] = tempfile.mkdtemp()
import vllm
import vllm.models.inkling2.amd.loop  # noqa: F401  (import check of the edited loop)
from vllm.entrypoints.openai import aeon_variants_api as api
from fastapi import HTTPException
assert vllm.__file__.startswith(os.path.expanduser("~/inkling/engines/aeon-cleanroom/wt-r5")), vllm.__file__
D = 2304
R = api.RegisterVariantRequest
run = lambda c: asyncio.run(c)
print(run(api.register_variant(R(name="ro", resid_dirs=[1.0] + [0.0] * (D - 1), resid_scales=[1.0] * 48)))["resid_nonzero_apps"][:3])
print(run(api.register_variant(R(name="both", dirs=[0.5] * D, scales=[1.0] * 48, resid_dirs=[0.1] * D, resid_scales=[0.5] * 48))))
print(run(api.register_variant(R(name="op", dirs=[0.5] * D, scales=[1.0] * 48)))["resid_rank"])
for bad in (R(name="x"), R(name="y", resid_dirs=[1.0] * D), R(name="stock", dirs=[1.0] * D, scales=[1.0] * 48)):
    try:
        run(api.register_variant(bad)); print("NO ERROR", bad)
    except HTTPException as e:
        print("400:", e.detail)
p = os.path.join(os.environ["AEON_VARIANT_DIR"], "both.safetensors")
print("by path:", run(api.register_variant(R(name="both2", path=p)))["resid_rank"])
print([ (v["name"], v["rank"], v["resid_rank"]) for v in run(api.get_variants())["variants"]])
print("API OK")
