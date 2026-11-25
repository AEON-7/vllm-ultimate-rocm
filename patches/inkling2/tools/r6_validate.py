# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AEON-7
"""R6-1 validation: offline bake (r5_bake.py --readers consumer|all) vs the
runtime hook, same process, same row position (row 0 of a 4-row batch),
16 prompts (8 harmful + 8 benign eval prompts), greedy, --gen-tokens.

  prep  (CPU) variants + overlays under runs/r6/
  run   (GPU, one free device) comparisons + expert-fraction measurement

Variants (no residual variant files are referenced in round6 ENGINE_REQUESTS /
STATE; the two named R6 variants are o_proj-only):
  n_reason      round6 conv_reason_blend_l6-23_s1.25 (o_proj, apps 6-47, rank 5)
  n_lsv1        round6 conv_lsv1_l4-23_s1.25         (o_proj, apps 4-47, rank 5)
  n_reason_r    n_reason + residual edits at apps 41-45, 47 (stock DoM
                none_last_v{app}, s = 1)              (stand-in residual part)
  n_reason_rx   n_reason + "extended hook": at app c the residual dirs of
                every edited app <= c, ascending (rank <= 6); the runtime
                equivalent of the plain all-readers fold on attention
                inputs + unembed (routers excluded) -> checks bake correctness
Overlays: *_r5 (--readers consumer), *_r5f32 (same math in fp32 -> a second
rounding of the same edit: the bf16-rounding floor), n_reason_r_all
(oblique), _allplain, _allq (oblique + queries), n_reason_rx_attn
(plain all-readers minus routers).
"""
import argparse, json, os, subprocess, sys, time
import torch

BASE = os.path.expanduser("~/inkling/engines/aeon-cleanroom")
RUN = f"{BASE}/runs/r6"
MODEL = os.path.expanduser("~/inkling/models/Inkling-2-Mini")
CK = f"{MODEL}/model-00001-of-00001.safetensors"
VDIR = f"{RUN}/variants"
WT = f"{BASE}/wt-r6"
R6V = os.path.expanduser("~/inkling/round6/variants")
ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["prep", "run"])
ap.add_argument("--gen-tokens", type=int, default=64)
ap.add_argument("--k", type=int, default=50)
args = ap.parse_args()
os.makedirs(VDIR, exist_ok=True)
D = 2304
R_APPS = [41, 42, 43, 44, 45, 47]
sys.path.insert(0, f"{WT}/tools/inkling2")
import r5_bake as rb  # noqa: E402


def dom():
    src = torch.load(os.path.expanduser("~/inkling/directions/inkling2mini_stock_dom_directions.pt"),
                     map_location="cpu", weights_only=False)["directions"]
    return {a: src[f"none_last_v{a:02d}"].float() for a in range(48)}


if args.mode == "prep":
    os.environ["AEON_VARIANT_DIR"] = VDIR
    sys.path.insert(0, WT)
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from vllm.models.inkling2.amd import aeon_variants as av
    assert av.__file__.startswith(WT), av.__file__
    torch.set_num_threads(16)
    nr = load_file(f"{R6V}/conv_reason_blend_l6-23_s1.25.safetensors")
    nl = load_file(f"{R6V}/conv_lsv1_l4-23_s1.25.safetensors")
    dv = dom()
    rd = torch.zeros(48, 1, D); rs = torch.zeros(48, 1)
    for a in R_APPS:
        rd[a, 0] = dv[a] / dv[a].norm(); rs[a, 0] = 1.0
    rx = torch.zeros(48, len(R_APPS), D); rxs = torch.zeros(48, len(R_APPS))
    for c in range(41, 48):
        prev = [a for a in R_APPS if a <= c]
        for k, a in enumerate(prev):
            rx[c, k] = rd[a, 0]; rxs[c, k] = 1.0
    V = {"n_reason": dict(dirs=nr["dirs"], scales=nr["scales"]),
         "n_lsv1": dict(dirs=nl["dirs"], scales=nl["scales"]),
         "n_reason_r": dict(dirs=nr["dirs"], scales=nr["scales"], resid_dirs=rd, resid_scales=rs),
         "n_reason_rx": dict(dirs=nr["dirs"], scales=nr["scales"], resid_dirs=rx, resid_scales=rxs)}
    for name, kw in V.items():
        print(name, av.save_variant(name, kw["dirs"], kw["scales"], resid_dirs=kw.get("resid_dirs"),
                                    resid_scales=kw.get("resid_scales")))
    vp = lambda n: f"{VDIR}/{n}.safetensors"
    jobs = {"n_reason_r5": ([vp("n_reason")], {}), "n_lsv1_r5": ([vp("n_lsv1")], {}),
            "n_reason_r_r5": ([vp("n_reason_r")], {}),
            "n_reason_r_all": ([vp("n_reason_r")], dict(readers="all")),
            "n_reason_r_allplain": ([vp("n_reason_r")], dict(readers="all", gain_mode="plain")),
            "n_reason_r_allq": ([vp("n_reason_r")], dict(readers="all", fold_queries=True)),
            "n_reason_r_allplain_rx": ([vp("n_reason_r")], dict(readers="all", gain_mode="plain"))}
    reports = {}
    for out, (vs, opt) in jobs.items():
        t0 = time.time()
        baked, rep = rb.bake(MODEL, vs, **opt)
        if out == "n_reason_r_allplain_rx":      # attention inputs + unembed only (no routers)
            baked = {k: v for k, v in baked.items() if not k.endswith("mlp.gate.weight")}
            out = "n_reason_rx_attn"
        rb.write_overlay(baked, rep, f"{RUN}/overlay_{out}.safetensors", MODEL)
        rep["secs"] = round(time.time() - t0, 1); rep["overlay"] = f"{RUN}/overlay_{out}.safetensors"
        rep["n_tensors"] = len(baked)
        reports[out] = rep
        print(out, len(baked), "tensors", rep["secs"], "s", flush=True)
    # fp32-math second rounding of the same o_proj edits (floor)
    for n in ("n_reason", "n_lsv1"):
        o, _ = rb.read_variant(vp(n))
        edits = rb.plan([(vp(n), o, None)])
        ov = {}
        with safe_open(CK, "pt") as f:
            for suffix, side, pairs, path, apps in edits:
                W = f.get_tensor(rb.PREFIX + suffix).float()
                for r, s in pairs:
                    r = r.float(); W = W - s * torch.outer(r, r @ W)
                ov[rb.PREFIX + suffix] = W.bfloat16().contiguous()
        ref = load_file(f"{RUN}/overlay_{n}_r5.safetensors")
        diff = sum(int((ov[k] != ref[k]).sum()) for k in ov); tot = sum(ov[k].numel() for k in ov)
        save_file(ov, f"{RUN}/overlay_{n}_r5f32.safetensors")
        reports[f"{n}_r5f32"] = dict(n_tensors=len(ov), elements_differing_from_fp64_bake=diff / tot)
        print(n, "fp32 alt bake differs in", diff / tot, "of elements", flush=True)
    json.dump(reports, open(f"{RUN}/bake_reports.json", "w"), indent=1, default=str)
    sys.exit(0)

# ---------------------------------------------------------------- run (GPU)
gpu = os.environ.get("ROCR_VISIBLE_DEVICES", "?")
smi = subprocess.run(["rocm-smi", "--showmemuse"], capture_output=True, text=True).stdout
used = [int(l.split(":")[-1]) for l in smi.splitlines() if f"GPU[{gpu}]" in l and "VRAM%" in l]
if not used or used[0] > 15:
    sys.exit(f"GPU {gpu} not free (VRAM% {used}); refusing to start a full-model engine")
os.environ["AEON_VARIANT_DIR"] = VDIR
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
sys.path.insert(0, f"{BASE}/aeon")
from abl_compare_lib import cmp_gen  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402
import vllm  # noqa: E402
assert vllm.__file__.startswith(WT), f"engine not from the aeon-r6 worktree: {vllm.__file__}"
from vllm.models.inkling2.amd.aeon_hooks import HOOKS  # noqa: E402
from vllm.models.inkling2.amd.aeon_variants import VARIANTS  # noqa: E402
import vllm.v1.worker.gpu.model_runner as mr  # noqa: E402

llm = LLM(model=MODEL, max_model_len=4096, max_num_seqs=16, max_num_batched_tokens=2048,
          gpu_memory_utilization=0.80, enforce_eager=True, limit_mm_per_prompt={"image": 0},
          enable_prefix_caching=False, max_logprobs=args.k)
tok = llm.get_tokenizer()


def chat(prompts):
    out = []
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True, reasoning_effort="none")
        out.append(tok(text, add_special_tokens=False)["input_ids"])
    return out


def load_eval(fn, n):
    return [json.loads(l)["prompt"] for l in list(open(os.path.expanduser(f"~/inkling/data/{fn}")))[:n]]


prompts = chat(load_eval("harmful_eval.jsonl", 8) + load_eval("benign_eval.jsonl", 8))


def sp(v, **kw):
    return SamplingParams(extra_args={"aeon_variant": v} if v else None, **kw)


def gens(v):
    out = []
    for p in prompts:
        o = llm.generate([TokensPrompt(prompt_token_ids=p)] * 4,
                         [sp(v, temperature=0.0, max_tokens=args.gen_tokens, logprobs=args.k)] * 4, use_tqdm=False)[0]
        out.append(dict(ids=list(o.outputs[0].token_ids),
                        lps=[{t: l.logprob for t, l in d.items()} for d in o.outputs[0].logprobs]))
    return out


def set_hooks(on):
    def f(model):
        if on:
            HOOKS.enabled, HOOKS.variants, mr._AEON_VARIANTS = True, VARIANTS, VARIANTS
        else:
            HOOKS.enabled, HOOKS.variants, mr._AEON_VARIANTS = False, None, None
    llm.apply_model(f)


def swap(tensors):
    """Load checkpoint-named tensors into the live model through the model's own
    renames / fused shards / weight loaders; refresh router weight_f32."""
    def f(model):
        params = dict(model.named_parameters())
        gates = set()
        for name, w in tensors.items():
            for old, new in model._renames:
                name = name.replace(old, new)
            shard = None
            for old, (new, sh) in model._shards.items():
                if old in name:
                    name = name.replace(old, new); shard = sh; break
            p = params[name]
            assert p.dtype == w.dtype, (name, p.dtype, w.dtype)
            loader = getattr(p, "weight_loader", None)
            if loader is None:
                p.data.copy_(w.to(p.device))
            elif shard is None:
                loader(p, w)
            else:
                loader(p, w, shard)
            if name.endswith("mlp.gate.weight"):
                gates.add(name[: -len(".weight")])
        for g in gates:
            m = model.get_submodule(g)
            m.process_weights_after_loading()
        return len(tensors), len(gates)
    return llm.apply_model(f)[0]


def stock_of(names):
    with safe_open(CK, "pt") as f:
        return {n: f.get_tensor(n) for n in names}


def snapshot(names_engine_filter):
    def f(model):
        return {n: p.detach().float().sum().item() for n, p in model.named_parameters()
                if any(s in n for s in names_engine_filter)}
    return llm.apply_model(f)[0]


R = {"engine": vllm.__file__, "gen_tokens": args.gen_tokens, "prompts": len(prompts)}
t0 = time.time()
set_hooks(True); gens(None)          # warm
G = {}

# S0: stock round trip through swap() is bitwise (params) for the largest overlay's tensors
ov_all = load_file(f"{RUN}/overlay_n_reason_r_allq.safetensors")
filt = ("layers.18.", "layers.23.", "lm_head", "attn_res", "layers.20.mlp.gate")
def full_params(model):
    return {n: p.detach().clone().cpu() for n, p in model.named_parameters() if any(s in n for s in filt)}
before = llm.apply_model(full_params)[0]
swap(ov_all); swap(stock_of(list(ov_all)))
after = llm.apply_model(full_params)[0]
gate_ok = llm.apply_model(lambda m: bool(torch.equal(m.get_submodule("model.layers.20.mlp.gate").weight_f32,
                                                     m.get_submodule("model.layers.20.mlp.gate").weight.float())))[0]
R["S0_stock_roundtrip_bitwise"] = dict(params=len(before), equal=all(torch.equal(before[k], after[k]) for k in before),
                                       router_weight_f32_refreshed=gate_ok)
print("S0", R["S0_stock_roundtrip_bitwise"], flush=True)
del before, after

# stock and hooks
set_hooks(False); G["stock"] = gens(None)
set_hooks(True)
for v in ("n_reason", "n_lsv1", "n_reason_r", "n_reason_rx"):
    G[f"hook:{v}"] = gens(v); print("hook", v, time.time() - t0, flush=True)
G["hook_repeat:n_reason"] = gens("n_reason")
set_hooks(False)
for o in ("n_reason_r5", "n_reason_r5f32", "n_lsv1_r5", "n_lsv1_r5f32", "n_reason_r_r5", "n_reason_r_all",
          "n_reason_r_allplain", "n_reason_r_allq", "n_reason_rx_attn"):
    ov = load_file(f"{RUN}/overlay_{o}.safetensors")
    swap(ov); G[f"bake:{o}"] = gens(None); swap(stock_of(list(ov)))
    print("bake", o, time.time() - t0, flush=True)
G["stock_after"] = gens(None)

K = ("n", "exact", "kl_mean", "lp_absdiff_max")
def c(a, b):
    r = cmp_gen(G[a], G[b]); return {k: r[k] for k in K if k in r} | {"full": r}
pairs = {
    "stock_vs_stock_after_swaps": ("stock", "stock_after"),
    "hook_repeat_n_reason": ("hook:n_reason", "hook_repeat:n_reason"),
    "n_reason: r5 bake vs hook": ("hook:n_reason", "bake:n_reason_r5"),
    "n_reason: r5f32 bake vs hook": ("hook:n_reason", "bake:n_reason_r5f32"),
    "n_reason: r5 bake vs r5f32 bake (rounding floor)": ("bake:n_reason_r5", "bake:n_reason_r5f32"),
    "n_reason: hook vs stock (effect)": ("stock", "hook:n_reason"),
    "n_lsv1: r5 bake vs hook": ("hook:n_lsv1", "bake:n_lsv1_r5"),
    "n_lsv1: r5f32 bake vs hook": ("hook:n_lsv1", "bake:n_lsv1_r5f32"),
    "n_lsv1: r5 bake vs r5f32 bake (rounding floor)": ("bake:n_lsv1_r5", "bake:n_lsv1_r5f32"),
    "n_lsv1: hook vs stock (effect)": ("stock", "hook:n_lsv1"),
    "n_reason_r: r5 bake vs hook": ("hook:n_reason_r", "bake:n_reason_r_r5"),
    "n_reason_r: r6 all (oblique) vs hook": ("hook:n_reason_r", "bake:n_reason_r_all"),
    "n_reason_r: r6 all (plain gains) vs hook": ("hook:n_reason_r", "bake:n_reason_r_allplain"),
    "n_reason_r: r6 all + queries vs hook": ("hook:n_reason_r", "bake:n_reason_r_allq"),
    "n_reason_r: r6 all vs r5 bake": ("bake:n_reason_r_r5", "bake:n_reason_r_all"),
    "n_reason_r: hook vs n_reason hook (resid part effect)": ("hook:n_reason", "hook:n_reason_r"),
    "n_reason_r: hook vs stock (effect)": ("stock", "hook:n_reason_r"),
    "n_reason_rx: ext fold (attn+unembed, plain) vs ext hook": ("hook:n_reason_rx", "bake:n_reason_rx_attn"),
    "n_reason_rx: ext hook vs local hook": ("hook:n_reason_r", "hook:n_reason_rx"),
}
R["compare"] = {k: c(*v) for k, v in pairs.items()}
for k, v in R["compare"].items():
    print(f"{k:60s} exact {v['exact']}/{v['n']} kl {v['kl_mean']:.3e}", flush=True)

# expert fraction: stock activations of the 16 prompts' prefill, layers 18-23
cap = {}
def install(model):
    hs = []
    for L in range(18, 24):
        layer = model.model.layers[L]
        def pa(mod, a, L=L):
            xs = [a[1]] + ([a[2], a[3]] if len(a) > 3 and torch.is_tensor(a[2]) else [])
            cap.setdefault(("attn", L), []).append([x.float().cpu() for x in xs])
        def pm(mod, a, L=L):
            cap.setdefault(("mlp", L), []).append(a[0].float().cpu())
        hs += [layer.attn.register_forward_pre_hook(pa), layer.mlp.register_forward_pre_hook(pm)]
    def pf(mod, a):
        cap.setdefault(("final", 0), []).append(a[1].float().cpu())
    hs.append(model.logits_processor.register_forward_pre_hook(pf))
    return hs
hs = llm.apply_model(install)[0]
llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], [sp(None, max_tokens=1)] * len(prompts), use_tqdm=False)
llm.apply_model(lambda m: [h.remove() for h in hs])
with safe_open(CK, "pt") as f:
    gn = lambda n: f.get_tensor("model.llm." + n).double()
    dv = dom()
    V = {a: (dv[a] / dv[a].norm()).double() for a in R_APPS}
    def gref(a):
        return gn("norm.weight") if a == 47 else gn(f"layers.{a - 23}.attn_norm.weight") if a - 23 != 23 else gn("layers.23.q_input_norm.weight")
    def delta(y, visit, h, own_app):
        """y - E y for reader input y (post-gain h) under the all-readers edits of apps < visit (s = 1)."""
        z = y.double()
        for a in [a for a in R_APPS if a < visit]:
            v = V[a]
            if a == own_app:
                p = q = v
            else:
                g = gref(a); p = h * v / g; q = g * v / h
            z = z - (z @ q)[:, None] * p[None]
        return y.double() - z
    E = {"attn": 0.0, "mlp": 0.0, "final": 0.0}; rel = {}
    for L in range(18, 24):
        visit = L + 24
        x_m = torch.cat(cap[("mlp", L)]); dm = delta(x_m, visit, gn(f"layers.{L}.mlp_norm.weight"), None)
        E["mlp"] += float(dm.square().sum())
        rel[L] = dict(mlp_rel=float(dm.norm(dim=-1).mean() / x_m.double().norm(dim=-1).mean()))
        streams = list(zip(*cap[("attn", L)]))
        names = ["q_input_norm", "k_input_norm", "v_input_norm"] if L == 23 else ["attn_norm"]
        ra = []
        for si, xs in enumerate(streams):
            xa = torch.cat(xs); da = delta(xa, visit, gn(f"layers.{L}.{names[si]}.weight"), visit - 1)
            E["attn"] += float(da.square().sum()); ra.append(float(da.norm(dim=-1).mean() / xa.double().norm(dim=-1).mean()))
        rel[L]["attn_rel"] = ra
    xf = torch.cat(cap[("final", 0)]); df = delta(xf, 48, gn("norm.weight"), 47)
    E["final"] = float(df.square().sum())
    tot = sum(E.values())
    R["expert_fraction"] = dict(
        definition=("share of the all-readers edit energy sum ||y - E y||^2 (stock activations, prefill of the 16 "
                    "prompts, layers 18-23 + final hidden, n_reason_r residual dirs, s = 1) that lands on the "
                    "MLP-input mix: the router part is folded, the expert part is not"),
        energy=E, expert_unfolded_fraction=E["mlp"] / tot, per_layer=rel,
        tokens=int(xf.shape[0]))
print("expert fraction", R["expert_fraction"]["expert_unfolded_fraction"], rel, flush=True)
R["elapsed_s"] = time.time() - t0
json.dump(R, open(f"{RUN}/validate.json", "w"), indent=1, default=str)
with open(f"{RUN}/gens_ids.jsonl", "w") as fo:
    for k, rows in G.items():
        fo.write(json.dumps(dict(run=k, ids=[r["ids"] for r in rows])) + "\n")
print("DONE r6_validate", R["elapsed_s"])
