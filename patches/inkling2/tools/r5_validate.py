# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AEON-7
"""R5-1 full-model validation of per-request residual-stream edits (NOT run
yet: needs one free GPU). In-process engine from the aeon-r5 worktree via
PYTHONPATH (see r5_run.sh), eager, AEON_VARIANT_DIR on (AEON_BATCH_INVARIANT
defaults on). Comparisons are same process and same row position.

  prep   (CPU) variant files + consumer-fold overlay       python r5_validate.py prep
  run    (GPU) checks A-H below, writes runs/r5/validate.json and gens

A  stock rows: plain engine vs variant engine with mixed batches - bitwise
B  every variant: mixed batch vs batch of itself, same row position - bitwise
C  r_zero (resid keys, all-zero scales) == stock - bitwise
D  post-edit |r . x| on edited rows ~ 0 for s = 1 (apps 10, 17->pass start, 30, 47)
E  hot add: r_late (copy of r_res) registered while serving == r_res - bitwise
F  consumer fold: r_cons (resid edits only where the consumer is a single
   local layer: apps 0-3, 41-45) vs W' = W P on those layers' attention input
   weights (in-process swap): greedy exact, KL, logprob diff
G  generations for the refusal judge: stock / r_res / r_res_h / r_both on
   N harmful + N benign eval prompts (greedy), runs/r5/gens.jsonl
H  throughput: 8-row decode, stock-only vs mixed (stock + residual variants)
"""
import argparse, json, math, os, subprocess, sys, time
import torch

RUN = os.path.expanduser("~/inkling/engines/aeon-cleanroom/runs/r5")
MODEL = os.path.expanduser("~/inkling/models/Inkling-2-Mini")
VDIR = f"{RUN}/variants"
WT = os.path.expanduser("~/inkling/engines/aeon-cleanroom/wt-r5")
ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["prep", "run"])
ap.add_argument("--gen-tokens", type=int, default=64)
ap.add_argument("--judge-tokens", type=int, default=256)
ap.add_argument("--n-eval", type=int, default=100, help="harmful and benign prompts each for G")
ap.add_argument("--k", type=int, default=50)
ap.add_argument("--skip", default="", help="comma list of checks to skip, e.g. G,H")
args = ap.parse_args()
os.makedirs(RUN, exist_ok=True)
D = 2304
CONS_APPS = [0, 1, 2, 3, 41, 42, 43, 44, 45]        # consumer = layer app+1 (1-4) / app-23 (18-22), local


def consumer_layer(app):
    return app + 1 if app < 5 else app - 24 + 1     # 41 -> 18 ... 45 -> 22


def dirs_per_visit():
    src = torch.load(os.path.expanduser("~/inkling/directions/inkling2mini_stock_dom_directions.pt"),
                     map_location="cpu", weights_only=False)["directions"]
    return torch.stack([src[f"none_last_v{a:02d}"].float() for a in range(48)])


def build():
    pv = dirs_per_visit()
    layer_vis = list(range(0, 6)) + list(range(30, 42)) + list(range(42, 48))
    s1 = torch.zeros(48); s1[4:] = 1.0
    V = {
        "op_only": dict(dirs=pv[layer_vis], scales=s1),
        "r_res": dict(resid_dirs=pv, resid_scales=s1),
        "r_res_h": dict(resid_dirs=pv, resid_scales=s1 * 0.5),
        "r_both": dict(dirs=pv[layer_vis], scales=s1, resid_dirs=pv, resid_scales=s1),
        "r_zero": dict(resid_dirs=pv, resid_scales=torch.zeros(48)),
    }
    s = torch.zeros(48); s[CONS_APPS] = 1.0
    V["r_cons"] = dict(resid_dirs=pv, resid_scales=s)
    s = torch.zeros(48); s[47] = 1.0
    V["r_47"] = dict(resid_dirs=pv, resid_scales=s)
    return V


if args.mode == "prep":
    os.environ["AEON_VARIANT_DIR"] = VDIR
    sys.path.insert(0, WT)
    from vllm.models.inkling2.amd import aeon_variants as av
    assert av.__file__.startswith(WT), av.__file__
    for name, kw in build().items():
        print(name, av.save_variant(name, kw.get("dirs"), kw.get("scales"), resid_dirs=kw.get("resid_dirs"),
                                    resid_scales=kw.get("resid_scales")))
    # consumer fold: next layer's attention input weights W' = W (I - r r^T)
    from safetensors import safe_open
    from safetensors.torch import save_file
    rd, rs = av.canonicalize(build()["r_cons"]["resid_dirs"], build()["r_cons"]["resid_scales"])
    ov = {}
    with safe_open(f"{MODEL}/model-00001-of-00001.safetensors", "pt") as f:
        for app in CONS_APPS:
            L = consumer_layer(app); r = rd[app, 0]; sc = float(rs[app, 0])
            for k in f.keys():
                if k.startswith(f"model.llm.layers.{L}.attn.") and k.endswith(".weight"):
                    W = f.get_tensor(k)
                    if W.dim() == 2 and W.shape[1] == D and W.dtype == torch.bfloat16:
                        Wf = W.float()
                        ov[k] = (Wf - sc * torch.outer(Wf @ r, r)).to(W.dtype).contiguous()
    save_file(ov, f"{RUN}/overlay_r_cons.safetensors")
    print("consumer-fold overlay tensors:", sorted(ov))
    sys.exit(0)

# ---------------------------------------------------------------- run (GPU)
gpu = os.environ.get("ROCR_VISIBLE_DEVICES", "?")
smi = subprocess.run(["rocm-smi", "--showmemuse"], capture_output=True, text=True).stdout
used = [int(l.split(":")[-1]) for l in smi.splitlines() if f"GPU[{gpu}]" in l and "VRAM%" in l]
if not used or used[0] > 15:
    sys.exit(f"GPU {gpu} not free (VRAM% {used}); refusing to start a full-model engine")
os.environ["AEON_VARIANT_DIR"] = VDIR
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abl_compare_lib import cmp_gen, cmp_tf  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402
import vllm  # noqa: E402
assert vllm.__file__.startswith(WT), f"engine not from the aeon-r5 worktree: {vllm.__file__}"
from vllm.models.inkling2.amd import aeon_variants as av  # noqa: E402
from vllm.models.inkling2.amd.aeon_hooks import HOOKS  # noqa: E402
from vllm.models.inkling2.amd.aeon_variants import VARIANTS  # noqa: E402
import vllm.v1.worker.gpu.model_runner as mr  # noqa: E402

skip = set(args.skip.split(",")) - {""}
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


harm, ben = load_eval("harmful_eval.jsonl", args.n_eval), load_eval("benign_eval.jsonl", args.n_eval)
gen_prompts = chat(harm[:8] + ben[:8])


def sp(v, **kw):
    return SamplingParams(extra_args={"aeon_variant": v} if v else None, **kw)


def rows_gen(prompt, variants, n=None):
    outs = llm.generate([TokensPrompt(prompt_token_ids=prompt)] * len(variants),
                        [sp(v, temperature=0.0, max_tokens=n or args.gen_tokens, logprobs=args.k) for v in variants],
                        use_tqdm=False)
    return [dict(ids=list(o.outputs[0].token_ids),
                 lps=[{t: l.logprob for t, l in d.items()} for d in o.outputs[0].logprobs]) for o in outs]


state = {}


def set_mode(mode, arg=None):
    def f(model):
        if "orig" in state:
            p = dict(model.named_parameters())
            for k, v in state.pop("orig").items():
                p[k].data.copy_(v)
        if mode == "plain":
            HOOKS.enabled, HOOKS.variants, mr._AEON_VARIANTS = False, None, None
        elif mode == "multi":
            HOOKS.enabled, HOOKS.variants, mr._AEON_VARIANTS = True, VARIANTS, VARIANTS
        elif mode == "overlay":   # plain engine + consumer-folded weights
            # The engine fuses wq_du/wr_du/gate_attn/attn_sink_proj/wk_dv/wv_dv
            # into attn.qkvr; W' = W (I - s r r^T) is row-wise, so folding the
            # fused weight equals folding each checkpoint shard (the overlay
            # file written by prep). fp32 math, one bf16 rounding.
            HOOKS.enabled, HOOKS.variants, mr._AEON_VARIANTS = False, None, None
            rd, rs = arg
            p = dict(model.named_parameters()); orig = {}
            for app in CONS_APPS:
                L = consumer_layer(app)
                name = next(n for n in p if n.endswith(f"layers.{L}.attn.qkvr.weight"))
                W = p[name].data
                assert W.dim() == 2 and W.shape[1] == D and W.dtype == torch.bfloat16, (name, W.shape, W.dtype)
                orig[name] = W.clone()
                r = rd[app, 0].to(W.device); sc = float(rs[app, 0])
                Wf = W.float(); W.copy_((Wf - sc * torch.outer(Wf @ r, r)).to(W.dtype))
            state["orig"] = orig
            return len(orig)
        return mode
    return llm.apply_model(f)


R = {"engine": vllm.__file__}
t0 = time.time()
set_mode("multi")
ORDER = [None, "r_res", "r_both", "r_res_h", "r_cons", "r_47", "op_only", "r_zero"]
names = [v or "stock" for v in ORDER]
N = len(ORDER)
rows_gen(gen_prompts[0], ORDER)  # warm / autotune
set_mode("plain"); rows_gen(gen_prompts[0], [None] * N)

# A + B + C
res = {n: dict(mixed=[], alone=[]) for n in names}
plain = {}
for pi in (0, 9):
    p = gen_prompts[pi]
    set_mode("plain"); plain[pi] = rows_gen(p, [None] * N)
    set_mode("multi")
    alone = {n: rows_gen(p, [v] * N) for n, v in zip(names, ORDER)}
    for rot in range(N):
        perm = [ORDER[(j + rot) % N] for j in range(N)]
        mixed = rows_gen(p, perm)
        for j, v in enumerate(perm):
            n = v or "stock"
            res[n]["mixed"].append(mixed[j]); res[n]["alone"].append(alone[n][j])
            if v is None:
                res[n].setdefault("plain", []).append(plain[pi][j])
            if v == "r_zero":
                res[n].setdefault("stock_same_pos", []).append(alone["stock"][j])
R["B_mixed_vs_alone"] = {n: cmp_gen(res[n]["alone"], res[n]["mixed"]) for n in names}
R["A_stock_mixed_vs_plain"] = cmp_gen(res["stock"]["plain"], res["stock"]["mixed"])
R["C_r_zero_vs_stock"] = cmp_gen(res["r_zero"]["stock_same_pos"], res["r_zero"]["mixed"])
R["effect_vs_stock_alone"] = {n: cmp_gen(res["stock"]["alone"], res[n]["alone"]) for n in names if n != "stock"}
print("A", R["A_stock_mixed_vs_plain"], "\nC", R["C_r_zero_vs_stock"], flush=True)
for n in names:
    print("B", n, {k: R["B_mixed_vs_alone"][n][k] for k in ("n", "exact", "kl_mean")}, flush=True)

# D: post-edit dot with r on edited rows
if "D" not in skip:
    rd, rs = av.canonicalize(build()["r_res"]["resid_dirs"], build()["r_res"]["resid_scales"])
    rec = {}
    orig_apply = VARIANTS.apply_resid

    def spy(x, app):
        before = x.clone()
        y = orig_apply(x, app)
        if app in (10, 17, 30, 47):
            r = rd[app, 0].to(x.device, torch.float32)
            vid = VARIANTS.vid[: VARIANTS.rows]
            rows = vid == VARIANTS.slots["r_res"]
            b = (before.float()[..., rows, :] @ r).abs().mean().item()
            a = (y.float()[..., rows, :] @ r).abs().mean().item()
            st = vid == 0
            stock_eq = bool(torch.equal(before[..., st, :], y[..., st, :]))
            rec.setdefault(app, []).append(dict(before=b, after=a, dtype=str(x.dtype), stock_rows_bitwise=stock_eq))
        return y
    VARIANTS.apply_resid = spy
    llm.generate([TokensPrompt(prompt_token_ids=p) for p in gen_prompts[:4]],
                 [sp(v, max_tokens=2) for v in (None, "r_res", None, "r_res")], use_tqdm=False)
    VARIANTS.apply_resid = orig_apply
    R["D_post_edit_dot"] = {a: v[0] for a, v in rec.items()}
    print("D", R["D_post_edit_dot"], flush=True)

# E: hot add while serving
if "E" not in skip:
    kw = build()["r_res"]
    av.save_variant("r_late", None, None, resid_dirs=kw["resid_dirs"], resid_scales=kw["resid_scales"])
    p = gen_prompts[3]
    a = rows_gen(p, ["r_res", None, "r_late", None])
    b = rows_gen(p, ["r_late", None, "r_res", None])
    R["E_hot_add"] = dict(late_vs_res_pos0=cmp_gen([a[0]], [b[0]]), late_vs_res_pos2=cmp_gen([a[2]], [b[2]]))
    print("E", R["E_hot_add"], flush=True)

# F: consumer fold vs runtime hook
if "F" not in skip:
    set_mode("multi"); hook = [rows_gen(p, ["r_cons"] * 4)[0] for p in gen_prompts]
    print("folded qkvr tensors", set_mode("overlay", av.canonicalize(build()["r_cons"]["resid_dirs"],
                                                                     build()["r_cons"]["resid_scales"])))
    bake = [rows_gen(p, [None] * 4)[0] for p in gen_prompts]
    set_mode("plain"); stock = [rows_gen(p, [None] * 4)[0] for p in gen_prompts]
    R["F_consumer_fold"] = dict(bake_vs_hook=cmp_gen(hook, bake), stock_vs_hook=cmp_gen(stock, hook),
                                stock_vs_bake=cmp_gen(stock, bake))
    print("F", {k: {kk: v[kk] for kk in ("n", "exact", "kl_mean", "lp_absdiff_max")} for k, v in R["F_consumer_fold"].items()}, flush=True)
    set_mode("multi")

# G: generations for the judge
if "G" not in skip:
    set_mode("multi")
    G = chat(harm) + chat(ben)
    kinds = ["harmful"] * len(harm) + ["benign"] * len(ben)
    with open(f"{RUN}/gens.jsonl", "w") as fo:
        for v in (None, "r_res", "r_res_h", "r_both", "op_only"):
            outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in G],
                                [sp(v, temperature=0.0, max_tokens=args.judge_tokens)] * len(G), use_tqdm=False)
            for i, o in enumerate(outs):
                fo.write(json.dumps(dict(variant=v or "stock", kind=kinds[i], idx=i,
                                         prompt=(harm + ben)[i], text=o.outputs[0].text)) + "\n")
            print("G", v, "done", flush=True)

# H: throughput
if "H" not in skip:
    set_mode("multi")
    def tps(vs, reps=3, n=256):
        best = 0.0
        for _ in range(reps):
            t = time.time()
            outs = llm.generate([TokensPrompt(prompt_token_ids=gen_prompts[i]) for i in range(len(vs))],
                                [sp(v, temperature=0.0, max_tokens=n, ignore_eos=True) for v in vs], use_tqdm=False)
            best = max(best, sum(len(o.outputs[0].token_ids) for o in outs) / (time.time() - t))
        return best
    R["H_tok_s"] = dict(stock8=tps([None] * 8),
                        mixed8=tps([None, "r_res", "r_both", "r_res_h", None, "r_res", "op_only", None]),
                        resid8=tps(["r_res"] * 8))
    set_mode("plain"); R["H_tok_s"]["plain8"] = tps([None] * 8); set_mode("multi")
    print("H", R["H_tok_s"], flush=True)

R["elapsed_s"] = time.time() - t0
json.dump(R, open(f"{RUN}/validate.json", "w"), indent=1, default=str)
print("DONE r5_validate", R["elapsed_s"])
