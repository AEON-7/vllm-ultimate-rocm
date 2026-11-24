# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AEON-7
"""Validation runs for AEON per-request ablation variants. One mode per
process (each mode needs its own engine env):
  prep         build the variant files and the baked overlay (CPU only)
  plain        plain M1 engine (no AEON env)
  ablate_eq    server-wide AEON_ABLATE = V_eq
  ablate_pass  server-wide AEON_ABLATE = V_pass (per-pass loop scales)
  overlay_eq   AEON_WEIGHT_OVERLAY with V_eq baked into wo_ud
  multi        AEON_VARIANT_DIR server: stock, per-request variants alone,
               a mixed batch of all variants, throughput
Outputs RUN/<mode>.pt"""
import argparse, json, os, sys, time
import torch

RUN = os.path.expanduser("~/inkling/engines/aeon-cleanroom/runs/abl")
MODEL = os.path.expanduser("~/inkling/models/Inkling-2-Mini")
ap = argparse.ArgumentParser()
ap.add_argument("mode")
ap.add_argument("--gen-tokens", type=int, default=96)
ap.add_argument("--k", type=int, default=50)
args = ap.parse_args()
os.makedirs(RUN, exist_ok=True)
VDIR = f"{RUN}/variants"
D = 2304


def loop_apps(pass_idx):
    return range(6 + 12 * pass_idx, 18 + 12 * pass_idx)


def build_variants():
    """dict name -> (dirs, scales) in the canonical [48,K,D]/[48,K] form."""
    src = torch.load(os.path.expanduser("~/inkling/directions/inkling2mini_stock_dom_directions.pt"),
                     map_location="cpu", weights_only=False)["directions"]
    per_visit = torch.stack([src[f"none_last_v{a:02d}"].float() for a in range(48)])  # [48, D]
    # per-layer directions: loop layers use their last-pass visit (30..41)
    layer_vis = list(range(0, 6)) + list(range(30, 42)) + list(range(42, 48))
    per_layer = per_visit[layer_vis]  # [24, D]
    g = torch.Generator().manual_seed(7)
    V = {}
    s = torch.zeros(48); s[4:] = 1.0
    V["v_eq"] = (per_layer, s)                      # overlay-expressible (same scale every pass)
    s = torch.zeros(48); s[4:6] = 1.0; s[42:] = 1.0
    s[list(loop_apps(0))] = 1.0; s[list(loop_apps(1))] = 0.6; s[list(loop_apps(2))] = 0.3
    V["v_pass"] = (per_visit, s)                    # per-pass loop scales, per-visit dirs
    for j, sc in enumerate((0.25, 0.5, 0.75, 1.25, 1.5)):
        s = torch.zeros(48); s[4:] = sc
        V[f"v_s{j}"] = (per_layer, s)
    # rank-2 and rank-3 variants
    r2 = torch.stack([per_visit, torch.randn(48, D, generator=g)], 1)
    s = torch.zeros(48, 2); s[8:24, 0] = 1.0; s[8:24, 1] = 2.0
    V["v_r2"] = (r2, s)
    r3 = torch.stack([per_visit, per_visit.roll(1, 0), torch.randn(48, D, generator=g)], 1)
    s = torch.zeros(48, 3); s[4:, 0] = 0.8; s[20:40, 1] = 0.5; s[30:, 2] = 1.0
    V["v_r3"] = (r3, s)
    return V


def chat_prompts(tok):
    rows = []
    for fn, n in (("harmful_eval.jsonl", 8), ("benign_eval.jsonl", 8)):
        for line in list(open(os.path.expanduser(f"~/inkling/data/{fn}")))[:n]:
            p = json.loads(line)["prompt"]
            text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                           add_generation_prompt=True, reasoning_effort="none")
            rows.append(tok(text, add_special_tokens=False)["input_ids"])
    return rows


def tf_sequences(tok, n=16):
    kl_dir = os.path.expanduser("~/inkling/phase3/stock_kl")
    idx = json.load(open(f"{kl_dir}/index.json"))
    gens = {json.loads(l)["id"]: json.loads(l) for l in open(f"{kl_dir}/stock_kl_gens.jsonl")}
    out = []
    for row in idx["rows"]:
        take = row.get("n_answer_tokens") or 0
        if take <= 0:
            continue
        g = gens[row["id"]]
        text = tok.apply_chat_template([{"role": "user", "content": g["prompt"]}], tokenize=False,
                                       add_generation_prompt=True, reasoning_effort="none")
        pids = tok(text, add_special_tokens=False)["input_ids"]
        full = pids + g["gen_ids"][: row["answer_start"] + take]
        out.append((full, len(pids)))
        if len(out) >= n:
            break
    return out


if args.mode == "prep":
    sys.path.insert(0, os.path.expanduser("~/inkling/engines/aeon-cleanroom/wt-abl"))
    os.environ["AEON_VARIANT_DIR"] = VDIR
    from vllm.models.inkling2.amd import aeon_variants as av
    from safetensors import safe_open
    from safetensors.torch import save_file
    V = build_variants()
    for name, (d, s) in V.items():
        print(av.save_variant(name, d, s)["nonzero_apps"][:6], name)
    for name in ("v_eq", "v_pass"):  # AEON_ABLATE files (rank 1, [48, D] + [48])
        d, s = av.canonicalize(*V[name])
        save_file({"dirs": d[:, 0].contiguous(), "scales": s[:, 0].contiguous()}, f"{RUN}/ablate_{name}.safetensors")
    d, s = av.canonicalize(*V["v_eq"])
    ov = {}
    with safe_open(f"{MODEL}/model-00001-of-00001.safetensors", "pt") as f:
        for layer in range(24):
            app = 6 + (layer - 6) if 6 <= layer < 18 else (layer if layer < 6 else layer + 24)
            sc = float(s[app, 0]); r = d[app, 0]
            key = f"model.llm.layers.{layer}.attn.wo_ud.weight"
            W = f.get_tensor(key)
            if sc == 0.0:
                continue
            Wf = W.float()
            ov[key] = (Wf - sc * torch.outer(r, r @ Wf)).to(W.dtype).contiguous()
    save_file(ov, f"{RUN}/overlay_v_eq.safetensors")
    print("overlay tensors", len(ov))
    sys.exit(0)

if args.mode == "ablate_eq":
    os.environ["AEON_ABLATE"] = f"{RUN}/ablate_v_eq.safetensors"
elif args.mode == "ablate_pass":
    os.environ["AEON_ABLATE"] = f"{RUN}/ablate_v_pass.safetensors"
elif args.mode == "overlay_eq":
    os.environ["AEON_WEIGHT_OVERLAY"] = f"{RUN}/overlay_v_eq.safetensors"
elif args.mode in ("multi", "bench"):
    os.environ["AEON_VARIANT_DIR"] = VDIR

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

llm = LLM(model=MODEL, max_model_len=4096, max_num_seqs=16, max_num_batched_tokens=2048,
          gpu_memory_utilization=0.80, enforce_eager=True, limit_mm_per_prompt={"image": 0},
          enable_prefix_caching=False, max_logprobs=args.k)
tok = llm.get_tokenizer()
gen_prompts = chat_prompts(tok)
tf = tf_sequences(tok)


def sp(variant, **kw):
    return SamplingParams(extra_args={"aeon_variant": variant} if variant else None, **kw)


def run_gen(variants):
    """variants: list (one per request) over gen_prompts repeated."""
    reqs, sps = [], []
    for v in variants:
        for p in gen_prompts:
            reqs.append(TokensPrompt(prompt_token_ids=p))
            sps.append(sp(v, temperature=0.0, max_tokens=args.gen_tokens, logprobs=args.k))
    outs = llm.generate(reqs, sps, use_tqdm=False)
    res = []
    for o in outs:
        c = o.outputs[0]
        res.append(dict(ids=list(c.token_ids),
                        lps=[{t: l.logprob for t, l in d.items()} for d in c.logprobs]))
    return res


def run_tf(variants):
    reqs, sps = [], []
    for v in variants:
        for full, _ in tf:
            reqs.append(TokensPrompt(prompt_token_ids=full))
            sps.append(sp(v, temperature=0.0, max_tokens=1, prompt_logprobs=args.k))
    outs = llm.generate(reqs, sps, use_tqdm=False)
    res = []
    for o, (full, start) in zip(outs, tf * len(variants)):
        pl = o.prompt_logprobs
        res.append([{t: l.logprob for t, l in pl[i].items()} for i in range(start, len(full))])
    return res


def bench(variants, tokens=256, reps=3):
    # distinct prompts, so MoE routing diverges as in real screening traffic
    prompts = [TokensPrompt(prompt_token_ids=gen_prompts[8 + i % 8]) for i in range(len(variants))]
    llm.generate(prompts, [sp(v, max_tokens=16) for v in variants], use_tqdm=False)
    rates = []
    for _ in range(reps):
        t0 = time.time()
        outs = llm.generate(prompts,
                            [sp(v, temperature=0.0, max_tokens=tokens, ignore_eos=True) for v in variants],
                            use_tqdm=False)
        rates.append(sum(len(o.outputs[0].token_ids) for o in outs) / (time.time() - t0))
    rates.sort()
    return dict(median=rates[len(rates) // 2], runs=rates)


R = {"mode": args.mode}
if args.mode == "det":
    R["gen"] = run_gen([None]); R["tf"] = run_tf([None])
    R["gen2"] = run_gen([None]); R["tf2"] = run_tf([None])
elif args.mode == "bench":  # interleaved overhead bench, same process
    names = ["v_eq", "v_pass", "v_s0", "v_s1", "v_s2", "v_s3", "v_s4", "v_r2"]
    for rep in range(3):
        for key, vs in (("stock", [None] * 8), ("variants8", names), ("one_variant", ["v_eq"] * 8),
                        ("half", [None, "v_eq", None, "v_pass", None, "v_r2", None, "v_s1"])):
            R.setdefault(f"bench_{key}_bs8", []).append(bench(vs, reps=1)["median"])
        R.setdefault("bench_stock_bs1", []).append(bench([None], tokens=128, reps=1)["median"])
        R.setdefault("bench_variant_bs1", []).append(bench(["v_eq"], tokens=128, reps=1)["median"])
    R = {k: (dict(median=sorted(v)[1], runs=v) if k.startswith("bench") else v) for k, v in R.items()}
elif args.mode == "multi":
    names = ["v_eq", "v_pass", "v_s0", "v_s1", "v_s2", "v_s3", "v_s4", "v_r2", "v_r3"]
    R["bench_stock_bs8"] = bench([None] * 8)
    R["bench_stock_bs1"] = bench([None], tokens=128)
    R["gen"] = {"stock": run_gen([None])}
    R["tf"] = {"stock": run_tf([None])}
    for n in names:
        R["gen"][n] = run_gen([n])
        R["tf"][n] = run_tf([n])
        print("alone", n, flush=True)
    # mixed: every request of every variant (incl. stock) in one call, interleaved
    order = [None] + names
    reqs, sps, keys = [], [], []
    for i, p in enumerate(gen_prompts):
        for j in range(len(order)):
            v = order[(i + j) % len(order)]
            reqs.append(TokensPrompt(prompt_token_ids=p))
            sps.append(sp(v, temperature=0.0, max_tokens=args.gen_tokens, logprobs=args.k))
            keys.append((v or "stock", i))
    t0 = time.time()
    outs = llm.generate(reqs, sps, use_tqdm=False)
    R["mixed_s"] = time.time() - t0
    R["mixed_gen"] = {f"{k[0]}|{k[1]}": dict(ids=list(o.outputs[0].token_ids),
                                             lps=[{t: l.logprob for t, l in d.items()} for d in o.outputs[0].logprobs])
                      for k, o in zip(keys, outs)}
    reqs, sps, keys = [], [], []
    for i, (full, start) in enumerate(tf):
        for j in range(len(order)):
            v = order[(i + j) % len(order)]
            reqs.append(TokensPrompt(prompt_token_ids=full))
            sps.append(sp(v, temperature=0.0, max_tokens=1, prompt_logprobs=args.k))
            keys.append((v or "stock", i, start, len(full)))
    outs = llm.generate(reqs, sps, use_tqdm=False)
    R["mixed_tf"] = {f"{k[0]}|{k[1]}": [{t: l.logprob for t, l in o.prompt_logprobs[p].items()}
                                         for p in range(k[2], k[3])] for k, o in zip(keys, outs)}
    # control: identical request structure, every request stock
    reqs = [TokensPrompt(prompt_token_ids=p) for p in gen_prompts for _ in order]
    outs = llm.generate(reqs, [sp(None, temperature=0.0, max_tokens=args.gen_tokens, logprobs=args.k)
                               for _ in reqs], use_tqdm=False)
    R["control_mixed_stock_gen"] = [dict(ids=list(o.outputs[0].token_ids),
                                         lps=[{t: l.logprob for t, l in d.items()} for d in o.outputs[0].logprobs])
                                    for o in outs[::len(order)]]
    reqs = [TokensPrompt(prompt_token_ids=full) for full, _ in tf for _ in order]
    outs = llm.generate(reqs, [sp(None, temperature=0.0, max_tokens=1, prompt_logprobs=args.k) for _ in reqs],
                        use_tqdm=False)
    R["control_mixed_stock_tf"] = [[{t: l.logprob for t, l in o.prompt_logprobs[p].items()} for p in range(start, len(full))]
                                   for o, (full, start) in zip(outs[::len(order)], tf)]
    R["bench_variants_bs8"] = bench(names[:8])
    R["bench_one_variant_bs8"] = bench(["v_eq"] * 8)
    R["bench_mixed_half_bs8"] = bench([None, "v_eq", None, "v_pass", None, "v_r2", None, "v_s1"])
    R["bench_variant_bs1"] = bench(["v_eq"], tokens=128)
    R["texts"] = {n: tok.decode(R["gen"][n][0]["ids"]) for n in ["stock"] + names}
else:
    R["gen"] = run_gen([None])
    R["tf"] = run_tf([None])
    if args.mode == "plain":
        R["bench_stock_bs8"] = bench([None] * 8)
        R["bench_stock_bs1"] = bench([None], tokens=128)
torch.save(R, f"{RUN}/{args.mode}.pt")
print(json.dumps({k: v for k, v in R.items() if k.startswith("bench") or k == "mixed_s"}, indent=1))
print("DONE", args.mode)
