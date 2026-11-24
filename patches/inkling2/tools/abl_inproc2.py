# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AEON-7
"""Follow-up checks (same process):
 A) T3 same-shape batches with position control: one prompt x 10 rows, mixed
    variants (rotated so each variant sits at every row position) vs a batch
    of 10 copies of that variant, comparing the SAME row position; plus the
    engine's own position noise (row 0 vs row j of identical requests).
 B) T2 first-edited-layer check: o_proj output at app 4 (first edited
    application, identical input for every path) captured for the request
    variant, AEON_ABLATE and the baked overlay; max diff in bf16 ulps."""
import glob, json, os, sys, time
import torch

RUN = os.path.expanduser("~/inkling/engines/aeon-cleanroom/runs/abl")
os.environ["AEON_VARIANT_DIR"] = f"{RUN}/variants"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
sys.argv = [sys.argv[0], "inproc2", "--gen-tokens", "64"]
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
exec(open(os.path.join(here, "abl_validate.py")).read().split("\nR = {\"mode\"")[0])
from abl_compare_lib import cmp_gen  # noqa: E402
src = open(os.path.join(here, "abl_inproc.py")).read()
exec(src[src.index("from vllm.models.inkling2.amd.aeon_hooks import HOOKS"):src.index("def both(")])

order = [None, "v_eq", "v_pass", "v_s0", "v_s1", "v_s2", "v_s3", "v_s4", "v_r2", "v_r3"]
names = [v or "stock" for v in order]
N = len(order)


def rows_gen(prompt, variants):
    outs = llm.generate([TokensPrompt(prompt_token_ids=prompt)] * len(variants),
                        [sp(v, temperature=0.0, max_tokens=args.gen_tokens, logprobs=args.k) for v in variants],
                        use_tqdm=False)
    return [dict(ids=list(o.outputs[0].token_ids), lps=[{t: l.logprob for t, l in d.items()} for d in o.outputs[0].logprobs])
            for o in outs]


set_mode("multi")
rows_gen(gen_prompts[0], order)  # warm
res = {n: dict(mixed=[], alone=[], pos_noise=[]) for n in names}
for pi in (0, 9):
    p = gen_prompts[pi]
    alone = {n: rows_gen(p, [v] * N) for n, v in zip(names, order)}
    for n in names:
        res[n]["pos_noise"].append(cmp_gen([alone[n][0]] * (N - 1), alone[n][1:]))
    for rot in range(N):
        perm = [order[(j + rot) % N] for j in range(N)]
        mixed = rows_gen(p, perm)
        for j, v in enumerate(perm):
            n = v or "stock"
            res[n]["mixed"].append(mixed[j]); res[n]["alone"].append(alone[n][j])
    print("prompt", pi, "done", flush=True)
A = {}
for n in names:
    c = cmp_gen(res[n]["alone"], res[n]["mixed"])
    pn = res[n]["pos_noise"]
    A[n] = dict(same_position=dict(exact=c["exact"], n=c["n"], kl_mean=c["kl_mean"], first_div=c["first_div"]),
                engine_position_noise=dict(exact=sum(x["exact"] for x in pn), n=sum(x["n"] for x in pn)))
    print("A", n, A[n], flush=True)

# B) first edited application capture
cap = f"{RUN}/cap_b"
os.makedirs(cap, exist_ok=True)
seqs = [TokensPrompt(prompt_token_ids=full) for full, _ in tf[:8]]


def capture(tag, variant=None):
    for f in glob.glob(f"{cap}/*.pt"):
        os.remove(f)
    def on(model):
        HOOKS.capture_dir, HOOKS.capture_apps, HOOKS.capture_what = cap, {4, 5, 30}, {"oproj_out", "oproj_in"}
        HOOKS.enabled = True
    def off(model):
        HOOKS.capture_dir = None
    llm.apply_model(on)
    llm.generate(seqs, [sp(variant, max_tokens=1)] * len(seqs), use_tqdm=False)
    llm.apply_model(off)
    files = sorted(glob.glob(f"{cap}/*.pt"))
    data = [torch.load(f) for f in files]
    return {k: torch.cat([d[k][4] for d in data]) for k in ("oproj_out", "oproj_in")}


def ulps(a, b):
    a, b = a.bfloat16().float(), b.bfloat16().float()
    ulp = torch.where(a.abs() > 0, 2.0 ** (torch.floor(torch.log2(a.abs().clamp_min(1e-30))) - 7), torch.tensor(2.0 ** -133))
    d = (a - b).abs() / ulp
    return dict(max_ulps=d.max().item(), frac_bitwise_equal=(a == b).float().mean().item())


set_mode("multi"); Cs = capture("stock"); Cr = capture("req", "v_eq")
set_mode("ablate", f"{RUN}/ablate_v_eq.safetensors"); Ca = capture("ablate")
set_mode("overlay", f"{RUN}/overlay_v_eq.safetensors"); Co = capture("overlay")
set_mode("plain")
from safetensors.torch import load_file
from vllm.models.inkling2.amd.aeon_variants import canonicalize, apply_rows_reference
d, s = canonicalize(*[load_file(f"{RUN}/variants/v_eq.safetensors")[k] for k in ("dirs", "scales")])
x = Cs["oproj_out"]
ref = apply_rows_reference(x.bfloat16(), torch.ones(x.shape[0], dtype=torch.int32), torch.stack([torch.zeros_like(d), d]),
                           torch.stack([torch.zeros_like(s), s]), 4).float()
r = d[4, 0]
B = dict(rows=x.shape[0],
         same_input=dict(req_vs_ablate=(Cr["oproj_in"] == Ca["oproj_in"]).all().item(),
                         req_vs_overlay=(Cr["oproj_in"] == Co["oproj_in"]).all().item()),
         req_vs_fp32_reference=ulps(Cr["oproj_out"], ref),
         req_vs_ablate_bf16=ulps(Cr["oproj_out"], Ca["oproj_out"]),
         req_vs_overlay=ulps(Cr["oproj_out"], Co["oproj_out"]),
         ablate_vs_overlay=ulps(Ca["oproj_out"], Co["oproj_out"]),
         stock_vs_req=ulps(Cs["oproj_out"], Cr["oproj_out"]),
         residual_r_dot_y=dict(stock=(x @ r).abs().mean().item(), req=(Cr["oproj_out"] @ r).abs().mean().item(),
                               ablate=(Ca["oproj_out"] @ r).abs().mean().item(), overlay=(Co["oproj_out"] @ r).abs().mean().item()))
print("B", json.dumps(B, indent=1))
json.dump(dict(A=A, B=B), open(f"{RUN}/inproc2.json", "w"), indent=1)
print("DONE inproc2")
