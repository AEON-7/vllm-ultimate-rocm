# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AEON-7
"""Same-process validation of AEON per-request variants (removes the
cross-process Triton-autotune noise of this engine). The engine runs
in-process; between phases the AEON hooks are switched on the live model:
  plain   = exact M1 code path (HOOKS off, no variant runner hook)
  multi   = AEON_VARIANT_DIR server path
  ablate  = server-wide AEON_ABLATE (bf16 runtime edit)
  overlay = wo_ud weights swapped for the baked (I - s r r^T) W, then restored
Writes runs/abl/inproc.pt and prints the comparisons."""
import json, os, sys, time
import torch

RUN = os.path.expanduser("~/inkling/engines/aeon-cleanroom/runs/abl")
os.environ["AEON_VARIANT_DIR"] = f"{RUN}/variants"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
sys.argv = [sys.argv[0], "inproc"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "abl_validate.py")).read().split("\nR = {\"mode\"")[0])
from abl_compare_lib import cmp_gen, cmp_tf, brief  # noqa: E402

from vllm.models.inkling2.amd.aeon_hooks import HOOKS
from vllm.models.inkling2.amd.aeon_variants import VARIANTS
import vllm.v1.worker.gpu.model_runner as mr

state = {}


def set_mode(mode, arg=None):
    def f(model):
        if "orig" in state and mode != "overlay":
            p = dict(model.named_parameters())
            for k, v in state.pop("orig").items():
                p[k].data.copy_(v)
        HOOKS.ablate_path, HOOKS._dirs = None, None
        if mode == "plain":
            HOOKS.enabled, HOOKS.variants, mr._AEON_VARIANTS = False, None, None
        elif mode == "multi":
            HOOKS.enabled, HOOKS.variants, mr._AEON_VARIANTS = True, VARIANTS, VARIANTS
        elif mode == "ablate":
            HOOKS.enabled, HOOKS.variants, mr._AEON_VARIANTS = True, None, None
            HOOKS.ablate_path = arg
        elif mode == "overlay":
            HOOKS.enabled, HOOKS.variants, mr._AEON_VARIANTS = False, None, None
            from safetensors.torch import load_file
            ov = load_file(arg)
            p = dict(model.named_parameters())
            orig = {}
            for key, w in ov.items():
                layer = key.split(".layers.")[1].split(".")[0]
                name = next(n for n in p if n.endswith(f"layers.{layer}.attn.wo_ud.weight"))
                orig[name] = p[name].data.clone()
                p[name].data.copy_(w.to(p[name].device))
            state["orig"] = orig
            return len(orig)
        return mode
    return llm.apply_model(f)


def both(v=None):
    return dict(gen=run_gen([v]), tf=run_tf([v]))


R = {}
t0 = time.time()
set_mode("multi"); both(None); both("v_eq"); both("v_r3")         # warm-up / autotune
set_mode("plain"); both(None)
R["plain1"] = both(None); R["plain2"] = both(None)
set_mode("multi")
R["multi_stock"] = both(None)
for n in ("v_eq", "v_pass"):
    R[f"req_{n}"] = both(n)
set_mode("ablate", f"{RUN}/ablate_v_eq.safetensors"); R["ablate_v_eq"] = both(None)
set_mode("ablate", f"{RUN}/ablate_v_pass.safetensors"); R["ablate_v_pass"] = both(None)
print("overlay tensors swapped", set_mode("overlay", f"{RUN}/overlay_v_eq.safetensors"))
R["overlay_v_eq"] = both(None)
set_mode("plain"); R["plain_after_restore"] = both(None)
set_mode("multi")
# T3a: same-shape batches (one prompt x 10 rows): mixed variants vs all-one-variant
order = [None, "v_eq", "v_pass", "v_s0", "v_s1", "v_s2", "v_s3", "v_s4", "v_r2", "v_r3"]
names = [v or "stock" for v in order]


def rows_gen(prompt, variants):
    outs = llm.generate([TokensPrompt(prompt_token_ids=prompt)] * len(variants),
                        [sp(v, temperature=0.0, max_tokens=args.gen_tokens, logprobs=args.k) for v in variants],
                        use_tqdm=False)
    return [dict(ids=list(o.outputs[0].token_ids), lps=[{t: l.logprob for t, l in d.items()} for d in o.outputs[0].logprobs])
            for o in outs]


T3a = {n: dict(alone=[], mixed=[]) for n in names}
for pi in (0, 3, 9, 12):
    p = gen_prompts[pi]
    mixed = rows_gen(p, order)
    for n, v, m in zip(names, order, mixed):
        T3a[n]["mixed"].append(m)
        T3a[n]["alone"].append(rows_gen(p, [v] * len(order))[0])
R["T3a"] = T3a
# T3b: all 16 prompts x 10 variants interleaved in one call vs each variant alone
alone = {n: run_gen([v]) for n, v in zip(names, order)}
alone_tf = {n: run_tf([v]) for n, v in zip(names, order)}
reqs, sps, keys = [], [], []
for i, p in enumerate(gen_prompts):
    for j in range(len(order)):
        v = order[(i + j) % len(order)]
        reqs.append(TokensPrompt(prompt_token_ids=p)); keys.append((v or "stock", i))
        sps.append(sp(v, temperature=0.0, max_tokens=args.gen_tokens, logprobs=args.k))
outs = llm.generate(reqs, sps, use_tqdm=False)
mixed = {k: dict(ids=list(o.outputs[0].token_ids), lps=[{t: l.logprob for t, l in d.items()} for d in o.outputs[0].logprobs])
         for k, o in zip(keys, outs)}
reqs, sps, keys = [], [], []
for i, (full, start) in enumerate(tf):
    for j in range(len(order)):
        v = order[(i + j) % len(order)]
        reqs.append(TokensPrompt(prompt_token_ids=full)); keys.append((v or "stock", i, start, len(full)))
        sps.append(sp(v, temperature=0.0, max_tokens=1, prompt_logprobs=args.k))
outs = llm.generate(reqs, sps, use_tqdm=False)
mixed_tf = {k[:2]: [{t: l.logprob for t, l in o.prompt_logprobs[q].items()} for q in range(k[2], k[3])]
            for k, o in zip(keys, outs)}
# control: same structure, all stock
outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in gen_prompts for _ in order],
                    [sp(None, temperature=0.0, max_tokens=args.gen_tokens, logprobs=args.k)] * (len(gen_prompts) * len(order)),
                    use_tqdm=False)
ctrl = [dict(ids=list(o.outputs[0].token_ids), lps=[{t: l.logprob for t, l in d.items()} for d in o.outputs[0].logprobs])
        for o in outs[::len(order)]]
outs = llm.generate([TokensPrompt(prompt_token_ids=full) for full, _ in tf for _ in order],
                    [sp(None, temperature=0.0, max_tokens=1, prompt_logprobs=args.k)] * (len(tf) * len(order)),
                    use_tqdm=False)
ctrl_tf = [[{t: l.logprob for t, l in o.prompt_logprobs[q].items()} for q in range(s, len(f))]
           for o, (f, s) in zip(outs[::len(order)], tf)]

out = {}
out["noise_plain_repeat"] = dict(gen=cmp_gen(R["plain1"]["gen"], R["plain2"]["gen"]), tf=cmp_tf(R["plain1"]["tf"], R["plain2"]["tf"]))
out["T1_multi_stock_vs_plain"] = dict(gen=cmp_gen(R["plain1"]["gen"], R["multi_stock"]["gen"]), tf=cmp_tf(R["plain1"]["tf"], R["multi_stock"]["tf"]))
out["T1_plain_after_overlay_restore"] = dict(gen=cmp_gen(R["plain1"]["gen"], R["plain_after_restore"]["gen"]), tf=cmp_tf(R["plain1"]["tf"], R["plain_after_restore"]["tf"]))
for a, b in (("req_v_eq", "ablate_v_eq"), ("req_v_eq", "overlay_v_eq"), ("ablate_v_eq", "overlay_v_eq"),
             ("req_v_pass", "ablate_v_pass"), ("plain1", "req_v_eq"), ("plain1", "req_v_pass")):
    out[f"T2_{a}_vs_{b}"] = dict(gen=cmp_gen(R[b]["gen"], R[a]["gen"]), tf=cmp_tf(R[b]["tf"], R[a]["tf"]))
for n in names:
    out[f"T3a_sameshape_{n}"] = dict(gen=cmp_gen(T3a[n]["alone"], T3a[n]["mixed"]))
    out[f"T3b_mixed_{n}"] = dict(gen=cmp_gen(alone[n], [mixed[(n, i)] for i in range(len(gen_prompts))]),
                                 tf=cmp_tf(alone_tf[n], [mixed_tf[(n, i)] for i in range(len(tf))]))
out["T3b_control_stock_in_mixed_structure"] = dict(gen=cmp_gen(alone["stock"], ctrl), tf=cmp_tf(alone_tf["stock"], ctrl_tf))
out["elapsed_s"] = time.time() - t0
torch.save(dict(R=R, out=out), f"{RUN}/inproc.pt")
json.dump(out, open(f"{RUN}/inproc.json", "w"), indent=1)
for k, v in out.items():
    if isinstance(v, dict):
        print(k, "::", brief(v))
print("DONE inproc")
