# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AEON-7
"""Teacher-forced KL / top-1 of the AEON ROCm vLLM build vs the transformers
bf16 reference logprobs of the stock model (~/inkling/phase3/stock_kl:
full-vocab float16 log-softmax over the first <=32 answer tokens of 200
effort-none prompts).

vLLM returns prompt logprobs for its top-K tokens (K=1000), so the KL per
position is KL(ref || vllm) over the K tokens plus one bucket for the rest
of the vocabulary (a coarsening; a lower bound on the full KL that is tight
when the bucket mass is tiny, which we report)."""
import argparse, json, math, os, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.expanduser("~/inkling/models/Inkling-2-Mini"))
ap.add_argument("--kl-dir", default=os.path.expanduser("~/inkling/phase3/stock_kl"))
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--k", type=int, default=1000)
ap.add_argument("--out", default="kl_result.json")
ap.add_argument("--gpu-util", type=float, default=0.80)
ap.add_argument("--eager", action="store_true")
ap.add_argument("--cpu-offload-gb", type=float, default=0.0)
ap.add_argument("--max-batched", type=int, default=2048)
ap.add_argument("--max-len", type=int, default=4096)
ap.add_argument("--kv-bytes", type=int, default=0)
args = ap.parse_args()

idx = json.load(open(f"{args.kl_dir}/index.json"))
gens = {}
for line in open(f"{args.kl_dir}/stock_kl_gens.jsonl"):
    r = json.loads(line)
    gens[r["id"]] = r
ref = np.memmap(idx["mmap"] if os.path.exists(idx["mmap"]) else f"{args.kl_dir}/logprobs.f16",
                dtype=np.float16, mode="r", shape=(idx["n_rows"], idx["t"], idx["vocab"]))

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(args.model)

rows, prompts, mism = [], [], 0
for row in idx["rows"]:
    take = row.get("n_answer_tokens") or 0
    if take <= 0:
        continue
    g = gens[row["id"]]
    text = tok.apply_chat_template([{"role": "user", "content": g["prompt"]}], tokenize=False,
                                   add_generation_prompt=True, reasoning_effort="none")
    pids = tok(text, add_special_tokens=False)["input_ids"]
    if len(pids) != row["n_input_tokens"]:
        mism += 1
        continue
    start = row["answer_start"]
    full = pids + g["gen_ids"][: start + take]
    assert g["gen_ids"][start:start + take] == row["answer_ids"][:take]
    rows.append((row, len(pids) + start, take))
    prompts.append({"prompt_token_ids": full})
    if len(rows) >= args.n:
        break
print(f"rows={len(rows)} token-count mismatches={mism}", flush=True)

from vllm import LLM, SamplingParams
t0 = time.time()
llm = LLM(model=args.model, max_logprobs=args.k, max_model_len=args.max_len, cpu_offload_gb=args.cpu_offload_gb, **({"kv_cache_memory_bytes": args.kv_bytes} if args.kv_bytes else {}), max_num_seqs=8,
          max_num_batched_tokens=args.max_batched, gpu_memory_utilization=args.gpu_util,
          enforce_eager=args.eager, limit_mm_per_prompt={"image": 0},
          enable_prefix_caching=False)
print(f"load {time.time()-t0:.1f}s", flush=True)
sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=args.k)
t0 = time.time()
outs = llm.generate(prompts, sp)
print(f"prefill+logprobs {time.time()-t0:.1f}s", flush=True)

per_prompt, all_kl, top1, tails, firsts = [], [], [], [], []
for (row, pos0, take), o in zip(rows, outs):
    plp = o.prompt_logprobs
    kls = []
    for j in range(take):
        d = plp[pos0 + j]
        ids = np.fromiter(d.keys(), dtype=np.int64)
        lv = np.array([v.logprob for v in d.values()], dtype=np.float64)
        ranks = np.array([v.rank for v in d.values()])
        rl = ref[row["index"], j].astype(np.float64)
        pr = np.exp(rl)
        ids_k = ids[ids < rl.shape[0]]
        lv_k = lv[ids < rl.shape[0]]
        # drop the sampled-token extra entry if it is outside the top-K
        pr_s = pr[ids_k]
        pv_s = np.exp(lv_k)
        kl = float(np.sum(pr_s * (rl[ids_k] - lv_k)))
        tail_r = max(1.0 - pr_s.sum(), 1e-12)
        tail_v = max(1.0 - pv_s.sum(), 1e-12)
        kl += tail_r * math.log(tail_r / tail_v)
        kls.append(max(kl, 0.0))
        tails.append(tail_r)
        v_top = int(ids[np.argmin(ranks)])
        top1.append(int(v_top == int(np.argmax(rl))))
    per_prompt.append(sum(kls) / len(kls))
    firsts.append(kls[0])
    all_kl.extend(kls)

res = dict(
    n_prompts=len(per_prompt), n_positions=len(all_kl),
    kl_mean_prompt=float(np.mean(per_prompt)), kl_mean_position=float(np.mean(all_kl)),
    kl_p95_position=float(np.percentile(all_kl, 95)), kl_max_position=float(np.max(all_kl)),
    kl_first_mean=float(np.mean(firsts)), top1_agree=float(np.mean(top1)),
    ref_mass_outside_topk_mean=float(np.mean(tails)), k=args.k,
)
print(json.dumps(res, indent=1))
json.dump(dict(summary=res, per_prompt=per_prompt), open(args.out, "w"), indent=1)
