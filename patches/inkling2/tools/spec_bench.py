# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AEON-7
"""(spec_bench: gen_bench plus --spec N for DSpark speculative decoding and
saved greedy token ids for the spec-vs-greedy identity check.)
Multi-turn chat sanity (greedy, capped tokens, loop check) and decode
throughput (bs1 and 8 concurrent, 256 new tokens, ignore_eos) for the AEON
ROCm vLLM build."""
import argparse, json, os, time, zlib

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.expanduser("~/inkling/models/Inkling-2-Mini"))
ap.add_argument("--out", default="gen_bench.json")
ap.add_argument("--gpu-util", type=float, default=0.80)
ap.add_argument("--eager", action="store_true")
ap.add_argument("--cpu-offload-gb", type=float, default=0.0)
ap.add_argument("--max-batched", type=int, default=2048)
ap.add_argument("--max-len", type=int, default=4096)
ap.add_argument("--kv-bytes", type=int, default=0)
ap.add_argument("--spec", type=int, default=0, help="DSpark draft tokens (0: off)")
ap.add_argument("--bench-only", action="store_true")
ap.add_argument("--bs-list", default="1,8")
ap.add_argument("--spec-extra", default="{}", help="extra speculative_config JSON")
ap.add_argument("--prompt", default="story", choices=["story", "code", "math"])
args = ap.parse_args()

from vllm import LLM, SamplingParams
llm = LLM(model=args.model, max_model_len=args.max_len, cpu_offload_gb=args.cpu_offload_gb, **({"kv_cache_memory_bytes": args.kv_bytes} if args.kv_bytes else {}), max_num_seqs=8, max_num_batched_tokens=args.max_batched,
          gpu_memory_utilization=args.gpu_util, enforce_eager=args.eager,
          limit_mm_per_prompt={"image": 0}, disable_log_stats=False,
          **({"speculative_config": {"method": "dspark", "num_speculative_tokens": args.spec, **json.loads(args.spec_extra)}} if args.spec else {}))

def loopy(text):
    if len(text) < 200:
        return False
    tail = text[-600:].encode()
    return len(zlib.compress(tail)) / len(tail) < 0.2

kw = dict(chat_template_kwargs={"reasoning_effort": "none"})
greedy = SamplingParams(temperature=0.0, max_tokens=400)
convs = [
    ["Say hello in one short sentence and tell me what 2+2 equals.",
     "Now multiply that result by 7 and explain in one sentence."],
    ["Give me three tips for writing clean Python code.",
     "Expand only on the second tip with a short code example."],
    ["What is the capital of Australia?",
     "And what is its approximate population? Answer briefly."],
    ["Write a haiku about autumn leaves.",
     "Now write one about winter snow in the same style."],
    ["Explain what a hash map is to a beginner in two sentences.",
     "What is its average lookup time complexity and why?"],
]
samples = []
for turns in ([] if args.bench_only else convs):
    msgs = []
    for user in turns:
        msgs.append({"role": "user", "content": user})
        out = llm.chat([msgs], greedy, **kw)[0].outputs[0]
        msgs.append({"role": "assistant", "content": out.text})
        samples.append(dict(user=user, reply=out.text, n_tokens=len(out.token_ids), ids=list(out.token_ids),
                            finish=out.finish_reason, loop=loopy(out.text)))
        print(f"\n### USER: {user}\n### MODEL ({len(out.token_ids)} tok, {out.finish_reason}): {out.text}", flush=True)

bench_sp = SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=True)
PROMPTS = {
    "story": "Write a long story about a lighthouse keeper.",
    "code": "Write a Python module implementing Dijkstra's shortest path algorithm with a binary heap, type hints, docstrings, and a small unittest suite.",
    "math": "A train leaves at 3:15 PM traveling 84 km/h; a second leaves the same station at 4:00 PM at 108 km/h on the same track. When and where does the second catch the first? Show each step.",
}
prompt = [{"role": "user", "content": PROMPTS[args.prompt]}]
llm.chat([prompt], SamplingParams(max_tokens=16), **kw)  # warm
res = {}
story = {}
for bs in [int(b) for b in args.bs_list.split(",")]:
    runs = []
    for _ in range(3):
        t0 = time.time()
        outs = llm.chat([prompt] * bs, bench_sp, **kw)
        dt = time.time() - t0
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        runs.append(ntok / dt)
        story[f"bs{bs}"] = [list(o.outputs[0].token_ids) for o in outs]
    runs.sort()
    res[f"bs{bs}_tok_s"] = runs[1]
    print(f"bs={bs} tok/s runs={['%.1f' % r for r in runs]}", flush=True)
spec_m = {}
try:
    for m in llm.get_metrics():
        if "spec_decode" in m.name:
            spec_m[m.name] = getattr(m, "value", None) if hasattr(m, "value") else str(getattr(m, "values", ""))[:200]
except Exception as e:  # noqa: BLE001
    spec_m["error"] = repr(e)
print("SPEC_METRICS", json.dumps(spec_m), flush=True)
json.dump(dict(spec_metrics=spec_m, samples=samples, bench=res, eager=args.eager, spec=args.spec, story=story), open(args.out, "w"))
print(json.dumps(res))
