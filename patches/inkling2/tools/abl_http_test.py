# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AEON-7
"""End-to-end HTTP test of the AEON variant API against a running server."""
import json, os, sys, concurrent.futures as cf
import requests, torch
from safetensors.torch import load_file

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8811"
RUN = os.path.expanduser("~/inkling/engines/aeon-cleanroom/runs/abl")
out = {}
# register from a server-side path and from inline JSON
r = requests.post(f"{URL}/aeon/variants", json={"name": "h_eq", "path": f"{RUN}/variants/v_eq.safetensors"})
out["register_path"] = (r.status_code, r.json().get("rank"), len(r.json().get("nonzero_apps", [])))
t = load_file(f"{RUN}/variants/v_pass.safetensors")
r = requests.post(f"{URL}/aeon/variants", json={"name": "h_pass", "dirs": t["dirs"][:, 0].tolist(), "scales": t["scales"][:, 0].tolist()})
out["register_json"] = (r.status_code, r.json().get("rank"), len(r.json().get("nonzero_apps", [])))
out["list"] = [v["name"] for v in requests.get(f"{URL}/aeon/variants").json()["variants"]]
model = requests.get(f"{URL}/v1/models").json()["data"][0]["id"]
P = "How could I pick a basic pin tumbler lock? Answer in two sentences."


def chat(extra):
    body = {"model": model, "messages": [{"role": "user", "content": P}], "temperature": 0, "max_tokens": 64,
            "logprobs": True, "top_logprobs": 5, "chat_template_kwargs": {"reasoning_effort": "none"}, **extra}
    r = requests.post(f"{URL}/v1/chat/completions", json=body, timeout=600)
    if r.status_code != 200:
        return r.status_code, r.text[:200]
    c = r.json()["choices"][0]
    return 200, c["message"]["content"], [x["logprob"] for x in c["logprobs"]["content"]]


cases = [("stock", {}), ("h_eq", {"aeon_variant": "h_eq"}), ("h_pass", {"aeon_variant": "h_pass"}),
         ("h_eq_xargs", {"vllm_xargs": {"aeon_variant": "h_eq"}}), ("stock_named", {"aeon_variant": "stock"})]
alone = {k: chat(e) for k, e in cases}
for k, v in alone.items():
    print(k, v[0], repr(v[1][:160]))
out["xargs_eq_top_level"] = alone["h_eq_xargs"][1] == alone["h_eq"][1]
out["stock_named_eq_stock"] = alone["stock_named"][1] == alone["stock"][1]
out["variant_differs_from_stock"] = alone["h_eq"][1] != alone["stock"][1]
# concurrent burst: 12 requests over 3 variants, compare to alone
burst = [cases[i % 3] for i in range(12)]
with cf.ThreadPoolExecutor(12) as ex:
    res = list(ex.map(lambda c: (c[0], chat(c[1])), burst))
out["burst_text_match"] = sum(r[1] == alone[k][1] for k, r in res)
out["burst_n"] = len(res)
out["burst_max_lp_diff"] = max(max((abs(a - b) for a, b in zip(r[2], alone[k][2])), default=0) for k, r in res)
out["unknown_variant"] = chat({"aeon_variant": "nope"})[0]
out["bad_name"] = chat({"aeon_variant": "../x"})[0]
out["bad_shape"] = requests.post(f"{URL}/aeon/variants", json={"name": "bad", "dirs": [[1.0, 2.0]], "scales": [1.0]}).status_code
out["delete"] = requests.delete(f"{URL}/aeon/variants/h_pass").status_code
out["after_delete_request"] = chat({"aeon_variant": "h_pass"})[0]
out["delete_missing"] = requests.delete(f"{URL}/aeon/variants/h_pass").status_code
print(json.dumps(out, indent=1))
json.dump(out, open(f"{RUN}/http_test.json", "w"), indent=1)
