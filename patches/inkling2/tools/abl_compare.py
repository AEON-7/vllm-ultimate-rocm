# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 AEON-7
"""Compare the abl_validate.py runs; writes runs/abl/compare.json."""
import json, math, os
import torch

RUN = os.path.expanduser("~/inkling/engines/aeon-cleanroom/runs/abl")
L = {m: torch.load(f"{RUN}/{m}.pt", weights_only=False) for m in
     ("multi", "plain", "ablate_eq", "ablate_pass", "overlay_eq", "bench") if os.path.exists(f"{RUN}/{m}.pt")}


def kl(p, q):
    """KL(p||q) over p's top-k, q missing -> q's min (top-k lower bound)."""
    qmin = min(q.values())
    return sum(math.exp(lp) * (lp - q.get(t, qmin)) for t, lp in p.items())


def maxdiff(p, q):
    common = set(p) & set(q)
    return max((abs(p[t] - q[t]) for t in common), default=0.0)


def cmp_gen(A, B):
    """A, B: lists of {ids, lps}. Exact match rate, first divergence, KL up to divergence."""
    exact, kls, mds, div = 0, [], [], []
    for a, b in zip(A, B):
        n = min(len(a["ids"]), len(b["ids"]))
        d = next((i for i in range(n) if a["ids"][i] != b["ids"][i]), None)
        if d is None and len(a["ids"]) == len(b["ids"]):
            exact += 1
        else:
            div.append(d if d is not None else n)
        upto = (d + 1) if d is not None else n
        for i in range(upto):
            kls.append(kl(a["lps"][i], b["lps"][i])); mds.append(maxdiff(a["lps"][i], b["lps"][i]))
    return dict(n=len(A), exact=exact, first_div=sorted(div), kl_mean=sum(kls) / max(1, len(kls)),
                kl_max=max(kls, default=0.0), lp_absdiff_max=max(mds, default=0.0), positions=len(kls))


def cmp_tf(A, B):
    kls, mds, top1 = [], [], 0
    for a, b in zip(A, B):
        for p, q in zip(a, b):
            kls.append(kl(p, q)); mds.append(maxdiff(p, q))
            top1 += max(p, key=p.get) == max(q, key=q.get)
    return dict(positions=len(kls), kl_mean=sum(kls) / len(kls), kl_max=max(kls),
                lp_absdiff_max=max(mds), top1_agree=top1 / len(kls))


M = L["multi"]
out = {}
if "plain" in L:
    out["T1_stock_vs_plainM1"] = dict(gen=cmp_gen(L["plain"]["gen"], M["gen"]["stock"]),
                                      tf=cmp_tf(L["plain"]["tf"], M["tf"]["stock"]))
T2 = {}
for ref, var in (("ablate_eq", "v_eq"), ("overlay_eq", "v_eq"), ("ablate_pass", "v_pass")):
    if ref in L:
        T2[f"{var}_request_vs_{ref}"] = dict(gen=cmp_gen(L[ref]["gen"], M["gen"][var]),
                                             tf=cmp_tf(L[ref]["tf"], M["tf"][var]))
if "ablate_eq" in L and "overlay_eq" in L:
    T2["noise_ref_ablate_eq_vs_overlay_eq"] = dict(gen=cmp_gen(L["ablate_eq"]["gen"], L["overlay_eq"]["gen"]),
                                                   tf=cmp_tf(L["ablate_eq"]["tf"], L["overlay_eq"]["tf"]))
for var in ("v_eq", "v_pass"):
    T2[f"effect_{var}_vs_stock"] = dict(gen=cmp_gen(M["gen"]["stock"], M["gen"][var]),
                                        tf=cmp_tf(M["tf"]["stock"], M["tf"][var]))
out["T2"] = T2
T3 = {}
n_prompts = len(M["gen"]["stock"])
n_tf = len(M["tf"]["stock"])
tot_exact = tot = 0
for var in M["gen"]:
    mixed = [M["mixed_gen"][f"{var}|{i}"] for i in range(n_prompts)]
    mtf = [M["mixed_tf"][f"{var}|{i}"] for i in range(n_tf)]
    T3[var] = dict(gen=cmp_gen(M["gen"][var], mixed), tf=cmp_tf(M["tf"][var], mtf))
    tot_exact += T3[var]["gen"]["exact"]; tot += n_prompts
out["T3_mixed_vs_alone"] = T3
if "control_mixed_stock_gen" in M:
    out["T3_control_stock_alone_vs_stock_in_mixed_structure"] = dict(
        gen=cmp_gen(M["gen"]["stock"], M["control_mixed_stock_gen"]),
        tf=cmp_tf(M["tf"]["stock"], M["control_mixed_stock_tf"]))
if os.path.exists(f"{RUN}/det.pt"):
    Dt = torch.load(f"{RUN}/det.pt", weights_only=False)
    out["noise_same_process_repeat"] = dict(gen=cmp_gen(Dt["gen"], Dt["gen2"]), tf=cmp_tf(Dt["tf"], Dt["tf2"]))
    out["noise_cross_process_plain_vs_plain"] = dict(gen=cmp_gen(L["plain"]["gen"], Dt["gen"]),
                                                     tf=cmp_tf(L["plain"]["tf"], Dt["tf"]))
out["T3_summary"] = dict(variants=len(M["gen"]), requests=tot, exact=tot_exact,
                         kl_mean_max=max(v["tf"]["kl_mean"] for v in T3.values()),
                         kl_max=max(v["tf"]["kl_max"] for v in T3.values()), mixed_s=M.get("mixed_s"))
out["bench"] = {f"{m}:{k}": v["median"] for m, R in L.items() for k, v in R.items() if k.startswith("bench")}
json.dump(out, open(f"{RUN}/compare.json", "w"), indent=1)


def brief(d):
    g, t = d["gen"], d["tf"]
    return (f"gen exact {g['exact']}/{g['n']} firstdiv {g['first_div'][:6]} klmean {g['kl_mean']:.2e} | "
            f"tf klmean {t['kl_mean']:.2e} klmax {t['kl_max']:.2e} top1 {t['top1_agree']:.4f} lpdiff {t['lp_absdiff_max']:.3g}")


for k in ("T3_control_stock_alone_vs_stock_in_mixed_structure", "noise_same_process_repeat",
          "noise_cross_process_plain_vs_plain"):
    if k in out:
        print("NOISE", k, brief(out[k]))
if "T1_stock_vs_plainM1" in out:
    print("T1", brief(out["T1_stock_vs_plainM1"]))
for k, v in T2.items():
    print("T2", k, brief(v))
for k, v in T3.items():
    print("T3", k, brief(v))
print("T3 summary", out["T3_summary"])
print("bench", json.dumps(out["bench"], indent=1))
