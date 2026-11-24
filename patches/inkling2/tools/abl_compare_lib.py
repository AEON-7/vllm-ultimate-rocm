import math


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



def brief(d):
    g, t = d["gen"], d.get("tf")
    s = f"gen exact {g['exact']}/{g['n']} firstdiv {g['first_div'][:6]} klmean {g['kl_mean']:.2e} lpdiff {g['lp_absdiff_max']:.3g}"
    if t is None:
        return s
    return (f"gen exact {g['exact']}/{g['n']} firstdiv {g['first_div'][:6]} klmean {g['kl_mean']:.2e} | "
            f"tf klmean {t['kl_mean']:.2e} klmax {t['kl_max']:.2e} top1 {t['top1_agree']:.4f} lpdiff {t['lp_absdiff_max']:.3g}")
