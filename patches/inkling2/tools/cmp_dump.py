# AEON-7 debug: compare ref vs vllm activations
import torch
R = torch.load("/home/center/inkling/engines/aeon-cleanroom/runs/ref_dump.pt")
V = torch.load("/home/center/inkling/engines/aeon-cleanroom/runs/vllm_dump.pt")
assert R["ids"] == V["ids"]
order = sorted({k for k in R["cap"]} | {k for k in V["cap"]}, key=lambda k: (int(k[1:].split(".")[0]) if k[0]=="L" else 99, k))
for k in order:
    r = R["cap"].get(k, []); v = V["cap"].get(k, [])
    for p in range(max(len(r), len(v))):
        if p >= len(r) or p >= len(v):
            print(f"{k:16s} p{p} missing r={len(r)} v={len(v)}"); continue
        a, b = r[p], v[p]
        if a.shape != b.shape: print(k, p, "shape", a.shape, b.shape); continue
        rel = ((a-b).norm()/a.norm().clamp_min(1e-12)).item()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        print(f"{k:16s} p{p} rel={rel:.4f} cos_min={cos.min().item():.4f} cos_mean={cos.mean().item():.4f} |r|={a.norm().item():.1f} |v|={b.norm().item():.1f}")
