# AEON-7: tl.dot path (3+ rows per expert) vs FMA multi-row, random vs identical tokens
src = open(__file__.replace("test_moe_dot.py", "test_moe_rows.py")).read()
src = src.replace('__file__.replace("test_moe_rows.py", "test_moe_decode.py")', '__file__.replace("test_moe_dot.py", "test_moe_decode.py")')
exec(src.split("import itertools\nbase")[0])
for dm in (3, 2):
  mdec.DOT_MIN = dm; print('DOT_MIN', dm)
  for T in (4, 8, 16):
      for mode in ("random", "same"):
          if mode == "random":
              xs = [torch.randn(T, 2304, device=dev).to(torch.bfloat16) for _ in Ls]
          else:
              xs = [torch.randn(1, 2304, device=dev).to(torch.bfloat16).expand(T, 2304).contiguous() for _ in Ls]
          mdec.AUTO_ROWS = False
          ref = mdec.moe_decode(xs[0], *Ls[0][:2], 8, 8.0, *Ls[0][2:], 0, 4.0, 25.0)[0]
          t_row = rbench(xs)
          mdec.AUTO_ROWS = True
          mdec.AUTO_DOT = False
          t_fma = rbench(xs)
          mdec.AUTO_DOT = True
          res = []
          for bk13, bk2, bn, nw in [(128, 64, 16, 2), (64, 64, 16, 2), (128, 128, 16, 2), (256, 64, 16, 2), (128, 64, 32, 2), (64, 32, 16, 2)]:
              mdec._DOT_BK[(0, 13)], mdec._DOT_BK[(0, 2)] = bk13, bk2
              c13, c2 = mdec._AUTO_CFG[(0, 13)], mdec._AUTO_CFG[(0, 2)]
              mdec._AUTO_CFG[(0, 13)] = (bn,) + c13[1:4] + (nw,); mdec._AUTO_CFG[(0, 2)] = (bn,) + c2[1:4] + (nw,)
              try:
                  o = mdec.moe_decode(xs[0], *Ls[0][:2], 8, 8.0, *Ls[0][2:], 0, 4.0, 25.0)[0]
                  rel = ((o.float() - ref.float()).norm() / ref.float().norm()).item()
                  res.append((round(rbench(xs)), (bk13, bk2, bn, nw), f"{rel:.1e}"))
              except Exception as e:
                  print("fail", bk13, bk2, bn, nw, str(e)[:200])
              mdec._AUTO_CFG[(0, 13)], mdec._AUTO_CFG[(0, 2)] = c13, c2
          res.sort()
          print(f"T={T} {mode}: per-row {t_row:.0f}us fma-auto {t_fma:.0f}us dot {res[:3]}", flush=True)
