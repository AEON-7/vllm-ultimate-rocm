# AEON-7: MXFP8 auto-kernel config sweep (cnt==1 GEMV branch + dot branch)
src = open(__file__.replace("test_auto_mx.py", "test_moe_dot_mx.py")).read()
src = src.replace('__file__.replace("test_moe_dot_mx.py", "test_moe_rows.py")', '__file__.replace("test_auto_mx.py", "test_moe_rows.py")')
src = src.replace('\'__file__.replace("test_moe_dot_mx.py", "test_moe_decode.py")\'', '\'__file__.replace("test_auto_mx.py", "test_moe_decode.py")\'')
exec(src.split("for dm in")[0])
mdec._DOT_BK[(1, 13)], mdec._DOT_BK[(1, 2)] = 64, 64
for T in (2, 4, 8):
    for mode in ("random", "same"):
        xs = [torch.randn(T, 2304, device=dev).to(torch.bfloat16) for _ in Ls] if mode == "random" else \
             [torch.randn(1, 2304, device=dev).to(torch.bfloat16).expand(T, 2304).contiguous() for _ in Ls]
        mdec.AUTO_ROWS = False; t_row = rbench(xs); mdec.AUTO_ROWS = True
        res = []
        for c13, c2 in [((16, 128, 64, 32, 4), (16, 64, 64, 32, 4)), ((16, 256, 64, 32, 8), (8, 128, 64, 32, 8)),
                        ((16, 256, 64, 32, 4), (16, 128, 64, 32, 4)), ((16, 256, 128, 32, 8), (16, 128, 64, 32, 8)),
                        ((16, 128, 64, 32, 8), (16, 128, 64, 32, 2)), ((16, 256, 64, 32, 2), (16, 128, 64, 32, 2))]:
            mdec._AUTO_CFG[(1, 13)], mdec._AUTO_CFG[(1, 2)] = c13, c2
            try: res.append((round(rbench(xs)), c13, c2))
            except Exception as e: print("fail", c13, c2, str(e)[:150])
        res.sort()
        print(f"T={T} {mode}: per-row {t_row:.0f} auto best {res[:3]}", flush=True)
