# AEON-7: per-step GPU busy vs wall from a rocprofv3 kernel trace csv
import csv, sys, glob, collections
f = sorted(glob.glob(sys.argv[1] + "/**/*kernel_trace.csv", recursive=True))[-1]
rows = [(int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]) for r in csv.DictReader(open(f))]
rows.sort()
lm = [i for i, r in enumerate(rows) if "MT16x32x256" in r[2] or "BSS_BH" in r[2]]
print("lm_head-like kernels:", len(lm))
steps = lm[-60:]
walls, busy = [], []
per = collections.Counter()
for a, b in zip(steps, steps[1:]):
    seg = rows[a + 1: b + 1]
    walls.append(rows[b][1] - rows[a][1])
    # union of kernel intervals
    tot = 0; cur_s = cur_e = None
    for s, e, n in seg:
        per[n[:90]] += e - s
        if cur_e is None or s > cur_e:
            if cur_e is not None: tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    tot += cur_e - cur_s
    busy.append(tot)
n = len(walls)
print(f"steps {n}: wall {sum(walls)/n/1e6:.2f} ms/step, gpu busy {sum(busy)/n/1e6:.2f} ms/step, kernels/step {sum(b-a for a,b in zip(steps,steps[1:]))/n:.0f}")
for k, v in per.most_common(30):
    print(f"{v/n/1e3:9.1f}us  {k}")
gaps = []
big = []
for a, b in zip(steps, steps[1:]):
    seg = rows[a: b + 1]
    g = [seg[i + 1][0] - seg[i][1] for i in range(len(seg) - 1)]
    gaps += g
    big.append(max(g))
import statistics
gs = sorted(gaps)
print(f"inter-kernel gaps: n={len(gs)} median {gs[len(gs)//2]/1e3:.1f}us p90 {gs[int(len(gs)*.9)]/1e3:.1f}us sum/step {sum(gs)/n/1e6:.2f}ms; max gap/step mean {sum(big)/n/1e6:.2f}ms")
small = sum(x for x in gs if x < 50000) / n / 1e6
print(f"sum of gaps <50us per step {small:.2f} ms; >=50us {sum(x for x in gs if x >= 50000)/n/1e6:.2f} ms")
byk = collections.defaultdict(list)
for a, b in zip(steps, steps[1:]):
    seg = rows[a: b + 1]
    for i in range(len(seg) - 1):
        byk[(seg[i][2][:50], seg[i + 1][2][:50])].append(seg[i + 1][0] - seg[i][1])
print("gap before kernel (prev -> next): mean us, count/step")
for kk, v in sorted(byk.items(), key=lambda kv: -sum(kv[1]))[:25]:
    print(f"{sum(v)/len(v)/1e3:7.2f}us x{len(v)/n:6.1f}  {kk[0]} -> {kk[1]}")
