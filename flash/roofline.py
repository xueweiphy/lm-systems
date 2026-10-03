# roofline (A2 4.2.2): place the measured attention forwards on the RTX Pro 6000 roofline
# run from the repo root:  python flash/roofline.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

d = pd.read_csv("results/flash_rtx6000.csv")
d = d[(d.dtype == "torch.bfloat16") & (d["T"] >= 4096)].dropna()
d = d[d.impl.isin(["flash64", "naive", "compiled", "sdpa"])]

# FLOPs: tiled kernels skip the masked half; bytes: DRAM floor for tiled,
# T x T round trips for the others (10 naive, ~5 after torch.compile fusion); b = 2 for bf16
tiled = d.impl.isin(["flash64", "sdpa"])
T, dd = d["T"], d["d"]
rt = np.where(d.impl == "compiled", 5, 10)
d["flops"] = np.where(tiled, 2*T**2*dd, 4*T**2*dd)
d["bytes"] = np.where(tiled, 4*T*dd*2, rt*T**2*2)
d["I"] = d.flops / d.bytes
d["perf"] = d.flops / (d.fwd * 1e-3) / 1e12

# measured on the card: 1.46 TB/s (clone of 4 GiB), 408 TFLOP/s (bf16 8192^2 matmul); ridge 280
x = np.logspace(0, 4, 100)
plt.figure()
plt.loglog(x, np.minimum(x * 1.46, 408), "k-", label="roof")
plt.axvline(280, ls=":", color="gray")
for name, g in d.groupby("impl"):
    plt.loglog(g.I, g.perf, "o", label=name)
plt.legend(loc="lower right")
plt.xlabel("arithmetic intensity I  [FLOP/byte]")
plt.ylabel("achieved throughput  [TFLOP/s]")
plt.title("Causal attention forward, bf16, RTX Pro 6000, T = 4096-65536")
plt.savefig("results/roofline_rtx6000.png", dpi=150)
