"""Figures for report_rmsnorm.md from the arrays in <RUN>/analysis/rmsnorm (numpy + matplotlib only).
python make_figures.py <analysis/rmsnorm dir> <output dir>"""

import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

A, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(exist_ok=True)
C_O, C_D, C_X = "#4477AA", "#EE6677", "#228833"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# 1. rms ratio, alpha, alpha/ratio at "=" per inner norm
b = np.load(A / "base.npz")
t = np.arange(1, 62)
ratio = (b["rms_d"][t, :, 4] / b["rms_o"][t, :, 4]).mean(1)
alpha = b["alpha"][t, :, 4].mean(1)
perp = b["perp"][t, :, 4].mean(1)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(t / 2, ratio, color=C_D, label="rms ratio (dec / orig)")
ax[0].plot(t / 2, alpha, color=C_O, label="alpha (stream projection on original)")
ax[0].plot(t / 2, alpha / ratio, color=C_X, ls="--", label="alpha / rms ratio (what a reader sees)")
ax[0].axhline(1, color="k", lw=0.5)
ax[0].set(
    xlabel="block (attn norm = integer, mlp norm = +0.5)", title="stream scale at '=' , dec vs orig"
)
ax[0].legend(fontsize=8, frameon=False)
ax[1].plot(t / 2, perp, color=C_D, label="|unread error| / |orig stream|")
ax[1].plot(
    t / 2,
    np.sqrt(np.maximum(ratio**2 - alpha**2, 0)),
    color="gray",
    ls=":",
    label="sqrt(ratio^2 - alpha^2)",
)
ax[1].set(xlabel="block", title="error energy carried in the stream")
ax[1].legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig1_scale.png", dpi=150)

# 2. single-norm +-10 % sensitivity
fig, ax = plt.subplots(figsize=(8, 3.6))
for m, c, lab in (("orig", C_O, "original Llama"), ("dec", C_D, "decomposed (-05, masked)")):
    z = np.load(A / f"robust_single_{m}.npz")
    s = np.array(
        [(z[f"scale0.9_t{i}__kl"].mean() + z[f"scale1.1_t{i}__kl"].mean()) / 2 for i in range(62)]
    )
    ax.semilogy(np.arange(62) / 2, s + 1e-6, "o-", ms=3, color=c, label=lab)
ax.set(
    xlabel="block",
    ylabel="KL vs own clean run",
    title="one norm's rms scaled by 0.9 / 1.1 (mean of both)",
)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig2_single_norm.png", dpi=150)

# 3. noise robustness
fig, ax = plt.subplots(figsize=(5.5, 3.6))
sig = [0.03, 0.1, 0.2]
for m, c, lab in (("orig", C_O, "original"), ("dec", C_D, "decomposed")):
    j = json.loads((A / f"robust_noise_{m}.json").read_text())
    ax.loglog(
        sig, [j[f"sigma{s}_all"]["kl"] for s in sig], "o-", color=c, label=f"{lab}, every token"
    )
    ax.loglog(
        sig,
        [j[f"sigma{s}_eq"]["kl"] for s in sig],
        "s--",
        color=c,
        mfc="none",
        label=f"{lab}, only at '='",
    )
ax.set(
    xlabel="sigma of log-normal rms noise, all inner norms (blocks 0-30)",
    ylabel="KL vs own clean run",
)
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig3_noise.png", dpi=150)

# 4. per-component direct vs norm-only effect, blocks 0-30
S, Ab = {}, {}
for f in sorted(glob.glob(str(A / "split/shard*of4.npz"))):
    z = np.load(f)
    for k in ("names", "kl_norm_only", "kl_direct_only"):
        S.setdefault(k, []).append(z[k])
for f in sorted(glob.glob(str(A / "ablate/shard*of4.npz"))):
    z = np.load(f)
    for k in ("kl_free", "n_act"):
        Ab.setdefault(k, []).append(z[k])
S = {k: np.concatenate(v) for k, v in S.items()}
Ab = {k: np.concatenate(v) for k, v in Ab.items()}
L = np.array([int(n.split(".")[0][1:]) for n in S["names"]])
sel = (L <= 30) & (Ab["n_act"].sum(1) > 0)
do, no = S["kl_direct_only"].mean(1)[sel], S["kl_norm_only"].mean(1)[sel]
fig, ax = plt.subplots(figsize=(4.8, 4.4))
ax.loglog(np.maximum(do, 1e-7), np.maximum(no, 1e-7), ".", ms=2.5, color=C_O, alpha=0.5)
ax.plot([1e-7, 1], [1e-7, 1], "k", lw=0.5)
ax.set(
    xlabel="direct-only KL (write dropped, energy kept)",
    ylabel="norm-only KL (write kept, energy removed)",
    title="11k components, blocks 0-30",
)
fig.tight_layout()
fig.savefig(OUT / "fig4_direct_vs_norm.png", dpi=150)
print("done")
