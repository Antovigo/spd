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

# 5. forced-original-rms windows (dec model)
j = json.loads((A / "robust_forced.json").read_text())
wins = ["b0-7", "b8-15", "b16-23", "b24-30", "b0-30"]
fig, ax = plt.subplots(figsize=(6, 3.6))
x = np.arange(len(wins))
ax.bar(
    x - 0.2, [j[f"dec_forced_orig_{w}"]["kl"] for w in wins], 0.4, color=C_D, label="all positions"
)
ax.bar(
    x + 0.2, [j[f"dec_forced_orig_{w}_eq"]["kl"] for w in wins], 0.4, color=C_O, label="only at '='"
)
ax.axhline(j["dec_free"]["kl"], color="k", lw=0.8, ls="--", label="own norms (0.051)")
ax.set(
    xticks=x,
    xticklabels=wins,
    ylabel="KL to original",
    yscale="log",
    title="decomposed model, rms forced to the original's",
)
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig5_forced_windows.png", dpi=150)

# 6. whole-block ablation, free vs frozen norms
g = np.load(A / "groups.npz")
nm = list(g["names"])
blk = [i for i in range(31) if f"L{i}" in nm]
idx = [nm.index(f"L{i}") for i in blk]
kf = g["kl_free"].mean(1)[idx]
kz = np.nan_to_num(g["kl_frozen"], nan=np.inf, posinf=np.inf).mean(1)[idx]
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.semilogy(blk, kf, "o-", ms=3, color=C_O, label="norms free")
ax.semilogy(blk, kz, "o-", ms=3, color=C_D, label="norms frozen at the unablated rms")
ax.set(
    xlabel="block whose alive components are all ablated", ylabel="KL to unablated decomposed model"
)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig6_block_ablation_free_frozen.png", dpi=150)

# 7. single-block hybrid: free vs forced original rms (median KL)
h = np.load(A / "hybrid.npz")
fr, fo = [], []
for li in range(31):
    o = h[f"block{li}_forced_orig__kl"]
    fr.append(np.median(h[f"block{li}_free__kl"]))
    fo.append(np.median(o[np.isfinite(o)]))
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(range(31), fr, "o-", ms=3, color=C_O, label="norms free")
ax.plot(range(31), fo, "o-", ms=3, color=C_D, label="all norms forced to the original's rms")
ax.set(
    xlabel="the one decomposed block",
    ylabel="median KL to original",
    title="original model with one block decomposed",
)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig7_hybrid_forced.png", dpi=150)

# 8. where the stream difference comes from: removal vs drift (accounting.npz), medians at "="
ac = np.load(A / "accounting.npz")
oo, To, Po, TT, PP, TP = (ac[k][1:] for k in ("oo", "To", "Po", "TT", "PP", "TP"))
blocks = np.arange(1, 32)
med = np.median
size_t, size_p = med(np.sqrt(TT / oo), 1), med(np.sqrt(PP / oo), 1)
size_d = med(np.sqrt(np.maximum(TT + PP + 2 * TP, 0) / oo), 1)
r_full = med(np.sqrt(1 + 2 * (To + Po) / oo + (TT + PP + 2 * TP) / oo), 1)
r_rem = med(np.sqrt(1 + 2 * To / oo + TT / oo), 1)
r_drift = med(np.sqrt(1 + 2 * Po / oo + PP / oo), 1)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(blocks, size_t, color=C_D, label="removed content (accumulated)")
ax[0].plot(blocks, size_p, color=C_O, label="downstream drift (accumulated)")
ax[0].plot(blocks, size_d, color="k", label="their sum = total difference")
ax[0].set(
    xlabel="block",
    ylabel="size relative to |original stream|",
    title="what makes the streams differ",
)
ax[0].legend(frameon=False)
ax[1].plot(blocks, r_rem, color=C_D, label="removal only")
ax[1].plot(blocks, r_drift, color=C_O, label="drift only")
ax[1].plot(blocks, r_full, color="k", label="both (what the model has)")
ax[1].axhline(1, color="k", lw=0.5)
ax[1].set(xlabel="block", ylabel="RMS ratio vs original", title="effect of each part on the RMS")
ax[1].legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig8_removal_vs_drift.png", dpi=150)
