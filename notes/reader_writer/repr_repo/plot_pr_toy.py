"""3D toy examples of the angular metrics of plot_angular.py: theta_axis and the tangent participation ratio PR.

Each panel: 400 unit vectors xhat on the sphere S^2 in R^3 (drawn from a Gaussian around the north pole with the
given standard deviations along the two tangent axes e1, e2, then normalised). mhat = their unit mean direction
(arrow). theta_axis = RMS angle of xhat to mhat (degrees). PR = (sum lambda)^2 / sum lambda^2 over the eigenvalues
lambda of sum_a t(a) t(a)^T, t(a) = xhat(a) - (xhat(a) . mhat) mhat. On S^2 the tangent plane is 2-dimensional,
so PR lies between 1 (spread along one direction) and 2 (spread equally along two).

    python plot_pr_toy.py --out <png>
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CASES = (
    ("narrow round cap", 0.12, 0.12),
    ("wide round cap", 0.45, 0.45),
    ("elongated cap", 0.45, 0.15),
    ("arc (one direction)", 0.6, 0.0),
)


def metrics(U: np.ndarray) -> tuple[np.ndarray, float, float]:
    m = U.mean(0)
    m /= np.linalg.norm(m)
    T = U - np.outer(U @ m, m)
    lam = np.linalg.svd(T, compute_uv=False) ** 2
    theta = float(np.sqrt(np.mean(np.degrees(np.arccos(np.clip(U @ m, -1, 1))) ** 2)))
    return m, theta, float(lam.sum() ** 2 / (lam**2).sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(15, 4.6))
    u, v = np.mgrid[0 : 2 * np.pi : 40j, 0 : np.pi : 20j]
    for i, (name, s1, s2) in enumerate(CASES):
        X = np.stack(
            [s1 * rng.standard_normal(400), s2 * rng.standard_normal(400), np.ones(400)], 1
        )
        U = X / np.linalg.norm(X, axis=1, keepdims=True)
        m, theta, pr = metrics(U)
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        ax.plot_wireframe(
            np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color="#cccccc", lw=0.3
        )
        ax.scatter(*U.T, s=4, color="#2a5db0", depthshade=False)
        ax.quiver(0, 0, 0, *(1.25 * m), color="#d9730d", lw=2, arrow_length_ratio=0.1)
        ax.set_box_aspect((1, 1, 0.75), zoom=1.35)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-0.2, 1.3)
        ax.view_init(elev=35, azim=-60)
        ax.set_axis_off()
        ax.set_title(f"{name}\nθ_axis {theta:.0f}°, PR {pr:.2f}", fontsize=10, y=0.98)
    fig.text(0.5, 0.02, "blue: 400 unit vectors x̂(a); orange arrow: mean direction m̂. θ_axis = width of the cone, "
             "PR = number of directions it spreads over (at most 2 on a 3D sphere)", ha="center", fontsize=9)  # fmt: skip
    fig.subplots_adjust(left=0, right=1, top=0.9, bottom=0.08, wspace=0)
    fig.savefig(args.out, dpi=120)


if __name__ == "__main__":
    main()
