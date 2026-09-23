"""One read point's full analysis (plan sections 3-4).

    python -m param_decomp.arith_repr.scripts.analyse --resid <raw residual dir> --v <V_alive.npz>
        --out <dir> --keys attn_in.18 mlp_in.18 [--n_null 3] [--folds 5] [--seed 0]

Per key writes `<out>/<key>.json` (every scalar and small array) and `<out>/<key>.npz`
(subspace bases, cluster axes, centroids)."""

import argparse
import json
import time
from pathlib import Path
from typing import Any, cast

import numpy as np

from param_decomp.arith_repr.fit import HypothesisFit, fit_hypotheses, orthonormal_union
from param_decomp.arith_repr.hypotheses import (
    QUANTITIES_BY_POSITION,
    Labels,
    ValueFolds,
    pooled_quantities,
)
from param_decomp.arith_repr.resid import load_read_input
from param_decomp.arith_repr.separability import (
    cluster_geometry,
    principal_angles,
    separability,
)

KINDS = {"attn_in": ("q_proj", "k_proj", "v_proj"), "mlp_in": ("gate_proj", "up_proj")}
N_AXES = 6


def site_name(layer: int, kind: str) -> str:
    return f"layers.{layer}.{'self_attn' if kind.endswith(('q_proj', 'k_proj', 'v_proj')) else 'mlp'}.{kind}"


def read_basis(v_alive: Any, key: str) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """`B` (d x k), the reads in read coordinates `v` (k x n), kind per read, ids per read."""
    stream, layer = key.split(".")
    blocks, kinds, ids = [], [], []
    for kind in KINDS[stream]:
        site = site_name(int(layer), kind)
        if site in v_alive.files:
            V = v_alive[site]
            blocks.append(V)
            kinds += [kind] * V.shape[1]
            ids.append(v_alive[site + ".ids"])
    V_all = np.concatenate(blocks, axis=1).astype(np.float64)
    U, s, _ = np.linalg.svd(V_all, full_matrices=False)
    B = U[:, s > 1e-6 * s[0]]
    return B, B.T @ V_all, kinds, np.concatenate(ids)


def load_labels(meta: dict[str, Any]) -> Labels:
    rows = meta["labels"]
    op = np.array([0 if r[0] == "add" else 1 for r in rows], np.int64)
    a = np.array([r[1] for r in rows], np.int64)
    b = np.array([r[2] for r in rows], np.int64)
    return Labels(op, a, b)


def load_position(resid_dir: Path, key: str, position: int) -> np.ndarray:
    """The read point's post-norm input at `position`, recovered from the raw stream."""
    return load_read_input(resid_dir, key, position)


def fit_record(f: HypothesisFit) -> dict[str, Any]:
    return {
        "name": f.hypothesis.name,
        "quantity": f.hypothesis.quantity,
        "kind": f.hypothesis.kind,
        "period": f.hypothesis.period,
        "dim_hypothesis": f.hypothesis.dim,
        "marginal": f.marginal,
        "unique": f.unique,
        "generalising": f.generalising_energy,
        "dim_S": int(f.kept.sum()),
        "singular_values": f.singular_values.tolist(),
        "heldout_r2": f.heldout_r2.tolist(),
        "kept": f.kept.tolist(),
        "linear_r2": f.linear_r2,
    }


def analyse_key(
    key: str,
    resid_dir: Path,
    v_alive: Any,
    labels_all: Labels,
    folds: ValueFolds,
    out_dir: Path,
    n_null: int,
    seed: int,
) -> None:
    t0 = time.time()
    B, v, kinds, ids = read_basis(v_alive, key)
    k = B.shape[1]
    kinds_arr = np.array(kinds)
    record: dict[str, Any] = {"key": key, "k": k, "n_reads": len(kinds), "positions": {}}
    arrays: dict[str, np.ndarray] = {
        "B": B.astype(np.float32),
        "v": v.astype(np.float32),
        "ids": ids,
    }
    arrays["kinds"] = kinds_arr
    keep_prompt = (labels_all.op == 0) | (labels_all.a >= labels_all.b)
    op_sets = {
        "both": np.ones(labels_all.n, bool),
        "add": labels_all.op == 0,
        "sub": labels_all.op == 1,
    }
    for position, quantities in QUANTITIES_BY_POSITION.items():
        X = load_position(resid_dir, key, position).astype(np.float64)
        Y_all = X @ B
        outside = 1.0 - float(np.sum(Y_all**2) / max(np.sum(X**2), 1e-300))
        pos_record: dict[str, Any] = {"outside_basis_energy": outside, "ops": {}}
        per_op_fits: dict[str, list[HypothesisFit]] = {}
        per_op_labels: dict[str, Labels] = {}
        for op_name, op_mask in op_sets.items():
            if position < 2 and op_name != "both":
                continue  # the operation token is not yet visible
            mask = op_mask & keep_prompt
            labels = labels_all.subset(mask)
            Y = Y_all[mask]
            fitted = pooled_quantities(quantities) if op_name == "both" else quantities
            fits, summary = fit_hypotheses(Y, labels, fitted, folds, n_null, seed)
            per_op_fits[op_name], per_op_labels[op_name] = fits, labels
            Yc = Y - Y.mean(axis=0)
            union = orthonormal_union([f.S for f in fits])
            explained = (
                float(np.sum((Yc @ union) ** 2) / summary["total_energy"]) if union.size else 0.0
            )
            _, sy, _ = np.linalg.svd(Yc, full_matrices=False)
            op_record: dict[str, Any] = {
                **summary,
                "explained_by_kept": explained,
                "y_spectrum": (sy**2 / summary["total_energy"]).tolist(),
                "fits": [fit_record(f) for f in fits],
                "separability": {},
                "clusters": {},
            }
            for kind_name in (*KINDS[key.split(".")[0]], "all"):
                cmask = None if kind_name == "all" else kinds_arr == kind_name
                if cmask is not None and not cmask.any():
                    continue
                sep = separability(fits, v, k, cmask)
                op_record["separability"][kind_name] = {
                    "names": sep.names,
                    "separable": sep.separable.tolist(),
                    "clusters": [[sep.names[i] for i in c] for c in sep.clusters],
                    "n_reads": int(sep.reads.shape[1]),
                    "unread": [sep.names[i] for i in sep.unread],
                    "reads_count": sep.reads.sum(axis=1).tolist(),
                    "max_overlap": sep.overlaps.max(axis=1).tolist() if sep.overlaps.size else [],
                }
                arrays[f"p{position}/{op_name}/overlap/{kind_name}"] = sep.overlaps.astype(
                    np.float32
                )
                if kind_name == "all":
                    geoms = []
                    for ci, members in enumerate(sep.clusters):
                        g = cluster_geometry(fits, members, Y, labels, N_AXES)
                        geoms.append(
                            {
                                "members": g.members,
                                "dim": g.dim,
                                "spectrum": g.spectrum.tolist(),
                                "counts": {n: c.tolist() for n, c in g.counts.items()},
                                "norms": {n: c.tolist() for n, c in g.norms.items()},
                                "centroids": {n: c.tolist() for n, c in g.centroids.items()},
                            }
                        )
                        arrays[f"p{position}/{op_name}/cluster{ci}/axes"] = g.axes.astype(
                            np.float32
                        )
                    op_record["clusters"] = geoms
            for f in fits:
                arrays[f"p{position}/{op_name}/S/{f.hypothesis.name}"] = f.S.astype(np.float32)
            pos_record["ops"][op_name] = op_record
        if "add" in per_op_fits and "sub" in per_op_fits:
            angles: dict[str, list[float]] = {}
            sub_by = {f.hypothesis.name: f for f in per_op_fits["sub"]}
            for f in per_op_fits["add"]:
                g = sub_by.get(f.hypothesis.name)
                if g is not None:
                    angles[f.hypothesis.name] = principal_angles(f.S, g.S).tolist()
            pos_record["add_vs_sub_cosines"] = angles
        record["positions"][str(position)] = pos_record
        print(f"{key} p{position}: {time.time() - t0:.0f}s", flush=True)
    (out_dir / f"{key}.json").write_text(json.dumps(record))
    np.savez_compressed(out_dir / f"{key}.npz", **cast(dict[str, Any], arrays))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resid", type=Path, required=True)
    parser.add_argument("--v", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--keys", nargs="+", required=True)
    parser.add_argument("--n_null", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.resid / "pool.json").read_text())
    labels = load_labels(meta)
    folds = ValueFolds.make(args.folds, np.arange(1, 101), args.seed)
    v_alive = np.load(args.v)
    for key in args.keys:
        analyse_key(key, args.resid, v_alive, labels, folds, args.out, args.n_null, args.seed)


if __name__ == "__main__":
    main()
