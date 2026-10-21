"""The representation repository of run p-ba8b0c08: representations found in the model, and how well they
explain each reader.

Stored in `<run>/analysis/representation_repository/`:

    meta.json               schema version, model / mask / data provenance, the response threshold
    representations.json    {"representations": [Representation, ...]}
    vectors/<rep id>.npz    basis (4096, d) float64 orthonormal columns; plus per-representation extras
    readers.json            {"readers": [Reader, ...]}: every alive gate / up component of the ceiling
                            filter that the learned mask "kl_int, sampled mask, λ=5e-05" keeps

Representation (one subspace of one block's MLP input at one token position):
    id            "L<block>.p<position>.<name>", e.g. "L0.p1.a_mod10.k1"
    block, position, space   where the vectors live (space: a text description of the activation)
    dim           subspace dimension d
    encodes       the variable it carries, e.g. "a mod 10"
    kind          "fourier_plane" | "fourier_line" (more kinds as they are found)
    details       kind-specific fields (frequency, residue coordinates, ...)
    vectors       path of the npz, relative to the repository root
    provenance    script, commit, date

Reader (one component):
    id            "L<block>.<gate|up>.<component id>", e.g. "L0.up.54"
    block, kind, component
    positions     {"<t>": Position or null}, t in 1..4 (1: a, 2: op, 3: b, 4: =); null = not measured yet
    Position:
      r2            share of the reader's inner-activation variance (over the prompts, at that position)
                    explained by projecting its input onto the span of `responds_to`; 0 if that is empty
      responds_to   [representation id, ...]: representations whose own r2 >= meta.response_threshold
      r2_by_rep     {representation id: r2 of that representation alone}
      method        how r2 was measured
      updated       ISO date
      writers       optional, the block's down components this reader feeds (seed_l0_mod10_writers.py):
        entries     [{writer "L<block>.down.<id>", site, component, phi, reader_fraction, routing?}]
                    phi: share of the writer's variance credited to the reader (variance split,
                    plan_variance.md); reader_fraction: |a| / sum over writers |a|, a = phi times the
                    writer's residual-stream variance; routing (up readers only): phi_sub,
                    phi_mean_gate and gates [{reader, phi_J}] (plan eq. 5, 8,000-prompt subsample)
        covered_fraction, source, method, provenance, updated
"""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08/analysis/representation_repository")
SCHEMA = 1
POSITIONS = ("1", "2", "3", "4")


def _write_json(path: Path, doc: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    os.replace(tmp, path)


class Repo:
    def __init__(self, root: Path = ROOT) -> None:
        self.root = root
        self.meta: dict[str, Any] = json.loads((root / "meta.json").read_text())
        assert self.meta["schema"] == SCHEMA, self.meta["schema"]
        self.representations: dict[str, dict[str, Any]] = {
            r["id"]: r
            for r in json.loads((root / "representations.json").read_text())["representations"]
        }
        self.readers: dict[str, dict[str, Any]] = {
            r["id"]: r for r in json.loads((root / "readers.json").read_text())["readers"]
        }
        self.validate()

    @staticmethod
    def create(root: Path, meta: dict[str, Any], readers: list[dict[str, Any]]) -> "Repo":
        assert not (root / "meta.json").exists(), f"{root} already holds a repository"
        (root / "vectors").mkdir(parents=True, exist_ok=True)
        _write_json(root / "meta.json", {"schema": SCHEMA} | meta)
        _write_json(root / "representations.json", {"representations": []})
        _write_json(root / "readers.json", {"readers": readers})
        return Repo(root)

    def basis(self, rep_id: str) -> np.ndarray:
        return np.load(self.root / self.representations[rep_id]["vectors"])["basis"]

    def add_representation(self, rep: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        assert rep["id"] not in self.representations, rep["id"]
        basis = arrays["basis"]
        assert basis.ndim == 2 and basis.shape[1] == rep["dim"], (basis.shape, rep["dim"])
        assert np.allclose(basis.T @ basis, np.eye(rep["dim"]), atol=1e-8), (
            "basis must be orthonormal"
        )
        rep = rep | {"vectors": f"vectors/{rep['id']}.npz"}
        np.savez(self.root / rep["vectors"], **arrays)
        self.representations[rep["id"]] = rep

    def set_position(self, reader_id: str, position: int, entry: dict[str, Any]) -> None:
        self.readers[reader_id]["positions"][str(position)] = entry

    def validate(self) -> None:
        for rep in self.representations.values():
            assert (self.root / rep["vectors"]).exists(), rep["vectors"]
        thr = self.meta["response_threshold"]
        for rd in self.readers.values():
            assert set(rd["positions"]) == set(POSITIONS), rd["id"]
            for pos in rd["positions"].values():
                if pos is None:
                    continue
                for rid in [*pos["responds_to"], *pos["r2_by_rep"]]:
                    assert rid in self.representations, (rd["id"], rid)
                assert set(pos["responds_to"]) == {
                    r for r, v in pos["r2_by_rep"].items() if v >= thr
                }, rd["id"]

    def save(self) -> None:
        self.validate()
        reps = sorted(self.representations.values(), key=lambda r: r["id"])
        readers = sorted(
            self.readers.values(), key=lambda r: (r["block"], r["kind"], r["component"])
        )
        _write_json(self.root / "representations.json", {"representations": reps})
        _write_json(self.root / "readers.json", {"readers": readers})
