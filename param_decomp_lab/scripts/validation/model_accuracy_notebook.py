"""Marimo notebook for the `measure_model_accuracy.py` output.

Lives in a run's `model_accuracy/` folder next to the `accuracy[_<ablation>].json` files it
plots. Pick an `original` and an `ablated` JSON with the dropdowns; the cells redraw reactively:

- offset curves: mean P over the answer window (offset 0 = correct), shaded ±1 std across
  prompts, in a 4×2 grid — rows are operand-parity classes, columns are original vs ablated;
- heatmaps: P(correct answer) over the `(a, b)` grid for both models.

Run with `marimo edit model_accuracy_notebook.py` (or `marimo run ...`).
"""

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import numpy as np
    from matplotlib import pyplot as plt

    return Path, json, mo, np, plt


@app.cell
def _(mo):
    mo.md(
        """
        # Model accuracy

        Per-prompt arithmetic accuracy from `measure_model_accuracy.py`. Choose the **original**
        and **ablated** result files (suffixed with the ablated subcomponents).
        """
    )
    return


@app.cell
def _(Path, mo):
    here = Path(__file__).parent
    files = sorted(p.name for p in here.glob("accuracy*.json"))
    assert files, f"no accuracy*.json found in {here}"
    abl_files = [f for f in files if f != "accuracy.json"]
    original = mo.ui.dropdown(
        files, value="accuracy.json" if "accuracy.json" in files else files[0], label="original"
    )
    ablated = mo.ui.dropdown(
        files, value=abl_files[0] if abl_files else files[0], label="ablated"
    )
    mo.hstack([original, ablated], justify="start", gap=2)
    return ablated, here, original


@app.cell
def _(ablated, here, json, original):
    orig = json.loads((here / original.value).read_text())
    abl = json.loads((here / ablated.value).read_text())
    return abl, orig


@app.cell
def _(mo, orig):
    _a = orig["ablated"]
    mo.md(
        f"**op** `{orig['symbol']}` · **range** ±{orig['range']} · "
        f"**{orig['n_prompts']}** prompts · answer at the `=` position."
    )
    return


@app.cell
def _(abl, np, orig, plt):
    _parities = [
        ("a,b both even", lambda a, b: a % 2 == 0 and b % 2 == 0),
        ("a,b both odd", lambda a, b: a % 2 == 1 and b % 2 == 1),
        ("a even, b odd", lambda a, b: a % 2 == 0 and b % 2 == 1),
        ("a odd, b even", lambda a, b: a % 2 == 1 and b % 2 == 0),
    ]
    _models = [("original", orig), ("ablated", abl)]
    _offsets = sorted(int(k) for k in orig["prompts"][0]["offset_probs"])

    fig, _axes = plt.subplots(
        len(_parities), 2, figsize=(11, 11), sharex=True, sharey=True, squeeze=False
    )
    for _col, (_mname, _data) in enumerate(_models):
        for _row, (_cname, _pred) in enumerate(_parities):
            _ax = _axes[_row][_col]
            _rows = [p for p in _data["prompts"] if _pred(p["a"], p["b"])]
            _m = np.array(
                [[p["offset_probs"][str(o)] for o in _offsets] for p in _rows], dtype=float
            )
            _mean, _std = _m.mean(0), _m.std(0)
            _ax.fill_between(_offsets, _mean - _std, _mean + _std, alpha=0.25, color="C0")
            _ax.plot(_offsets, _mean, color="C0", marker="o", ms=3)
            _ax.axvline(0, color="0.5", lw=0.8, ls="--")
            _ax.margins(x=0.02)
            if _row == 0:
                _ax.set_title(f"{_mname}\naccuracy = {_data['accuracy']:.3f}")
            if _col == 0:
                _ax.set_ylabel(f"{_cname}\nmean P", fontsize=9)
            if _row == len(_parities) - 1:
                _ax.set_xlabel("offset from correct answer")
    fig.suptitle("Answer-probability profile by operand parity (±1 std across prompts)", y=1.0)
    fig.tight_layout()
    fig
    return (fig,)


@app.cell
def _(abl, np, orig, plt):
    def _grid(data):
        n = max(max(p["a"], p["b"]) for p in data["prompts"])
        g = np.full((n, n), np.nan)
        for p in data["prompts"]:
            g[p["b"] - 1, p["a"] - 1] = p["p_correct"]  # rows = b, cols = a
        return g, n

    fig_hm, _axes = plt.subplots(1, 2, figsize=(11, 4.8), squeeze=False)
    for _ax, (_mname, _data) in zip(_axes[0], [("original", orig), ("ablated", abl)]):
        _g, _n = _grid(_data)
        _im = _ax.imshow(
            _g, origin="lower", extent=(1, _n, 1, _n), vmin=0, vmax=1, cmap="magma", aspect="auto"
        )
        _ax.set_title(f"{_mname} · P(correct)\naccuracy = {_data['accuracy']:.3f}")
        _ax.set_xlabel("a (first operand)")
        _ax.set_ylabel("b (second operand)")
        fig_hm.colorbar(_im, ax=_ax, fraction=0.046, pad=0.04, label="P(correct)")
    fig_hm.tight_layout()
    fig_hm
    return (fig_hm,)


if __name__ == "__main__":
    app.run()
