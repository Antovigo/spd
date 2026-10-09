"""Candidates for the agentic run's quantities (agentic/run1/t<t>/<site>/final.py), so that qdirs.py and
qapplet.py can read them as `extra` candidates: one Cand per distinct formula at the position, named
by the formula. The displayed value is the builder's variable (x mod P for circle(x, P); gate unwrapped).
"""

import ast
import sys
from pathlib import Path

import numpy as np
from qdata import RUN, Position
from qfeat import Cand

AGENTIC = Path(__file__).resolve().parents[1] / "agentic"
sys.path.insert(0, str(AGENTIC))
import qa  # noqa: E402

RUN1 = RUN / "analysis/quantities/agentic/run1"


def formulas(t: int) -> list[str]:
    out: list[str] = []
    for f in sorted((RUN1 / f"t{t}").glob("*/final.py")):
        out += [src for _, src in qa.load_spec(f) if src not in out]
    return out


def shown(src: str, pos: Position) -> np.ndarray:
    node = ast.parse(src, mode="eval").body
    while isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "gate":
        node = node.args[1]
    env = {
        "a": pos.a.astype(np.int64),
        "b": pos.b.astype(np.int64),
        "op": pos.op.astype(np.int64),
        **qa.FUNCS,
    }
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in qa.BUILDERS
        and node.args
    ):
        x = eval(compile(ast.Expression(node.args[0]), "<x>", "eval"), {"__builtins__": {}}, env)  # noqa: S307
        x = np.broadcast_to(np.asarray(x, float), (pos.D,)).copy()
        if node.func.id == "circle":
            x = np.mod(x, float(ast.literal_eval(node.args[1])))
        return x
    return qa.evaluate(src, pos.op, pos.a, pos.b)[:, 0]


def cands(pos: Position) -> list[Cand]:
    return [
        Cand(src, "", qa.evaluate(src, pos.op, pos.a, pos.b), shown(src, pos))
        for src in formulas(pos.t)
    ]
