"""Map the base model's ability to complete `<a><op><b>=` arithmetic prompts.

Tests the ORIGINAL (undecomposed) model — no decomposition is loaded — on every
`(a, b)` pair with `1 <= a, b <= --max-operand`, across several prompt renderings:

- digit-symbol: `1+1=`, `1-1=`, and multiplication with each of `* x X ×`;
- words:        `one plus one equals` (also minus / times);
- digits+words: `1 plus 1 equals` (also minus / times).

("plus" / "minus" / "times" + "equals" is the standard English phrasing.)

For each prompt it records the model's next-token prediction, the first token of the
correct answer (the operation's result rendered to match the prompt's style — e.g. a
negative subtraction result starts with `-`/`minus`), whether they match, and the model's
probability on the correct token. Per condition it also writes an interactive HTML heatmap
of P(correct token) over the (a, b) grid (hover shows the predicted/correct tokens etc.).

An 8B forward pass needs a GPU; pass `--slurm` to submit as a single-GPU SLURM job.

Usage:
    python -m param_decomp_lab.scripts.validation.map_arithmetic \
        [--model-name=meta-llama/Llama-3.1-8B] [--max-operand=100] [--batch-size=128] \
        [--output-dir=PATH] [--slurm [--partition=... --slurm-time=1:00:00 --slurm-mem=...]]

Output (default `PARAM_DECOMP_OUT_DIR/arithmetic_map/`):
- `results.tsv`           — one row per (condition, a, b)
- `accuracy_summary.tsv`  — per-condition accuracy + mean P(correct)
- `heatmaps/<condition>.html`, `index.html`
"""

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import fire
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from param_decomp.log import logger
from param_decomp.torch_helpers import bf16_autocast
from param_decomp_lab.distributed import ensure_cached_and_call, get_device
from param_decomp_lab.infra.settings import DEFAULT_PARTITION_NAME, PARAM_DECOMP_OUT_DIR
from param_decomp_lab.scripts.validation.common import SlurmOptions, submit_self_to_slurm

_MODULE = "param_decomp_lab.scripts.validation.map_arithmetic"

# RdPu (colorbrewer 9-class), matching the other validation figures.
_RDPU = [
    "#fff7f3",
    "#fde0dd",
    "#fcc5c0",
    "#fa9fb5",
    "#f768a1",
    "#dd3497",
    "#ae017e",
    "#7a0177",
    "#49006a",
]

_ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _words_below_1000(n: int) -> str:
    assert 0 <= n < 1000
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")
    hundreds, rest = divmod(n, 100)
    return _ONES[hundreds] + " hundred" + (f" {_words_below_1000(rest)}" if rest else "")


def _words(n: int) -> str:
    """English words for `n` (American style, no 'and'); negatives prefixed with 'minus'."""
    if n < 0:
        return f"minus {_words(-n)}"
    if n < 1000:
        return _words_below_1000(n)
    thousands, rest = divmod(n, 1000)
    return (
        _words_below_1000(thousands) + " thousand" + (f" {_words_below_1000(rest)}" if rest else "")
    )


@dataclass(frozen=True)
class Condition:
    name: str
    description: str
    prompt_fn: Callable[[int, int], str]
    answer_fn: Callable[[int, int], str]  # the correct continuation, including any leading space


_RESULT: dict[str, Callable[[int, int], int]] = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
}
_WORD = {"add": "plus", "sub": "minus", "mul": "times"}
_SYM_NAME = {"+": "plus", "-": "minus", "*": "star", "x": "x", "X": "Xcap", "×": "times"}


def _conditions() -> list[Condition]:
    conds: list[Condition] = []
    for op, sym in [
        ("add", "+"),
        ("sub", "-"),
        ("mul", "*"),
        ("mul", "x"),
        ("mul", "X"),
        ("mul", "×"),
    ]:
        res = _RESULT[op]
        conds.append(
            Condition(
                name=f"digit_{op}_{_SYM_NAME[sym]}",
                description=f"{op} digit-symbol '{sym}'  (e.g. '1{sym}1=')",
                prompt_fn=lambda a, b, sym=sym: f"{a}{sym}{b}=",
                answer_fn=lambda a, b, res=res: str(res(a, b)),
            )
        )
    for op in ("add", "sub", "mul"):
        res, word = _RESULT[op], _WORD[op]
        conds.append(
            Condition(
                name=f"words_{op}",
                description=f"{op} words  (e.g. 'one {word} one equals')",
                prompt_fn=lambda a, b, word=word: f"{_words(a)} {word} {_words(b)} equals",
                answer_fn=lambda a, b, res=res: f" {_words(res(a, b))}",
            )
        )
    for op in ("add", "sub", "mul"):
        res, word = _RESULT[op], _WORD[op]
        conds.append(
            Condition(
                name=f"digits_words_{op}",
                description=f"{op} digits+words  (e.g. '1 {word} 1 equals')",
                prompt_fn=lambda a, b, word=word: f"{a} {word} {b} equals",
                answer_fn=lambda a, b, res=res: f" {res(a, b)}",
            )
        )
    return conds


@dataclass
class _Pred:
    predicted_id: int
    p_predicted: float
    p_correct: float


def _predict(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    correct_ids: list[int],
    device: torch.device,
    batch_size: int,
) -> list[_Pred]:
    """Next-token prediction + P(correct token) at the last real position of each prompt."""
    preds: list[_Pred] = []
    with torch.no_grad(), bf16_autocast():
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            enc = tokenizer(chunk, return_tensors="pt", padding=True)
            input_ids = cast(Tensor, enc["input_ids"]).to(device)
            attn = cast(Tensor, enc["attention_mask"]).to(device)
            logits = model(input_ids=input_ids, attention_mask=attn).logits

            rows = torch.arange(input_ids.shape[0], device=device)
            last = attn.sum(dim=1) - 1  # right-padded, so last real position
            probs = logits[rows, last].float().softmax(dim=-1)
            predicted = probs.argmax(dim=-1)
            correct = torch.tensor(correct_ids[start : start + batch_size], device=device)
            p_pred = probs[rows, predicted].tolist()
            p_corr = probs[rows, correct].tolist()
            for i, pid in enumerate(predicted.tolist()):
                preds.append(_Pred(pid, p_pred[i], p_corr[i]))
    return preds


def _first_token(tokenizer: PreTrainedTokenizerBase, answer: str) -> int:
    ids = tokenizer.encode(answer, add_special_tokens=False)
    assert ids, f"answer {answer!r} tokenized to nothing"
    return ids[0]


_HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script></head>
<body><div id="hm" style="width:840px;height:820px;"></div>
<script>
Plotly.newPlot("hm", [{trace}], {layout}, {{responsive:true}});
</script></body></html>
"""


def _write_heatmap(
    path: Path,
    condition: Condition,
    n: int,
    accuracy: float,
    cells: dict[tuple[int, int], dict[str, Any]],
) -> None:
    """Interactive Plotly heatmap of P(correct token) over the (a, b) grid (CDN, no deps)."""
    operands = list(range(1, n + 1))
    z = [[cells[(a, b)]["p_correct"] for a in operands] for b in operands]
    customdata = [
        [
            [
                cells[(a, b)]["predicted"],
                cells[(a, b)]["correct"],
                "yes" if cells[(a, b)]["match"] else "no",
                round(cells[(a, b)]["p_predicted"], 4),
            ]
            for a in operands
        ]
        for b in operands
    ]
    trace = {
        "type": "heatmap",
        "z": z,
        "x": operands,
        "y": operands,
        "zmin": 0.0,
        "zmax": 1.0,
        "colorscale": [[i / (len(_RDPU) - 1), c] for i, c in enumerate(_RDPU)],
        "colorbar": {"title": "P(correct)"},
        "customdata": customdata,
        "hovertemplate": (
            "a=%{x} · b=%{y}<br>P(correct %{customdata[1]}) = %{z:.3f}<br>"
            "predicted %{customdata[0]} (p=%{customdata[3]:.3f})<br>"
            "match: %{customdata[2]}<extra></extra>"
        ),
    }
    layout = {
        "title": f"{condition.description}<br>accuracy = {accuracy:.3f}",
        "xaxis": {"title": "a (first operand)", "constrain": "domain"},
        "yaxis": {"title": "b (second operand)", "scaleanchor": "x", "constrain": "domain"},
    }
    html = _HTML_TEMPLATE.format(trace=json.dumps(trace), layout=json.dumps(layout))
    path.write_text(html)


def _write_index(path: Path, summary: list[dict[str, Any]]) -> None:
    rows = "\n".join(
        f'<tr><td><a href="heatmaps/{s["condition"]}.html">{s["condition"]}</a></td>'
        f"<td>{s['description']}</td><td>{s['accuracy']:.3f}</td>"
        f"<td>{s['mean_p_correct']:.3f}</td></tr>"
        for s in summary
    )
    path.write_text(
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>arithmetic map</title>"
        "<style>body{font-family:sans-serif}td,th{padding:4px 10px;text-align:left}</style></head>"
        "<body><h2>Llama arithmetic capability map</h2><table border=1>"
        "<tr><th>condition</th><th>description</th><th>accuracy</th><th>mean P(correct)</th></tr>"
        f"{rows}</table></body></html>"
    )


_TSV_FIELDS = [
    "condition",
    "a",
    "b",
    "prompt",
    "correct_answer",
    "correct_token",
    "correct_token_id",
    "predicted_token",
    "predicted_token_id",
    "correct",
    "p_correct",
    "p_predicted",
]


def map_arithmetic(
    model_name: str = "meta-llama/Llama-3.1-8B",
    max_operand: int = 100,
    batch_size: int = 128,
    output_dir: str | None = None,
    slurm: bool = False,
    partition: str | None = DEFAULT_PARTITION_NAME,
    gpus: int = 1,
    slurm_time: str = "1:00:00",
    slurm_mem: str | None = None,
) -> Path | None:
    """Run the full arithmetic capability map. Returns the output folder (or None on submit)."""
    out_dir = (
        Path(output_dir).expanduser() if output_dir else PARAM_DECOMP_OUT_DIR / "arithmetic_map"
    )
    if slurm:
        argv = [
            f"--model-name={model_name}",
            f"--max-operand={max_operand}",
            f"--batch-size={batch_size}",
            f"--output-dir={out_dir}",
        ]
        opts = SlurmOptions(
            partition=partition, gpus=gpus, slurm_time=slurm_time, slurm_mem=slurm_mem
        )
        submit_self_to_slurm(_MODULE, argv, opts, job_name="val-map-arithmetic")
        return None

    device = torch.device(get_device())
    tokenizer = ensure_cached_and_call(AutoTokenizer.from_pretrained, model_name)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = ensure_cached_and_call(AutoModelForCausalLM.from_pretrained, model_name, dtype="auto")
    model = model.to(device).eval()

    heatmaps_dir = out_dir / "heatmaps"
    heatmaps_dir.mkdir(parents=True, exist_ok=True)
    operands = list(range(1, max_operand + 1))

    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for condition in _conditions():
        prompts, correct_ids, ab = [], [], []
        for b in operands:
            for a in operands:
                prompt = condition.prompt_fn(a, b)
                answer = condition.answer_fn(a, b)
                prompts.append(prompt)
                correct_ids.append(_first_token(tokenizer, answer))
                ab.append((a, b))

        preds = _predict(model, tokenizer, prompts, correct_ids, device, batch_size)

        cells: dict[tuple[int, int], dict[str, Any]] = {}
        n_correct = 0
        sum_p_correct = 0.0
        for (a, b), prompt, cid, pred in zip(ab, prompts, correct_ids, preds, strict=True):
            match = pred.predicted_id == cid
            n_correct += int(match)
            sum_p_correct += pred.p_correct
            predicted_tok = repr(tokenizer.decode([pred.predicted_id]))
            correct_tok = repr(tokenizer.decode([cid]))
            cells[(a, b)] = {
                "p_correct": pred.p_correct,
                "p_predicted": pred.p_predicted,
                "predicted": predicted_tok,
                "correct": correct_tok,
                "match": match,
            }
            rows.append(
                {
                    "condition": condition.name,
                    "a": a,
                    "b": b,
                    "prompt": repr(prompt),
                    "correct_answer": repr(condition.answer_fn(a, b)),
                    "correct_token": correct_tok,
                    "correct_token_id": cid,
                    "predicted_token": predicted_tok,
                    "predicted_token_id": pred.predicted_id,
                    "correct": int(match),
                    "p_correct": pred.p_correct,
                    "p_predicted": pred.p_predicted,
                }
            )

        accuracy = n_correct / len(prompts)
        mean_p_correct = sum_p_correct / len(prompts)
        summary.append(
            {
                "condition": condition.name,
                "description": condition.description,
                "accuracy": accuracy,
                "mean_p_correct": mean_p_correct,
            }
        )
        _write_heatmap(
            heatmaps_dir / f"{condition.name}.html", condition, max_operand, accuracy, cells
        )
        logger.info(
            f"{condition.name}: accuracy {accuracy:.3f}, mean P(correct) {mean_p_correct:.3f}"
        )

    with (out_dir / "results.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_TSV_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "accuracy_summary.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["condition", "description", "accuracy", "mean_p_correct"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(summary)
    _write_index(out_dir / "index.html", summary)

    logger.info(f"Wrote {len(rows)} rows + {len(summary)} heatmaps to {out_dir}")
    return out_dir


if __name__ == "__main__":
    fire.Fire(map_arithmetic)
