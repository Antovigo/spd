# Finding the quantities read from the residual stream: instructions for the agent

## The model and the task
A language model (Llama-3.1-8B, decomposed into components) reads prompts "a op b =" and answers
with the result: a and b are integers 1..100, op is + or - (op = 0 for +, 1 for -). The prompt has
five token positions t: 0 <BOS>, 1 a, 2 op, 3 b, 4 "=". At each position the residual stream (a
4096-vector) is read 64 times, at the attention input and the MLP input of each of the 32 blocks
(read sites, written "L<block>.attn" and "L<block>.mlp", in stream order L0.attn, L0.mlp, L1.attn,
...). At a site, each reader (a component that reads the stream: its inner activation is the stream,
normalised by its RMS, dotted with the reader's read direction) responds to the stream along its own
read direction; the readers of a site together see the stream only inside the span of their read
directions (k dimensions). The CI (causal importance, 0..1) of a reader on a prompt says how much the
model's output depends on that reader there; CI > 0.01 counts as important.

What the stream at t can depend on: t = 1: a only (op and b come later); t = 2: op and a; t = 3, 4:
op, a and b. Each distinct input is one row of the data ("domain row"): 100 rows at t = 1, 200 at
t = 2, 20000 at t = 3, 4. A fifth of them (values of a at t = 1, 2; (a, b) pairs at t = 3, 4) are
held out for the final evaluation; the tools never show them.

## What you are looking for
Quantities: functions of (op, a, b) that the stream carries along fixed directions in the readers'
span, so that the stream at the site is approximately its mean plus, for each quantity, its features
times a set of directions (one direction per feature, the same for all readers). The tool fits all
directions jointly (least squares with a small ridge) for the list of quantities you give it.

A good list explains the readers' inner activations where they are causally important with few,
simple quantities. The tool scores a list by its description length (bits, lower is better):
- data bits: how many bits the residual of the stream costs (Gaussian code);
- parameter bits: 1/2 log2(N) bits per fitted coefficient: each feature dimension of each quantity
  costs one coefficient per span dimension (k_eff of them), so a 20-function basis costs 20 times a
  scalar;
- formula bits: the length of the quantity's formula (about 5 bits per symbol, plus the bits of its
  numbers; integers cost more the larger they are).
A quantity is worth keeping when removing it would raise the total ("drop_one_delta_bits" > 0).
Smooth bases (spline, bumps) can fit almost anything with enough functions, and pay for every
function: use them only when no simpler formula explains the pattern.

## The loop at one site (at most 4 fits per site)
1. Look at the readers' inner activations (`show`): the most causally important readers, their
   activations over the domain (curves over a at t = 1, 2; (a, b) grids for + and - at t = 3, 4, a
   down, b across; grey cells are held out) and their CI, and the stream's principal components.
2. Write a list of candidate quantities (a spec file, below), from what you see. At every site after
   the first of your block, start from the previous site's final list: the stream is passed on from
   site to site, so most quantities persist; drop or replace what does not fit here.
3. Fit it (`fit`).
4. Look at the result: total bits and the drop-one delta of each quantity; the residual figures of the
   readers with the largest CI-weighted residual (the errors that matter); the residual's singular
   functions (patterns shared by many readers); which variables' values still explain residual
   variance (excess over chance). If there is no large regular discrepancy left in the causally
   important readers' activations, stop and keep this list. Otherwise:
   a) replace a quantity by a better one (for example, a pattern fit by a sine wave that leaves
      structure at the same period may be a rounded or step version, a different encoding, or a
      different variable);
   b) add quantities that explain the residual;
   c) remove quantities whose drop-one delta is <= 0;
   d) anything else that seems like a good idea.
5. Go back to 3. After the 4th fit, keep the best list you have (lowest total bits among lists with no
   quantity of negative drop-one delta) and note what remains unexplained.

## Spec files and the formula language
A spec is a Python file defining `QUANTITIES = [(name, formula), ...]`. The name is free text
describing the quantity; the formula is an expression in:
- variables: `a`, `b` (integers 1..100), `op` (0 for +, 1 for -);
- operators: `+ - * // %`, comparisons `== != < <= > >=`, `& | ~`, unary `-`;
- functions: `log`, `abs`, `sqrt`, `where(cond, x, y)`, `minimum`, `maximum`;
- feature builders (the formula's value must be one of these, or a numeric array):
  `scalar(x)` (one feature); `circle(x, P)` (cos and sin of 2 pi x / P); `classes(x)` (one indicator
  per distinct value of x); `onehot(x, [v1, v2, ...])` (indicators of the listed values);
  `bumps(x, lo, hi, step, width)` (Gaussian bumps of the given width centred every `step` from `lo`
  to `hi`); `spline(x, lo, hi, n)` (n cubic B-spline functions of x on [lo, hi]); `gate(cond, F)`
  (the features F where cond holds, 0 elsewhere);
- numbers (integers or decimals).
Example:
```python
QUANTITIES = [
    ("square of a", "scalar(a * a)"),
    ("a mod 7 on a circle", "circle(a, 7)"),
    ("a - b on subtraction prompts only", "gate(op == 1, scalar(a - b))"),
]
```

## Tools
From `notes/arith_representations/quantities/agentic/` (they run on a compute machine and copy the
outputs back; a fit takes about 10 s):
- `./qa_remote.sh sites <t>`: the sites of position t in stream order (and which are constant:
  skip those);
- `./qa_remote.sh show <t> <site> <dir>`: prints the readers ranked by CI-weighted variance and
  writes `show_readers.png`, `show_components.png` in `<dir>`;
- `./qa_remote.sh fit <t> <site> <spec.py> <dir>`: prints the report (bits, drop-one deltas, R^2 of
  the reads in-sample and on held-out folds of your training rows, the important residual share,
  the worst readers, the variable table) and writes `<spec>_reads.png`, `<spec>_residual.png`,
  `<spec>_residual_svd.png` in `<dir>`.
Look at the figures with your file-reading tool. Do not modify the tools.

## What to write (in your work directory, one folder per site: `<workdir>/t<t>/<site>/`)
- `spec_r1.py`, `spec_r2.py`, ... (one per fit) and the tool's outputs;
- `final.py`: a copy of the list you keep;
- `notes.md`: for each round, two or three lines: what you saw, what you changed, why.
At the end of your block, `<workdir>/t<t>/summary_<first site>-<last site>.md`: per site, the final
quantities and what remains unexplained.

## Rules
- Use only the tools above and the files in `agentic/` and your work directory. Do not read other
  files in the repository or in the run's analysis folders (reports, earlier results, other agents'
  folders, the data files directly): the point is to find the quantities from the activations.
- Every quantity must be justified by something you saw, and must earn its bits.
- Keep notes short; do not repeat the tool output.
