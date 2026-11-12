"""The integer-only pipeline's pieces on a tiny all-sites decomposed GLU (CPU): the factored
forward and its inner activations against the decomposition's own masked forward, `KL_int`
against a hand computation, the mask gradient against finite differences, the learner, the λ
rule, the collection's column selection and bit packing, the component-list schema, the
prompt-file pool, and the applet slices."""

import base64
import json
import re
from collections.abc import Mapping
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from param_decomp.ci_filter.step import all_logits, lm_batch, masked_output
from param_decomp.core.components import ComponentStacks, SiteC, require_full_emission
from param_decomp.core.init_placed import (
    init_model_component_stacks_placed,
    random_component_initializer,
)
from param_decomp.core.model import MaterializedMasking
from param_decomp.core.placement import from_config
from param_decomp.core.precision import COMPUTE_DT
from param_decomp.core.sharding import place_target, single_device_mesh
from param_decomp.experiments.lm.arithmetic_eval import component_activation_model
from param_decomp.experiments.lm.load_run import PlacedTransformer
from param_decomp.integer_only import model as im
from param_decomp.integer_only.applet import operation_grid, slice_documents
from param_decomp.integer_only.collect import (
    columns_of,
    ever_on_bits,
    ever_on_max_ci,
    make_inner_columns,
    pack_rows,
    unpack_column,
)
from param_decomp.integer_only.config import IntegerOnlyConfig
from param_decomp.integer_only.learn import Context, is_finished, learn_mask
from param_decomp.integer_only.objective import (
    QUANTITIES,
    int_log_q,
    kl_int,
    log_softmax32,
    mask_gradient,
    probe_scores,
    scores,
)
from param_decomp.integer_only.pick import choose, mask_json, read_mask_json
from param_decomp.integer_only.pool import (
    build_pool,
    check_answer_tokens,
    draw_check_set,
    integer_token_ids,
)
from param_decomp.targets.testing import tiny_glu_cfg, tiny_glu_decomposed_lm
from param_decomp.targets.transformer import (
    KIND_ORDER,
    TransformerDecomposedModel,
    glu_site_specs,
    site_name,
)

C = 8
VOCAB = 64

type Setup = tuple[
    Mesh,
    PlacedTransformer,
    ComponentStacks,
    dict[str, np.ndarray],
    im.Arch,
    im.Shared,
    im.Factors,
    im.DenseWeights,
    im.Slots,
    jax.Array,
]


@pytest.fixture(scope="module")
def setup() -> Setup:
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(
        cfg,
        tuple(
            SiteC(site_name(layer, kind), C) for layer in range(cfg.n_layer) for kind in KIND_ORDER
        ),
    )
    mesh = single_device_mesh()
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    rules = from_config("ddp", mesh, sites)
    placed: PlacedTransformer = place_target(model, rules)
    with jax.set_mesh(mesh):
        components: ComponentStacks = init_model_component_stacks_placed(
            placed, jax.random.PRNGKey(1), rules, random_component_initializer
        )
    rng = np.random.default_rng(0)
    keep = {s.name: rng.random(C) < 0.7 for s in sites}
    for name in keep:  # at least one per site, and one site with every component
        keep[name][0] = True
    keep[site_name(3, "down")][:] = True
    assert isinstance(placed.model, TransformerDecomposedModel)
    arch, shared, dense = im.build_target(placed.model)
    factors, slots = im.build_factors(components, keep, arch.n_layer)
    tokens = jnp.asarray(rng.integers(1, VOCAB, size=(6, 5)), jnp.int32)
    return mesh, placed, components, keep, arch, shared, factors, dense, slots, tokens


def _native_masks(keep: dict[str, np.ndarray], shape: tuple[int, ...]) -> dict[str, jax.Array]:
    return {
        name: jnp.broadcast_to(jnp.asarray(k, COMPUTE_DT), (*shape, k.size))
        for name, k in keep.items()
    }


def test_factored_forward_matches_the_decomposition(setup: Setup) -> None:
    mesh, placed, components, keep, arch, shared, factors, dense, slots, tokens = setup
    mask = slots.mask(np.ones(slots.n, bool), arch.n_layer)
    ours = im.factored_logits(arch, shared, factors, mask, tokens, "fp32", "fp32")
    with jax.set_mesh(mesh):
        prepared = placed.prepare_compute_weights(components)
        masking = MaterializedMasking(component_masks=_native_masks(keep, tuple(tokens.shape)))
        theirs = all_logits(masked_output(placed, prepared, lm_batch(tokens), masking, remat=False))
        clean = all_logits(placed.clean_forward(lm_batch(tokens)).output)[:, -1, :]
    # The decomposition computes in bf16; ours is the same model in fp32.
    gap = kl_full(log_softmax32(ours), log_softmax32(theirs[:, -1, :]))
    assert float(jnp.max(gap)) < 1e-3, gap
    target = im.dense_logits(arch, shared, dense, tokens, "fp32", "fp32")
    assert float(jnp.max(kl_full(log_softmax32(target), log_softmax32(clean)))) < 1e-3
    assert float(jnp.mean(kl_full(log_softmax32(target), log_softmax32(ours)))) > 1e-3


def kl_full(logp_ref: jax.Array, logp: jax.Array) -> jax.Array:
    return jnp.sum(jnp.exp(logp_ref) * (logp_ref - logp), axis=-1)


def test_inner_columns_match_the_decomposition(setup: Setup) -> None:
    """The collection's inner activations: `x V_c` at every position in the masked model,
    canonical order, equal to the decomposition's own masked component activations."""
    mesh, placed, components, keep, arch, shared, factors, _dense, slots, tokens = setup
    mask = slots.mask(np.ones(slots.n, bool), arch.n_layer)
    ours = np.asarray(make_inner_columns(slots)(arch, shared, factors, mask, tokens, "fp32"))
    assert ours.shape == (*tokens.shape, slots.n)
    with jax.set_mesh(mesh):
        prepared = placed.prepare_compute_weights(components)
        masking = MaterializedMasking(component_masks=_native_masks(keep, tuple(tokens.shape)))
        acts = component_activation_model(placed).masked_component_activations(
            prepared, lm_batch(tokens), masking, placement=placed.placement
        )
    theirs = np.concatenate(
        [
            np.asarray(require_full_emission(acts[s]), np.float32)[:, :, np.flatnonzero(keep[s])]
            for s in sorted(keep)
        ],
        axis=-1,
    )
    scale = np.abs(theirs).max()
    np.testing.assert_allclose(ours, theirs, atol=0.02 * scale)


def test_kl_int_and_scores_by_hand() -> None:
    """`q` renormalizes over the integer tokens (here 3 of a 5-token vocabulary)."""
    logits = np.array([[1.0, 2.0, 0.5, -1.0, 3.0]], np.float32)
    ref_logits = np.array([[0.0, 1.0, 1.0, 2.0, -2.0]], np.float32)
    int_ids = np.array([4, 1, 3], np.int32)  # integers 0, 1, 2 are tokens 4, 1, 3

    def q_of(z: np.ndarray) -> np.ndarray:
        e = np.exp(z[0, int_ids])
        return e / e.sum()

    q, q_ref = q_of(logits), q_of(ref_logits)
    want = float(np.sum(q_ref * np.log(q_ref / q)))
    logp, logp_ref = (log_softmax32(jnp.asarray(z)) for z in (logits, ref_logits))
    got = kl_int(int_log_q(logp_ref, jnp.asarray(int_ids)), int_log_q(logp, jnp.asarray(int_ids)))
    assert float(got[0]) == pytest.approx(want, rel=1e-5)
    r = jnp.asarray([2])
    y = jnp.asarray([int_ids[2]])
    s = np.asarray(scores(logp, logp_ref, r, y, jnp.asarray(int_ids)))[:, 0]
    p = np.exp(logits[0]) / np.exp(logits[0]).sum()
    p_ref = np.exp(ref_logits[0]) / np.exp(ref_logits[0]).sum()
    expected = {
        "kl_int": want,
        "kl_full": float(np.sum(p_ref * np.log(p_ref / p))),
        "ce": float(-np.log(p[3])),
        "ce_int": float(-np.log(q[2])),
        "p_int": float(p[int_ids].sum()),
        "entropy_int": float(-np.sum(q * np.log(q))),
        "correct": 0.0,  # argmax q is integer 0 (token 4), the answer is 2
    }
    for i, name in enumerate(QUANTITIES):
        assert s[i] == pytest.approx(expected[name], rel=1e-5, abs=1e-6), name


def test_mask_gradient_matches_finite_differences(setup: Setup) -> None:
    *_, arch, shared, factors, dense, slots, tokens = setup
    rng = np.random.default_rng(2)
    on = rng.random(slots.n) < 0.8
    mask = slots.mask(on, arch.n_layer)
    int_ids = jnp.asarray(rng.permutation(VOCAB)[:40], jnp.int32)
    log_q_ref = int_log_q(
        log_softmax32(im.dense_logits(arch, shared, dense, tokens, "fp32", "fp32")), int_ids
    )
    weight = jnp.ones((tokens.shape[0],))
    grad = slots.gather(
        mask_gradient(arch, shared, factors, mask, tokens, log_q_ref, weight, int_ids, "fp32")
    )
    r = jnp.zeros((tokens.shape[0],), jnp.int32)

    def f(m: im.Mask) -> float:
        kl = probe_scores(arch, shared, factors, m, tokens, log_q_ref, r, int_ids, "fp32")[0]
        return float(jnp.sum(kl))

    h = 2e-2  # central difference: O(h²) bias, and fp32 sums need a step well above 1e-3
    for j in (0, int(np.flatnonzero(~on)[0]), slots.n // 2, slots.n - 1):
        kind = KIND_ORDER[slots.kind[j]]
        hi, lo = dict(mask), dict(mask)
        hi[kind] = mask[kind].at[slots.layer[j], slots.slot[j]].add(h)
        lo[kind] = mask[kind].at[slots.layer[j], slots.slot[j]].add(-h)
        np.testing.assert_allclose(grad[j], (f(hi) - f(lo)) / (2 * h), rtol=2e-2, atol=1e-4)


def _config(tmp_path: Path, **update: object) -> IntegerOnlyConfig:
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("x\n")
    base = {
        "step": 0,
        "prompts": str(prompts),
        "templates": {"add": "{a} plus {b}"},
        "check_size": 8,
        "lambdas": [1.0],
        "steps": 4,
        "batch": 8,
        "microbatch": 4,
        "lr": 0.5,
        "init": 0.5,
        "eval_every": 2,
        "eval_size": 8,
        "precision": "fp32",
        "score_batch": 8,
    }
    return IntegerOnlyConfig.model_validate(base | update)


def test_learned_mask_prunes_and_resumes(
    setup: Setup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A large λ prunes candidates; a run interrupted after its step-2 save resumes to exactly
    the uninterrupted result; a resume under other settings refuses."""
    from param_decomp.integer_only import learn

    *_, arch, shared, factors, dense, slots, _tokens = setup
    rng = np.random.default_rng(3)
    n_rows = 40
    tokens = rng.integers(1, VOCAB, size=(n_rows, 5)).astype(np.int32)
    int_ids = jnp.asarray(rng.permutation(VOCAB)[:40], jnp.int32)
    logits = im.dense_logits(arch, shared, dense, jnp.asarray(tokens), "fp32", "fp32")
    check = rng.permutation(n_rows)[:8]
    ctx = Context(
        arch=arch, shared=shared, factors=factors, slots=slots, tokens=tokens,
        r=rng.integers(0, 40, size=n_rows),
        log_q_ref=np.asarray(int_log_q(log_softmax32(logits), int_ids)), int_ids=int_ids,
        train=np.setdiff1d(np.arange(n_rows), check), check=check,
    )  # fmt: skip
    config = _config(tmp_path)
    whole = tmp_path / "whole"
    facts = learn_mask(config, ctx, 1.0, whole)
    assert is_finished(whole, config) and facts["steps_run"] == 4
    assert int(facts["size"]) < slots.n  # pyright: ignore[reportArgumentType]
    log = [json.loads(line) for line in (whole / "log.jsonl").read_text().splitlines()]
    assert [row["step"] for row in log] == [2, 4]

    interrupted = tmp_path / "interrupted"
    real_probe = learn.probe
    calls: list[int] = []

    def failing_probe(
        ctx: Context, on: np.ndarray, rows: np.ndarray, config: IntegerOnlyConfig
    ) -> tuple[float, float]:
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real_probe(ctx, on, rows, config)

    monkeypatch.setattr(learn, "probe", failing_probe)
    with pytest.raises(KeyboardInterrupt):
        learn_mask(config, ctx, 1.0, interrupted)
    monkeypatch.setattr(learn, "probe", real_probe)
    assert int(np.load(interrupted / "state.npz")["step"]) == 2
    facts = learn_mask(config, ctx, 1.0, interrupted)
    assert facts["steps_run"] == 2
    np.testing.assert_array_equal(
        np.load(interrupted / "state.npz")["logit"], np.load(whole / "state.npz")["logit"]
    )
    with pytest.raises(AssertionError, match="learn settings differ"):
        learn_mask(config.model_copy(update={"lr": 0.1}), ctx, 1.0, whole)


def test_lambda_rule() -> None:
    bar = 0.16479748860001564
    assert choose({5e-5: 0.05, 1e-4: 0.12, 2e-4: 0.17, 5e-4: 0.4}, bar) == 1e-4
    assert choose({5e-5: 0.05, 1e-4: 0.2, 2e-4: 0.1, 5e-4: 0.4}, bar) == 2e-4  # highest, not first
    assert choose({5e-5: bar, 1e-4: 0.3}, bar) is None  # strictly below
    assert choose({5e-5: 0.2, 1e-4: 0.3}, bar) is None


def test_collection_columns_are_the_ever_on_pairs(setup: Setup) -> None:
    *_, slots, _tokens = setup
    rng = np.random.default_rng(4)
    seq_len, threshold = 5, 0.1
    max_ci = {site: rng.random((seq_len, C)).astype(np.float32) * 0.2 for site in set(slots.sites)}
    columns = columns_of(ever_on_max_ci(slots, max_ci, threshold, seq_len))
    want = {
        (t, j)
        for t in range(seq_len)
        for j in range(slots.n)
        if max_ci[slots.sites[j]][t, slots.index[j]] > threshold
    }
    got = set(zip(columns.position.tolist(), columns.component.tolist(), strict=True))
    assert got == want
    assert np.all(np.diff(columns.position) >= 0)
    for t in range(seq_len):
        assert np.all(np.diff(columns.at(t)) > 0)  # canonical order within a position
    # A component on nowhere has no column; one on at a single position has exactly one.
    j_off, j_one = 0, 1
    for t in range(seq_len):
        max_ci[slots.sites[j_off]][t, slots.index[j_off]] = 0.0
        max_ci[slots.sites[j_one]][t, slots.index[j_one]] = 0.5 if t == 2 else 0.0
    columns = columns_of(ever_on_max_ci(slots, max_ci, threshold, seq_len))
    assert j_off not in columns.component
    assert columns.position[columns.component == j_one].tolist() == [2]


def test_collection_columns_from_the_pass_bits() -> None:
    """Stage 5 stores exactly the (t, j) pairs ON on some prompt of its own CI pass, by position
    then component, and each stored column's bits are the pass's."""
    rng = np.random.default_rng(7)
    n_prompts, seq_len, n = 21, 4, 6
    on = rng.random((n_prompts, seq_len, n)) < 0.05
    on[:, 1, :] = False  # a position where nothing is ever on
    on[:, :, 2] = False  # a component never on anywhere
    on[5, 3, 2] = True  # ... except once
    packed = pack_rows(np.concatenate([on, np.zeros((3, seq_len, n), bool)]))  # 24 rows
    columns = columns_of(ever_on_bits(packed))
    want = {(t, j) for t in range(seq_len) for j in range(n) if on[:, t, j].any()}
    assert set(zip(columns.position.tolist(), columns.component.tolist(), strict=True)) == want
    assert columns.at(1).size == 0 and columns.at(3).tolist().count(2) == 1
    assert np.all(np.diff(columns.position) >= 0)
    for t in range(seq_len):
        cols = columns.at(t)
        assert np.all(np.diff(cols) > 0)
        stored = packed[:, t][:, cols]
        for k, j in enumerate(cols):
            np.testing.assert_array_equal(unpack_column(stored[:, k], n_prompts), on[:, t, j])


def test_collection_stores_the_union_of_both_passes(setup: Setup) -> None:
    """A pair ON only in stage 1's max CI, or only in the collection's own pass, is stored."""
    *_, slots, _tokens = setup
    seq_len, threshold = 3, 0.1
    max_ci = {site: np.zeros((seq_len, C), np.float32) for site in set(slots.sites)}
    max_ci[slots.sites[0]][1, slots.index[0]] = 0.2  # (t=1, j=0) ON in stage 1 only
    on = np.zeros((8, seq_len, slots.n), bool)
    on[3, 2, 5] = True  # (t=2, j=5) ON in the collection only
    on[0, 1, 0] = True  # (t=1, j=0) ON in both
    first = ever_on_max_ci(slots, max_ci, threshold, seq_len)
    here = ever_on_bits(pack_rows(on))
    columns = columns_of(first | here)
    pairs = set(zip(columns.position.tolist(), columns.component.tolist(), strict=True))
    assert pairs == {(1, 0), (2, 5)}
    on[0, 1, 0] = False
    columns = columns_of(first | ever_on_bits(pack_rows(on)))
    assert set(zip(columns.position.tolist(), columns.component.tolist(), strict=True)) == pairs


def test_bit_packing_round_trip() -> None:
    rng = np.random.default_rng(5)
    bits = rng.random((24, 3)) < 0.3
    packed = pack_rows(bits)
    assert packed.shape == (3, 3) and packed.dtype == np.uint8
    for k in range(3):
        np.testing.assert_array_equal(unpack_column(packed[:, k], 21), bits[:21, k])


def test_component_list_schema(setup: Setup, tmp_path: Path) -> None:
    *_, slots, _tokens = setup
    on = np.zeros(slots.n, bool)
    on[[0, 3, slots.n - 1]] = True
    sizes = {site: C for site in set(slots.sites)}
    doc = mask_json(slots, on, sizes, rule="r", source="s", threshold=0.1)
    assert set(doc) == {"rule", "source", "threshold", "n_alive", "n_components", "components"}
    assert doc["n_alive"] == 3 and doc["n_components"] == C * len(sizes)
    comps = doc["components"]
    assert isinstance(comps, dict) and set(comps) == set(sizes)
    path = tmp_path / "integer-only.json"
    path.write_text(json.dumps(doc))
    back = read_mask_json(path)
    assert {(s, c) for s, ids in back.items() for c in ids} == {
        (slots.sites[j], int(slots.index[j])) for j in (0, 3, slots.n - 1)
    }


class WordTokenizer:
    """Llama-3-like: `<BOS>`, numbers up to 3 digits as one token, a space before a word kept
    with it, a lone space its own token."""

    PATTERN = re.compile(r"[0-9]{1,3}| ?[A-Za-z]+|:| ")

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {"<BOS>": 0}

    def _id(self, piece: str) -> int:
        return self.vocab.setdefault(piece, len(self.vocab))

    def encode(self, text: str, /, *, add_special_tokens: bool) -> list[int]:
        pieces = self.PATTERN.findall(text)
        assert "".join(pieces) == text, text
        return ([0] if add_special_tokens else []) + [self._id(p) for p in pieces]

    def decode(self, token_ids: list[int], /) -> str:
        inverse = {i: p for p, i in self.vocab.items()}
        return "".join(inverse[i] for i in token_ids)

    def __len__(self) -> int:
        return len(self.vocab)


def test_prompt_file_pool(tmp_path: Path) -> None:
    templates = {"add": "Answer: {a} plus {b} is ", "sub": "Answer: {a} minus {b} is "}
    lines = [templates["add"].format(a=a, b=b) for a in range(4) for b in range(4)]
    lines += [templates["sub"].format(a=a, b=b) for a in range(4) for b in range(a + 1)]
    path = tmp_path / "prompts.txt"
    path.write_text("\n".join(lines) + "\n")
    tok = WordTokenizer()
    int_ids = integer_token_ids(tok)
    pool = build_pool(path, templates, tok)  # pyright: ignore[reportArgumentType]
    assert pool.ops == ("add", "sub") and pool.n == 16 + 10
    assert pool.position_labels == ("<BOS>", "Answer", ":", "␣", "a", "op", "␣", "b", "␣is", "␣")
    assert pool.op[16] == 1 and (pool.a[16], pool.b[16], pool.r[16]) == (0, 0, 0)
    assert pool.r[:16].tolist() == [a + b for a in range(4) for b in range(4)]
    rows = pool.problem_rows()
    assert rows.size == pool.n  # subtraction keeps a >= b: every answer is >= 0
    check_answer_tokens(path, pool, rows, int_ids, tok)
    q = draw_check_set(rows, 10, seed=0)
    assert q.size == 10 and np.unique(q).size == 10 and set(q) <= set(rows)
    assert not np.all(np.diff(q) > 0)  # draw order, not sorted
    np.testing.assert_array_equal(draw_check_set(rows, 10, seed=0), q)
    pool.save(tmp_path / "pool.npz")
    back = type(pool).load(tmp_path / "pool.npz")
    assert back.position_labels == pool.position_labels and back.ops == pool.ops
    np.testing.assert_array_equal(back.tokens, pool.tokens)


def _decoded(doc: Mapping[str, object], key: str, dtype: type[np.generic]) -> np.ndarray:
    value = doc[key]
    assert isinstance(value, str), key
    return np.frombuffer(base64.b64decode(value), dtype)


def test_applet_slice_greys_cells_outside_the_pool() -> None:
    """A triangular subtraction block: absent cells are NaN; a grid wider than 100 splits."""
    for side in (6, 101):
        a, b = (x.ravel() for x in np.meshgrid(np.arange(side), np.arange(side), indexing="ij"))
        keep = a >= b
        a, b = a[keep], b[keep]
        grid, cell = operation_grid(a, b, "sub")
        assert (grid.n_a, grid.n_b) == (side, side)
        rng = np.random.default_rng(6)
        on = rng.random((a.size, 2)) < 0.5
        inner = rng.standard_normal((a.size, 2)).astype(np.float32)
        site = "layers.0.mlp.up_proj"
        cols = {site: (np.array([1, 5]), on, inner)}
        index, modules = slice_documents(
            grid, cell, 9, 10, 0, cols, {site: 8, "layers.0.mlp.down_proj": 8}
        )
        entries = index["modules"]
        assert isinstance(entries, list)
        (entry,) = [m for m in entries if m["name"] == site]
        mean = _decoded(entry, "mean_ci", np.float32)
        np.testing.assert_allclose(mean[[1, 5]], on.mean(axis=0), rtol=1e-6)
        assert mean[[0, 2, 3, 4, 6, 7]].max() == 0.0
        if side > 100:
            assert "thumb" in index and site in modules
            doc = modules[site]
        else:
            assert "thumb" not in index and not modules
            doc = entry
        grids = _decoded(doc, "inner", np.float16).reshape(2, side, side)
        lower = np.tril(np.ones((side, side), bool))
        assert np.isnan(grids[0][~lower]).all() and not np.isnan(grids[0][lower]).any()
        ci = _decoded(doc, "ci", np.uint8).reshape(2, side, side)
        assert set(np.unique(ci)) <= {0, 255}
        full = np.zeros((side, side), bool)
        full[a, b] = on[:, 0]
        np.testing.assert_array_equal(ci[0] == 255, full)
