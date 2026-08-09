"""Target-owned one-to-one activation vocabulary shared by transformer targets.

Every key names one physical forward activation. Matrix sites and captured activations are
separate vocabularies: a site names a decomposed weight, while a tap names one array in
the forward. Public capture keys therefore cannot alias the same array.
"""

from collections.abc import Callable, Hashable
from dataclasses import dataclass

from param_decomp.core.family import ArchFamily

_RESID_PREFIX = "resid."
_POST_ATTN_PREFIX = "post_attn."
_ATTN_IN_PREFIX = "attn_in."
_ATTN_OUT_PREFIX = "attn_out."
_MLP_IN_PREFIX = "mlp_in."
_MLP_HIDDEN_PREFIX = "mlp_hidden."
_SITE_OUTPUT_SUFFIX = ".out"


def resid_tap_key(boundary: int) -> str:
    """Raw residual boundary: 0 enters the first block; ``n_layer`` exits the last."""
    return f"{_RESID_PREFIX}{boundary}"


def post_attention_tap_key(block: int) -> str:
    """Raw residual after block ``block``'s attention add and before its MLP norm."""
    return f"{_POST_ATTN_PREFIX}{block}"


def attention_input_tap_key(block: int) -> str:
    """Normalized residual consumed by the block's q/k/v projections."""
    return f"{_ATTN_IN_PREFIX}{block}"


def attention_output_tap_key(block: int) -> str:
    """Attention-core output consumed by the block's o projection."""
    return f"{_ATTN_OUT_PREFIX}{block}"


def mlp_input_tap_key(block: int) -> str:
    """Normalized post-attention residual consumed by the block's MLP input projections."""
    return f"{_MLP_IN_PREFIX}{block}"


def mlp_hidden_tap_key(block: int) -> str:
    """Post-nonlinearity MLP activation consumed by the block's down projection."""
    return f"{_MLP_HIDDEN_PREFIX}{block}"


def site_output_tap_key(site: str) -> str:
    """Linear output of ``site``, before a following bias, nonlinearity, or residual add."""
    return f"{site}{_SITE_OUTPUT_SUFFIX}"


@dataclass(frozen=True, kw_only=True)
class ResidualBoundary:
    boundary: int


@dataclass(frozen=True, kw_only=True)
class PostAttentionResidual:
    block: int


@dataclass(frozen=True, kw_only=True)
class BlockTap:
    name: str
    block: int
    width: int


@dataclass(frozen=True, kw_only=True)
class SiteOutput:
    name: str
    block: int


TransformerPoint = ResidualBoundary | PostAttentionResidual | BlockTap | SiteOutput


def _parse_numbered_key(key: str, prefix: str, upper: int, noun: str) -> int:
    suffix = key.removeprefix(prefix)
    assert suffix.isdigit(), f"malformed {noun} {key!r}: expected {prefix}{{integer}}"
    index = int(suffix)
    assert 0 <= index <= upper, f"{noun} {key!r} out of range: expected 0..{upper}"
    return index


@dataclass(frozen=True, kw_only=True)
class TransformerTapGrammar:
    """One transformer target's closed, one-to-one point grammar.

    ``family`` owns matrix-output syntax. The four named block taps are the
    transformer's actual forward vectors, not aliases derived from matrix sites. Every
    query parses fail-closed; core never imports or interprets this type.
    """

    family: ArchFamily
    n_layer: int
    d_resid: int
    d_attention_output: int
    d_mlp_hidden: int
    d_out_of: Callable[[str], int]

    @staticmethod
    def _block_tap_key(name: str, block: int) -> str:
        return f"{name}.{block}"

    def _block_tap_widths(self) -> dict[str, int]:
        return {
            "attn_in": self.d_resid,
            "attn_out": self.d_attention_output,
            "mlp_in": self.d_resid,
            "mlp_hidden": self.d_mlp_hidden,
        }

    def _block_taps(self) -> dict[str, BlockTap]:
        return {
            self._block_tap_key(name, block): BlockTap(name=name, block=block, width=width)
            for block in range(self.n_layer)
            for name, width in self._block_tap_widths().items()
        }

    def block_tap_keys(self, blocks: tuple[int, ...]) -> tuple[str, ...]:
        """Return every distinct block tap in the requested blocks."""
        assert len(set(blocks)) == len(blocks), blocks
        assert all(0 <= block < self.n_layer for block in blocks), (blocks, self.n_layer)
        return tuple(
            self._block_tap_key(name, block)
            for block in blocks
            for name in self._block_tap_widths()
        )

    def parse(self, key: str) -> TransformerPoint:
        if key.startswith(_RESID_PREFIX):
            return ResidualBoundary(
                boundary=_parse_numbered_key(key, _RESID_PREFIX, self.n_layer, "residual boundary")
            )
        if key.startswith(_POST_ATTN_PREFIX):
            block = _parse_numbered_key(
                key, _POST_ATTN_PREFIX, self.n_layer - 1, "post-attention residual"
            )
            return PostAttentionResidual(block=block)
        if key.endswith(_SITE_OUTPUT_SUFFIX):
            name = key.removesuffix(_SITE_OUTPUT_SUFFIX)
            block, _kind = self.family.parse(name)
            self._assert_block(name, block)
            return SiteOutput(name=name, block=block)
        point = self._block_taps().get(key)
        assert point is not None, f"unknown transformer activation {key!r}"
        return point

    def _assert_block(self, key: str, block: int) -> None:
        assert 0 <= block < self.n_layer, (
            f"site point {key!r} out of range: target blocks are 0..{self.n_layer - 1}"
        )

    def resolve[SourceT: Hashable](
        self, keys: tuple[str, ...], source_of: Callable[[TransformerPoint], SourceT]
    ) -> tuple[SourceT, ...]:
        """Validate ``keys`` and return their physical sources in request order."""
        sources = tuple(source_of(self.parse(key)) for key in keys)
        assert len(set(sources)) == len(sources), (
            "multiple capture keys name one physical activation",
            keys,
            sources,
        )
        return sources

    def block_of(self, key: str) -> int:
        match self.parse(key):
            case ResidualBoundary(boundary=boundary):
                return boundary
            case (
                PostAttentionResidual(block=block) | BlockTap(block=block) | SiteOutput(block=block)
            ):
                return block

    def assert_hidden_acts_reconstruction_points(
        self,
        keys: tuple[str, ...],
        decomposed_sites: tuple[str, ...],
        same_block_dependencies: Callable[[TransformerPoint], frozenset[str]],
    ) -> None:
        """Reject points no decomposed matrix can influence.

        A point can move if any earlier block is decomposed, or if a decomposed matrix in
        the same block is in that point's target-specific dependency set. Capture remains
        more permissive: invariant points are useful diagnostics, but putting one in a mean
        loss silently weakens its coefficient.
        """
        sites_by_block: dict[int, set[str]] = {}
        for site in decomposed_sites:
            block, kind = self.family.parse(site)
            sites_by_block.setdefault(block, set()).add(kind)

        dead: list[str] = []
        for key in keys:
            point = self.parse(key)
            match point:
                case ResidualBoundary(boundary=block):
                    dependencies = frozenset()
                case (
                    PostAttentionResidual(block=block)
                    | BlockTap(block=block)
                    | SiteOutput(block=block)
                ):
                    dependencies = same_block_dependencies(point)
            changed_before = any(site_block < block for site_block in sites_by_block)
            changed_here = bool(sites_by_block.get(block, set()) & dependencies)
            if not changed_before and not changed_here:
                dead.append(key)

        assert not dead, (
            f"hidden_acts_reconstruction points {tuple(dead)} cannot change under masking and would "
            "contribute guaranteed zeros to the point mean"
        )

    def width_of(self, key: str) -> int:
        match self.parse(key):
            case ResidualBoundary() | PostAttentionResidual():
                return self.d_resid
            case BlockTap(width=width):
                return width
            case SiteOutput(name=name):
                return self.d_out_of(name)
