"""Token geometry for equal-coverage windows and complete receiver-step patches.

This module has no tokenizer or model dependency. Callers pass token IDs from the
actual rendered scoring prompt and its offset-aligned, end-exclusive step span.
``full_step`` always targets every receiver-step token, including when the donor
step is shorter. Its donor mapping is an operational alignment, not a claim of
semantic equivalence. It never caps either step or shortens the receiver window.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from numbers import Integral
from typing import Any

WINDOW_NAMES = ("last_token", "suffix_k", "divergence_k", "full_step")
DEFAULT_K_CANDIDATES = (4, 8, 16, 32)
DONOR_MAPPING = "endpoint_aligned_nearest_half_up; singleton_receiver_uses_donor_last"
IDENTITY_RULE = "same_absolute_position_and_identical_complete_causal_token_prefix"


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _span(value: Sequence[int], *, name: str, token_count: int | None = None) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain start and end")
    start, end = (_integer(v, name) for v in value)
    if start < 0 or end <= start or (token_count is not None and end > token_count):
        raise ValueError(f"{name} must be nonempty, end-exclusive, and within its token sequence")
    return start, end


def common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    """Count the identical leading token IDs, without crediting matching suffixes."""

    for position, (left_token, right_token) in enumerate(zip(left, right, strict=False)):
        if left_token != right_token:
            return position
    return min(len(left), len(right))


def monotone_donor_positions(
    receiver_positions: Sequence[int],
    *,
    receiver_span: tuple[int, int],
    donor_span: tuple[int, int],
) -> tuple[int, ...]:
    """Map unique sorted targets to nearest donor positions with aligned endpoints.

    A receiver offset ``j`` in a length ``R`` span maps to
    ``floor(j * (D - 1) / (R - 1) + 1/2)`` in a length ``D`` donor span.
    Integer arithmetic fixes half-ties upward and avoids floating-point drift.
    For ``R=1`` use the donor's final token, preserving the last-token interface.
    A shorter donor can supply the same vector to multiple distinct targets;
    capture its unique positions once, then gather in this returned order.
    """

    receiver_start, receiver_end = _span(receiver_span, name="receiver_span")
    donor_start, donor_end = _span(donor_span, name="donor_span")
    positions = tuple(_integer(position, "receiver position") for position in receiver_positions)
    if not positions:
        raise ValueError("receiver_positions must be nonempty")
    if any(left >= right for left, right in pairwise(positions)):
        raise ValueError("receiver_positions must be strictly increasing with no duplicate target")
    if positions[0] < receiver_start or positions[-1] >= receiver_end:
        raise ValueError("receiver_positions lie outside receiver_span")
    receiver_length = receiver_end - receiver_start
    donor_length = donor_end - donor_start
    if receiver_length == 1:
        return (donor_end - 1,)
    denominator = receiver_length - 1
    return tuple(
        donor_start
        + (2 * (position - receiver_start) * (donor_length - 1) + denominator)
        // (2 * denominator)
        for position in positions
    )


@dataclass(frozen=True)
class WindowGeometry:
    name: str
    requested_k: int
    receiver_step_span: tuple[int, int]
    donor_step_span: tuple[int, int]
    receiver_positions: tuple[int, ...]
    donor_positions: tuple[int, ...]
    identity_positions: tuple[int, ...]
    common_prefix_length: int
    divergence_position: int | None
    status: str = "ok"
    reason: str | None = None
    donor_mapping: str = DONOR_MAPPING
    identity_rule: str = IDENTITY_RULE
    overlap_windows: tuple[str, ...] = ()
    identical_windows: tuple[str, ...] = ()
    overlap_counts: tuple[tuple[str, int], ...] = ()

    @property
    def applicable(self) -> bool:
        return self.status == "ok"

    @property
    def coverage(self) -> int:
        return len(self.receiver_positions)

    @property
    def donor_unique_positions(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.donor_positions)))

    @property
    def donor_repeat_count(self) -> int:
        return len(self.donor_positions) - len(self.donor_unique_positions)

    @property
    def nonidentity_positions(self) -> tuple[int, ...]:
        """Positions not proven identity; actual activation change still needs measurement."""

        identity = set(self.identity_positions)
        return tuple(position for position in self.receiver_positions if position not in identity)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            applicable=self.applicable,
            coverage=self.coverage,
            donor_unique_positions=self.donor_unique_positions,
            donor_repeat_count=self.donor_repeat_count,
            nonidentity_positions=self.nonidentity_positions,
            receiver_step_length=self.receiver_step_span[1] - self.receiver_step_span[0],
            donor_step_length=self.donor_step_span[1] - self.donor_step_span[0],
            receiver_full_step_covered=(
                self.applicable
                and self.receiver_positions
                == tuple(range(self.receiver_step_span[0], self.receiver_step_span[1]))
            ),
        )
        return payload


def _centered_window(span: tuple[int, int], divergence: int, k: int) -> tuple[int, int]:
    # An even k puts the anchor at the left of the two central positions.
    anchor = min(divergence, span[1] - 1)
    start = min(max(anchor - (k - 1) // 2, span[0]), span[1] - k)
    return start, start + k


def build_pair_windows(
    receiver_token_ids: Sequence[int],
    donor_token_ids: Sequence[int],
    *,
    receiver_step_span: tuple[int, int],
    donor_step_span: tuple[int, int],
    k: int,
) -> dict[str, WindowGeometry]:
    """Construct four windows for one direction of one clean/error pair.

    ``suffix_k`` and ``divergence_k`` are available only when *both* steps have
    at least k tokens. Each then supplies exactly k receiver and k donor tokens.
    The divergence anchor is the first difference in the complete causal token
    sequences, not a repeated token match later in the step. It must lie within
    both step spans or at the shorter step's right boundary. An identical step,
    or divergence solely outside the target step, is recorded as not applicable.
    Both centered windows are clipped *within* the respective step while keeping
    k tokens; clipping and overlap do not justify describing them as distinct
    positions when their actual target/donor coordinates coincide.

    Identity certification assumes the same deterministic model, position IDs,
    and causal attention convention for both unpadded sequences. IDs must include
    the entire model-facing prefix. For self/matched controls, inherit these
    receiver targets rather than recomputing a self-pair divergence window.
    """

    k = _integer(k, "k")
    if k < 1:
        raise ValueError("k must be positive")
    receiver_span = _span(
        receiver_step_span, name="receiver_step_span", token_count=len(receiver_token_ids)
    )
    donor_span = _span(donor_step_span, name="donor_step_span", token_count=len(donor_token_ids))
    prefix = common_prefix_length(receiver_token_ids, donor_token_ids)
    divergence = (
        None
        if prefix == len(receiver_token_ids) == len(donor_token_ids)
        else prefix
    )
    receiver_length = receiver_span[1] - receiver_span[0]
    donor_length = donor_span[1] - donor_span[0]
    windows: dict[str, WindowGeometry] = {}

    def add(
        name: str,
        target_span: tuple[int, int] | None,
        source_span: tuple[int, int] | None,
        reason: str | None = None,
    ) -> None:
        targets = () if target_span is None else tuple(range(*target_span))
        sources = (
            ()
            if target_span is None or source_span is None
            else monotone_donor_positions(
                targets, receiver_span=target_span, donor_span=source_span
            )
        )
        identity = tuple(
            target
            for target, source in zip(targets, sources, strict=True)
            if target == source and target < prefix
        )
        windows[name] = WindowGeometry(
            name=name,
            requested_k=k,
            receiver_step_span=receiver_span,
            donor_step_span=donor_span,
            receiver_positions=targets,
            donor_positions=sources,
            identity_positions=identity,
            common_prefix_length=prefix,
            divergence_position=divergence,
            status="ok" if reason is None else "not_applicable",
            reason=reason,
        )

    add("last_token", (receiver_span[1] - 1, receiver_span[1]), (donor_span[1] - 1, donor_span[1]))
    if min(receiver_length, donor_length) < k:
        reason = "receiver_or_donor_step_shorter_than_k"
        add("suffix_k", None, None, reason)
        add("divergence_k", None, None, reason)
    else:
        add("suffix_k", (receiver_span[1] - k, receiver_span[1]), (donor_span[1] - k, donor_span[1]))
        if divergence is None:
            add("divergence_k", None, None, "no_token_divergence")
        elif not (
            receiver_span[0] <= divergence <= receiver_span[1]
            and donor_span[0] <= divergence <= donor_span[1]
            and divergence < max(receiver_span[1], donor_span[1])
        ):
            add("divergence_k", None, None, "first_causal_divergence_outside_target_step")
        else:
            add(
                "divergence_k",
                _centered_window(receiver_span, divergence, k),
                _centered_window(donor_span, divergence, k),
            )
    add("full_step", receiver_span, donor_span)

    for name, window in windows.items():
        if not window.applicable:
            continue
        overlaps = tuple(
            (other_name, len(set(window.receiver_positions) & set(other.receiver_positions)))
            for other_name, other in windows.items()
            if other_name != name and other.applicable
        )
        windows[name] = replace(
            window,
            overlap_windows=tuple(other for other, count in overlaps if count),
            overlap_counts=overlaps,
            identical_windows=tuple(
                other_name
                for other_name, other in windows.items()
                if other_name != name
                and other.applicable
                and window.receiver_positions == other.receiver_positions
                and window.donor_positions == other.donor_positions
            ),
        )
    return windows


def calibrate_window_k(
    length_pairs: Iterable[tuple[int, int]],
    *,
    candidates: Sequence[int] = DEFAULT_K_CANDIDATES,
    min_pair_coverage: float = 0.9,
) -> dict[str, Any]:
    """Choose k independently per actual model tokenizer before any effects exist.

    Choose the largest candidate with both step lengths >= k in at least the
    requested fraction of pairs. If no candidate qualifies, use k=1 and mark the
    fallback. This records length eligibility only; divergence NA and window
    overlap must still be reported from :func:`build_pair_windows`. Callers must
    verify the discovery cohort and model/tokenizer revisions themselves.
    """

    if not 0 < min_pair_coverage <= 1:
        raise ValueError("min_pair_coverage must lie in (0, 1]")
    candidate_values = tuple(sorted({_integer(value, "candidate k") for value in candidates}))
    if not candidate_values or candidate_values[0] < 1:
        raise ValueError("candidates must contain positive integers")
    pairs: list[tuple[int, int]] = []
    for pair in length_pairs:
        if len(pair) != 2:
            raise ValueError("each length pair must contain two lengths")
        clean_length, error_length = (_integer(value, "step length") for value in pair)
        if min(clean_length, error_length) < 1:
            raise ValueError("step lengths must be positive")
        pairs.append((clean_length, error_length))
    if not pairs:
        raise ValueError("cannot calibrate k without real tokenized length pairs")
    counts = {
        candidate: sum(min(clean, error) >= candidate for clean, error in pairs)
        for candidate in candidate_values
    }
    eligible = [
        candidate
        for candidate in candidate_values
        if counts[candidate] / len(pairs) >= min_pair_coverage
    ]
    selected = max(eligible) if eligible else 1
    selected_count = sum(min(clean, error) >= selected for clean, error in pairs)
    return {
        "selected_k": selected,
        "pair_count": len(pairs),
        "min_pair_coverage": min_pair_coverage,
        "eligible_pair_count": selected_count,
        "coverage": selected_count / len(pairs),
        "fallback_to_one": not eligible,
        "candidate_coverage": [
            {"k": value, "eligible_pair_count": counts[value], "coverage": counts[value] / len(pairs)}
            for value in candidate_values
        ],
        "selection_rule": "largest_candidate_with_both_step_lengths_at_least_k_at_min_pair_coverage",
        "scope": "per_model_real_discovery_tokenization; no effect values used",
    }
