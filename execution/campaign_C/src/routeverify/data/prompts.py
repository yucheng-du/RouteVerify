"""Prompt rendering and token span alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from routeverify.data.schema import TokenizedTrace, TraceExample

STEP_BINARY_CANDIDATES = ("CORRECT", "INCORRECT")


@dataclass(frozen=True)
class PromptRender:
    prompt: str
    step_char_spans: tuple[tuple[int, int], ...]
    query_char_span: tuple[int, int] | None = None


class OffsetTokenizer(Protocol):
    def __call__(
        self,
        text: str,
        *,
        return_offsets_mapping: bool = False,
        add_special_tokens: bool = True,
    ) -> dict[str, Any]: ...


def _append_step(lines: list[str], step: str, index: int, cursor: int) -> tuple[int, int, int]:
    prefix = f"{index + 1}. "
    line = f"{prefix}{step.strip()}\n"
    lines.append(line)
    start = cursor + len(prefix)
    end = start + len(step.strip())
    return start, end, cursor + len(line)


def plain_trace(example: TraceExample) -> PromptRender:
    lines = [f"Problem:\n{example.problem.strip()}\n\nSolution:\n"]
    cursor = len(lines[0])
    spans: list[tuple[int, int]] = []
    for index, step in enumerate(example.steps):
        start, end, cursor = _append_step(lines, step, index, cursor)
        spans.append((start, end))
    return PromptRender("".join(lines), tuple(spans))


def step_binary(example: TraceExample, *, step_index: int) -> PromptRender:
    if step_index < 0 or step_index >= len(example.steps):
        raise IndexError("step_index out of range")
    instruction = (
        "You are checking a partial solution. Decide whether the final displayed "
        "solution step is valid given the problem and the previous displayed steps. "
        "Answer exactly one word: CORRECT or INCORRECT.\n\n"
    )
    prefix = [instruction, f"Problem:\n{example.problem.strip()}\n\nSolution so far:\n"]
    cursor = len("".join(prefix))
    spans: list[tuple[int, int]] = []
    for index, step in enumerate(example.steps[: step_index + 1]):
        start, end, cursor = _append_step(prefix, step, index, cursor)
        spans.append((start, end))
    prefix.append("\nAnswer:")
    prompt = "".join(prefix)
    return PromptRender(prompt, tuple(spans), (0, len(instruction.strip())))


def first_error(example: TraceExample) -> PromptRender:
    base = plain_trace(example)
    suffix = (
        "\nFind the first step that contains an error. "
        "Return -1 if all steps are correct; otherwise return the step number only.\nAnswer:"
    )
    query_start = len(base.prompt) + 1
    return PromptRender(base.prompt + suffix, base.step_char_spans, (query_start, len(base.prompt) + len(suffix)))


def shift_render_into_prompt(render: PromptRender, *, prompt: str, offset: int) -> PromptRender:
    """Return render spans shifted into a wrapper prompt containing the original prompt."""

    if offset < 0:
        raise ValueError("offset must be non-negative")
    return PromptRender(
        prompt=prompt,
        step_char_spans=tuple((start + offset, end + offset) for start, end in render.step_char_spans),
        query_char_span=(
            None
            if render.query_char_span is None
            else (render.query_char_span[0] + offset, render.query_char_span[1] + offset)
        ),
    )


def continuation_pair(clean: TraceExample, corrupt: TraceExample) -> tuple[str, str, str]:
    if clean.pair_id != corrupt.pair_id:
        raise ValueError("continuation_pair requires examples from the same pair")
    if corrupt.first_error is None:
        raise ValueError("corrupt example needs first_error")
    boundary = corrupt.first_error
    if clean.steps[:boundary] != corrupt.steps[:boundary]:
        raise ValueError("clean/corrupt prefixes differ before first_error")
    shared = TraceExample(
        sample_id=f"{corrupt.sample_id}/prefix",
        dataset=corrupt.dataset,
        subset=corrupt.subset,
        problem=corrupt.problem,
        steps=corrupt.steps[:boundary] or ("No prior steps.",),
        first_error=None,
        error_steps=(),
        is_all_correct=True,
        pair_id=corrupt.pair_id,
        variant="natural",
    )
    prompt = plain_trace(shared).prompt
    return prompt, clean.steps[boundary], corrupt.steps[boundary]


def align_token_spans(
    *,
    offset_mapping: list[tuple[int, int]],
    step_char_spans: tuple[tuple[int, int], ...],
) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...]]:
    """Map character spans to token spans using tokenizer offset mappings."""

    token_spans: list[tuple[int, int]] = []
    end_tokens: list[int] = []
    for char_start, char_end in step_char_spans:
        indices = [
            index
            for index, (tok_start, tok_end) in enumerate(offset_mapping)
            if tok_end > char_start and tok_start < char_end and tok_end > tok_start
        ]
        if not indices:
            raise ValueError(f"no tokens overlap step span {(char_start, char_end)}")
        token_spans.append((indices[0], indices[-1] + 1))
        end_tokens.append(indices[-1])
    return tuple(token_spans), tuple(end_tokens)


def tokenize_render(
    render: PromptRender,
    *,
    sample_id: str,
    tokenizer: OffsetTokenizer,
    add_special_tokens: bool = True,
) -> TokenizedTrace:
    encoded = tokenizer(
        render.prompt,
        return_offsets_mapping=True,
        add_special_tokens=add_special_tokens,
    )
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    step_token_spans, step_end_tokens = align_token_spans(
        offset_mapping=offsets,
        step_char_spans=render.step_char_spans,
    )
    query_token_span = None
    if render.query_char_span is not None:
        query_token_spans, _ = align_token_spans(
            offset_mapping=offsets,
            step_char_spans=(render.query_char_span,),
        )
        query_token_span = query_token_spans[0]
    return TokenizedTrace(
        sample_id=sample_id,
        input_ids=encoded["input_ids"],
        attention_mask=encoded.get("attention_mask"),
        step_char_spans=render.step_char_spans,
        step_token_spans=step_token_spans,
        step_end_tokens=step_end_tokens,
        query_token_span=query_token_span,
    )
