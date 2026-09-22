"""Native Mistral Small 4 text/chat tokenization and verified byte span alignment.

Chat IDs always come from mistral-common's v15 message encoder, including model
settings. The facade's strings are *registered displays of native IDs*, never HF
chat templates or strings re-encoded by guessing the meaning of control markers.
Tekken's public ``id_to_byte_piece`` provides exact bytes: projecting their byte
ranges onto Python character indices gives true character coverage even when two
byte tokens share one Unicode character. Such overlapping offsets are explicit.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from routeverify.data.prompts import PromptRender, shift_render_into_prompt
from routeverify.data.schema import stable_hash

MISTRAL_NVFP4_MODEL_ID = "mistralai/Mistral-Small-4-119B-2603-NVFP4"
MISTRAL_NVFP4_REVISION = "b1a9048590131d38491bd23a7c9f6ed0962f0358"
MISTRAL_NVFP4_TOKENIZER_SHA256 = "b1272b956bd6edd2d2c674c76896c7661308c9e723997b0afb55ecb429cb5dc7"
MISTRAL_COMMON_VERSION = "1.11.7"


@dataclass(frozen=True)
class NativeTokenDisplay:
    """A reversible display and exact byte/character coverage of native token IDs."""

    input_ids: tuple[int, ...]
    text: str
    offset_mapping: tuple[tuple[int, int], ...]
    byte_offsets: tuple[tuple[int, int], ...]
    exact_char_boundaries: tuple[tuple[bool, bool], ...]
    special_tokens_mask: tuple[int, ...]


@dataclass(frozen=True)
class NativeChatEncoding:
    display: NativeTokenDisplay
    user_char_span: tuple[int, int]
    user_token_span: tuple[int, int]

    @property
    def input_ids(self) -> tuple[int, ...]:
        return self.display.input_ids


@dataclass(frozen=True)
class NativeCandidateEncoding:
    """Complete native chat IDs, with an exclusive candidate-only score interval.

    The input includes the assistant EOS. ``target_ids`` excludes that EOS, model
    settings, and every user token. A caller uses logits[start-1:end-1] for scoring.
    """

    input_ids: tuple[int, ...]
    target_start: int
    target_end: int
    target_ids: tuple[int, ...]
    rendered_text: str

    @property
    def target_token_start(self) -> int:
        return self.target_start

    @property
    def target_token_end(self) -> int:
        return self.target_end


class NativeBatchEncoding(dict[str, Any]):
    """Minimal tensor-moving mapping needed by existing RouteVerify model code."""

    def to(self, device: Any) -> NativeBatchEncoding:
        for name, value in self.items():
            if hasattr(value, "to"):
                self[name] = value.to(device)
        return self


def display_from_token_bytes(tokenizer: Any, token_ids: tuple[int, ...]) -> NativeTokenDisplay:
    """Recover exact byte extents, then character *coverage*, never guessed offsets.

    A byte boundary inside a UTF-8 code point projects outward to cover that code
    point. Adjacent tokens may consequently overlap in their character offsets;
    byte_offsets and exact_char_boundaries preserve the more precise information.
    """
    from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

    pieces = [
        tokenizer.id_to_byte_piece(token, special_token_policy=SpecialTokenPolicy.KEEP)
        for token in token_ids
    ]
    if any(not isinstance(piece, bytes) or not piece for piece in pieces):
        raise ValueError("every displayed native token must have a nonempty byte representation")
    raw = b"".join(pieces)
    text = raw.decode("utf-8", errors="strict")
    native_text = tokenizer.decode(list(token_ids), special_token_policy=SpecialTokenPolicy.KEEP)
    if native_text != text:
        raise ValueError("native token bytes and native decoded display disagree")
    boundaries = [0]
    for char in text:
        boundaries.append(boundaries[-1] + len(char.encode("utf-8")))
    boundary_set = set(boundaries)
    cursor = 0
    offsets = []
    byte_offsets = []
    exact = []
    for piece in pieces:
        end = cursor + len(piece)
        offsets.append((bisect_right(boundaries, cursor) - 1, bisect_left(boundaries, end)))
        byte_offsets.append((cursor, end))
        exact.append((cursor in boundary_set, end in boundary_set))
        cursor = end
    return NativeTokenDisplay(
        input_ids=token_ids,
        text=text,
        offset_mapping=tuple(offsets),
        byte_offsets=tuple(byte_offsets),
        exact_char_boundaries=tuple(exact),
        special_tokens_mask=tuple(int(token < tokenizer.num_special_tokens) for token in token_ids),
    )


class NativeMistralTokenizer:
    """A pinned native v15 encoder and a ``tokenize_render`` compatible facade.

    ``from_file`` performs no network access. Downloading an authorized tokenizer
    is a separate operation. ``__call__`` accepts only previously registered chat
    or candidate displays, rejecting arbitrary control-token strings; callers
    must use ``chat_completion_render`` or ``encode_chat_prompt`` first.
    """

    def __init__(
        self,
        tokenizer_path: str | Path,
        *,
        expected_sha256: str = MISTRAL_NVFP4_TOKENIZER_SHA256,
        reasoning_effort: str = "none",
        expected_mistral_common_version: str = MISTRAL_COMMON_VERSION,
    ) -> None:
        path = Path(tokenizer_path)
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_sha256:
            raise ValueError(
                f"Mistral tokenizer SHA256 mismatch: {actual_hash} != {expected_sha256}"
            )
        try:
            from mistral_common.protocol.instruct.normalize import get_normalizer
            from mistral_common.protocol.instruct.validator import ValidationMode, get_validator
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        except ImportError as exc:
            raise RuntimeError(
                f"Install mistral-common=={MISTRAL_COMMON_VERSION} for native Mistral encoding"
            ) from exc
        version = importlib.metadata.version("mistral-common")
        if version != expected_mistral_common_version:
            raise ValueError(
                f"mistral-common version mismatch: {version} != {expected_mistral_common_version}"
            )
        self.native = MistralTokenizer.from_file(path, mode=ValidationMode.test)
        self.base = self.native.instruct_tokenizer.tokenizer
        if self.native.version.value != "v15":
            raise ValueError("Mistral Small 4 requires the pinned native v15 tokenizer")
        if not callable(getattr(self.base, "id_to_byte_piece", None)):
            raise TypeError("native byte-piece API is required; fabricated offsets are unsupported")
        # The same native instruct tokenizer has a finetuning validator only to
        # encode a *complete* assistant answer plus EOS; no model training occurs.
        self._complete_native = MistralTokenizer(
            instruct_tokenizer=self.native.instruct_tokenizer,
            validator=get_validator(self.native.version, mode=ValidationMode.finetuning),
            request_normalizer=get_normalizer(
                self.native.version, self.base.model_settings_builder
            ),
        )
        self.reasoning_effort = reasoning_effort
        self.source_path = path
        self.source_sha256 = actual_hash
        self.mistral_common_version = version
        self.bos_token_id = self.base.bos_id
        self.eos_token_id = self.base.eos_id
        self.pad_token_id = self.base.pad_id
        self._chat_cache: dict[str, NativeChatEncoding] = {}
        self._candidate_cache: dict[tuple[str, str], NativeCandidateEncoding] = {}
        self._displays: dict[str, NativeTokenDisplay] = {}

    @classmethod
    def from_file(cls, tokenizer_path: str | Path, **kwargs: Any) -> NativeMistralTokenizer:
        return cls(tokenizer_path, **kwargs)

    def _register(self, ids: tuple[int, ...]) -> NativeTokenDisplay:
        display = display_from_token_bytes(self.base, ids)
        prior = self._displays.get(display.text)
        if prior is not None and prior.input_ids != ids:
            raise ValueError(
                "two different native token sequences have the same display; refusing ambiguous string lookup"
            )
        self._displays[display.text] = display
        return display

    def chat_encoding(self, raw_user_text: str) -> NativeChatEncoding:
        if not isinstance(raw_user_text, str) or not raw_user_text:
            raise ValueError("raw user text must be a nonempty string")
        cached = self._chat_cache.get(raw_user_text)
        if cached is not None:
            return cached
        from mistral_common.protocol.instruct.messages import UserMessage
        from mistral_common.protocol.instruct.request import ChatCompletionRequest

        result = self.native.encode_chat_completion(
            ChatCompletionRequest(
                messages=[UserMessage(content=raw_user_text)],
                reasoning_effort=self.reasoning_effort,
                truncate_for_context_length=False,
            )
        )
        ids = tuple(result.tokens)
        begin_id = self.base.get_special_token("[INST]")
        end_id = self.base.get_special_token("[/INST]")
        begin_indices = [index for index, token in enumerate(ids) if token == begin_id]
        end_indices = [index for index, token in enumerate(ids) if token == end_id]
        if len(begin_indices) != 1 or len(end_indices) != 1:
            raise ValueError("single-user native chat must contain exactly one INST delimiter pair")
        start, end = begin_indices[0] + 1, end_indices[0]
        raw_ids = tuple(self.base.encode(raw_user_text, bos=False, eos=False))
        if ids[start:end] != raw_ids:
            raise ValueError(
                "native chat changed raw user tokenization; exact span mapping cannot be assumed"
            )
        display = self._register(ids)
        char_start = display.offset_mapping[start][0]
        char_end = display.offset_mapping[end - 1][1]
        if (
            display.text[char_start:char_end] != raw_user_text
            or not display.exact_char_boundaries[start][0]
            or not display.exact_char_boundaries[end - 1][1]
        ):
            raise ValueError(
                "native user-text token slice does not match exact raw character boundaries"
            )
        encoding = NativeChatEncoding(
            display=display, user_char_span=(char_start, char_end), user_token_span=(start, end)
        )
        self._chat_cache[raw_user_text] = encoding
        return encoding

    def encode_chat_prompt(self, raw_user_text: str) -> tuple[int, ...]:
        return self.chat_encoding(raw_user_text).input_ids

    def chat_completion_render(self, render: PromptRender) -> PromptRender:
        for start, end in (
            *render.step_char_spans,
            *(() if render.query_char_span is None else (render.query_char_span,)),
        ):
            if not (0 <= start < end <= len(render.prompt)):
                raise ValueError("render contains a character span outside the raw prompt")
        encoding = self.chat_encoding(render.prompt)
        return shift_render_into_prompt(
            render, prompt=encoding.display.text, offset=encoding.user_char_span[0]
        )

    def encode_candidate(self, raw_user_text: str, candidate: str) -> NativeCandidateEncoding:
        if not isinstance(candidate, str) or not candidate:
            raise ValueError("candidate must be a nonempty string")
        key = (raw_user_text, candidate)
        cached = self._candidate_cache.get(key)
        if cached is not None:
            return cached
        from mistral_common.protocol.instruct.messages import AssistantMessage, UserMessage
        from mistral_common.protocol.instruct.request import ChatCompletionRequest

        prompt_ids = self.encode_chat_prompt(raw_user_text)
        complete = self._complete_native.encode_chat_completion(
            ChatCompletionRequest(
                messages=[UserMessage(content=raw_user_text), AssistantMessage(content=candidate)],
                reasoning_effort=self.reasoning_effort,
                truncate_for_context_length=False,
            )
        )
        ids = tuple(complete.tokens)
        target_ids = tuple(self.base.encode(candidate, bos=False, eos=False))
        if not target_ids or ids != (*prompt_ids, *target_ids, self.eos_token_id):
            raise ValueError(
                "native complete candidate does not preserve the exact chat prefix/candidate/EOS contract"
            )
        start, end = len(prompt_ids), len(prompt_ids) + len(target_ids)
        display = self._register(ids)
        encoding = NativeCandidateEncoding(
            input_ids=ids,
            target_start=start,
            target_end=end,
            target_ids=target_ids,
            rendered_text=display.text,
        )
        self._candidate_cache[key] = encoding
        return encoding

    def render_provenance(
        self, raw_user_text: str, *, include_text: bool = False
    ) -> dict[str, Any]:
        chat = self.chat_encoding(raw_user_text)
        result = {
            "render_family": "mistral_common_native_v15",
            "reasoning_effort": self.reasoning_effort,
            "tokenizer_sha256": self.source_sha256,
            "mistral_common_version": self.mistral_common_version,
            "rendered_prompt_hash": stable_hash({"prompt": chat.display.text}, length=64),
            "input_ids_hash": stable_hash({"input_ids": list(chat.input_ids)}, length=64),
            "user_char_span": list(chat.user_char_span),
            "user_token_span": list(chat.user_token_span),
            "offset_alignment": "exact_native_token_bytes_projected_to_unicode_character_coverage",
            "tokens_with_partial_character_boundaries": sum(
                not all(pair) for pair in chat.display.exact_char_boundaries
            ),
        }
        if include_text:
            result["rendered_prompt_text"] = chat.display.text
        return result

    def __call__(
        self,
        text: str | list[str],
        *,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
        return_tensors: str | None = None,
        padding: bool | str = False,
        **kwargs: Any,
    ) -> NativeBatchEncoding:
        if kwargs:
            raise TypeError(f"unsupported native tokenizer facade arguments: {sorted(kwargs)}")
        if not isinstance(add_special_tokens, bool):
            raise TypeError("add_special_tokens must be boolean")
        # Registered displays already include every native control token, including
        # BOS. Both flag values preserve those explicit IDs and never add another BOS.
        texts = [text] if isinstance(text, str) else text
        if not texts or any(not isinstance(value, str) for value in texts):
            raise ValueError("text must be one registered display or a nonempty list of displays")
        try:
            displays = [self._displays[value] for value in texts]
        except KeyError as exc:
            raise ValueError(
                "unregistered native display: call chat_completion_render/encode_chat_prompt/encode_candidate first"
            ) from exc
        if padding not in (False, True, "longest"):
            raise ValueError("only no padding or padding to the longest native input is supported")
        lengths = [len(display.input_ids) for display in displays]
        max_length = max(lengths)
        if return_tensors not in (None, "pt"):
            raise ValueError("only return_tensors='pt' is supported")
        if return_tensors and not padding and len(set(lengths)) > 1:
            raise ValueError("native tensor batches with unequal lengths require padding=True")
        batch: dict[str, list[Any]] = {"input_ids": [], "attention_mask": []}
        if return_offsets_mapping:
            for field in (
                "offset_mapping",
                "byte_offsets",
                "exact_char_boundaries",
                "special_tokens_mask",
            ):
                batch[field] = []
        for display in displays:
            count = max_length - len(display.input_ids) if padding else 0
            batch["input_ids"].append([*display.input_ids, *([self.pad_token_id] * count)])
            batch["attention_mask"].append([1] * len(display.input_ids) + [0] * count)
            if return_offsets_mapping:
                batch["offset_mapping"].append([*display.offset_mapping, *([(0, 0)] * count)])
                batch["byte_offsets"].append([*display.byte_offsets, *([(0, 0)] * count)])
                batch["exact_char_boundaries"].append(
                    [*display.exact_char_boundaries, *([(True, True)] * count)]
                )
                batch["special_tokens_mask"].append([*display.special_tokens_mask, *([1] * count)])
        if return_tensors == "pt":
            import torch

            return NativeBatchEncoding(
                {name: torch.tensor(values) for name, values in batch.items()}
            )
        if isinstance(text, str):
            return NativeBatchEncoding({name: values[0] for name, values in batch.items()})
        return NativeBatchEncoding(batch)
