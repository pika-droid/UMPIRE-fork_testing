"""
Transformers >= 4.38 compatibility monkey-patches for LLaVA architectures.

Strips unsupported generation kwargs (cache_position, num_logits_to_keep,
matryoshka_vis_token_scale) to prevent TypeError on modern HuggingFace Transformers.
"""

from __future__ import annotations

import functools
from typing import Any

import torch


def patch_transformers_quantization() -> None:
    """Strips load_in_4bit/8bit kwargs if quantization_config is passed (transformers >= 4.38 conflict)."""
    try:
        import transformers

        if hasattr(transformers, "PreTrainedModel") and hasattr(
            transformers.PreTrainedModel, "from_pretrained"
        ):
            orig_from_pretrained = transformers.PreTrainedModel.from_pretrained.__func__

            @classmethod
            def safe_from_pretrained(
                cls: Any, pretrained_model_name_or_path: Any, *model_args: Any, **kwargs: Any
            ) -> Any:
                if "quantization_config" in kwargs:
                    kwargs.pop("load_in_4bit", None)
                    kwargs.pop("load_in_8bit", None)
                return orig_from_pretrained(
                    cls, pretrained_model_name_or_path, *model_args, **kwargs
                )

            setattr(transformers.PreTrainedModel, "from_pretrained", safe_from_pretrained)  # noqa: B010
    except Exception:
        pass


def apply_transformers_compatibility_patches(model: Any) -> None:
    """
    Applies 3 essential monkey-patches to prevent transformers >= 4.38 GenerationMixin crashes:
    1. Removes cache_position and num_logits_to_keep from forward kwargs.
    2. Removes cache_position and num_logits_to_keep from prepare_inputs_for_generation.
    3. Protects matryoshka_vis_token_scale in _validate_model_kwargs.
    """
    orig_forward = model.forward

    @functools.wraps(orig_forward)
    def patched_forward(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop("cache_position", None)
        kwargs.pop("num_logits_to_keep", None)
        out = orig_forward(*args, **kwargs)
        if hasattr(out, "logits") and out.logits is not None and torch.is_tensor(out.logits):
            out.logits = torch.nan_to_num(out.logits, nan=-1e4, posinf=1e4, neginf=-1e4)
        return out

    model.forward = patched_forward

    orig_prep = model.prepare_inputs_for_generation

    @functools.wraps(orig_prep)
    def patched_prep(input_ids: Any, past_key_values: Any = None, **kwargs: Any) -> Any:
        model_inputs = orig_prep(input_ids, past_key_values=past_key_values, **kwargs)
        model_inputs.pop("cache_position", None)
        model_inputs.pop("num_logits_to_keep", None)
        return model_inputs

    model.prepare_inputs_for_generation = patched_prep

    if hasattr(model, "_validate_model_kwargs"):
        orig_validate = model._validate_model_kwargs

        @functools.wraps(orig_validate)
        def patched_validate(model_kwargs: dict[str, Any]) -> Any:
            kw_copy = model_kwargs.copy()
            kw_copy.pop("matryoshka_vis_token_scale", None)
            return orig_validate(kw_copy)

        model._validate_model_kwargs = patched_validate
