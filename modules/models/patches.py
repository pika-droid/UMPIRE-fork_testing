"""
Transformers >= 4.38 compatibility monkey-patches for LLaVA architectures.

Strips unsupported generation kwargs (cache_position, num_logits_to_keep,
matryoshka_vis_token_scale) to prevent TypeError on modern HuggingFace Transformers.
"""

from __future__ import annotations

import contextlib
import functools
from typing import Any

import torch


def _make_safe_rotary_forward(orig_fn: Any) -> Any:
    @functools.wraps(orig_fn)
    def safe_rotary_forward(*args: Any, **kwargs: Any) -> Any:
        if not args:
            return orig_fn(*args, **kwargs)

        if hasattr(args[0], "inv_freq"):
            self_obj = args[0]
            x = args[1] if len(args) > 1 else None
            rest_args = args[2:]
        else:
            self_obj = getattr(orig_fn, "__self__", None)
            x = args[0]
            rest_args = args[1:]

        target_dev = getattr(x, "device", None)
        if target_dev is None and len(rest_args) > 0:
            target_dev = getattr(rest_args[0], "device", None)
        if target_dev is None and "position_ids" in kwargs:
            target_dev = getattr(kwargs["position_ids"], "device", None)

        if (
            self_obj is not None
            and target_dev is not None
            and hasattr(self_obj, "inv_freq")
            and self_obj.inv_freq is not None
            and self_obj.inv_freq.device != target_dev
        ):
            self_obj.inv_freq = self_obj.inv_freq.to(target_dev)

        return orig_fn(*args, **kwargs)

    safe_rotary_forward._is_device_safe = True  # type: ignore[attr-defined]
    return safe_rotary_forward


def patch_llama_rotary_embedding() -> None:
    """
    Ensures inv_freq buffer in LlamaRotaryEmbedding is dynamically moved
    to the target execution device (e.g. cuda:0) if loaded on CPU during 4-bit quantization.
    """
    with contextlib.suppress(Exception):
        from transformers.models.llama import modeling_llama

        classes_to_patch = [
            getattr(modeling_llama, name)
            for name in [
                "LlamaRotaryEmbedding",
                "LlamaLinearScalingRotaryEmbedding",
                "LlamaDynamicNTKScalingRotaryEmbedding",
            ]
            if hasattr(modeling_llama, name)
        ]
        for cls in classes_to_patch:
            if hasattr(cls, "forward"):
                orig_fwd = cls.forward
                if getattr(orig_fwd, "_is_device_safe", False):
                    continue
                cls.forward = _make_safe_rotary_forward(orig_fwd)


def patch_transformers_quantization() -> None:
    """Strips load_in_4bit/8bit kwargs if quantization_config is passed (transformers >= 4.38 conflict)."""
    patch_llama_rotary_embedding()
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

            if hasattr(transformers.PreTrainedModel, "to"):
                orig_ptm_to = transformers.PreTrainedModel.to

                def safe_ptm_to(self: Any, *args: Any, **kwargs: Any) -> Any:
                    try:
                        return orig_ptm_to(self, *args, **kwargs)
                    except ValueError as e:
                        err_msg = str(e).lower()
                        if (
                            "not supported for `4-bit`" in err_msg
                            or "not supported for `8-bit`" in err_msg
                            or "bitsandbytes" in err_msg
                        ):
                            return self
                        raise

                setattr(transformers.PreTrainedModel, "to", safe_ptm_to)  # noqa: B010

        orig_to = torch.nn.Module.to

        def safe_to(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                return orig_to(self, *args, **kwargs)
            except ValueError as e:
                err_msg = str(e).lower()
                if (
                    "not supported for `4-bit`" in err_msg
                    or "not supported for `8-bit`" in err_msg
                    or "bitsandbytes" in err_msg
                ):
                    return self
                raise

        torch.nn.Module.to = safe_to  # type: ignore[assignment]
    except Exception:
        pass


def apply_transformers_compatibility_patches(model: Any) -> None:
    """
    Applies essential monkey-patches to prevent transformers >= 4.38 GenerationMixin crashes:
    1. Removes cache_position and num_logits_to_keep from forward kwargs.
    2. Removes cache_position and num_logits_to_keep from prepare_inputs_for_generation.
    3. Protects matryoshka_vis_token_scale in _validate_model_kwargs.
    4. Ensures rotary embedding inv_freq buffers are on the model's active device.
    """
    patch_llama_rotary_embedding()

    target_device = None
    if hasattr(model, "parameters"):
        for p in model.parameters():
            if hasattr(p, "device") and p.device.type == "cuda":
                target_device = p.device
                break
    if target_device is None and torch.cuda.is_available():
        target_device = torch.device("cuda:0")

    if target_device is not None and hasattr(model, "modules"):
        for m in model.modules():
            if hasattr(m, "inv_freq") and getattr(m, "inv_freq", None) is not None:
                with contextlib.suppress(Exception):
                    m.inv_freq = m.inv_freq.to(target_device)

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
