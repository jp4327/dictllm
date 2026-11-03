"""Native LLaMA-only adapters for the unchanged identity quantizer."""

from __future__ import annotations

import weakref

PACKED_MODEL_TYPES = {"llama": "identity_packed_share_llama"}


def architecture(config):
    if (config.get("model_type") != "llama" or not config.get("dictllm_enabled")
            or "DictLlamaForCausalLM" not in config.get("architectures", [])):
        raise ValueError("Expected a native DictLLM LLaMA factorized checkpoint.")
    if config.get("dictllm_quantization", {}).get("packed", False):
        raise ValueError("Rotation-based legacy packed checkpoints are not supported.")
    return "native_llama"


def base_config(config):
    from transformers import LlamaConfig

    architecture(config)
    return LlamaConfig.from_dict(config)


def restore_base_config(config):
    metadata = config.get("offline_nvfp4", {})
    if (metadata.get("base_model_type") != "llama"
            or config.get("model_type") != PACKED_MODEL_TYPES["llama"]
            or metadata.get("requires_custom_loader") is not True
            or metadata.get("transform_class") != "identity"
            or not isinstance(metadata.get("base_architectures"), list)):
        raise ValueError("Expected a verified identity-packed DictLLM LLaMA checkpoint.")
    restored = dict(config, model_type="llama", architectures=metadata["base_architectures"])
    architecture(restored)
    return restored


def model_class(config):
    from dictllm.modeling_llama import DictLlamaForCausalLM

    architecture(config)
    return DictLlamaForCausalLM


def load_tokenizer(source, config):
    from dictllm.models import load_tokenizer as load_native_tokenizer

    architecture(config)
    return load_native_tokenizer(source)


def factor_modules(model, targets):
    from dictllm.modeling import FactorizedLinear

    architecture(model.config.to_dict())
    if "basis" in targets:
        for key, module in model.model.dictllm_dictionaries.items():
            name = f"model.dictllm_dictionaries.{key}.weight"
            yield name, module, "basis", [name]
    if "coefficient" in targets:
        for name, module in model.named_modules():
            if isinstance(module, FactorizedLinear):
                key = name + ".coefficient.weight"
                yield key, module.coefficient, "coefficient", [key]


def capture_dictionary_bindings(model):
    from dictllm.modeling import FactorizedLinear

    paths = {id(module): f"model.dictllm_dictionaries.{key}"
             for key, module in model.model.dictllm_dictionaries.items()}
    return [(module, paths[id(module.dictionary)]) for module in model.modules()
            if isinstance(module, FactorizedLinear)]


def restore_dictionary_bindings(model, bindings):
    for module, path in bindings:
        object.__setattr__(module, "_dictionary_ref", weakref.ref(model.get_submodule(path)))


def restore_native_rotary_buffers(model):
    import torch

    for module in model.modules():
        if (hasattr(module, "inv_freq") and module.inv_freq.device.type == "meta"
                and hasattr(module, "compute_default_rope_parameters")):
            fresh = type(module)(config=module.config, device=torch.device("cpu"))
            for name, buffer in fresh.named_buffers(recurse=False):
                module.register_buffer(name, buffer, persistent=name not in fresh._non_persistent_buffers_set)
            module.attention_scaling = fresh.attention_scaling
