"""Native Hugging Face LLaMA adapter for DictLLM checkpoints."""

from transformers.models.llama.modeling_llama import LlamaForCausalLM

from .modeling import install_empty_dictllm


class DictLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        if getattr(config, "dictllm_enabled", False):
            install_empty_dictllm(self)

