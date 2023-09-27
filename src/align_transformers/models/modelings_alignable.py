from .llama.modelings_alignable_llama import AlignableLlamaForCausalLM
from .gpt2.modelings_alignable_gpt2 import AlignableGPT2LMHeadModel
from transformers import AutoConfig

class AutoAlignableModel:
    @staticmethod
    def from_pretrained(model_path, alignment_config=None, torch_dtype=None, cache_dir="~/.cache"):
        """
        Currently, we should always assume we are aligning a pretrained model.
        """
        if AutoConfig.from_pretrained(model_path).architectures[0] in ["AlignableLlamaForCausalLM", "LLaMAForCausalLM", "LlamaForCausalLM"]:
            model_class = AlignableLlamaForCausalLM
        elif AutoConfig.from_pretrained(model_path).architectures[0] == "GPT2LMHeadModel":
            model_class = AlignableGPT2LMHeadModel
        return model_class.from_pretrained(
            model_path,
            alignment_config=alignment_config,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir
        )
