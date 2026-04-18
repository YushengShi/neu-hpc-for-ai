"""Model wrapper for Qwen3."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


class Model:
    def __init__(self, model_path: str, device: str = "cuda", dtype: str = "float16"):
        self.device = device
        self.dtype = getattr(torch, dtype)

        self.config = AutoConfig.from_pretrained(model_path)
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_key_value_heads
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.vocab_size = self.config.vocab_size

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=self.dtype, device_map=device,
        )
        self.model.eval()

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, past_key_values=None,
                position_ids=None, attention_mask=None):
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        return outputs.logits, outputs.past_key_values


class Tokenizer:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id