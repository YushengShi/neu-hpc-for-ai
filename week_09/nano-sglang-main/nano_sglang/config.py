"""Configuration for nano-sglang."""

from dataclasses import dataclass


@dataclass
class Config:
    model_path: str
    max_batch_size: int = 64
    max_seq_len: int = 2048
    block_size: int = 16  # tokens per block (used in paged KV cache)
    device: str = "cuda"
    dtype: str = "float16"
