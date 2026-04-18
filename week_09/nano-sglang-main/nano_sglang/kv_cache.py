"""Part 1: KV Cache

Stores key/value tensors from previous forward passes so we don't
recompute them. Turns O(n^2) decode into O(n).
"""

import torch


class KVCache:
    def __init__(self, num_layers: int, num_heads: int, head_dim: int,
                 max_seq_len: int, max_batch_size: int, device: str = "cuda",
                 dtype: torch.dtype = torch.float16):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size

        # Shape per layer: [max_batch_size, num_heads, max_seq_len, head_dim]
        self.keys = [
            torch.zeros(max_batch_size, num_heads, max_seq_len, head_dim,
                        device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.values = [
            torch.zeros(max_batch_size, num_heads, max_seq_len, head_dim,
                        device=device, dtype=dtype)
            for _ in range(num_layers)
        ]

    def update(self, layer_idx: int, batch_idx: int,
               key: torch.Tensor, value: torch.Tensor, start_pos: int):
        """Write key/value into cache. key/value shape: [1, num_heads, seq_len, head_dim]"""
        seq_len = key.shape[2]
        self.keys[layer_idx][batch_idx, :, start_pos:start_pos + seq_len, :] = key[0]
        self.values[layer_idx][batch_idx, :, start_pos:start_pos + seq_len, :] = value[0]
    
    

    def get(self, layer_idx: int, batch_idx: int, seq_len: int):
        """Read cached key/value. Returns two [1, num_heads, seq_len, head_dim] tensors."""
        k = self.keys[layer_idx][batch_idx:batch_idx + 1, :, :seq_len, :]
        v = self.values[layer_idx][batch_idx:batch_idx + 1, :, :seq_len, :]
        return k, v

    def clear(self, batch_idx: int):
        """Zero out cache for a finished sequence."""
        for layer_idx in range(self.num_layers):
            self.keys[layer_idx][batch_idx].zero_()
            self.values[layer_idx][batch_idx].zero_()