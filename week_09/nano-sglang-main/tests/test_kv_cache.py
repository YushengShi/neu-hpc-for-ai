"""Tests for KV Cache (provided code - run this to verify setup)"""

import torch
from nano_sglang.kv_cache import KVCache


def test_update_and_get():
    cache = KVCache(num_layers=2, num_heads=4, head_dim=32,
                    max_seq_len=128, max_batch_size=4, device="cpu", dtype=torch.float32)
    key = torch.randn(1, 4, 5, 32)
    value = torch.randn(1, 4, 5, 32)
    cache.update(layer_idx=0, batch_idx=0, key=key, value=value, start_pos=0)
    k_out, v_out = cache.get(layer_idx=0, batch_idx=0, seq_len=5)
    assert torch.allclose(k_out, key)
    assert torch.allclose(v_out, value)


def test_append():
    cache = KVCache(num_layers=1, num_heads=2, head_dim=16,
                    max_seq_len=64, max_batch_size=1, device="cpu", dtype=torch.float32)
    k1 = torch.randn(1, 2, 3, 16)
    v1 = torch.randn(1, 2, 3, 16)
    cache.update(0, 0, k1, v1, start_pos=0)
    k2 = torch.randn(1, 2, 1, 16)
    v2 = torch.randn(1, 2, 1, 16)
    cache.update(0, 0, k2, v2, start_pos=3)
    k_out, _ = cache.get(0, 0, seq_len=4)
    assert torch.allclose(k_out[:, :, :3, :], k1)
    assert torch.allclose(k_out[:, :, 3:4, :], k2)


def test_clear():
    cache = KVCache(num_layers=1, num_heads=2, head_dim=16,
                    max_seq_len=64, max_batch_size=2, device="cpu", dtype=torch.float32)
    cache.update(0, 0, torch.ones(1, 2, 5, 16), torch.ones(1, 2, 5, 16), start_pos=0)
    cache.clear(batch_idx=0)
    k_out, _ = cache.get(0, 0, seq_len=5)
    assert torch.all(k_out == 0)


def test_independent_slots():
    cache = KVCache(num_layers=1, num_heads=2, head_dim=16,
                    max_seq_len=64, max_batch_size=4, device="cpu", dtype=torch.float32)
    k0 = torch.ones(1, 2, 3, 16)
    k1 = torch.ones(1, 2, 3, 16) * 2
    cache.update(0, 0, k0, k0, start_pos=0)
    cache.update(0, 1, k1, k1, start_pos=0)
    k_out_0, _ = cache.get(0, 0, seq_len=3)
    k_out_1, _ = cache.get(0, 1, seq_len=3)
    assert torch.allclose(k_out_0, k0)
    assert torch.allclose(k_out_1, k1)