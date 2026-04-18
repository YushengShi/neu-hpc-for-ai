"""Part 2: Inference Engine

Two phases:
  Prefill:  process entire prompt in one pass -> compute-bound
  Decode:   generate one token at a time from cache -> memory-bound
"""

import torch
import torch.nn.functional as F
from transformers import DynamicCache
from .model import Model, Tokenizer
from .sampling import SamplingParams, sample_token
from .sequence import Sequence, SequenceStatus


class Engine:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model = Model(model_path, device=device)
        self.tokenizer = Tokenizer(model_path)
        self.device = device

    def prefill(self, seq: Sequence, sampling_params: SamplingParams) -> int:
        """Process all prompt tokens in one forward pass, return first generated token."""
        input_ids = torch.tensor([seq.prompt_token_ids], device=self.device)
        logits, past_key_values = self.model.forward(input_ids)
        next_token = sample_token(logits[:, -1, :], sampling_params).item()
        seq.past_key_values = past_key_values
        seq.status = SequenceStatus.DECODING
        return next_token

    def decode_step(self, seq: Sequence, sampling_params: SamplingParams) -> int:
        """Generate one token for a single sequence using cached KV."""
        last_token = seq.output_token_ids[-1]
        input_ids = torch.tensor([[last_token]], device=self.device)
        logits, past_key_values = self.model.forward(input_ids, past_key_values=seq.past_key_values)
        next_token = sample_token(logits[:, -1, :], sampling_params).item()
        seq.past_key_values = past_key_values
        return next_token

    def decode_batch(self, sequences: list[Sequence], sampling_params: SamplingParams) -> list[int]:
        """Generate one token for multiple sequences in a single GPU forward pass."""
        if not sequences:
            return []
        if len(sequences) == 1:
            return [self.decode_step(sequences[0], sampling_params)]

        n = len(sequences)
        input_ids = torch.tensor(
            [[seq.output_token_ids[-1]] for seq in sequences], device=self.device,
        )

        cache_lens = [seq.past_key_values.get_seq_length() for seq in sequences]
        max_len = max(cache_lens)

        # Use to_legacy_cache() to extract tensors, build batched cache, then convert back
        # Legacy format: tuple of (key, value) tuples per layer,
        # each tensor shape [batch, heads, seq_len, head_dim]
        legacy_caches = [seq.past_key_values.to_legacy_cache() for seq in sequences]
        num_layers = len(legacy_caches[0])

        batched_legacy = []
        for layer_idx in range(num_layers):
            padded_keys, padded_values = [], []
            for i, lc in enumerate(legacy_caches):
                k = lc[layer_idx][0]  # [1, heads, seq_len, head_dim]
                v = lc[layer_idx][1]
                pad = max_len - cache_lens[i]
                if pad > 0:
                    k = F.pad(k, (0, 0, pad, 0))
                    v = F.pad(v, (0, 0, pad, 0))
                padded_keys.append(k)
                padded_values.append(v)
            batched_legacy.append((
                torch.cat(padded_keys, dim=0),
                torch.cat(padded_values, dim=0),
            ))

        batched_cache = DynamicCache.from_legacy_cache(tuple(batched_legacy))

        attn_mask = torch.zeros(n, max_len + 1, device=self.device, dtype=torch.long)
        for i, cl in enumerate(cache_lens):
            attn_mask[i, max_len - cl:] = 1

        position_ids = torch.tensor([[cl] for cl in cache_lens], device=self.device)

        logits, new_cache = self.model.forward(
            input_ids, past_key_values=batched_cache,
            position_ids=position_ids, attention_mask=attn_mask,
        )

        tokens = sample_token(logits[:, -1, :], sampling_params)

        # Split batched cache back into per-sequence caches
        new_legacy = new_cache.to_legacy_cache()
        for i, seq in enumerate(sequences):
            real_len = cache_lens[i] + 1
            pad = max_len - cache_lens[i]
            per_seq_legacy = []
            for layer_idx in range(num_layers):
                k = new_legacy[layer_idx][0][i:i+1, :, pad:pad + real_len, :]
                v = new_legacy[layer_idx][1][i:i+1, :, pad:pad + real_len, :]
                per_seq_legacy.append((k.clone(), v.clone()))
            seq.past_key_values = DynamicCache.from_legacy_cache(tuple(per_seq_legacy))

        return [t.item() for t in tokens]

    def generate(self, prompt: str, sampling_params: SamplingParams = None) -> str:
        """Generate text for a single prompt."""
        if sampling_params is None:
            sampling_params = SamplingParams()

        seq = Sequence(
            seq_id=0,
            prompt_token_ids=self.tokenizer.encode(prompt),
            max_tokens=sampling_params.max_tokens,
        )

        first_token = self.prefill(seq, sampling_params)
        seq.output_token_ids.append(first_token)
        seq.status = SequenceStatus.DECODING

        eos_id = self.tokenizer.eos_token_id
        while (len(seq.output_token_ids) < sampling_params.max_tokens
               and seq.output_token_ids[-1] != eos_id):
            next_token = self.decode_step(seq, sampling_params)
            seq.output_token_ids.append(next_token)

        seq.status = SequenceStatus.FINISHED
        return self.tokenizer.decode(seq.output_token_ids)