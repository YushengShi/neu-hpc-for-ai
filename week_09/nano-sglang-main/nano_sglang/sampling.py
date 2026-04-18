"""Sampling strategies for token generation."""

import torch
from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 256


def sample_token(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    """Sample next token from logits.

    Args:
        logits: shape [batch_size, vocab_size] (logits for the last position)
        params: sampling parameters

    Returns:
        token_ids: shape [batch_size] sampled token IDs
    """
    if params.temperature <= 0:
        # Greedy
        return logits.argmax(dim=-1)

    logits = logits / params.temperature

    if params.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above top_p
        sorted_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= params.top_p
        sorted_logits[sorted_mask] = float("-inf")
        logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
