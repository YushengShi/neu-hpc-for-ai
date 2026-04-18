"""Sequence tracking.

Each incoming request becomes a Sequence object that tracks its state
as it moves through prefill -> decode -> finished.
"""

from enum import Enum
from dataclasses import dataclass, field


class SequenceStatus(Enum):
    WAITING = "waiting"      # queued, not yet prefilled
    PREFILLING = "prefilling" # currently being prefilled
    DECODING = "decoding"    # prefill done, generating tokens
    FINISHED = "finished"    # hit EOS or max_tokens


@dataclass
class Sequence:
    seq_id: int
    prompt_token_ids: list[int]     # original prompt tokens
    output_token_ids: list[int] = field(default_factory=list)  # generated tokens so far
    status: SequenceStatus = SequenceStatus.WAITING
    max_tokens: int = 256
    past_key_values: object = None  # HuggingFace past_key_values (set after prefill)

    @property
    def num_generated(self) -> int:
        return len(self.output_token_ids)

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED
