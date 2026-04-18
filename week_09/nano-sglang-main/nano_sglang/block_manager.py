"""Part 5 (stretch): Paged KV Cache

Fixed-size blocks instead of contiguous allocation.
Same idea as OS virtual memory pages.
"""

import torch


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, num_layers: int,
                 num_heads: int, head_dim: int, device: str = "cuda",
                 dtype: torch.dtype = torch.float16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers

        self.k_pool = [
            torch.zeros(num_blocks, num_heads, block_size, head_dim,
                        device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.v_pool = [
            torch.zeros(num_blocks, num_heads, block_size, head_dim,
                        device=device, dtype=dtype)
            for _ in range(num_layers)
        ]

        self.free_blocks: list[int] = list(range(num_blocks))
        self.seq_to_blocks: dict[int, list[int]] = {}

    def allocate(self, seq_id: int, num_tokens: int) -> list[int]:
        """Allocate blocks for a sequence. Returns list of block IDs."""
        import math
        num_blocks_needed = math.ceil(num_tokens / self.block_size)
        if len(self.free_blocks) < num_blocks_needed:
            raise RuntimeError(
                f"OOM: need {num_blocks_needed} blocks but only {len(self.free_blocks)} free"
            )
        allocated = self.free_blocks[:num_blocks_needed]
        self.free_blocks = self.free_blocks[num_blocks_needed:]
        self.seq_to_blocks[seq_id] = allocated
        return allocated

    def free(self, seq_id: int):
        """Free all blocks for a finished sequence."""
        blocks = self.seq_to_blocks.pop(seq_id, [])
        self.free_blocks.extend(blocks)

    def get_block_ids(self, seq_id: int) -> list[int]:
        return self.seq_to_blocks.get(seq_id, [])

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)