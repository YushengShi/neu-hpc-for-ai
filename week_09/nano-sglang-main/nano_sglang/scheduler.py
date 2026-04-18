"""Part 3: Continuous Batching Scheduler

Prefill each request one at a time, then batch all decodes together
using engine.decode_batch() for GPU efficiency.
"""

from .sampling import SamplingParams
from .sequence import Sequence, SequenceStatus
from .engine import Engine


class Scheduler:
    def __init__(self, model_path: str, max_batch_size: int = 64, device: str = "cuda"):
        self.engine = Engine(model_path, device=device)
        self.tokenizer = self.engine.tokenizer
        self.max_batch_size = max_batch_size

        self.next_seq_id = 0
        self.waiting_queue: list[Sequence] = []
        self.running: list[Sequence] = []
        self.finished: list[Sequence] = []

    def add_request(self, prompt: str, sampling_params: SamplingParams = None):
        """Tokenize prompt, create Sequence, add to waiting queue."""
        if sampling_params is None:
            sampling_params = SamplingParams()
        token_ids = self.tokenizer.encode(prompt)
        seq = Sequence(
            seq_id=self.next_seq_id,
            prompt_token_ids=token_ids,
            max_tokens=sampling_params.max_tokens,
        )
        self.waiting_queue.append(seq)
        self.next_seq_id += 1

    def _prefill_waiting(self, sampling_params: SamplingParams):
        """Prefill one request from the waiting queue and move it to running."""
        if not self.waiting_queue:
            return
        if len(self.running) >= self.max_batch_size:
            return
        seq = self.waiting_queue.pop(0)
        first_token = self.engine.prefill(seq, sampling_params)
        seq.output_token_ids.append(first_token)
        if first_token == self.tokenizer.eos_token_id:
            seq.status = SequenceStatus.FINISHED
            self.finished.append(seq)
        else:
            self.running.append(seq)

    def _decode_running(self, sampling_params: SamplingParams):
        """Decode all running sequences in one batched forward pass."""
        if not self.running:
            return

        next_tokens = self.engine.decode_batch(self.running, sampling_params)

        eos_id = self.tokenizer.eos_token_id
        still_running = []
        for seq, token in zip(self.running, next_tokens):
            seq.output_token_ids.append(token)
            if token == eos_id or seq.num_generated >= min(seq.max_tokens, sampling_params.max_tokens):
                seq.status = SequenceStatus.FINISHED
                self.finished.append(seq)
            else:
                still_running.append(seq)
        self.running = still_running

    def step(self, sampling_params: SamplingParams = None):
        """One scheduling iteration."""
        if sampling_params is None:
            sampling_params = SamplingParams()
        self._prefill_waiting(sampling_params)
        self._decode_running(sampling_params)

    def run_to_completion(self, sampling_params: SamplingParams = None) -> list[str]:
        """Run all requests to completion, return generated texts in order."""
        if sampling_params is None:
            sampling_params = SamplingParams()
        while self.waiting_queue or self.running:
            self.step(sampling_params)
        self.finished.sort(key=lambda s: s.seq_id)
        return [self.tokenizer.decode(s.output_token_ids) for s in self.finished]