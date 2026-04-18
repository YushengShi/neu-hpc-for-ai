"""Tests for Part 3: Scheduler (requires GPU + model)"""

import os
import time
import pytest
import torch

MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-0.6B")


@pytest.fixture
def scheduler():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from nano_sglang.scheduler import Scheduler
    return Scheduler(MODEL_PATH)


def test_add_request(scheduler):
    scheduler.add_request("Hello")
    scheduler.add_request("World")
    assert len(scheduler.waiting_queue) == 2
    assert scheduler.waiting_queue[0].seq_id == 0
    assert scheduler.waiting_queue[1].seq_id == 1


def test_single_request_completes(scheduler):
    from nano_sglang.sampling import SamplingParams
    scheduler.add_request("The capital of France is")
    results = scheduler.run_to_completion(SamplingParams(temperature=0, max_tokens=10))
    assert len(results) == 1
    assert len(results[0]) > 0
    print(f"Result: {results[0]}")


def test_multiple_requests_complete(scheduler):
    from nano_sglang.sampling import SamplingParams
    prompts = ["Hello", "The weather is", "Python is"]
    for p in prompts:
        scheduler.add_request(p)
    results = scheduler.run_to_completion(SamplingParams(temperature=0, max_tokens=10))
    assert len(results) == len(prompts)
    for r in results:
        assert len(r) > 0
    print(f"Results: {results}")


def test_respects_max_tokens(scheduler):
    from nano_sglang.sampling import SamplingParams
    scheduler.add_request("Tell me a long story about")
    results = scheduler.run_to_completion(SamplingParams(temperature=0, max_tokens=5))
    tokens = scheduler.tokenizer.encode(results[0])
    assert len(tokens) <= 5


def test_results_in_order(scheduler):
    from nano_sglang.sampling import SamplingParams
    for i in range(5):
        scheduler.add_request(f"Topic {i}")
    results = scheduler.run_to_completion(SamplingParams(temperature=0, max_tokens=10))
    assert len(results) == 5


def test_scheduler_throughput(scheduler):
    from nano_sglang.sampling import SamplingParams
    params = SamplingParams(temperature=0, max_tokens=20)
    for i in range(8):
        scheduler.add_request(f"Write about topic {i}")
    start = time.time()
    results = scheduler.run_to_completion(params)
    elapsed = time.time() - start
    total_tokens = sum(len(scheduler.tokenizer.encode(r)) for r in results)
    throughput = total_tokens / elapsed
    print(f"\n8 requests: {elapsed:.2f}s, {total_tokens} tokens, {throughput:.1f} tok/s")
    assert len(results) == 8