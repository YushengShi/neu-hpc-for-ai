"""Tests for Part 2: Engine (requires GPU + model)"""

import os
import pytest
import torch

MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-0.6B")


@pytest.fixture(scope="module")
def engine():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from nano_sglang.engine import Engine
    return Engine(MODEL_PATH)


def test_decode_step_works(engine):
    """Verify provided decode_step() works (need prefill first)."""
    from nano_sglang.sequence import Sequence, SequenceStatus
    from nano_sglang.sampling import SamplingParams
    seq = Sequence(seq_id=0, prompt_token_ids=engine.tokenizer.encode("Hello"))
    params = SamplingParams(temperature=0)
    first = engine.prefill(seq, params)
    seq.output_token_ids.append(first)
    second = engine.decode_step(seq, params)
    assert isinstance(second, int)
    assert 0 <= second < engine.model.vocab_size


def test_prefill(engine):
    from nano_sglang.sequence import Sequence, SequenceStatus
    from nano_sglang.sampling import SamplingParams
    seq = Sequence(seq_id=0, prompt_token_ids=engine.tokenizer.encode("Hello"))
    token = engine.prefill(seq, SamplingParams(temperature=0))
    assert isinstance(token, int)
    assert 0 <= token < engine.model.vocab_size
    assert seq.status == SequenceStatus.DECODING
    assert seq.past_key_values is not None


def test_generate_returns_text(engine):
    from nano_sglang.sampling import SamplingParams
    text = engine.generate("The capital of France is", SamplingParams(temperature=0, max_tokens=20))
    assert isinstance(text, str)
    assert len(text) > 0
    print(f"Generated: {text}")


def test_generate_stops_at_max_tokens(engine):
    from nano_sglang.sampling import SamplingParams
    text = engine.generate("Hello", SamplingParams(temperature=0, max_tokens=5))
    tokens = engine.tokenizer.encode(text)
    assert len(tokens) <= 5


def test_generate_deterministic(engine):
    from nano_sglang.sampling import SamplingParams
    params = SamplingParams(temperature=0, max_tokens=10)
    text1 = engine.generate("Once upon a time", params)
    text2 = engine.generate("Once upon a time", params)
    assert text1 == text2