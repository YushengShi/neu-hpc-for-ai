# nano-sglang

A minimal LLM inference engine. Model: Qwen3-0.6B.

## Run tests

```bash
pytest tests/test_kv_cache.py -v          # local, no GPU
modal run modal_run.py::test              # all tests on GPU
```

## Assignment: Build nano-sglang Core

**Part 1: KV Cache** — `kv_cache.py`
- Implement `update()` and `get()`
- Store and retrieve key/value tensors across layers

**Part 2: Engine** — `engine.py`
- Implement `prefill()` — process all prompt tokens, store KV cache, return first token
- Implement `generate()` — wire prefill + decode loop, stop at EOS or max_tokens

**Part 3: Scheduler** — `scheduler.py`
- Implement `_decode_running()` — use provided `decode_batch()` to run all decodes in one GPU call
- Implement `step()` and `run_to_completion()` — manage request lifecycle: waiting → running → finished

**Part 4: Benchmark**
- Measure throughput (tokens/sec) vs. number of concurrent requests
- Compare batched scheduler vs. generating one request at a time

**Part 5 (stretch): Paged KV Cache** — `block_manager.py`
- Implement `allocate()` and `free()`

## Reference

- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
- [nano-vllm walkthrough](https://neutree.ai/blog/nano-vllm-part-1)
