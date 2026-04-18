"""Run nano-sglang on Modal.

Usage:
    modal run modal_run.py::run        # single generation sanity check
    modal run modal_run.py::test       # run all tests on GPU
    modal run modal_run.py::benchmark  # Part 4: throughput vs concurrency
"""

import modal

MODEL_NAME = "Qwen/Qwen3-0.6B"


def download_model():
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_NAME)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "huggingface_hub", "pytest", "accelerate")
    .run_function(download_model)
    .add_local_dir("nano_sglang", remote_path="/root/nano_sglang")
    .add_local_dir("tests", remote_path="/root/tests")
)

app = modal.App("nano-sglang")


@app.function(image=image, gpu="A100-40GB", timeout=600)
def run():
    """Single generation — sanity check that the engine works end-to-end."""
    import sys
    sys.path.insert(0, "/root")

    from nano_sglang.engine import Engine
    from nano_sglang.sampling import SamplingParams

    print(f"Loading {MODEL_NAME}...")
    engine = Engine(MODEL_NAME)
    params = SamplingParams(temperature=0, max_tokens=50)
    output = engine.generate("The capital of France is", params)
    print(f"Output: {output}")


@app.function(image=image, gpu="A100-40GB", timeout=600)
def test():
    """Run the full test suite (Parts 1-3)."""
    import os
    import sys
    import subprocess
    sys.path.insert(0, "/root")

    subprocess.run(
        ["python", "-m", "pytest", "/root/tests/", "-v", "--tb=short"],
        check=False,
        env={**os.environ, "MODEL_PATH": MODEL_NAME, "PYTHONPATH": "/root"},
    )


@app.function(image=image, gpu="A100-40GB", timeout=600)
def benchmark():
    """Part 4: throughput (tokens/sec) vs. number of concurrent requests.
    Compares batched Scheduler vs. sequential engine.generate().
    """
    import os
    import sys
    import time
    sys.path.insert(0, "/root")

    from nano_sglang.engine import Engine
    from nano_sglang.scheduler import Scheduler
    from nano_sglang.sampling import SamplingParams

    CONCURRENCY_LEVELS = [1, 2, 4, 8, 16]
    MAX_TOKENS = 100
    params = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)

    PROMPTS = [
        "The capital of France is",
        "The tallest mountain in the world is",
        "Python is a programming language that",
        "The theory of relativity states that",
        "Machine learning is a field of",
        "The human brain contains approximately",
        "The speed of light is approximately",
        "The most widely spoken language is",
        "Climate change refers to long-term",
        "The Internet was invented in",
        "Quantum computing uses principles of",
        "The first moon landing occurred in",
        "Artificial intelligence can be defined as",
        "The Amazon rainforest is located in",
        "DNA stands for",
        "The largest ocean on Earth is",
    ]

    print("=" * 58)
    print(f"  nano-sglang Benchmark  (max_tokens={MAX_TOKENS})")
    print("=" * 58)
    print(f"  {'N':>3}  {'Sequential tok/s':>18}  {'Batched tok/s':>15}  {'Speedup':>8}")
    print("-" * 58)

    seq_throughputs = {}
    bat_throughputs = {}

    for n in CONCURRENCY_LEVELS:
        prompts = PROMPTS[:n]

        # ── Sequential: one request at a time ──────────────────
        engine = Engine(MODEL_NAME)
        t0 = time.perf_counter()
        outputs = [engine.generate(p, params) for p in prompts]
        seq_time = time.perf_counter() - t0
        seq_tokens = sum(len(engine.tokenizer.encode(o)) for o in outputs)
        seq_tps = seq_tokens / seq_time
        seq_throughputs[n] = seq_tps

        # ── Batched: scheduler with decode_batch ───────────────
        scheduler = Scheduler(MODEL_NAME)
        for p in prompts:
            scheduler.add_request(p, params)
        t0 = time.perf_counter()
        results = scheduler.run_to_completion(params)
        bat_time = time.perf_counter() - t0
        bat_tokens = sum(len(scheduler.tokenizer.encode(r)) for r in results)
        bat_tps = bat_tokens / bat_time
        bat_throughputs[n] = bat_tps

        speedup = bat_tps / seq_tps
        print(f"  {n:>3}  {seq_tps:>18.1f}  {bat_tps:>15.1f}  {speedup:>7.2f}x")

    # ── ASCII chart ────────────────────────────────────────────
    print()
    print("  Batched throughput (tokens/sec)")
    print()
    max_tps = max(bat_throughputs.values())
    bar_width = 40
    for n, tps in bat_throughputs.items():
        bar = "█" * int(bar_width * tps / max_tps)
        print(f"  N={n:>2}  {bar:<{bar_width}}  {tps:.0f}")
    print()
    print("  Sequential throughput (tokens/sec)")
    print()
    for n, tps in seq_throughputs.items():
        bar = "█" * int(bar_width * tps / max_tps)
        print(f"  N={n:>2}  {bar:<{bar_width}}  {tps:.0f}")


@app.local_entrypoint()
def main():
    run.remote()