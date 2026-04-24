# LLM Alignment via Direct Preference Optimization (DPO)

Fine-tuning Llama-3.2-1B-Instruct using an automated preference dataset pipeline with GPT-4o-mini as an LLM-as-a-Judge, trained with DPO on a memory-constrained T4 GPU.

---

## Overview

This project implements an end-to-end LLM alignment pipeline that:

1. Generates candidate response pairs from Llama-3.2-1B at different sampling temperatures
2. Uses GPT-4o-mini as an LLM-as-a-Judge to automatically label preferences
3. Fine-tunes the model with Direct Preference Optimization (DPO) using LoRA adapters
4. Evaluates the fine-tuned model against the base model on novel instructions

The result is a parameter-efficient fine-tuned model stored as a 6.83MB LoRA adapter — a 99.7% reduction compared to the 2.47GB base model — without exceeding the 15GB VRAM limit of a free Colab T4 GPU.

---

## Pipeline Architecture

```
Alpaca Dataset (200 prompts)
        ↓
Llama-3.2-1B-Instruct
  ├── Response A (temperature=0.3)   ← focused, coherent
  └── Response B (temperature=1.2)   ← creative, noisier
        ↓
GPT-4o-mini Judge
  └── "A is better" → { prompt, chosen: A, rejected: B }
        ↓
DPO Training (LoRA, 4-bit NF4)
        ↓
Fine-tuned Adapter → HuggingFace
```

---

## Key Results

| Metric | Value |
|--------|-------|
| Preference dataset size | 200 examples |
| Train / Test split | 180 / 20 |
| Training loss (step 10 → 40) | 0.691 → 0.509 (26.3% reduction) |
| Adapter size vs full model | 6.83MB vs 2.47GB (99.7% smaller) |
| Avg response length — Base | 954 characters |
| Avg response length — DPO | 961 characters (+0.7%) |
| DPO response longer than base | 5/10 novel prompts |

---

## Technical Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| Student model | Llama-3.2-1B-Instruct | Small enough for T4, instruction-tuned |
| Judge model | GPT-4o-mini via OpenAI | Cost-efficient, reliable, no rate limits |
| Quantization | 4-bit NF4 + double quant | Fits 2.47GB model in 15GB VRAM |
| Fine-tuning method | LoRA (r=16, alpha=32) | Parameter-efficient, 99.7% size reduction |
| Training framework | TRL DPOTrainer | Native DPO support with PEFT integration |
| Compute | Google Colab T4 GPU | Free tier, 15GB VRAM |

---

## Project Structure

```
├── DPO_Finetune_Yusheng_Shi.ipynb   # Main notebook
├── preference_dataset.json           # 200 labeled preference pairs
├── comparative_analysis.csv          # Base vs DPO model outputs on 10 novel prompts
└── dpo-llama-1b-openai-adapter/      # Saved LoRA adapter (local)
```

---

## HuggingFace Model

The trained LoRA adapter is publicly available at:

👉 [alanshi31/dpo-llama-1b-openai-judge](https://huggingface.co/alanshi31/dpo-llama-1b-openai-judge)

To load and run inference:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "alanshi31/dpo-llama-1b-openai-judge")
tokenizer = AutoTokenizer.from_pretrained("alanshi31/dpo-llama-1b-openai-judge")
```

---

## LLM-as-a-Judge Design

The judge prompt was designed with four evaluation criteria — accuracy, clarity, helpfulness, and completeness — to provide a comprehensive assessment of response quality. The output was constrained to a single token (`A` or `B`) to eliminate ambiguity and ensure deterministic, parseable labels. Temperature was set to `0.0` to make judgments reproducible across runs.

**Example evaluation:**

> **Prompt:** "Explain what machine learning is in simple terms."
>
> **Response A (temp=0.3):** Structured explanation with a clear child-learning analogy, lists supervised/unsupervised/reinforcement learning types.
>
> **Response B (temp=1.2):** Starts well but drifts into an unrelated GPS metaphor and trails off mid-sentence.
>
> **Verdict: A** — preferred for accuracy, coherence, and completeness.

---

## Qualitative Analysis

The DPO fine-tuned model demonstrated marginally improved technical precision across several prompts. In the virus vs. bacteria prompt, the DPO model correctly referenced peptidoglycan cell walls in its structural description, while the base model only mentioned a generic cell wall. In the internet explanation prompt, the DPO model introduced a more organized taxonomy covering ISPs, DNS, and web browsers, compared to the base model's narrative step-by-step explanation.

Quantitatively, response lengths were nearly identical (954 vs 961 characters, 0.7% difference), suggesting DPO improved quality rather than verbosity — consistent with the preference optimization objective.

---

## Limitations

- 200 preference pairs is a small dataset; larger DPO runs typically use 5,000–50,000 examples
- Both models exhibit response truncation at the 200 token generation limit
- Training loss of 0.509 indicates the model is only marginally better than random at preference discrimination — more epochs and data would push this lower
- The base model is already instruction-tuned (Llama-3.2-1B-**Instruct**), leaving less room for DPO to improve behavior

---

## Suggestions for Future Work

- Scale preference dataset to 500+ examples with more diverse prompt categories to reduce distribution shift
- Experiment with higher `beta` values in DPO to enforce stronger preference boundaries
- Implement iterative DPO (Self-Rewarding Language Models) — using the model as its own judge across multiple rounds — to progressively improve alignment without external APIs
- Try a larger base model (3B or 7B) where DPO has more capacity to produce visible behavioral changes

---

## Setup

```bash
pip install transformers trl peft accelerate bitsandbytes datasets huggingface_hub openai
```

Set your API keys:

```python
import os
os.environ["OPENAI_API_KEY"] = "your_key_here"

from huggingface_hub import login
login(token="your_hf_token_here")
```

Then run all cells in `DPO_Finetune_Yusheng_Shi.ipynb` in order.

---

## Author

**Yusheng Shi**
