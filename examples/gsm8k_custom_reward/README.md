# GSM8K Custom Reward Demo (Single-Turn)

A demo showing how to customize a reward function for single-turn GSM8K training. The default reward is binary (1.0 correct / 0.0 wrong). This example replaces it with tiered scoring that gives partial credit for close answers and format compliance.

## What's different from the default

| Outcome | Default (`gsm8k.compute_score`) | This example |
|---|---|---|
| Correct answer | 1.0 | 1.0 |
| Within 5% of correct | 0.0 | 0.5 |
| Parsed but wrong | 0.0 | 0.1 |
| Unparseable | 0.0 | 0.0 |

## Files

```
examples/gsm8k_custom_reward/
├── README.md
├── gsm8k_custom_reward.py                  # Custom compute_score function
└── run_qwen3-4b_gsm8k_custom_reward.sh     # Launch script (8xH100)
```

## How it works

Single-turn rewards use a plain Python function scored after generation completes:

```
Prompt
  → SGLang rollout (single generation)
  → Decode response to text
  → compute_score(data_source, solution_str, ground_truth) → tiered reward
  → GRPO advantage estimation
  → Policy gradient update
```

You provide the function and point to it with two Hydra CLI overrides:

```
reward.custom_reward_function.path=examples/gsm8k_custom_reward/gsm8k_custom_reward.py
reward.custom_reward_function.name=compute_score
```

verl dynamically loads the function and calls it for every generated response.

### Reward tiers

- **1.0** — Exact match with ground truth
- **0.5** — Numerically within 5% (e.g. rounding error)
- **0.1** — A number was parsed but it's wrong (format credit)
- **0.0** — No number could be parsed from the answer

## Quick start

### 1. Preprocess data (one-time)

```bash
python examples/data_preprocess/gsm8k.py
```

This downloads the GSM8K dataset, wraps each problem with a prompt, and writes `~/data/gsm8k/train.parquet` and `~/data/gsm8k/test.parquet`.

**Note:** The multi-turn demo uses a different preprocessor (`gsm8k_multiturn_w_tool.py`) that writes to the same path. If you're switching between demos, re-run the correct preprocessor.

### 2. Run training

```bash
bash examples/gsm8k_custom_reward/run_qwen3-4b_gsm8k_custom_reward.sh
```

Requires 8x H100 GPUs. Key settings:

| Setting | Value |
|---|---|
| Model | `Qwen3-4B-Instruct-2507` (local path) |
| Algorithm | GRPO |
| Rollout engine | SGLang (TP=2) |
| Samples per prompt | 16 |
| Batch size | 256 |
| Learning rate | 1e-6 |
| KL loss | `low_var_kl`, coef=0.001 |
| Epochs | 15 |
| Logging | console + W&B (`gsm8k_async_rl` project) |

### Customizing

All settings can be overridden as CLI args appended to the script:

```bash
bash examples/gsm8k_custom_reward/run_qwen3-4b_gsm8k_custom_reward.sh \
    trainer.total_epochs=5 \
    actor_rollout_ref.rollout.n=8
```

## See also: [`gsm8k_detailed_reward`](../gsm8k_detailed_reward/) (multi-turn version)

Both demos apply the same tiered scoring (1.0 / 0.5 / 0.1 / 0.0) to GSM8K, but they use verl's two different reward customization mechanisms. Choose based on whether you need the model to interact with the reward signal during generation.

| | This demo (single-turn) | [`gsm8k_detailed_reward`](../gsm8k_detailed_reward/) (multi-turn) |
|---|---|---|
| **Mechanism** | `compute_score()` function only | `BaseTool` subclass (mid-generation) + `compute_score()` (final reward) |
| **Config** | `reward.custom_reward_function.path/name` | Tool YAML + `rollout.multi_turn.tool_config_path` + `reward.custom_reward_function` |
| **When reward runs** | After generation only | During generation (tool feedback) + after generation (final reward) |
| **Feedback to model** | None — model never sees the score | Yes — descriptive text injected as tool response |
| **Model can self-correct** | No — single shot | Yes — up to 5 turns to refine its answer |
| **Data preprocessing** | `gsm8k.py` (standard single-turn format) | `gsm8k_multiturn_w_tool.py` (embeds tool kwargs + system prompt) |
| **Boilerplate** | One function, two CLI flags | Tool class + YAML config + PYTHONPATH setup |

**When to use which:**

- **Single-turn** (this demo) — simpler setup, faster training (one generation per sample), good baseline. Use when you just want to reshape the reward signal without changing the generation process.
- **Multi-turn** — the model gets feedback and can retry, which teaches self-correction. More compute per sample but can learn richer behaviors. Use when the reward includes actionable information the model should learn to respond to.

## Writing your own reward function

1. Create a `.py` file with a `compute_score` function:

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    # data_source: dataset name (e.g. "openai/gsm8k")
    # solution_str: the model's full decoded response
    # ground_truth: the expected answer string from the dataset
    # extra_info: dict with dataset metadata (split, index, etc.)
    ...
    return score  # float, or dict with "score" key + extra metrics
```

2. Point to it in the run script:

```
reward.custom_reward_function.path=path/to/your_reward.py
reward.custom_reward_function.name=compute_score
```

You can also pass extra kwargs via `reward.custom_reward_function.reward_kwargs` — they'll be forwarded to your function as keyword arguments.
