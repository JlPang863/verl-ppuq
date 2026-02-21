# GSM8K Detailed Reward Demo (Multi-Turn)

A demo showing how to customize a reward function in verl's multi-turn tool pipeline. The standard `Gsm8kTool` returns binary 1.0/0.0 rewards. This example replaces it with tiered scoring that gives partial credit and detailed feedback, enabling the model to self-correct across turns.

## What's different from the default

| Outcome | Default (`Gsm8kTool`) | This example (`Gsm8kDetailedRewardTool`) |
|---|---|---|
| Correct answer | 1.0 | 1.0 |
| Within 5% of correct | 0.0 | 0.5 |
| Parsed but wrong | 0.0 | 0.1 |
| Unparseable | 0.0 | 0.0 |
| Feedback to model | `"Current parsed answer=... reward=..."` | Explains *why* the score was given and hints at what to fix |
| Reward tracking | Stores latest reward | Stores best reward across attempts |

The tool name stays `calc_gsm8k_reward`, so the existing preprocessed parquet data works as-is.

## Files

```
examples/gsm8k_detailed_reward/
├── README.md
├── gsm8k_detailed_reward_tool.py            # Custom tool class
├── config/tool_config/
│   └── gsm8k_detailed_tool_config.yaml      # Tool config pointing to the custom class
└── run_qwen3-4b_gsm8k_detailed_reward.sh    # Launch script (8xH100)
```

## How it works

This demo combines both of verl's reward customization mechanisms:

1. **During generation** — a `BaseTool` subclass scores each tool call and returns feedback text so the model can retry.
2. **After generation** — a custom `compute_score` function (reused from [`gsm8k_custom_reward`](../gsm8k_custom_reward/)) computes the final episode reward that drives GRPO training.

Both use the same tiered scoring, keeping the mid-generation feedback consistent with the final training signal.

```
Prompt (with system instructions to use the tool)
  → SGLang rollout (up to 5 assistant turns)
    → Model calls calc_gsm8k_reward(answer="...")
      → Tool scores the answer (tiered: 1.0 / 0.5 / 0.1 / 0.0)
      → Tool returns detailed feedback text
    → Feedback injected as tool message → model may retry
  → custom compute_score() produces the final tiered reward
  → GRPO advantage estimation
  → Policy gradient update
```

The tool is configured via a YAML config:

```
actor_rollout_ref.rollout.multi_turn.tool_config_path=.../gsm8k_detailed_tool_config.yaml
```

The final reward function is configured via Hydra CLI (pointing to the shared `gsm8k_custom_reward.py`):

```
reward.custom_reward_function.path=.../gsm8k_custom_reward/gsm8k_custom_reward.py
reward.custom_reward_function.name=compute_score
```

### Reward tiers

- **1.0** — Exact match with ground truth
- **0.5** — Numerically within 5% (e.g. rounding error)
- **0.1** — A number was parsed but it's wrong (format credit)
- **0.0** — No number could be parsed from the answer

### Feedback examples

| Score | Feedback |
|---|---|
| 1.0 | `"Correct! Your answer 42 matches the expected answer 42. Reward: 1.0"` |
| 0.5 | `"Close but not exact. Your answer 43 (parsed as 43) is within 5% of the expected answer. Check your arithmetic for rounding or off-by-one errors. Reward: 0.5"` |
| 0.1 | `"Incorrect. Your answer 100 (parsed as 100) does not match the expected answer. Please re-read the problem and try a different approach. Reward: 0.1"` |
| 0.0 | `"Could not parse a numeric answer from 'hello'. Please provide a plain number (e.g. 42 or -3.5). Reward: 0.0"` |

## Quick start

### 1. Preprocess data (one-time)

```bash
python examples/data_preprocess/gsm8k_multiturn_w_tool.py
```

This downloads the GSM8K dataset, wraps each problem with a system prompt instructing tool use, embeds the ground truth answer for the tool, and writes `~/data/gsm8k/train.parquet` and `~/data/gsm8k/test.parquet`.

**Note:** The single-turn demo uses a different preprocessor (`gsm8k.py`) that writes to the same path. If you're switching between demos, re-run the correct preprocessor.

### 2. Run training

```bash
bash examples/gsm8k_detailed_reward/run_qwen3-4b_gsm8k_detailed_reward.sh
```

Requires 8x H100 GPUs. Key settings:

| Setting | Value |
|---|---|
| Model | `Qwen3-4B-Instruct-2507` (local path) |
| Algorithm | GRPO |
| Rollout engine | SGLang (async, TP=2) |
| Samples per prompt | 16 |
| Max assistant turns | 5 |
| Batch size | 256 |
| Learning rate | 1e-6 |
| KL loss | `low_var_kl`, coef=0.001 |
| Epochs | 15 |
| Logging | console + W&B (`gsm8k_async_rl` project) |

### Customizing

All settings can be overridden as CLI args appended to the script:

```bash
bash examples/gsm8k_detailed_reward/run_qwen3-4b_gsm8k_detailed_reward.sh \
    trainer.total_epochs=5 \
    actor_rollout_ref.rollout.n=8
```

## See also: [`gsm8k_custom_reward`](../gsm8k_custom_reward/) (single-turn version)

Both demos apply the same tiered scoring (1.0 / 0.5 / 0.1 / 0.0) to GSM8K, but they use verl's two different reward customization mechanisms. Choose based on whether you need the model to interact with the reward signal during generation.

| | This demo (multi-turn) | [`gsm8k_custom_reward`](../gsm8k_custom_reward/) (single-turn) |
|---|---|---|
| **Mechanism** | `BaseTool` subclass (mid-generation) + `compute_score()` (final reward) | `compute_score()` function only |
| **Config** | Tool YAML + `rollout.multi_turn.tool_config_path` + `reward.custom_reward_function` | `reward.custom_reward_function.path/name` |
| **When reward runs** | During generation (tool feedback) + after generation (final reward) | After generation only |
| **Feedback to model** | Yes — descriptive text injected as tool response | None — model never sees the score |
| **Model can self-correct** | Yes — up to 5 turns to refine its answer | No — single shot |
| **Data preprocessing** | `gsm8k_multiturn_w_tool.py` (embeds tool kwargs + system prompt) | `gsm8k.py` (standard single-turn format) |
| **Boilerplate** | Tool class + YAML config + PYTHONPATH setup | One function, two CLI flags |

**When to use which:**

- **Single-turn** — simpler setup, faster training (one generation per sample), good baseline. Use when you just want to reshape the reward signal without changing the generation process.
- **Multi-turn** (this demo) — the model gets feedback and can retry, which teaches self-correction. More compute per sample but can learn richer behaviors. Use when the reward includes actionable information the model should learn to respond to.

## Writing your own tool

The tool extends `BaseTool` from `verl/tools/base_tool.py`. The interface:

| Method | What it does |
|---|---|
| `create(instance_id, **kwargs)` | Allocate per-trajectory state (receives ground truth from the dataset) |
| `execute(instance_id, parameters, **kwargs)` | Score the model's answer, return `(ToolResponse, step_reward, metrics)` |
| `calc_reward(instance_id, **kwargs)` | Compute the reward for the current tool state (used by `execute()` for step penalties) |
| `release(instance_id, **kwargs)` | Clean up state |

To make your own variant:

1. Copy `gsm8k_detailed_reward_tool.py` and modify `execute()` / `calc_reward()`
2. Update `class_name` in the YAML config to point to your new class
3. The run script already adds this directory to `PYTHONPATH`, so any module here is importable
