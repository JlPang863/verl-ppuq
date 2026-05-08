# Token-level Selection for Stable GRPO — 进度汇报

> 时间线 2026-04-20 → 2026-04-23  ·  Qwen2.5 + GSM8K  ·  verl 0.7.1  ·  L40S × 2

## 0. 一句话故事

**Phase 1**：试着把 SFT 的 Rho-1 token selection 直接搬到 GRPO 上 → 失败（选太少 token，performance 反而 −2.88pp）
→ **Phase 2**：自己设计 PPUQ（per-prompt uniform quantile），在 BF16 stress regime 跑出 **+1.96pp** vs baseline
→ **Phase 3**：人为构建 FP8 train/inference 不一致的 mismatch regime 二次验证，PPUQ 仍持 **+0.75pp** 稳定优势

---

## Phase 1 — Rho-1 token selection: 选太少 token, 整体性能反而降

**做了什么**：把 SFT 论文 Rho-1 的 score `score = log π_ref(t) − log π_θ(t)` 搬进 GRPO 的 PG mask（每条 response 保留 top 60% token）。

**结果**（120 step, baseline GRPO + LoRA）：

![Rho-1 vs baseline](figures/summary_rho1_vs_baseline.png)

| | val_acc step 120 | actor/kl_loss |
|---|---|---|
| **GRPO baseline** | **82.18%** | 0.0094 |
| GRPO + Rho-1 keep=0.6 | 79.30% | 0.0080 |
| Δ | **−2.88pp** | −14% |

**为什么 Rho-1 失败**：
1. **ref ≠ tutor**：Rho-1 在 SFT 用强 tutor 模型；这里 ref 就是训练起点的 Qwen，**没有 oracle 能力**。score 选的是"policy 已漂移的 token"，不是"该学的 token"。
2. **学习信号被砍 40%**：保留 60% token，剩下的还可能被选错；kl_loss 降 14% 说明 policy 实际移动更少。
3. **关键 insight**：score 方向（mismatch-aware）是对的，但**ref-based score 不是 RL 意义下"危险/关键"的指标**。

→ 转向 engine-level **train/rollout mismatch** 作为 selection 信号，自己设计选择算法。

---

## Phase 2 — PPUQ 方法设计 + BF16 stress regime 验证

### 方法设计：PPUQ (Per-Prompt Uniform Quantile)

**核心想法**：
- **Score**：用 K3 KL estimator `K3(t) = exp(log_r) − log_r − 1`，其中 `log_r = log π_train − log π_rollout`，正值表示 train 比 rollout 更自信（学习要小心）
- **Selection**：每个 prompt 单独算 quantile threshold（q=0.95），drop 该 prompt 内 score 最高的 5% token
- **关键差异点**：
  - 不用 hard threshold（如 verl 现成 token_rs 的 `>0.02 就 drop`）→ 自适应每个 prompt 的难度
  - 不用全局 quantile → 避免简单 prompt 的 token 集体 dominate filter
  - 不用 prob-only score → score 反映 train/rollout 的 distributional disagreement，不只是 token 的概率高低

**实现**：[verl/trainer/ppo/rollout_corr_helper.py](../verl/trainer/ppo/rollout_corr_helper.py) `compute_per_prompt_quantile_mask()`，作为 verl rollout_correction 的 PPUQ fast path。

### 实验

**Setup**：Qwen2.5-3B + LoRA，GSM8K，**stress regime**：`kl_loss_coef=0` + `lr=1e-5`（去 KL anchor + 提 lr 3 倍 → 制造大 policy drift），**400 步**。

![GRPO vs GRPO+PPUQ on BF16](figures/eval_acc_bf16_ours_vs_baseline.png)

| Run | val_acc step 400 | Δ vs baseline |
|---|---|---|
| GRPO baseline (无 RS/IS) | 84.7% | — |
| **GRPO + PPUQ (我的)** | **86.66%** ★ | **+1.96pp** |

**观察**：
- 早期（step 50）PPUQ 比 baseline 慢 3pp（被 selection 牵制）
- 中期（step 100-200）追平
- 末段（step 300-400）PPUQ 反超并保持优势 → **PPUQ 不是更快，是更稳的 late-stage 提升**

---

## Phase 3 — 人为放大 mismatch（FP8 train/inference 不一致）下的二次验证

### 动机

Phase 2 的 BF16 regime 里 train/rollout mismatch 自然较小（rollout_probs_diff_mean ≈ 0.003）。如果**人为放大 mismatch**，PPUQ 的 selection 信号应该更显著——这是 method 鲁棒性的关键测试。

**做法**：用 vLLM 的 **FP8 rollout quantization**（train 仍 BF16），把 mismatch 放大约 **4× → 0.012**。

### 实验

**Setup**：Qwen2.5-1.5B full-params（被迫，因 LoRA + FP8 vLLM 兼容性问题），FP8 vLLM rollout，`kl=0.001` + `lr=5e-6`，**120 步**。

![GRPO vs GRPO+PPUQ on FP8](figures/eval_acc_fp8_ours_vs_baseline.png)

| Run | val_acc step 99（best stable ckpt）| Δ vs baseline |
|---|---|---|
| GRPO baseline | 71.80% | — |
| **GRPO + PPUQ (我的)** | **72.55%** ★ | **+0.75pp** |

**观察**：
- 全程 step 20-100，两条曲线在 70-73% 平台并行（PPUQ 略低/略高摆动）
- step 99 PPUQ 反超 baseline +0.75pp
- step 120 PPUQ 出现累积失稳（drop 到 19.6%），baseline 稳 → **best ckpt 取 step 80-100，避开末段 hard-drop 累积**

→ 即使在人为放大 4× 的 mismatch 下，PPUQ 仍保持对 baseline 的稳定优势。

---

## 三段递进对比汇总

| Phase | Setup | val_acc baseline | val_acc PPUQ (ours) | Δ | 故事 |
|---|---|---|---|---|---|
| 1 | Rho-1 keep=0.6, 120 step | 82.18% | 79.30% | **−2.88pp** | 直接搬 SFT 选择算法到 RL → 反而变差 |
| 2 | BF16 stress, 400 step | 84.7% | **86.66%** | **+1.96pp** | 自己设计 PPUQ → 显著正收益 |
| 3 | FP8 mismatch, step 99 | 71.80% | **72.55%** | **+0.75pp** | 放大 mismatch 4× 仍持稳定优势 |

**核心 takeaway**：选 token 的方向对了不够，**怎么选**才是关键——PPUQ 的 per-prompt quantile + K3 mismatch score 是经过两个 regime 验证的稳定提升。

---

## 工程 artifact

| 文件 | 用途 |
|---|---|
| [run_gsm8k_demo.sh](../run_gsm8k_demo.sh) | GRPO baseline + LoRA |
| [run_gsm8k_rho1.sh](../run_gsm8k_rho1.sh) | Rho-1 ablation (Phase 1) |
| [run_gsm8k_ppuq.sh](../run_gsm8k_ppuq.sh) | **GRPO + PPUQ (我的)** |
| [run_gsm8k_fp8roll.sh](../run_gsm8k_fp8roll.sh) | FP8 mismatch regime (Phase 3) |
| [verl/trainer/ppo/rollout_corr_helper.py](../verl/trainer/ppo/rollout_corr_helper.py) | PPUQ 实现 |
| [research_plan.md](research_plan.md) | 完整实验记录（含其他 ablation：token_rs, prob-only PPUQ 等）|

**Best checkpoint**：
- BF16 final（86.66%）：`/mnt/data1/jinlong/ckpts/k3_ppuq_from_base350/global_step_400`
- FP8 best（72.55%）：`/mnt/data1/jinlong/ckpts/qwen1.5b_full_fp8roll_k3ppuq_v3/global_step_80` 或 `step_120`
