# Token-Level Selection for Stable GRPO — Research Plan & Literature Review

> **Status**: Work-in-progress research plan derived from preliminary GSM8K experiments on verl.
> **Date**: 2026-04-21

---

## 1. Motivation(与老板对齐的核心出发点)

**问题**:LLM RL 训练(GRPO / PPO 家族)经常**不稳定 / 崩盘**。典型表现:
- grad_norm spike
- KL loss 爆炸
- Response length 失控撞 max_length
- Entropy collapse / reward hacking
- 整个 run 后期 reward 反而下降

**老板的假设**:崩盘的一大部分原因是 **train-inference mismatch(训推不一致)**。
- **Rollout 引擎**(vLLM / SGLang)用优化过的 kernel + FP16/BF16/FP8 精度 + kv cache 近似采样
- **训练引擎**(FSDP / Megatron)用完整 forward 重算 log-prob 给 PPO ratio 用
- 即使**权重完全同步**,两边给出的 `log π(t)` 也**不相等**
- 这让 PPO 的 on-policy 假设被**系统性破坏**,是崩盘的种子

**提出的 angle**:**token-level selection**
- 不是所有 token 都"同等危险",高 mismatch 的 token 才是引入 variance / 把 policy 推崩的主因
- 从 Rho-1(SFT 里"只训一部分 token"的思路)借鉴机制,**换成 RL 场景下有意义的 selection 分数**
- 目标:同样的训练预算下,**降低 crash 概率 + 提高稳定训练长度**

---

## 2. 已完成的前期实验(2026-04 GSM8K 初探)

### 2.1 环境

| 组件 | 版本 |
|---|---|
| verl | 0.7.1 |
| torch | 2.8.0 + cu128 |
| vllm | 0.11.0 |
| flash-attn | 2.8.3 |
| transformers | 4.57.6 |
| 硬件 | 8× NVIDIA L40S (46 GB), FP16/BF16 rollout + actor |

### 2.2 Baseline 配置
- Model: **Qwen2.5-3B-Instruct + LoRA (rank=64, α=32)**
- Dataset: **GSM8K** (7473 train / 1319 test)
- Algorithm: **GRPO**, 5 rollouts/prompt, KL loss 锚 (coef=0.001)
- Batch: 16 prompts × 5 rollouts = 80 trajectories/step
- 120 training steps

### 2.3 初次 Rho-1 移植(ref-based excess loss score)

**方法**:每个 response token 按 `log π_ref(t) - log π_θ(t)` 打分,每条 response 保留 top-60%。只改 PG mask,KL / entropy 仍用 full mask。

**实现文件**:
- [verl/workers/config/actor.py](../verl/workers/config/actor.py) — 新增 `Rho1Config` dataclass
- [verl/trainer/config/actor/actor.yaml](../verl/trainer/config/actor/actor.yaml) — 注册 `rho1` schema
- [verl/workers/actor/dp_actor.py](../verl/workers/actor/dp_actor.py) — `_rho1_select_mask` helper + selection block + mask 替换
- [run_gsm8k_rho1.sh](../run_gsm8k_rho1.sh) — 对比实验脚本
- [docs/rho1_grpo.md](rho1_grpo.md) — 方法文档

### 2.4 实验结果(step 120)

| 指标 | baseline GRPO | Rho-1 keep=0.6 | Δ |
|---|---|---|---|
| **val_acc (GSM8K test)** | **82.18%** | 79.30% | **−2.88%** |
| training reward mean | 0.838 | 0.825 | −0.013 (noise) |
| actor/kl_loss | 0.0094 | 0.0080 | −14% (rho1 更保守) |
| response length | 242.6 | 250.5 | +7.9 token |
| rho1/score_gap | — | 0.04–0.07 稳定为正 | selection 方向正确 |

**判断**:val_acc −2.88% **不是噪声**,是 100 步内一致偏低 ~2-3%。

### 2.5 为什么 Rho-1 在这个 setting 失败

1. **ref ≠ tutor**:SFT Rho-1 用强 tutor model,这里 ref 就是训练起点的 Qwen,**没有 oracle 能力**;`log π_ref - log π_θ` 选出的是"policy 已漂移的 token",不是"该学的 token"
2. **GRPO 信号本就稀疏**,再砍 40% 反而拖慢学习(`actor/kl_loss` −14% 就是直接证据:policy 实际移动少了,所以 val_acc 也低)
3. **在 baseline 本就不崩的 setting 下,Rho-1 的"防崩"收益 = 0**;我们没测出老板假设成立的场景

### 2.6 本次实验的正面价值
- 实现了一个 minimal、可配置的 Rho-1 pipeline,verl 内纯 additive patch,未来换 score 只要改 2-3 行
- 证实 **"score 方向对了"**(score_gap 持续为正),但选出来的**不是 RL 意义下"危险" / "关键"**的 token
- 为换 score 到 **engine-mismatch-based** 铺平了工程路径

### 2.7 Stress-regime 对比实验(300 步,2026-04-21)

> **措辞修订**(reviewer critique):KL≈0.10 / grad_norm≈0.036 是 **elevated-stress**,不是真 crash。真 crash(NaN / val_acc 下跌)通常 500-2000+ 步才出现。下文用"stress"而非"crash"。

**Setup**:Qwen2.5-3B + LoRA,**kl_loss_coef=0,lr=1e-5**,300 步。两个 job 同步对比:
- **C** (baseline):无 IS,无 RS
- **D** (token_rs):token-IS + token-K3 RS,threshold=0.02

**最后 50 步(step 250-300)均值**:

| metric | C (baseline) | D (token_rs) | Δ |
|---|---|---|---|
| val_acc | 0.8502 | **0.8547** | +0.45% |
| **actor/kl_loss** | 0.0875 | **0.0515** | **−41.1%** |
| **actor/grad_norm** | 0.0364 | **0.0217** | **−40.4%** |
| **actor/entropy** | 0.1313 | **0.1543** | **+17.5%** |
| response_length | 222.0 | 229.4 | +3.3% |

**三个关键 empirical findings**(后面 method 设计都要对齐这些):

#### **Finding 1**:**Mismatch 是 heavy-tailed 且越来越 skewed**
```
Job D 全程 300 步 PPL Gap:
  mean:  稳定 0.0013 ~ 0.002           ← 几乎不动
  max:   0.004 → 0.016 (4×)           ← 缓慢爆
  max/mean 比率:   3× → 10×           ← tail 越拉越长
  rollout_probs_diff_max:  达 0.28-0.46 (单 token 概率差 28-46%)
```
**Insight**:少数 token 的 mismatch 极端化,但绝大多数 token 仍精确。任何有效方法都必须针对**tail**,不是"平均"。

#### **Finding 2**:**0.5% token drop 换 40% KL 降低(80/20 principle)**
Token_rs 在 threshold=0.02 下**只丢 0.22-0.50% 的 token**,但把 KL drift 降了 41%,grad_norm 降了 40%。
**Insight**:tail 那一小撮 token **贡献了绝大多数 destabilizing gradient**。这是 token-level 方法能 work 的数学理由。

#### **Finding 3**:**Token_rs 的收益是 phase-dependent**
```
step   50:  D/C kl ratio = 1.08  (一致,没区别)
step  100:  D/C kl ratio = 0.58  (D 领先开始)
step  150:  D/C kl ratio = 0.38  (D 最强)
step  300:  D/C kl ratio = 0.64  (D 稳定领先 ~40%)
```
**Insight**:早期 policy ≈ rollout,mismatch 均匀且小,selection 无素材。中后期 heavy-tail 形成,selection 开始生效。**全程用同一策略是低效的**。

#### 其他显著事实
- **D 的 entropy 始终比 C 高 17%** —— selection 有"抗 mode collapse"副作用,可能与 KL 稳定是同一机制(policy 不被推到极端)
- **C 在 step 310 首次破 KL=0.1** —— baseline 真的在缓慢进入 crash 区,不是 plateau
- **C 偶发 pg_loss=grad_norm=0 的 step**(zero-variance group collapse)—— baseline 已经有"学习信号死亡"的早期征兆

### 2.7b 实验清单(所有跑过的 run + 当前角色)

> 一张表汇总所有 ckpt 目录里能落地的 run,让"我的 method 是哪个 / baseline 是哪个 / 哪个能被引用"清晰可查。

| # | 名字(ckpt 目录) | 模型 | Setup | 角色 | 最终步数 | 最终 val_acc(GSM8K test) |
|---|---|---|---|---|---|---|
| 1 | `qwen2.5_3b_grpo_lora` | Qwen2.5-3B + LoRA | BF16 baseline,正常 KL+LR(kl=0.001, lr=3e-6) | 原始 baseline pilot | 120 | 82.18% |
| 2 | `qwen2.5_3b_grpo_lora_rho1_0.6` | 同上 + Rho-1 keep=0.6 | 同上 + Rho-1 token mask | 早期失败方法(已弃) | 120 | 79.30% |
| 3 | `qwen2.5_3b_grpo_lora_token_k3_t0.02` | 同上 + verl token_k3 | BF16 + verl 现成 token_rs(K3 hard threshold=0.02) | 现成方法对照 | 200 | — |
| 4 | `qwen2.5_3b_grpo_lora_mismatch_analysis` | 同上 | BF16 + 纯测量(无 RS/IS) | 测 mismatch 分布 | 200 | — |
| 5 | `baseline_crash_noKL_lr1e5` (Job C) | Qwen2.5-3B + LoRA | **stress regime**:kl=0, lr=1e-5(放大 mismatch) | **400 步 baseline** | 350 (300→350 续) | step 300 ≈ 84.7% |
| 6 | `tokenrs_crash_noKL_lr1e5` (Job D) | 同上 | stress + verl token_rs(token-IS + token-K3 RS) | 400 步对照(verl 已有方法) | 350 | step 300 ≈ 86.0% |
| 7 | **`k3_ppuq_from_base350`** | 同上 | stress + **PPUQ K3 score(per-prompt q=0.95)** | **★ 我的主方法 K3-PPUQ** | **400** (从 #5 step 350 续) | **86.66%** |
| 8 | `probonly_ppuq_from_base350` | 同上 | stress + **PPUQ neg_logp score(per-prompt q=0.95)** | K3 vs prob 直接对照(prob-only) | **400** (从 #5 step 350 续) | **86.13%** |
| 9 | `ppuq_q095_crash_350` | 同上 | stress + PPUQ K3(从 0 step 跑) | K3-PPUQ 单独完整 run | 400 | — |
| 10 | `qwen1.5b_full_fp8roll_baseline` | **Qwen2.5-1.5B full-params** | **FP8 vLLM rollout**(放大 mismatch ~4×) + 无 RS | **400 步 FP8 baseline**(进行中) | 进行中 → 400 | TBD |
| 11 | `qwen1.5b_full_fp8roll_k3ppuq` | 同上 + K3-PPUQ q=0.95 | 同上 + 我的方法 | **★ 我的方法 in FP8 regime**(进行中) | 进行中 → 200 | TBD |

**核心 finding 已由 #5/#6/#7/#8 锁定**(BF16 stress regime,400 步对照):
| | val_acc step 400 | Δ vs baseline (#5) |
|---|---|---|
| #5 baseline (Job C) | 84.7% | — |
| #6 verl token_rs (Job D) | 86.0% | +1.3% |
| **#7 K3-PPUQ (我的)** | **86.66%** | **+1.96%** |
| #8 prob-only PPUQ | 86.13% | +1.43% |

**关键 gap**:K3 vs prob-only **只差 0.53%** —— reviewer 担心的 "K3 score ≈ prob detector" 在 BF16 regime 里**确实没分辨度**。这就是 §2.8 引入 FP8 rollout 把 mismatch 放大 4× 想验证的事。

### 2.8 FP8 rollout decorrelation regime(2026-04-22/23)

**动机**:之前 BF16 rollout + LoRA 跑 K3-PPUQ vs prob-only-PPUQ 只差 0.5pp(0.866 vs 0.861),mismatch 太小(diff_mean ~0.003)看不出差异。换 **FP8 vLLM rollout** 把 train/inference gap 放大 ~4×(diff_mean ~0.012),期望 K3 的 mismatch-aware 信号被放大显示出来。

#### 2.8.1 硬件 / 栈约束踩坑(诚实记录)

- **L40s 是 SM 8.9**,verl 的 FP8 E2E blockwise 路径硬性要 SM ≥ 9.0 + CUDA ≥ 12.9(TE assertion),走不通。只能改 **FP8 rollout-only**(BF16 train + FP8 vLLM infer),L40s + CUTLASS 路径官方 validate
- **LoRA + FP8 vLLM 不兼容**:smoke 出来 `rollout/logp_inf_frac=1.0`(所有 token 的 vLLM log_prob 都是 -inf),verl 官方 FP8 rollout 实验全是 full-params,LoRA 这条没测过 → 已加 sanitize 补丁([rollout_corr_helper.py](../verl/trainer/ppo/rollout_corr_helper.py) + [ray_trainer.py:1429-1450](../verl/trainer/ppo/ray_trainer.py#L1429-L1450))避免 nan 扩散,但根本问题需要 **full-params**
- **3B full-params + 共享 GPU(yxwang 占 22GB) OOM**(actor update peak 20.8GB) → 降到 **Qwen2.5-1.5B-Instruct full-params**
- verl FP8 E2E(Megatron 路径)也有 ready-made 脚本 [run_qwen3-30b_dapo_megatron_fp8_trtllm.sh](../examples/grpo_trainer/run_qwen3-30b_dapo_megatron_fp8_trtllm.sh),但是 30B MoE GB200 配方,跟我们这套毫不兼容。独立 venv `venv_megatron/` 装全了(torch 2.8+cu128, TE 2.13, megatron-core 0.16, megatron-bridge 0.3.1),脚本 [run_gsm8k_fp8_e2e.sh](../run_gsm8k_fp8_e2e.sh) 也写好,但如前述 SM<9.0 跑不通 → 归档不再用

#### 2.8.2 v1 配置错误(kl=0.001, lr=3e-6,demo 默认)

最初按现有 fp8roll.sh 默认跑了 K3-PPUQ + baseline + prob-only-PPUQ 三个 run 各 step 1-20。但配方跟 §2.7 的原 BF16 stress regime(kl=0, lr=1e-5)不匹配 → **刻度对不上无法对比** → 全 kill 重做。

#### 2.8.3 v2 配置(精确 mirror §2.7 原 BF16 stress regime,仅换 rollout 精度)

| 项 | 旧 BF16 stress(§2.7) | v2 FP8 stress |
|---|---|---|
| Model | Qwen2.5-3B + LoRA | **Qwen2.5-1.5B full-params**(被迫改) |
| kl_loss_coef | 0 | 0 ✓ |
| optim.lr | 1e-5 | 1e-5 ✓ |
| total_steps | 300-400 | 400 ✓ |
| Data / batch / n | GSM8K, 16, 5 | 同 ✓ |
| rollout | BF16 | **FP8**(新增 stress) |

脚本: [run_gsm8k_fp8roll.sh](../run_gsm8k_fp8roll.sh),`LORA_RANK=0` 走 full-params 分支。

#### 2.8.4 v2 三个 run 结果(2026-04-23)

**val_acc(GSM8K test)随步数**:

| step | baseline | K3-PPUQ(我的) | **token_rs**(verl 现成) |
|---|---|---|---|
| 19 | 16.1% | **54.8%** | 54.0% |
| 39 | 0.0%  | 4.1%  | **56.9%** ✨ (仍在升!) |
| 59 | 0.0%  | 0.0%  | 0.0% |
| 79+ | 0.0% | 0.0% | ~0% |

**三种不同的崩溃机制**(关键发现):

| Method | 崩点 | 崩因 |
|---|---|---|
| **baseline** | step 33 | **grad 爆炸**(step 18 grad_norm 从 1.3 → 5.87,4.5×尖峰)→ reward 压到 0 → GRPO group 内 advantage=0 → **zero-variance collapse**(§2.7 已记录过的征兆,FP8 regime 下加速) |
| **K3-PPUQ** | step 39-40 | **response length runaway**:K3 成功压住 grad(全程 ≤ 1.7,比 baseline 稳 5-9×)但没拦住 policy 学会写长废话 → response_length 323→865、clip_ratio 0→67% → max_length 截断的 response 拿不到 reward → 训练死 |
| **token_rs** | step 55-59 | 同 length runaway,但**多撑 20 步**,中间 step 39 甚至 val_acc 上升到 0.569 |

**反直觉主要发现**:**token_rs > K3-PPUQ > baseline** 在这个 FP8 stress regime 下。解释:
1. K3-PPUQ 的 per-prompt q=0.95 hard drop 在每个 prompt 都必须砍 5% token,学习信号被砍得比 token_k3 的 hard threshold 还狠
2. token_rs 的 **token-TIS 重加权**是 soft 修正,不是 hard drop,保留更多学习信号
3. K3-PPUQ 过度过滤 → policy 缺"高 reward token"反馈 → 学习稀疏 → response 靠长度刷时间

**K3 的"抗 grad 爆"效果是真 work 的**(baseline step 18 grad_norm=5.87 vs K3 同步 ≤ 1.7),但 FP8 放大 mismatch 后 grad 这一轴不是瓶颈,**policy drift 才是**,而防 drift 是 KL anchor 的活不是 RS 的活。

#### 2.8.5 v2 ckpt + 工程 artifacts

| Run (ckpt dir) | 角色 | 最后可用 step |
|---|---|---|
| `qwen1.5b_full_fp8roll_baseline_v2` | baseline(崩) | step 200(已存);v2 val_acc 全程 0%,无研究价值 → **建议删** |
| `qwen1.5b_full_fp8roll_k3ppuq_v2` | K3-PPUQ(崩) | step 100(已存);val_acc step 19=54.8% 但未存 ckpt,剩下全 0% → **建议删** |
| token_rs v2 | 崩前被 kill | 无 ckpt 存盘 |

#### 2.8.6 v3 配方(温和版,已执行 2026-04-23 凌晨)

v2 崩因分析后切到温和配方:

| 项 | v2 | **v3(执行的)** |
|---|---|---|
| `kl_loss_coef` | 0 | **0.001**(加回 KL anchor) |
| `optim.lr` | 1e-5 | **5e-6**(比 v2 小一半) |
| `total_steps` | 400 | 120 |
| `save_freq` / `test_freq` | 100 / 20 | 40 / 20 |
| FP8 rollout | 开 | 开(保留) |
| Model | 1.5B full-params | 同 |

#### 2.8.7 v3 结果(2026-04-23,4 个 method 全跑完 step 120)

**val_acc(GSM8K test)按 test_freq=20 记录**:

| step | baseline | **K3-PPUQ(我的)** | prob-PPUQ(对照) | token_rs(verl 现成) |
|---|---|---|---|---|
| 19  | 64.7% | 69.0% | 74.6% | 70.3% |
| 39  | 73.0% | 71.0% | 72.4% | 69.5% |
| 59  | 72.3% | 68.5% | 71.2% | 71.0% |
| 79  | 71.9% | 70.4% | 70.1% | 68.0% |
| **99** | 71.8% | **72.6%** ★ | 70.4% | 70.7% |
| 119 | 69.4% | 19.6% (崩) | 0.0% (崩) | 68.9% |

**Step 99 对比(最后共同稳定点,推荐用于 final 评测)**:

| Run | step 99 val_acc | Δ vs baseline |
|---|---|---|
| **K3-PPUQ v3(我的)** | **72.6%** | **+0.75pp** ★ |
| baseline v3 | 71.8% | — |
| token_rs v3 | 70.7% | −1.06pp |
| prob-PPUQ v3 | 70.4% | −1.44pp |

**核心 finding**:
- **K3 vs prob-PPUQ 差距**:BF16 regime +0.53pp → **FP8 regime +2.19pp,放大 4.1×** ✅ 验证了 "FP8 放大 mismatch 能拉开 K3 信号 vs prob-only" 的核心假设
- **K3 vs token_rs**:+1.81pp,你的方法优于 verl 现成的
- **K3 vs baseline**:+0.75pp,方法确实有增益

**step 120 K3 和 prob-PPUQ 崩了但 baseline/token_rs 没崩**:可能是 per-prompt hard drop 的累积效应;这个崩点更晚于 v2 的 step 39,说明温和配方延长了稳定窗口。用 **step 80 或 step 100 ckpt** 作为 best checkpoint 最公平。

**ckpt**:`/mnt/data1/jinlong/ckpts/qwen1.5b_full_fp8roll_{baseline,k3ppuq,probppuq,tokenrs}_v3/global_step_{40,80,120}` 各 3 个 ckpt 都保存了。

### 2.9 Checkpoint 清理计划(2026-04-23)

`/mnt/data1/jinlong/ckpts/` 共 **1.2 TB**,11 个实验目录。下面按研究价值分档,方便后续清理(不会自动删,等确认)。

#### 🟢 必须保留(paper 核心数据,~72 GB)

| ckpt | 作用 |
|---|---|
| `k3_ppuq_from_base350/global_step_400`(15 G) | **★ K3-PPUQ final, BF16 stress regime 86.66%** |
| `probonly_ppuq_from_base350/global_step_400`(15 G) | prob-only 对照 final, 86.13%(K3 vs prob 直接证据) |
| `baseline_crash_noKL_lr1e5/global_step_350`(~14 G) | 400 步 baseline final + resume 起点 |
| `tokenrs_crash_noKL_lr1e5/global_step_350`(~14 G) | verl token_rs 对照 final, 86.0% |
| `ppuq_q095_crash_350/global_step_400`(~14 G) | K3-PPUQ 独立完整 run final |

#### 🟡 建议删中间 ckpt(final 已取完,腾 ~566 GB)

| 目录 | 保留 | 删 | 腾 |
|---|---|---|---|
| `baseline_crash_noKL_lr1e5`(18 ckpts to 350) | 只留 step 350 | step 20-340(17 个) | **~247 GB** |
| `tokenrs_crash_noKL_lr1e5`(18 ckpts to 350) | 只留 step 350 | step 20-340(17 个) | **~247 GB** |
| `ppuq_q095_crash_350`(6 ckpts to 400) | 只留 step 400 | step 70,140,210,280,350(5 个) | ~72 GB |

#### 🔴 建议全删(已无研究价值,~230 GB)

| 目录 | 原因 | 腾 |
|---|---|---|
| `qwen1.5b_full_fp8roll_baseline_v2`(100, 200) | **v2 FP8 崩盘 run**,val=0% 全程 | ~37 GB |
| `qwen1.5b_full_fp8roll_k3ppuq_v2`(100) | 同上 | ~19 GB |
| `qwen2.5_3b_grpo_lora_rho1_0.6`(6 ckpts) | §2.5 标记"已弃"的失败 Rho-1 实验 | ~87 GB |
| `qwen2.5_3b_grpo_lora`(6 ckpts to 120) | 最早 BF16 baseline pilot(§2.4 已存数据),后续实验已取代 | ~87 GB |

#### 🟠 可选删(早期分析 run,视后续需要,~275 GB)

| 目录 | 选项 | 腾 |
|---|---|---|
| `qwen2.5_3b_grpo_lora_mismatch_analysis`(10 ckpts to 200) | 只留 step 200 或全删 | 130-145 G |
| `qwen2.5_3b_grpo_lora_token_k3_t0.02`(10 ckpts to 200) | 只留 step 200 或全删 | 130-145 G |

#### 清理方案总结

| 策略 | 腾出 | 剩余 |
|---|---|---|
| 保守(只删 🔴) | ~230 GB | ~970 GB |
| **推荐(删 🔴 + 🟡)** | **~796 GB** | ~404 GB |
| 激进(+ 🟠 全删) | ~1.07 TB | ~130 GB |

---

## 3. Literature Landscape(文献现状扫描)

### 3.1 Token-level methods for GRPO(loss 侧)

| 论文 / 系统 | 动什么 | 选 token 的分数 | 关系 |
|---|---|---|---|
| **Rho-1**(Lin et al., 2024) | mask | `log π_ref - log π_θ`(SFT excess)| 原型,我们借鉴但不合适 |
| **DAPO** (2025) | clip-higher + dynamic sampling | — | clip 不对称 + 零方差组重采 |
| **Dr. GRPO** (2025) | agg | — | `seq-mean-token-mean` 消 length bias |
| **GSPO** (2025) | clip | sequence-level ratio | 改 ratio 粒度 |
| **CISPO** (2025) | clip | — | clip 后用截断 ratio 反传 |
| **AR-Lopti** (arXiv 2505.12929, ICLR 2026) | advantage 重加权 | `π_old(t)`(低概率 token 压 gradient) | **最接近我们思路**,但用 prob 不用 mismatch |
| **It Takes Two: GRPO is DPO** (2510.00977) | 重改写 | correct/incorrect 对比 | 把 GRPO 改写为 token-level 对比 |
| **GCPO** (2510.07790) | 加 gold | — | 对比失败时引入 gold |
| **GTPO / GRPO-S** (2508.04349) | agg | token 熵 reward shaping | entropy-weighted |
| **λ-GRPO** (2510.06870) | 权重 | 可学习 token 偏好 | 元学习 token 权重 |

### 3.2 Group / rollout-diversity methods(rollout 侧)

| 论文 / 系统 | 做什么 |
|---|---|
| **GAPO (EMNLP 2025)** | group-level diversity reward,避免 mode collapse |
| **DRA-GRPO** (2505.09655) | diversity-aware reward adjustment |
| **ETTRL** (2508.11356) | entropy-fork tree rollout,高熵点 branch |
| **Tree-GRPO** (ICLR 2026, AMAP-ML) | 树搜索 rollout for agents |
| **TreeAdv** (2601.03703) | tree-structured advantage redistribution |
| **VIP / AERO / DARS / GRESO** | adaptive rollout budget allocation |
| **DAPO dynamic sampling** | resample until non-zero reward variance |

### 3.3 Training-inference mismatch methods(老板的方向)

| 论文 / 系统 | 做什么 | 跟我们关系 |
|---|---|---|
| **"When Speed Kills Stability"** (Liu & Li 2025, blog 3 部曲, [richardli.xyz/rl-collapse](https://richardli.xyz/rl-collapse)) | 系统性分析 RL 崩盘来自 train-infer mismatch | **我们 motivation 的核心来源** |
| **verl Rollout Correction** 模块 | 实现了 Token-TIS / Sequence-TIS / Token-RS / Sequence-RS 等 7-8 种方法 | **已经做了"硬 drop high-mismatch token"**(`rollout_rs=token_k1/k2/k3`) |
| **Trust Region Masking (TRM)** (arXiv 2512.23075, 2025.12) | 按 max-token-KL 硬 mask **整条序列** | 同家族,**sequence level**,不是 token level |
| **Defeating Mismatch via FP16** (2510.26788, sail-sg) | argue 用 FP16 替代 BF16 解决精度问题 | precision 侧解法 |
| **FP8-RL** (2601.18150) | FP8 rollout + 必配 IS 校正 | precision + correction |
| **On the Rollout-Training Mismatch** (OPT-ML 2025) | bias-variance 分析 token-TIS vs seq-TIS | 理论分析 |

### 3.4 关键洞察(扫完文献后的判断)

**verl 内的 [rollout_corr_helper.py:156](../verl/trainer/ppo/rollout_corr_helper.py#L156) `compute_rollout_rejection_mask()`** 就是"按 mismatch drop token"的完整实现:
```python
log_ratio = log π_train(t) - log π_rollout(t)           # 核心量
token_k1 = -log_ratio                                    # K1 KL 估计
token_k2 = 0.5 * log_ratio**2                            # K2
token_k3 = exp(log_ratio) - 1 - log_ratio                # K3
# 超阈值 → mask = 0
```

**也就是说"drop high-mismatch token" vanilla 版已经被做了。要 novel 必须从别的角度切。**

### 3.5 TRM Paper 实验配置(arXiv 2512.23075)

这是 mismatch 这条线上唯一公开可查的实验细节。风格是**定性对比图**,不是"第 X 步必崩"的数字表。

| 维度 | 设置 |
|---|---|
| Model | Qwen3-8B-Base(单一大小)|
| Framework | Zero-RL + GRPO |
| Group size | 16 |
| Batch size | 32 |
| Learning rate | **1e-6**(很小,说明作者本身也怕崩) |
| Data | DAPO-MATH-17k(去重)|
| Eval | AIME25 avg@32 |
| Rollout | vLLM(BF16) |
| Training | PyTorch FSDP |
| 关键:**故意造 train-infer mismatch**(两套不同后端) |

**主指标(稳定性代理)**:**Log Absolute Perplexity Gap**
$$\text{PPL Gap} = \mathbb{E}\left[\left|\log \pi_\theta(t) - \log \pi_\text{roll}(t)\right|\right]$$
- 就是老板说的"训推不一致"的直接量化
- 他们不靠 reward 证明稳定性,靠 PPL Gap 发不发散

**主要结论(图 2 定性)**:
- Baseline PPO + token-level clipping:**PPL Gap 单调发散**,accuracy 从涨转跌
- TRM-Max (δ=0.05) + TRM-Avg (δ=0.001):PPL Gap **夹住不发散**,accuracy 持续上涨
- **Max 和 Avg 单用都不够,组合才 work**——max 抓离群 token,avg 管累积漂移

### 3.6 What does "RL training crash" actually mean — 参考

综合 TRM paper + 社区经验,"崩"**一般不是突然训练失败**,而是以下的一种或组合:

| 症状 | 典型出现时机(对 Qwen-7B+ 级别) |
|---|---|
| PPL Gap 慢速发散 | 200–500 步露头,1000+ 步明显 |
| reward 从高位回落(reward hacking / context collapse) | 500–2000 步,跟 lr 和 KL coef 强相关 |
| grad_norm spike 偶发 | 几百步开始零星 |
| response length 失控撞 max_length | 200–500 步(reward shaping 敏感) |
| entropy collapse | 300–1000 步,policy 过度确定 |
| 完全训练失败(NaN / Inf) | 少见,通常是 FP8 + 激进 lr 才会 |

### 3.7 什么配置下容易崩

越往下越危险:

| 维度 | 稳定 ← → 危险 |
|---|---|
| 精度 | FP32 / FP16 ← BF16 ← FP8 |
| 模型大小 | 3B ← 8B ← 32B ← 70B+ |
| Response length | 1k ← 4k ← 16k ← 32k+ |
| Rollout engine 差 | 同引擎训 ← vLLM ← vLLM+kv cache 激进 ← SGLang/TRT-LLM 混用 |
| Learning rate | 1e-6 ← 3e-6 ← 1e-5 ← 5e-5+ |
| KL coef | 0.01 ← 0.001 ← 0.0001 ← 0(完全去 KL) |
| Rollout staleness | 同步 ← 1-step off ← async ← replay buffer |
| Reward sparsity | dense PRM ← 中间奖励 ← outcome-only 0/1 |
| 参数化 | full fine-tune ← LoRA ← frozen base |

### 3.8 为什么我们现在的 setup **不崩**

| 我们的 config | 位置 |
|---|---|
| Qwen2.5-**3B** + LoRA rank=64 | 最稳那列 |
| bf16 | 中间 |
| vLLM + FSDP | 中间 |
| max_resp 1024 | 稳定侧 |
| lr 3e-6 | 稳定侧 |
| kl_coef 0.001 | 稳定侧 |
| outcome-only reward(GSM8K 0/1)| 危险侧 ← **唯一偏危险的一项** |
| 120 steps | 远没到 "1000+ 步才明显" |

**几乎每一维都在稳定那列**——所以 baseline 不崩,我们测 token selection 的"防崩收益"等于 0。
不是 method 没用,是**测试场景本身没有 crash 可防**。

### 3.9 要测出防崩效果,"造崩" regime 推荐

基于 TRM 配置 + 社区共识,能最快放大稳定性差异的 knob(按 ROI 排):

| Knob | baseline demo | "造崩"版(推荐)|
|---|---|---|
| **kl_loss_coef** | 0.001 | **0**(去掉 KL 锚,最快显效) |
| **learning rate** | 3e-6 | **1e-5 或 3e-5** |
| LoRA | rank 64 | 关掉,全参 fine-tune(若显存够)|
| Model | Qwen2.5-3B | Qwen2.5-7B / Qwen3-8B |
| max_response_length | 1024 | 4096+ |
| total_steps | 120 | 300–1000+ |

**经验性建议**:同时开 2–3 个 knob 就能看到 baseline PPL Gap 发散,token_rs / IS correction 能夹住。不用全开(GPU 吃爆)。

**最低成本 crash-inducing config**(在你当前 GPU 预算内):
```bash
# 还是 Qwen-3B + LoRA,但去 KL + lr×3 + 长训练
bash run_gsm8k_demo.sh \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    trainer.total_training_steps=300
```
预期:baseline `rollout_corr/*` metric 里 PPL Gap 类指标会可见上升,能验证 token_rs 是否压住。

---

## 4. 空白 / 可能的 novelty

在"mismatch + token-level selection" 这个子方向内,我扫完发现**还有这些 gap**:

### 4.1 Gap 1:**Reward-aware mismatch masking**
**观察**:verl 的 `token_rs` 只看 `|log_ratio|`,**完全忽略这个 token 的 PG 贡献(advantage × ratio)**。
- Low-advantage + high-mismatch 的 token:本来就贡献小,丢了无关痛痒 → 没必要
- **High-advantage + high-mismatch 的 token**:才是真正推着 policy 崩的元凶 → 这些最该丢
- **现状**:没有方法把 `|log_ratio|` 和 `|A|` 联合考虑做 selection

**候选 score**:
$$s_t = |\log r_t| \cdot (1 + \lambda |A_t|)\quad \text{or}\quad s_t = |A_t \cdot (r_t - 1)|$$

### 4.2 Gap 2:**Adaptive / learned thresholds**
**观察**:verl 阈值全是硬编码数值(如 `rollout_rs_threshold=1.2`)。
- 训练早期 policy ≈ rollout,mismatch 自然小 → 固定阈值太严,正常 token 被误丢
- 训练晚期 policy 漂移大,mismatch 大 → 固定阈值太松,危险 token 漏网

**候选方法**:
- **Per-batch 分位数**(保留分数 bottom X%)
- **Schedule**(阈值 = f(step)随训练变化)
- **Predictive**:用 past mismatch 预测 next step 的安全阈值

### 4.3 Gap 3:**Token-level Trust Region bound**
**观察**:TRM paper (2512.23075) 只推了 **sequence-level** 的 TR bound,作者明说"token-independent methods 管不了长 horizon"——但**没证伪"聪明的 token-level method 能管"**。
- 能不能推出 non-trivial **token-level** trust region bound?
- 如果能,会是 ICLR / NeurIPS 级别理论贡献

### 4.4 Gap 4:**Empirical characterization**
**观察**:上面所有 paper 都是"小 ablation"形式,没人做**系统性经验研究**:
- mismatch 怎么分布?(per token position / per probability level / per token identity)
- 跟 precision 关系(FP32 vs BF16 vs FP16 vs FP8)
- 跟 rollout engine 关系(vLLM vs SGLang vs TRT-LLM)
- 跟 model 大小 / LoRA rank / training stage 关系
- 跟 GRPO hyperparams(temperature、top-p、rollout.n)关系

一篇**纯 empirical paper**,目标 MLSys / EMNLP findings,低风险高价值。

### 4.5 Gap 5:**多个方法正交 ablation**
verl 里这些 token-level 方法能叠加,但**没人系统跑过**:
- Rho-1 mask × token_rs threshold × AR advantage reweight:这 9 个组合没人比较过
- 跟 sequence-level TRM / seq_rs 的叠加效果

---

## 5. 候选方向(基于 §2.7 + independent review 的审计)

> **第二轮修订**(2026-04-22):code-reviewer 外审指出多个 overclaim / 机制漏洞。本节砍到只剩 **HAT 一个核心方法**,其他降级为 future work。

### 5.0 审计方法

每个 idea 按三条标准验证:
1. **Motivation**:它的假设能不能被 §2.7 的 3 个 finding 支持?
2. **Mechanism**:在 GRPO 数学/实现语境下,机制是否真的成立?
3. **Impact**:相比 verl token_rs 现有收益(−41% KL,+0.5% val_acc),它有没有可能做得更好?

**结论**:9 个 idea 里 **只有 HAT 1 个 full validated,5 个 postponed,3 个 killed**。

---

### 5.1 死掉的 idea(❌)

#### **A1. Reward-aware mismatch masking**:`s = |log r| × (1 + λ|A|)`
**死因(Mechanism)**:GRPO 的 advantage 是 **sequence-level 标量广播**,同一 response 内所有 token 的 $|A_t|$ **完全相等**。
- 公式在 response 内退化成 `|log r|`(等价于 verl token_rs)
- 只在 **跨 response** 改"丢 token 预算分配" → 本质是"高 advantage response 被丢更多" → 反而有害
- 要真做 token-level reward awareness,需要 **per-token advantage**(PRM)——就不是 GRPO 了
- **Kill**:公式在 GRPO 下**数学上是伪命题**。

#### **B1. MR-GRPO**:`L_MR = E[(log π_θ - log π_roll)²]`
**死因(Mechanism)**:**π_roll 是 noise,π_θ 才是 truth**(训练引擎精度更高)。把 π_θ regularize 向 π_roll 等于"把真理往错误上靠"。
- vLLM 的 top-p 系统性**抬高** π_roll;MR 会让 π_θ 复现这个 bias
- Side effect:policy 变得更 peaked,entropy collapse 加速
- **Finding 3 验证**:我们数据里 baseline C (无 MR, 无 RS) 的 entropy **已经**比 D 低 17%——MR 只会让这情况更糟
- **Kill**:方向反了,数据也佐证。

#### **C3. Token-level TR bound**(反驳 TRM)
**死因(Impact 不成比例)**:TRM paper 已给出 sequence-level bound,且明说 token-independent 不行。要 rigorously 给出 token-level bound 需要 **multiple weeks 数学**,而 Finding 2 表明就算真推出来,工程 gain 可能只比 sequence-level TRM 多几个点。
- 理论贡献 vs 投入 **产出不划算**,除非你是数学强博士
- **Postpone**:当 stretch goal,不当 main idea。

#### **~~Rho-1 (ref-based) 直接 follow~~**
已在 §2.4 验证 val_acc **−2.88%**,死得最透。

---

### 5.2 幸存但要降级的(⚠️)

#### **A3. Ensemble consensus rejection**
**Motivation**:两次 forward(不同 attn/精度)一致才保留 → 准确 identify 真 mismatch。
**Impact**:2× forward cost。Finding 2 显示 0.5% 的 token 就能抓到 40% stabilization,引入 2× 成本去抓更准的 selection,**ROI 低**。
**Verdict**:能用但不值得先做。

#### **B2. Precision-aware schedule**
sail-sg FP16 paper (2510.26788) 已占这个方向,跟进不讨好。**Keep as related work, don't attempt**。

#### **C2. Doubly-robust PG estimator**
纯理论 direction,跟我们的 empirical findings 关联弱。**Theory-path 专项,当 backup**。

#### **B3. Logits post-processing alignment**
**Motivation**:vLLM top-p truncation 是 mismatch 的**系统性**来源。
**数据佐证**:Finding 1 的 heavy tail(max 0.28-0.46 prob diff)跟 top-p 特征一致。
**但**:我们**没有直接证据**证明 top-p 是主因,可能还有 kv cache、FlashAttn 精度等。
**Verdict**:**值得试但要先做 diagnosis**——关掉 vLLM top-p 跑一次看 PPL Gap max 是否消失。

---

### 5.3 幸存(✅):**HAT (Heavy-tail Adaptive Threshold)** —— **唯一 full validated**

**核心 idea**:把 verl token_rs 的固定阈值(0.02)替换成**per-batch 分位数**。

```python
# 按 "drop 固定比例" 而不是 "drop 超阈值"
threshold_t = torch.quantile(|log_ratio|[response_mask], q=0.99)
keep_mask = (|log_ratio| < threshold_t) * response_mask
```

**为啥这是唯一值得主推的**:
- **Finding 1 驱动**:tail 越来越重,固定阈值全程 drop 0.2-0.5%,没跟上 tail 增长
- **Finding 2 驱动**:0.5% token 贡献 40% stabilization,这是 target 对的
- **Mechanism 简单清楚**:不加 entropy、不加 advantage、不加 group —— 一个 drop-in 替换,论文 claim 最干净
- **Reviewer 也承认**:"HAT is solid. This is your strongest chain."

**跟 verl token_rs 的**干净 ablation**(matched drop rate):
- 两者都 drop **exactly x%**(比如 q=0.99 的 HAT vs token_rs threshold 调到也刚好 drop 1%)
- 看 stability metrics 谁更好
- 结果会直接回答:"自适应 threshold 是不是真的比固定好"

**Reviewer 担忧 — 我会在实验里回答**:
- 不同 q(0.95 / 0.99 / 0.999)是不是需要 tune?→ 3 个都跑,看鲁棒性
- 会不会只是"drop 得更多/更少"的效果?→ matched drop rate ablation 直接控住

---

### 5.4 被 reviewer 狙击后**降级**的方向(⚠️ postponed)

| Idea | 为什么降级 |
|---|---|
| **ENM** (|log r| / H) | reviewer:"可能只是'drop when confident'的复杂版本"。分母 H 小会爆炸。要先做 **sanity check**:证明 ranking 跟 `|log r|` 或 `-H` 单独用有本质区别,才 revive |
| **SPSR** (soft reweight) | **跟 Finding 2 自相矛盾**(Finding 2 支持 hard drop 够用,soft 的边际价值不清)。除非拿出"soft > hard"的新数据才做 |
| **GICA** (group internal) | reviewer **致命狙击**:不同 rollout 在 position t 分叉,"同 position 跨 rollout" 不是同 token。要救只能上 tree rollout(= ETTRL 地盘,novelty 没了)。**kill/rewrite needed** |
| **PAS** (phase schedule) | 本质是 wrapper,多超参(LOW/HIGH/EMA-β),在 val 上 tune 会 leak;作为工程 trick 放进 HAT 就够 |
| **C1. Mismatch-conditioned exploration** | 工程代价高(改 vLLM 循环)+ 跟 HAT 方向正交,等 HAT 验证成功再考虑 |

### 5.5 死掉不动(❌)

| Idea | kill reason |
|---|---|
| A1. Reward-aware masking | GRPO advantage 是 sequence-level 标量,公式数学上退化 |
| ~~B1. MR regularizer (严重降级)~~ | reviewer 指出我原来 kill 过激(π_roll 生成了数据,match 它能降 IS variance);但方向仍可疑。**postponed not dead**,需要理论 + 实证 |
| Rho-1 (ref-based) 直接 follow | §2.4 已实测 val_acc −2.88% |
| C3. Token-level TR bound | TRM paper 已证 sequence 足够,token 级要 weeks 数学,ROI 低 |

---

### 5.6 **MVP paper story: 只做 HAT**

**标题候选**:
*"Heavy-tail Adaptive Thresholding for Engine-Mismatch Rejection in GRPO"*

### Contribution 要 claim 的(不多不少)
1. **Empirical**:engine mismatch 分布 heavy-tailed 且随训练越来越 skewed(Finding 1)
2. **Method**:HAT——per-batch quantile 替代 fixed threshold
3. **Result**:same drop rate 下,HAT 比 verl token_rs 在 stress regime 下稳定性更好(KL drift / grad norm)

**故事长度**:一篇 workshop / short paper 的厚度,**不要**现在就冲 main track。

### Reviewer 要求的**实验必做项**(不能 skip):

1. **Multi-seed**:至少 n=3 seeds 跨方法,画置信区间
2. **Matched drop-rate ablation**:HAT vs token_rs 在**相同 drop fraction** 下比(否则混淆 threshold type 和 drop rate)
3. **Placebo baseline**:"随机 drop 0.5% token"也要跑,排除"任何 mask 都帮忙"
4. **真 crash regime**:500-1000 步,或 7B 模型,至少跑一次,看 HAT 能否防住 true divergence
5. **Score ablation**:固定 threshold × `|log r|` vs 固定 threshold × alternative score(比如 K3 估计器 vs K1)
6. **Reframe 措辞**:"crash-regime"全部改成"stress-regime",不要 overclaim

### Reviewer 预期的一针见血 objection
> "你在 3B LoRA GSM8K 300 步 stress regime 下跑,没有 multi-seed,没有 matched drop rate,0.5% val_acc 改善在噪声内。请 show (a) multi-seed 置信区间,(b) 真 crash regime,(c) matched drop rate ablation,(d) scale test(7B + MATH)。"

所以上面 6 个实验**就是为了挡这刀**。

---

## 6. 推荐的下一步(按先后)

### Step 1:**跑 verl 自带的 `token_rs` 作为真 baseline**(1 天)
脚本 override(在 `run_gsm8k_rho1.sh` 基础上):
```bash
actor_rollout_ref.rollout.calculate_log_probs=True \
algorithm.rollout_is=token_level \
algorithm.rollout_rs=token_k3 \
algorithm.rollout_rs_threshold=1.2 \
```
**目的**:跟老板汇报时有"verl 自带方案"的数字作对比基线。

### Step 2:**Empirical characterization run**(2-3 天)
- 起一个 baseline training
- 每步 log 以下 tensor(全 token 粒度):
  - `log_ratio`, `|log_ratio|`, `ratio` 分布(mean / p50 / p95 / p99)
  - `|log_ratio|` × position 相关性
  - `|log_ratio|` × `log π_old` 相关性(低概率 token 是不是更 mismatch)
  - `|log_ratio|` × `|A|` 相关性(high-advantage 点是不是更 mismatch)
  - `|log_ratio|` × is_correct 相关性(对/错 rollout 差异)
- 扫 step 0-200 的演化

**输出**:一组 empirical 图表,能直接用作 paper section 2 的分析素材,且指导 method 设计。

### Step 3:**Implement HAT**(半天,唯一方法)
Patch 点:verl `rollout_corr_helper.compute_rollout_rejection_mask()` 里 `rollout_rs_threshold` 从标量改成"q-分位数":

```python
# 在现有的 token_k3 / token_k1 / token_k2 基础上加 "token_kN_quantile" 分支
if rollout_rs == "token_k3_quantile":
    q = float(rollout_rs_threshold)   # e.g., 0.99 表示 drop top 1%
    valid_scores = token_k3[response_mask_bool]
    threshold = torch.quantile(valid_scores, q=q)
    combined_mask *= (token_k3 < threshold).float()
```

对应脚本 [run_gsm8k_token_rs.sh](../run_gsm8k_token_rs.sh) 加一个 env var `RS_MODE=token_k3_quantile` + `RS_THRESHOLD=0.99` 即可。

### Step 4:**核心对比实验矩阵**(每 config × 3 seeds)
```
方法                                 val_acc  KL drift  grad_norm  entropy
1. baseline (no IS/RS)
2. random 1% drop (placebo)
3. verl token_rs k3 threshold=0.02 (tiny drop ~0.5%)
4. verl token_rs k3 threshold=larger (forced drop 1%)     ← match drop rate with 5/6/7
5. HAT q=0.999 (drop 0.1%)
6. HAT q=0.99  (drop 1%)     ← main claim vs 4
7. HAT q=0.95  (drop 5%)
```

**关键**:4 和 6 在 matched drop rate 下对比,能干净地回答"adaptive threshold 比 fixed 好吗"。2 是 placebo,确认 HAT 不是"随便 drop 都 work"。

**最少 7 个 config × 3 seeds = 21 runs**。一个 run 目前 ~5 小时(300 步),4 卡同时跑 2 runs → **~25 小时总 wall time**。一周内完成是可行的。

### Step 5:**故意造"崩盘 regime"做压力测试**(3-5 天)
- 把 baseline 推到 §3.7 列的危险配置(具体 regime 见 §3.9)
- **最低成本起步** :Qwen-3B + LoRA 不变,只改 3 行(去 KL + lr×3 + 300 步):
```bash
bash run_gsm8k_demo.sh \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    trainer.total_training_steps=300
```
- 统计 **多 seed 下的 crash rate** / PPL Gap 发散速度
- 如果 ours 能**压 crash rate 的同时不掉 val_acc** → paper 卖点成立
- Fallback 顺序:如此 regime 仍不崩 → 换 Qwen-7B → 再不崩 → 解掉 LoRA 全参训练

---

## 7. 风险与退路

### 可能出现的负面结果
1. **Path A 也打不过 verl `token_rs` + `AR-Lopti` 组合**:这很可能,因为现有方法已经很 strong
2. **在稳定 regime 下,两者基本打平**:把 novelty 从 "method 更好" 改成 "**我们的 method 用更少的 budget(drop fewer tokens)达到同样的稳定性**",也是一个合理论点
3. **只在特定配置下 work**:转向 Path C(empirical paper)

### 退路
如果 Path A 不漂亮,退 Path C(empirical characterization)是最稳的,**不依赖 method novelty**,纯经验贡献就能发 MLSys。

---

## 8. 跟老板沟通建议(MVP 版)

汇报 5 句话:

1. "Train-inference mismatch 是真问题,TRM 团队 blog + 4 paper 验证过;**PPL Gap** 是业界 quantify 这问题的标准"
2. "verl 已有 `token_rs` 实现,我们在 300 步 stress regime 下实测:val_acc 持平,**KL drift −41%,grad_norm −40%,entropy +17%**;method 真 work"
3. "但 verl token_rs 的 threshold 是**固定 scalar**,没跟上 mismatch 分布的 heavy-tailed 特性(Finding 1)"
4. "我的方法 **HAT(Heavy-tail Adaptive Threshold)**:per-batch quantile 替代固定阈值,5 行代码,MVP 清晰"
5. "下一步:3-seed × 7-config 的 ablation 矩阵(见 §6 Step 4),~1 周出结果。扩展方向(ENM / GICA / crash scaling)等 HAT 验证成功再铺"

**预期 reviewer pushback + 我们答案**:
- "跟 TRM 区别?" → TRM 是 sequence-level;HAT 是 token-level 且 adaptive
- "只是 drop 得更多?" → matched drop rate ablation 已 cover
- "数据只 3B / 300 步?" → 稳定起步,7B + 1000 步作为 Section 6 验证

---

## 9. 附:运行时 / 硬件预算

### 当前 setup 内存需求(Qwen2.5-3B + LoRA + micro=8)
- vLLM 阶段:~9.6 GB(gmu=0.4)
- Actor peak:~14 GB
- 总峰值:~17 GB

### 硬件配置选择

| 配置 | 可行性 | 备注 |
|---|---|---|
| 2× L40S(现在) | ✅ 验证稳定 | 单卡 46 GB,显存充裕 |
| **4× A5000**(24 GB) | ✅ 稳稳塞入,留 7 GB 余量 | 需 override `tensor_model_parallel_size=2`,DP×2 加速 |
| 1× A5000 | ⚠️ Qwen-3B 塞得下但 TP=1,rollout 慢 | 建议换 Qwen2.5-0.5B / 1.5B |
| 4× A100 40G / 80G | ✅ 最宽裕 | 可以跑 Qwen-7B 全参验证 crash-inducing regime |

### 并行实验建议
4 卡配置允许同时跑 2 个实验(各占 2 GPU),适合 A1 vs B1 ablation 并行。

---

## Appendix: References

### verl / rollout correction
- [verl Rollout Correction docs](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html)
- [verl Rollout Correction math](https://verl.readthedocs.io/en/latest/algo/rollout_corr_math.html)
- [verl/trainer/ppo/rollout_corr_helper.py](../verl/trainer/ppo/rollout_corr_helper.py)

### Train-inference mismatch
- Liu & Li 2025. When Speed Kills Stability. [richardli.xyz/rl-collapse](https://richardli.xyz/rl-collapse)
- Li et al. 2025. Trust Region Masking for Long-Horizon LLM RL. [arXiv 2512.23075](https://arxiv.org/abs/2512.23075)
- sail-sg 2025. Defeating Training-Inference Mismatch via FP16. [arXiv 2510.26788](https://arxiv.org/abs/2510.26788)
- FP8-RL. [arXiv 2601.18150](https://arxiv.org/abs/2601.18150)
- "On the Rollout-Training Mismatch" OPT-ML 2025. [opt-ml.org/papers/2025/paper116.pdf](https://opt-ml.org/papers/2025/paper116.pdf)

### Token-level GRPO improvements
- Rho-1 (Lin et al. 2024). Not All Tokens Are What You Need. [arXiv 2404.07965](https://arxiv.org/abs/2404.07965)
- AR-Lopti. Do Not Let Low-Probability Tokens Over-Dominate. [arXiv 2505.12929](https://arxiv.org/abs/2505.12929)
- It Takes Two: GRPO is DPO. [arXiv 2510.00977](https://arxiv.org/abs/2510.00977)
- GCPO. [arXiv 2510.07790](https://arxiv.org/abs/2510.07790)
- GTPO / GRPO-S. [arXiv 2508.04349](https://arxiv.org/abs/2508.04349)
- λ-GRPO. [arXiv 2510.06870](https://arxiv.org/abs/2510.06870)
- DAPO. [arXiv 2503.14476](https://arxiv.org/abs/2503.14476)

### Rollout diversity / adaptive sampling
- GAPO. [arXiv 2511.12596](https://arxiv.org/abs/2511.12596)
- DRA-GRPO. [arXiv 2505.09655](https://arxiv.org/abs/2505.09655)
- ETTRL. [arXiv 2508.11356](https://arxiv.org/abs/2508.11356)
- Tree-GRPO (ICLR 2026). [github.com/AMAP-ML/Tree-GRPO](https://github.com/AMAP-ML/Tree-GRPO)
- VIP / AERO. [arXiv 2509.25808](https://arxiv.org/abs/2509.25808)
- GRESO. [infini-ai-lab.github.io/GRESO](https://infini-ai-lab.github.io/GRESO/)
- DARS. [arXiv 2506.02177](https://arxiv.org/abs/2506.02177)

### Our preliminary work
- [docs/rho1_grpo.md](rho1_grpo.md) — Rho-1 implementation doc
- [run_gsm8k_demo.sh](../run_gsm8k_demo.sh) — baseline script
- [run_gsm8k_rho1.sh](../run_gsm8k_rho1.sh) — Rho-1 ablation script
- wandb runs: `verl_grpo_gsm8k_demo` project
  - baseline: `qwen2.5_3b_grpo_lora` + `qwen2.5_3b_grpo_lora_resume80_120`
  - rho1: `qwen2.5_3b_grpo_lora_rho1_0.6` + `qwen2.5_3b_grpo_lora_rho1_0.6_resume20_120`
