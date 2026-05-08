# Rho-1 Token Selection 移植到 GRPO —— 设计与实现

## 0. 一句话说明

把 [Rho-1](https://arxiv.org/abs/2404.07965) 的"只对高价值 token 反向传播"思想嫁接到 verl 的 GRPO 训练上。
具体做法:在 `compute_policy_loss` 调用前,用 `log π_ref - log π_θ` 给每个 response token 打分,**每条 response 各自保留 top-K%(默认 60%)**,其余 token 的 mask 清零;**KL / entropy 仍走完整 mask**。

---

## 1. 背景与动机

### 1.1 GRPO loss 流水线

```
reward (sequence-level 0/1)
    ↓
advantage  A  (sequence-level scalar,广播到所有 response token)
    ↓
ratio_t = exp(log π_θ - log π_old)  (per-token)
    ↓
pg_loss_t = -A · clip(ratio_t)     (per-token, shape (bs, L))
    ↓
× response_mask                    ← ① 选哪些 token  (Rho-1 在这里改)
    ↓
agg_loss(loss_agg_mode=...)        ← ② 怎么聚合  (Dr. GRPO / DAPO 在这里改)
    ↓
+ β · KL(π_θ ‖ π_ref)              ← KL 锚
    ↓
backward
```

三个旋钮(clip / mask / agg)正交,可叠加。Rho-1 只动 mask。

### 1.2 为什么动 mask 可能有用

- GSM8K response 里大部分 token 是 boilerplate(空格、连接词、标点),它们在 ref / policy 下概率都很高,贡献的 PG 信号是噪音。
- 选 `log π_ref - log π_θ` 大的 token,等于过滤掉 boilerplate(那些位置 score ≈ 0),把梯度集中到关键的推理 token 上。
- 期望效果:variance reduction + 更聚焦的更新。

### 1.3 Score 选 `log π_ref - log π_θ` 而非别的

| 候选 | 公式 | 取舍 |
|---|---|---|
| **ref-vs-θ excess** ✅ 默认 | `log π_ref(t) - log π_θ(t)` | 最贴近原 Rho-1,需要 `use_kl_loss=True` |
| old-vs-θ ratio | `\|log π_θ(t) - log π_old(t)\|` | 不需要 ref,off-policy 更稳,后续可加为可选项 |
| advantage × ratio | `\|A · (ratio - 1)\|` | A 是 sequence 常数,在 GRPO 里实质等价于 ratio 项 |
| token entropy | `H_π(t)` | 选"决策点 token",DAPO/GSPO 思路 |

`log π_ref - log π_θ` 在数学上 **等价于** SFT Rho-1 的 `loss_θ - loss_ref`(差一个负号),写成 log_prob 形式更自然(GRPO 没有 SFT 意义上的 per-token CE loss)。

---

## 2. 三个已知的"坑"

### 坑 1:Loss 归一化被自动放大
- `loss_agg_mode=token-mean` 默认 `Σ(loss·mask) / mask.sum()`。
- Rho-1 把 mask.sum() 砍到 60%,分母变小 → 单 token 等效权重 ×1/0.6 ≈ 1.67。
- **副作用**:有效学习率被隐式放大,grad_norm 易飙。
- **对策**(可选):用原始 `response_mask.sum()` 做分母关闭放大效果。本实现**默认接受放大**(Rho-1 集中火力的本意)。

### 坑 2:与 KL 锚的拉锯
- Rho-1 score 高 = 已经偏离 ref → 在这些 token 上加大 PG → KL 项再拉回 → 内耗。
- **观察指标**:`actor/kl_loss` 涨速。如果显著快于 baseline,可调低 `kl_loss_coef` 或暂时关 KL。

### 坑 3:冷启动期 score ≈ 0
- 训练第 0 步 `π_ref = π_θ` → score 全是 0 + 浮点噪声 → 选出来的 token 近乎随机。
- **对策**:加 `warmup_steps`(本实现暂未提供,如需可在循环外加判断;GSM8K 上经验值 50-100 步)。

### 坑 4(实现本身):Rho-1 是 token selection,不是 reward shaping
- GRPO 的 advantage 是 sequence scalar,Rho-1 不会改变 reward 信号方向,只改变"reward 分摊到哪些 token"。
- 别期待它能帮 GRPO 做 step-level credit assignment。

---

## 3. 实现细节

### 3.1 改动点(总共 3 处)

文件:`verl/workers/actor/dp_actor.py`

1. **顶部加 helper**(import 区下方):`_rho1_select_mask()`
2. **Rho-1 selection 代码块**:第 614 行 `# Compute policy loss` 之前
3. **替换 PG 调用的 mask 参数**:第 619 行 `response_mask=response_mask` → `response_mask=pg_response_mask`

KL / entropy 那两块**不动**,继续用原 `response_mask`。

### 3.2 Helper 实现

```python
def _rho1_select_mask(score: torch.Tensor, base_mask: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    """对每条 response 各自取 score 的 top-keep_ratio token,返回 0/1 mask。

    Args:
        score: (bs, L) per-token 分数(高=该保留)
        base_mask: (bs, L) 原始 response_mask(0/1)
        keep_ratio: float in (0,1],保留比例

    Returns:
        (bs, L) 0/1 mask,保证 ⊂ base_mask
    """
    bsz, L = score.shape
    valid_n = base_mask.sum(dim=-1)                                    # (bs,)
    k_per = (valid_n * keep_ratio).clamp(min=1).long()                 # (bs,)
    score_m = score.masked_fill(base_mask == 0, float("-inf"))
    k_max = int(k_per.max().item())
    if k_max == 0:
        return torch.zeros_like(base_mask)
    topk_idx = torch.topk(score_m, k=k_max, dim=-1).indices            # (bs, k_max)
    keep = torch.arange(k_max, device=score.device).unsqueeze(0) < k_per.unsqueeze(-1)
    out = torch.zeros_like(base_mask)
    rows = torch.arange(bsz, device=score.device).unsqueeze(-1).expand_as(topk_idx)
    out[rows, topk_idx] = keep.to(out.dtype)
    return out * base_mask
```

### 3.3 Selection 代码块

```python
# === Rho-1 token selection (optional) ===
pg_response_mask = response_mask
rho1_cfg = self.config.get("rho1", None)
if rho1_cfg is not None and rho1_cfg.get("enable", False):
    if not self.config.use_kl_loss:
        raise ValueError("rho1 requires use_kl_loss=True so that ref_log_prob is loaded")
    ref_lp = model_inputs["ref_log_prob"]
    score = (ref_lp - log_prob.detach()) * response_mask               # excess
    keep_ratio = float(rho1_cfg.get("select_ratio", 0.6))
    pg_response_mask = _rho1_select_mask(score, response_mask, keep_ratio)

    with torch.no_grad():
        kept = pg_response_mask
        dropped = response_mask * (1.0 - pg_response_mask)
        denom_resp = response_mask.sum().clamp(min=1)
        denom_kept = kept.sum().clamp(min=1)
        denom_drop = dropped.sum().clamp(min=1)
        micro_batch_metrics["rho1/keep_ratio_actual"] = (kept.sum() / denom_resp).item()
        micro_batch_metrics["rho1/score_kept_mean"] = ((score * kept).sum() / denom_kept).item()
        micro_batch_metrics["rho1/score_dropped_mean"] = ((score * dropped).sum() / denom_drop).item()
        micro_batch_metrics["rho1/score_gap"] = (
            micro_batch_metrics["rho1/score_kept_mean"] - micro_batch_metrics["rho1/score_dropped_mean"]
        )
```

### 3.4 调用替换

```python
pg_loss, pg_metrics = policy_loss_fn(
    old_log_prob=old_log_prob,
    log_prob=log_prob,
    advantages=advantages,
    response_mask=pg_response_mask,    # ← 唯一改动
    loss_agg_mode=loss_agg_mode,
    config=self.config,
    rollout_is_weights=rollout_is_weights,
)
```

---

## 4. 怎么开关

### 默认:**关闭**(zero-cost when off)
不加任何参数 → 行为与原版 GRPO 完全一致。

### 启用 Rho-1
脚本里追加 hydra override(`rho1` 已是注册字段,**不要**加 `+` 前缀):
```bash
bash run_gsm8k_demo.sh \
    actor_rollout_ref.actor.rho1.enable=True \
    actor_rollout_ref.actor.rho1.select_ratio=0.6
```
或直接用现成的 [run_gsm8k_rho1.sh](../run_gsm8k_rho1.sh)。

### 关键参数
| 参数 | 默认 | 含义 |
|---|---|---|
| `actor.rho1.enable` | `False` | 是否启用 Rho-1 |
| `actor.rho1.select_ratio` | `0.6` | 每条 response 保留比例 ∈ (0, 1] |

---

## 5. wandb 上要看的指标

启用 Rho-1 后会多出来:

| metric | 健康范围 | 说明 |
|---|---|---|
| `rho1/keep_ratio_actual` | ≈ select_ratio | 实际保留比例,应贴近设定 |
| `rho1/score_kept_mean` | > 0 | 保留 token 的平均 score |
| `rho1/score_dropped_mean` | ≤ 0 | 丢弃 token 的平均 score |
| `rho1/score_gap` | **越大越好** | 反映 selection 信号强度;接近 0 → selection ≈ 随机,Rho-1 失效 |
| `actor/kl_loss` | 跟 baseline 相当 | 显著快于 baseline → 跟 KL 锚拉锯 |
| `val/test_score/openai/gsm8k` | 优于 baseline | 终极判据 |

---

## 6. 推荐对照实验

1. baseline:`bash run_gsm8k_demo.sh trainer.total_training_steps=300`
2. Rho-1 60%:`bash run_gsm8k_demo.sh trainer.total_training_steps=300 +actor_rollout_ref.actor.rho1.enable=True +actor_rollout_ref.actor.rho1.select_ratio=0.6`
3. Rho-1 80%:同上 + `+actor_rollout_ref.actor.rho1.select_ratio=0.8`
4. Rho-1 40%:同上 + `+actor_rollout_ref.actor.rho1.select_ratio=0.4`

观察 `val/test_score` + `actor/kl_loss` + `response_length/mean` 三条曲线。

---

## 7. 后续可扩展项(本实现暂未做)

- [ ] `rho1.warmup_steps`:前 N 步关闭 Rho-1,避开冷启动噪声
- [ ] `rho1.score_mode=ref|old|entropy`:不同 selection 分数
- [ ] `rho1.normalize=fixed_denom`:用 `response_mask.sum()` 而非 `pg_mask.sum()` 做分母,关闭"自动放大"
- [ ] Megatron 后端同步:`verl/workers/actor/megatron_actor.py:520` 附近重复一遍
- [ ] `policy_loss_fn` 签名重构,把 `ref_log_prob` 作为可选 kwarg 传入,搬到 `core_algos.py`

---

## 8. 改动文件清单

| 文件 | 变化 |
|---|---|
| `verl/workers/config/actor.py` | 新增 `Rho1Config` dataclass,挂到 `ActorConfig.rho1` 字段(默认关) |
| `verl/workers/actor/dp_actor.py` | 加 helper `_rho1_select_mask` + selection block + 1 行参数替换 |
| `run_gsm8k_demo.sh` | (无强制改动,末尾注释里给出启用示例) |
| `run_gsm8k_rho1.sh` | 新脚本,默认开 Rho-1,实验名/ckpt 跟 baseline 分开 |
| `docs/rho1_grpo.md` | 本文档 |
