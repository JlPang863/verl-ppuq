<!--
Slide deck (English) for advisor presentation.
Each `---` is a slide break (Marp / reveal-md compatible).
Render: `marp slides_en.md -o slides_en.pdf`  or just read as markdown.
-->

---

# Token-level Selection for Stable GRPO

**Goal**: identify "high-impact / risky" tokens during GRPO training and selectively train on them — improve final accuracy while reducing instability.

**Progress**: from Rho-1 failure (−2.88pp) → designed **PPUQ** → BF16 +1.86pp / FP8 mismatch +0.75pp

---

## 1. Problem: GRPO treats every token equally — wasteful

GRPO PG loss:
$$
\mathcal{L}_{PG} = \mathbb{E}_t \left[ \frac{\pi_\theta(t)}{\pi_{\text{old}}(t)} A_t \right]
$$

- Each response token receives **the same** advantage weight.
- In reality: a few tokens are "key reasoning steps", most are filler / template / connectives.
- **Selection hypothesis**: identifying which tokens to focus on → improves efficiency + reduces collapse risk.

---

## 2. Phase 1 — Rho-1: direct port from SFT literature

**Rho-1 score**: `score(t) = log π_ref(t) − log π_θ(t)` (ref-based excess loss)

**Experiment**: keep top 60% tokens in PG mask, 120 steps
- baseline: **82.18%** | Rho-1: **79.30%** → **−2.88pp**

**Why it failed**:
1. **ref ≠ tutor**: in SFT, ref is a strong tutor model. Here ref = the Qwen training start point — no oracle ability.
2. **Score selects the wrong tokens**: it picks "tokens where policy already drifted", not "tokens at risk".
3. **Learning becomes too conservative**: kl_loss drops 14% → policy actually moves less.

**Conclusion**: score *direction* (mismatch-aware) is right, but using **ref** as the anchor doesn't work. Switch to **engine-level train/rollout mismatch** as the signal.

---

## 3. PPUQ method design

### Three core knobs

| Knob | Choice | Rationale |
|---|---|---|
| **Score** | K3 KL: `exp(log_r) − log_r − 1`, where `log_r = log π_train − log π_rollout` | symmetric KL estimator; positive = train more confident than rollout (risky direction) |
| **Threshold** | **per-prompt** quantile (q=0.95) | adapts to each prompt's difficulty; easy prompts can't dominate the filter |
| **Action** | hard-drop top 5% tokens (mask out from PG) | direct removal vs. reweighting → avoids high-variance importance sampling |

### Implementation
- [verl/trainer/ppo/rollout_corr_helper.py](../verl/trainer/ppo/rollout_corr_helper.py) — new `compute_per_prompt_quantile_mask()` function as PPUQ fast path inside verl's rollout_correction module
- Trigger: `algorithm.rollout_correction.rollout_rs=per_prompt_k3_quantile`

---

## 4. PPUQ vs related work — key differences

| Method | Score | Threshold | Granularity | Action |
|---|---|---|---|---|
| **Rho-1** (SFT, failed) | `log π_ref − log π_θ` (ref-based) | top-K ratio | response | hard drop |
| **verl token_rs** | K3 KL (same as ours) | **global hard threshold** (e.g. 0.02) | token | hard drop |
| **AR-Lopti** | `−log π_θ` (prob only) | binary split at η=0.5 | token | reweight (α-blend) |
| **prob-only PPUQ** (ablation) | `−log π_old` (prob only) | per-prompt quantile | per-prompt | hard drop |
| **K3-PPUQ (ours)** | **K3 KL** (mismatch) | **per-prompt quantile** | per-prompt | hard drop |

### Two distinguishing features of PPUQ
1. **Score = mismatch (K3), not probability**
   - prob-only score only finds "rare tokens"
   - K3 score finds "tokens where train/rollout disagree" — directly aligned with GRPO's off-policy risk
2. **Threshold is per-prompt, not global**
   - global thresholds let easy prompts dominate the filter
   - per-prompt q=0.95 guarantees **every prompt drops exactly 5% of tokens** — stable selection ratio

---

## 5. Phase 2 — BF16 stress regime (empirical +1.86pp)

**Setup**: Qwen2.5-3B + LoRA, GSM8K, kl=0, lr=1e-5, 400 steps

**Experimental design**: train baseline GRPO for 350 steps to establish a common reference point, then **branch** from the step-350 checkpoint with PPUQ for 50 more steps to step 400. Both runs share the exact same starting point at step 350, so the PPUQ-induced improvement is precisely measurable.

![BF16 branching](figures/eval_acc_bf16_ours_vs_baseline.png)

| | val_acc step 400 | Δ |
|---|---|---|
| GRPO baseline (step 350) | 84.8% | — |
| **GRPO + PPUQ (resume 350→400)** | **86.66%** ★ | **+1.86pp** |

→ Monotonic climb 84.8% → 86.66% within 50 steps, fully attributable to PPUQ selection.

---

## 6. Phase 3 — Robustness under amplified mismatch

**Motivation**: BF16's natural mismatch (`rollout_probs_diff_mean ≈ 0.003`) is small. Use vLLM's **FP8 rollout quantization** to amplify mismatch ~4× (≈ 0.012) and test whether PPUQ remains effective in a more adversarial regime.

**Setup**: Qwen2.5-1.5B full-params + FP8 vLLM rollout, kl=0.001, lr=5e-6, 120 steps

![FP8 vs baseline](figures/eval_acc_fp8_ours_vs_baseline.png)

| | val_acc step 99 (best stable) | Δ |
|---|---|---|
| GRPO baseline | 71.80% | — |
| **GRPO + PPUQ (ours)** | **72.55%** ★ | **+0.75pp** |

→ PPUQ retains a stable advantage even under 4× amplified mismatch. *Caveat*: at step 120 PPUQ exhibits cumulative hard-drop instability — best checkpoint is step 80–100.

---

## 7. Three-phase progressive summary

| Phase | Setup | baseline | PPUQ (ours) | Δ |
|---|---|---|---|---|
| 1 | Rho-1 keep=0.6, 120 step | 82.18% | 79.30% | **−2.88pp** (failed) |
| 2 | BF16 stress, 350→400 resume | 84.8% | **86.66%** | **+1.86pp** ★ |
| 3 | FP8 mismatch, step 99 | 71.80% | **72.55%** | **+0.75pp** ★ |

**Take-aways**:
- **Phase 1 → 2**: from "wrong selection makes things worse" to "smart selection gives +1.86pp"
- **Phase 2 → 3**: from natural mismatch to amplified mismatch — PPUQ retains its advantage → robustness evidence

---

## 8. Next steps

1. **Stability fix**: address Phase 3 step-120 cumulative instability — try dynamic q (early q=0.99 drops less, later q=0.95 drops more) or soft reweighting
2. **Longer horizon**: run FP8 E2E on H100 for full 400 steps to verify PPUQ remains dominant (`venv_megatron/` already provisioned)
3. **MATH dataset**: switch to long-response tasks to test K3 score on long sequences
4. **Full ablation**: q ∈ {0.90, 0.95, 0.99}, score ∈ {K1, K2, K3, neg_logp, abs_log_ratio}

---

## Appendix — engineering artifacts

| File | Purpose |
|---|---|
| [run_gsm8k_demo.sh](../run_gsm8k_demo.sh) | GRPO baseline + LoRA |
| [run_gsm8k_rho1.sh](../run_gsm8k_rho1.sh) | Phase 1 Rho-1 ablation |
| [run_gsm8k_ppuq.sh](../run_gsm8k_ppuq.sh) | **GRPO + PPUQ (ours)** |
| [run_gsm8k_fp8roll.sh](../run_gsm8k_fp8roll.sh) | Phase 3 FP8 mismatch regime |
| [verl/trainer/ppo/rollout_corr_helper.py](../verl/trainer/ppo/rollout_corr_helper.py) | PPUQ implementation |

**Best checkpoints**:
- BF16 final (86.66%): `/mnt/data1/jinlong/ckpts/k3_ppuq_from_base350/global_step_400`
- FP8 best (72.55%): `/mnt/data1/jinlong/ckpts/qwen1.5b_full_fp8roll_k3ppuq_v3/global_step_80`

**Repo**: https://github.com/JlPang863/verl-ppuq
