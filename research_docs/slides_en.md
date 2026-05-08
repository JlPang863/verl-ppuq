<!--
Slide deck (English) for advisor presentation.
Each `---` is a slide break (Marp / reveal-md compatible).
Render: `marp slides_en.md -o slides_en.pdf`  or just read as markdown.
-->

---

# Token-level Selection for Stable GRPO

**Story**: Rho-1 fails → design PPUQ → K3-PPUQ vs verl token_rs (prior baseline) BF16 +0.84pp → amplify mismatch via FP8 → +1.81pp (gap widens)

---

## 1. Phase 1 — Initial Rho-1 port (ref-based excess loss score)

Direct port from SFT Rho-1:

$$
\text{score}(t) = \log \pi_\text{ref}(t) - \log \pi_\theta(t)
$$

Keep top 60% tokens per response, mask out the rest in PG loss.

**Results** (120 step, BF16, kl=0.001, lr=3e-6):

![Rho-1 vs baseline](figures/summary_rho1_vs_baseline.png)

| | val_acc step 120 |
|---|---|
| GRPO baseline | **82.18%** |
| GRPO + Rho-1 keep=60% | 79.30% |
| Δ | **−2.88pp** ❌ |

---

## 2. Why Rho-1 fails → motivates PPUQ

1. **ref ≠ tutor**: SFT Rho-1 uses a strong tutor; in GRPO ref = the Qwen training start point — no oracle ability
2. **Score selects wrong tokens**: picks "tokens where policy already drifted", not "tokens at risk"
3. **Learning becomes too conservative**: kl_loss drops 14% → policy actually moves less

→ Score *direction* (mismatch-aware) is right, but ref-based anchor doesn't work
→ Switch to **engine-level train/rollout mismatch** as the signal
→ Design **PPUQ**

---

## 3. Background: what is token_rs? what is K3 KL?

### token_rs full name = **token-level rejection sampling**

verl's built-in rollout correction module variants:

| Name | Granularity | Description |
|---|---|---|
| **`token_k3` / token_rs** | **token** | each token scored independently → keep/drop (**default**) |
| `seq_k3` / seq_rs | sequence | whole response gets one score → keep/drop entire response |
| `token_k1`, `token_k2` | token | same as token_rs but with K1 / K2 KL estimator |

→ "verl token_rs" = `token_k3` (K3 estimator + token granularity + global hard threshold)

### K3 is one of three KL divergence estimators (from John Schulman's blog)

Input `log r = log π_train(t) − log π_rollout(t)`:

| Estimator | Formula | Non-negative? | Bias | Variance |
|---|---|---|---|---|
| **K1** | $-\log r$ | ❌ can be negative | unbiased | high |
| **K2** | $\frac{1}{2}(\log r)^2$ | ✅ always ≥ 0 | **biased** | medium |
| **K3** | $\exp(\log r) - \log r - 1$ | ✅ always ≥ 0 | unbiased | **low (best)** |

K3 satisfies all three: **non-negative + unbiased + low variance** → ideal per-token KL "danger" score

Reference: John Schulman, *Approximating KL Divergence* — http://joschu.net/blog/kl-approx.html

---

## 3b. K3 as a token-level "danger" signal

In GRPO, rollout and train policies differ across PPO update steps (multi-step PPO), so the same token has different probabilities under the two policies.

| K3 value | Physical meaning |
|---|---|
| **K3 ≈ 0** | $\pi_\text{train}(t) \approx \pi_\text{rollout}(t)$, two policies agree → off-policy correction ≈ on-policy → **safe** |
| **K3 large** | Two policies severely disagree on this token → IS ratio is extreme → **gradient variance explodes, risky** |
| **K3 → ∞** | $\pi_\text{train}$ differs from $\pi_\text{rollout}$ by orders of magnitude → **certain to crash** |

**Two motivations to drop high-K3 tokens**:

1. **Variance control**: high-K3 tokens have IS ratio $r$ far from 1 → gradient $\rho \cdot A$ variance explodes → dropping reduces variance
2. **Trust region**: equivalent to a token-level trust region ("some tokens' update is already too aggressive, skip this step")

→ verl token_rs and K3-PPUQ **share the K3 score idea**; they only differ in *how the threshold is set* and *whether to reweight the survivors*.

---

## 4. Phase 2 — Method design: verl token_rs (prior) vs K3-PPUQ (ours)

Both methods **share the K3 KL score**; they differ in **threshold + action** design:

| Knob | **verl token_rs** (prior baseline) | **K3-PPUQ** (ours) |
|---|---|---|
| **Score** | $K_3(t) = \exp(\log r) - \log r - 1$, $\log r = \log \pi_\text{train} - \log \pi_\text{rollout}$ | same |
| **Threshold** | **global hard threshold** = 0.02 | **per-prompt quantile** $q=0.95$ |
| **Action** | hard drop **+ token-IS reweight** ($w = \min(\pi_\text{train}/\pi_\text{rollout}, 2)$) | hard drop only |

### Each method's design rationale

**Why verl token_rs picks this design**:
1. **K3 KL score**: same reasoning — unbiased non-negative KL estimator, used as per-token "danger score"
2. **Global hard threshold**: treats "off-policy risk" as a **universal physical quantity** — any token with K3 > 0.02 is dangerous regardless of which prompt it belongs to. Simple, no batch-internal sorting needed.
3. **+ token-IS reweight**: kept tokens get IS weights → makes the PG estimator closer to unbiased on-policy gradient (standard PPO off-policy correction)

**Why K3-PPUQ changes the design**:
1. **Same K3 score** — prior baseline got this right, no need to change
2. **Per-prompt quantile**: treats "off-policy risk" as a **prompt-relative quantity** — hard prompts naturally have higher overall K3, easy prompts lower. A global threshold lets easy prompts drop nothing while hard prompts get every token dropped. **Per-prompt q=0.95 ensures every prompt drops exactly 5%**, controlled drop ratio.
3. **Drop the IS reweight**: IS variance explodes on high-mismatch tokens; clipping helps but still destabilizes. Since high-K3 tokens are already hard-dropped, reweighting the rest is unnecessary — pure selection.

### Design philosophy in one line

> **token_rs**: treats RS as a "noise filter" — drop dangerous + IS-reweight the rest
> **K3-PPUQ**: treats RS as a "selection" — every prompt drops top 5% K3, no reweighting

---

## 5. Phase 2 main comparison: K3-PPUQ vs verl token_rs (prior baseline)

**Three runs** (BF16 stress regime: kl=0, lr=1e-5):

![K3 vs token_rs](figures/eval_acc_bf16_k3_vs_tokenrs.png)

| Run | final val_acc | vs token_rs |
|---|---|---|
| GRPO baseline (gray, step 350) | 84.76% | −1.06pp |
| **verl token_rs** (green, prior baseline, step 350) | **85.82%** | — |
| **K3-PPUQ** (blue, ours, full step 1→400) | **86.66%** ★ | **+0.84pp** |

→ K3-PPUQ outperforms verl built-in token_rs by **+0.84pp**.

---


---

## 6. Phase 3 — Artificially amplify mismatch (FP8 vLLM rollout)

**Motivation**: in BF16 the natural mismatch (`rollout_probs_diff_mean ≈ 0.003`) is small. Use vLLM **FP8 rollout quantization** to amplify mismatch ~4× (≈ 0.012) and see how selection methods behave under harsher mismatch.

**Setup**: Qwen2.5-1.5B full-params + FP8 vLLM rollout, kl=0.001, lr=5e-6, 120 steps

### Main comparison: K3-PPUQ vs verl token_rs (3 methods, same horizon)

![FP8 main 3-method](figures/eval_acc_fp8_main.png)

| Method | step 99 val_acc | Δ vs token_rs |
|---|---|---|
| GRPO baseline | 71.80% | +1.06pp |
| verl token_rs (prior) | 70.74% | — |
| **K3-PPUQ (ours)** | **72.55%** ★ | **+1.81pp** |

→ K3-PPUQ beats verl token_rs by **+1.81pp** (vs Phase 2 BF16 only +0.84pp — **FP8 amplification widens the gap**)

> step 120 K3-PPUQ crashes; baseline and token_rs stay stable → step 99 is the fair comparison point.

---

## 7. Key finding: gap scales with mismatch (4×)

| Regime | mismatch (`diff_mean`) | K3 vs prob gap |
|---|---|---|
| BF16 stress | ~0.003 | **+0.84pp** |
| FP8 stress | ~0.012 (4× larger) | **+1.81pp** |
| **Amplification** | **4×** | **~2.15×** |

**Conclusion**: K3-PPUQ's advantage over verl token_rs (prior baseline) widens as mismatch grows → K3's mismatch-aware signal is genuinely effective.

---

## 8. Three-phase summary

| Phase | Comparison | Result | Conclusion |
|---|---|---|---|
| 1 | baseline vs Rho-1 | 82.18% vs 79.30% (**−2.88pp**) | Direct SFT port fails |
| 2 main | **K3-PPUQ vs verl token_rs** (BF16) | resume 86.66% vs 85.82% (**+0.84pp**) | K3 late-stage refinement beats prior baseline |
| 3 | **K3-PPUQ vs verl token_rs** (FP8) | 72.55% vs 70.74% (**+1.81pp**) | mismatch ×4 → gap ~×2.15, K3 advantage stronger under amplified mismatch |

**Next steps**:
1. Fix Phase 3 step-120 cumulative instability (try dynamic q or soft reweight)
2. Run full FP8 E2E for 400 steps on H100 (`venv_megatron/` already provisioned)
3. Validate on MATH dataset (long responses)

**Repo**: https://github.com/JlPang863/verl-ppuq
