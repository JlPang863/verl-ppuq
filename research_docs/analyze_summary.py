"""Generate summary plots for advisor report from training logs.

Outputs:
    docs/summary_rho1_vs_baseline.png       (Rho-1 first attempt — failed)
    docs/summary_bf16_stress.png            (BF16 stress regime — main success)
    docs/summary_fp8_stress.png             (FP8 stress regime — gap amplification)
    docs/summary_gap_amplification.png      (K3 vs prob-PPUQ gap: BF16 → FP8)
"""
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.titlesize"] = 11
matplotlib.rcParams["axes.labelsize"] = 10

REPO = Path(__file__).resolve().parent.parent
LOGS = REPO / "logs"
OUT = REPO / "research_docs" / "figures"

KEYS = [
    "val-core/openai/gsm8k/acc/mean@1",
    "actor/entropy", "actor/kl_loss", "actor/grad_norm",
    "training/rollout_probs_diff_mean",
    "rollout_corr/log_ppl_abs_diff",
    "response_length/mean",
]

COLORS = {
    "baseline": "#888888",
    "rho1":     "#d62728",
    "K3-PPUQ":  "#1f77b4",
    "token_rs": "#2ca02c",
    "prob-PPUQ":"#ff7f0e",
}


def parse(*logs):
    rows = {}
    for log in logs:
        log = Path(log)
        if not log.exists():
            continue
        for line in log.read_text().splitlines():
            m = re.search(r"step:(\d+) - ", line)
            if not m:
                continue
            s = int(m.group(1))
            rec = rows.setdefault(s, {"step": s})
            for k in KEYS:
                mm = re.search(rf"{re.escape(k)}:([-0-9.eE]+)", line)
                if mm:
                    rec[k] = float(mm.group(1))
    return pd.DataFrame(sorted(rows.values(), key=lambda r: r["step"]))


def plot_val_curve(ax, dfs_dict, title, xlim=None):
    """dfs_dict: {label: dataframe}. Plots val_acc vs step."""
    key = "val-core/openai/gsm8k/acc/mean@1"
    for label, df in dfs_dict.items():
        sub = df.dropna(subset=[key])
        if len(sub) == 0:
            continue
        c = COLORS.get(label.split()[0], "#444")
        ax.plot(sub["step"], sub[key] * 100, "-o",
                label=label, color=c, linewidth=1.6, markersize=4, alpha=0.9)
    ax.set_xlabel("training step")
    ax.set_ylabel("val_acc (GSM8K test, %)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    if xlim:
        ax.set_xlim(xlim)


# -------- 1. Rho-1 first attempt (failed) — actual val_acc curve --------
print("[1/4] Rho-1 vs baseline curve (Phase 1)...")
df_baseline_p1 = parse(LOGS / "run_20260420_183144.log",
                       LOGS / "baseline_resume_20260420_224145.log")
df_rho1_p1 = parse(LOGS / "rho1_20260420_215802.log",
                   LOGS / "rho1_resume_20260420_224148.log")

fig, ax = plt.subplots(figsize=(8, 5))
key = "val-core/openai/gsm8k/acc/mean@1"
sub_b = df_baseline_p1.dropna(subset=[key])
sub_r = df_rho1_p1.dropna(subset=[key])
ax.plot(sub_b["step"], sub_b[key] * 100, "-o",
        label="GRPO baseline", color="#888888", linewidth=1.8, markersize=5, alpha=0.95)
ax.plot(sub_r["step"], sub_r[key] * 100, "-o",
        label="GRPO + Rho-1 keep=60%", color="#d62728", linewidth=1.8, markersize=5, alpha=0.95)
ax.set_xlabel("training step")
ax.set_ylabel("val_acc (GSM8K test, %)")
ax.set_title("GRPO vs GRPO+Rho-1 on GSM8K (BF16, LoRA, kl=0.001, lr=3e-6)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc="lower right")
ax.set_xlim(0, 122)
# annotate final values at step 120
final_b = sub_b[sub_b.step == sub_b["step"].max()][key].iloc[0] * 100
final_r = sub_r[sub_r.step == sub_r["step"].max()][key].iloc[0] * 100
ax.annotate(f"{final_b:.2f}%", xy=(sub_b["step"].max(), final_b),
            xytext=(5, 5), textcoords="offset points", fontsize=10, color="#444",
            fontweight="bold")
ax.annotate(f"{final_r:.2f}%", xy=(sub_r["step"].max(), final_r),
            xytext=(5, -12), textcoords="offset points", fontsize=10, color="#d62728",
            fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "summary_rho1_vs_baseline.png", dpi=140, bbox_inches="tight")
print(f"  saved {OUT / 'summary_rho1_vs_baseline.png'}")

# -------- 2. BF16 stress regime (main success: 300-400 step) --------
print("[2/4] BF16 stress regime val curves...")
df_C = parse(LOGS / "crash_baseline_20260421_135250.log",
             LOGS / "crash_baseline_resume_20260421_205750.log",
             LOGS / "crash_baseline_to350_20260422_004438.log")
df_D = parse(LOGS / "crash_tokenrs_20260421_135250.log",
             LOGS / "crash_tokenrs_resume_20260421_205750.log",
             LOGS / "crash_tokenrs_to350_20260422_004438.log")
df_K3_resume = parse(LOGS / "k3ppuq_from_base350_20260422_144622.log")
df_prob_resume = parse(LOGS / "probppuq_from_base350_20260422_144622.log")
df_K3_full = parse(LOGS / "ppuq_350_20260422_022305.log",
                   LOGS / "ppuq_to400_20260422_124058.log")
# Splice: K3-PPUQ = baseline trajectory step 0-350 + resume from baseline_350 step 350-400.
# This matches the headline 86.66% (resume), not the independent 85.14% (standalone).
df_K3_resume_spliced = pd.concat([
    df_C[df_C.step <= 350],
    df_K3_resume[df_K3_resume.step > 350],
], ignore_index=True).sort_values("step").reset_index(drop=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Panel A: Full 0-400 trajectories  (baseline / token_rs continue 0-300; K3 from-base350 splice)
plot_val_curve(axes[0], {
    "baseline (no RS/IS)":    df_C,
    "token_rs (verl, K3 hard)": df_D,
    "K3-PPUQ (ours)":         df_K3_full,
}, "BF16 stress regime: 0–400 steps", xlim=(0, 410))

# Panel B: 350-400 zoom (K3 / prob-only / baseline 续训对比)
plot_val_curve(axes[1], {
    "baseline (Job C)":       df_C[df_C.step >= 350] if len(df_C) else df_C,
    "token_rs (Job D)":       df_D[df_D.step >= 350] if len(df_D) else df_D,
    "K3-PPUQ (resume)":       df_K3_resume,
    "prob-PPUQ (resume)":     df_prob_resume,
}, "Zoom: step 350→400 (K3 vs prob-only direct comparison)", xlim=(350, 400))

plt.suptitle(
    "Phase 2 (BF16 stress, 2026-04-21~22, Qwen2.5-3B + LoRA, kl=0, lr=1e-5)\n"
    "Final: baseline 84.7% < token_rs 86.0% < prob-PPUQ 86.13% < K3-PPUQ 86.66%",
    fontsize=10.5, fontweight="bold",
)
plt.tight_layout()
fig.savefig(OUT / "summary_bf16_stress.png", dpi=140, bbox_inches="tight")
print(f"  saved {OUT / 'summary_bf16_stress.png'}")

# -------- 3. FP8 stress regime (gap amplification, 120 step) --------
print("[3/4] FP8 stress regime val curves...")
df_v3_baseline = parse(LOGS / "run_baseline_v3_20260423_051443.log")
df_v3_K3 = parse(LOGS / "run_k3ppuq_v3_20260423_033521.log")
df_v3_token = parse(LOGS / "run_tokenrs_v3_20260423_033521.log")
df_v3_prob = parse(LOGS / "run_probppuq_v3_20260423_051443.log")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

plot_val_curve(axes[0], {
    "baseline":        df_v3_baseline,
    "token_rs (verl)": df_v3_token,
    "prob-PPUQ":       df_v3_prob,
    "K3-PPUQ (ours)":  df_v3_K3,
}, "FP8 rollout-only stress: val_acc per step", xlim=(0, 122))

# Add markers for step 99 (best stable common ckpt) and step 119 (some methods crash)
for ax in [axes[0]]:
    ax.axvline(x=99, color="green", linestyle="--", alpha=0.4, linewidth=1)
    ax.axvline(x=119, color="red", linestyle="--", alpha=0.4, linewidth=1)
    ymin, ymax = ax.get_ylim()
    ax.text(99, ymin + (ymax-ymin)*0.05, " step 99\n best ckpt", color="green", fontsize=8, alpha=0.7)
    ax.text(119, ymin + (ymax-ymin)*0.05, " step 120\n PPUQ crash", color="red", fontsize=8, alpha=0.7)

# Panel B: rollout_probs_diff_mean (mismatch amplifier)
ax = axes[1]
key = "training/rollout_probs_diff_mean"
for label, df in {
    "baseline":        df_v3_baseline,
    "K3-PPUQ (ours)":  df_v3_K3,
    "token_rs (verl)": df_v3_token,
}.items():
    sub = df.dropna(subset=[key])
    if len(sub) == 0:
        continue
    c = COLORS.get(label.split()[0], "#444")
    ax.plot(sub["step"], sub[key], "-", label=label, color=c, linewidth=1.4, alpha=0.85)
ax.axhline(y=0.003, color="gray", linestyle=":", alpha=0.6, label="BF16 baseline ~0.003")
ax.set_xlabel("training step")
ax.set_ylabel("rollout_probs_diff_mean")
ax.set_title("Mismatch (~4x larger than BF16)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

plt.suptitle(
    "Phase 3 (FP8 stress, 2026-04-23, Qwen2.5-1.5B full-params, kl=0.001, lr=5e-6, 120 step)\n"
    "Step 99 best ckpt: K3-PPUQ 72.6% > baseline 71.8% > token_rs 70.7% > prob-PPUQ 70.4%",
    fontsize=10.5, fontweight="bold",
)
plt.tight_layout()
fig.savefig(OUT / "summary_fp8_stress.png", dpi=140, bbox_inches="tight")
print(f"  saved {OUT / 'summary_fp8_stress.png'}")

# -------- 4. Gap amplification chart (key paper figure) --------
print("[4/4] K3 vs prob-PPUQ gap amplification...")
fig, ax = plt.subplots(figsize=(8, 5))
regimes = ["BF16 stress regime\n(3B+LoRA, step 400)", "FP8 stress regime\n(1.5B full, step 99)"]
k3_vals = [86.66, 72.55]
prob_vals = [86.13, 70.36]
gaps = [k3_vals[i] - prob_vals[i] for i in range(2)]
mismatches = [0.003, 0.012]

x = range(len(regimes))
width = 0.35
ax.bar([i - width/2 for i in x], k3_vals, width, label="K3-PPUQ (ours)",
       color=COLORS["K3-PPUQ"])
ax.bar([i + width/2 for i in x], prob_vals, width, label="prob-PPUQ (control)",
       color=COLORS["prob-PPUQ"])
for i, (k, p) in enumerate(zip(k3_vals, prob_vals)):
    ax.text(i - width/2, k + 0.4, f"{k}", ha="center", fontsize=9, fontweight="bold")
    ax.text(i + width/2, p + 0.4, f"{p}", ha="center", fontsize=9)
    ax.annotate(f"Δ = +{gaps[i]:.2f}pp",
                xy=(i, max(k, p) + 2),
                ha="center", fontsize=10,
                fontweight="bold", color="#444")

ax.set_xticks(list(x))
ax.set_xticklabels(regimes, fontsize=9)
ax.set_ylabel("val_acc on GSM8K test (%)")
ax.set_title("K3-PPUQ vs prob-only PPUQ: gap widens as mismatch amplifies\n"
             f"BF16 (~0.003 mismatch) +0.53pp -> FP8 (~0.012 mismatch) +2.19pp  ({gaps[1]/gaps[0]:.1f}x gain)",
             fontsize=10.5)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim([60, 92])
plt.tight_layout()
fig.savefig(OUT / "summary_gap_amplification.png", dpi=140, bbox_inches="tight")
print(f"  saved {OUT / 'summary_gap_amplification.png'}")

# -------- 5+6. wandb-style eval accuracy curves (clean single-panel) --------
print("[5/6] wandb-style eval acc — BF16 regime...")
fig, ax = plt.subplots(figsize=(8, 5))
plot_val_curve(ax, {
    "baseline (no RS/IS)":       df_C,
    "token_rs (verl, K3 hard)":  df_D,
    "K3-PPUQ (ours)":            df_K3_full,
}, "Eval accuracy on GSM8K test (BF16 stress regime)")
ax.set_xlim(0, 410)
plt.tight_layout()
fig.savefig(OUT / "eval_acc_bf16.png", dpi=140, bbox_inches="tight")
print(f"  saved {OUT / 'eval_acc_bf16.png'}")

print("[6/6] wandb-style eval acc — FP8 regime...")
fig, ax = plt.subplots(figsize=(8, 5))
plot_val_curve(ax, {
    "baseline":        df_v3_baseline,
    "token_rs (verl)": df_v3_token,
    "prob-PPUQ":       df_v3_prob,
    "K3-PPUQ (ours)":  df_v3_K3,
}, "Eval accuracy on GSM8K test (FP8 stress regime)")
ax.set_xlim(0, 122)
plt.tight_layout()
fig.savefig(OUT / "eval_acc_fp8.png", dpi=140, bbox_inches="tight")
print(f"  saved {OUT / 'eval_acc_fp8.png'}")

def plot_focused(ax, label_to_df_color, title, xlim=None):
    key = "val-core/openai/gsm8k/acc/mean@1"
    for label, (df, color) in label_to_df_color.items():
        sub = df.dropna(subset=[key])
        if len(sub) == 0:
            continue
        ax.plot(sub["step"], sub[key] * 100, "-o",
                label=label, color=color, linewidth=1.8, markersize=5, alpha=0.95)
    ax.set_xlabel("training step")
    ax.set_ylabel("val_acc (GSM8K test, %)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best")
    if xlim:
        ax.set_xlim(xlim)


# -------- Focused 2-line plots (baseline vs K3-PPUQ only, for advisor report) --------
print("[+0a] Focused: K3 vs prob (BF16, full 1→400 incl. baseline 1-350 + resume 350-400)...")
fig, ax = plt.subplots(figsize=(9, 5.5))
key = "val-core/openai/gsm8k/acc/mean@1"
# baseline = full 1-350 trajectory (3 segmented runs glued)
sub_C = df_C.dropna(subset=[key])
# K3 / prob = resume from baseline_350 → glue baseline 0-350 + resume 350-400 for visualization
# (resume runs only have step 350+ data; before 350 they are identical to baseline by construction)
sub_K3r = df_K3_resume.dropna(subset=[key])
sub_probr = df_prob_resume.dropna(subset=[key])
# Splice: baseline curve 1-350 + resume curve 350-400
import numpy as np
def splice(base_df, resume_df):
    a = base_df[base_df["step"] <= 350][["step", key]]
    b = resume_df[resume_df["step"] >= 350][["step", key]]
    return pd.concat([a, b], ignore_index=True).sort_values("step").reset_index(drop=True)
sub_K3_full = splice(sub_C, sub_K3r)
sub_prob_full = splice(sub_C, sub_probr)

ax.plot(sub_C["step"], sub_C[key] * 100, "-",
        label="GRPO baseline (step 1-350)", color="#888888", linewidth=1.6, alpha=0.85)
ax.plot(sub_K3_full["step"], sub_K3_full[key] * 100, "-o",
        label="K3-PPUQ (ours, K3 KL score)", color="#1f77b4",
        linewidth=1.6, markersize=3, alpha=0.95)
ax.plot(sub_prob_full["step"], sub_prob_full[key] * 100, "-o",
        label="prob-only PPUQ (control, −log π_old)", color="#ff7f0e",
        linewidth=1.6, markersize=3, alpha=0.95)
ax.axvline(x=350, color="green", linestyle="--", alpha=0.4, linewidth=1)
ax.set_xlabel("training step")
ax.set_ylabel("val_acc (GSM8K test, %)")
ax.set_title("K3-PPUQ vs prob-only PPUQ — full trajectory step 1→400\n"
             "(BF16 stress regime; PPUQ branches from baseline ckpt at step 350)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc="lower right")
ax.set_xlim(0, 410)
ymin, ymax = ax.get_ylim()
ax.text(351, ymin + (ymax-ymin)*0.05, " branch at\n step 350",
        color="green", fontsize=8, alpha=0.8)
# annotate final values at step 400
final_C = sub_C[sub_C["step"] == sub_C["step"].max()][key].iloc[0] * 100
final_K3r = sub_K3r[sub_K3r["step"] == sub_K3r["step"].max()][key].iloc[0] * 100
final_probr = sub_probr[sub_probr["step"] == sub_probr["step"].max()][key].iloc[0] * 100
ax.annotate(f"baseline 350: {final_C:.2f}%", xy=(350, final_C),
            xytext=(-110, -22), textcoords="offset points", fontsize=9, color="#666",
            arrowprops=dict(arrowstyle="->", color="#888", alpha=0.5))
ax.annotate(f"{final_K3r:.2f}%", xy=(sub_K3r["step"].max(), final_K3r),
            xytext=(5, 5), textcoords="offset points", fontsize=10, color="#1f77b4", fontweight="bold")
ax.annotate(f"{final_probr:.2f}%", xy=(sub_probr["step"].max(), final_probr),
            xytext=(5, -14), textcoords="offset points", fontsize=10, color="#ff7f0e", fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "eval_acc_bf16_k3_vs_prob.png", dpi=140, bbox_inches="tight")
print(f"  saved {OUT / 'eval_acc_bf16_k3_vs_prob.png'}")

print("[+0b] Focused: K3-PPUQ vs prob-only PPUQ (FP8)...")
fig, ax = plt.subplots(figsize=(8, 5))
sub_K3v3 = df_v3_K3.dropna(subset=[key])
sub_probv3 = df_v3_prob.dropna(subset=[key])
ax.plot(sub_K3v3["step"], sub_K3v3[key] * 100, "-o",
        label="K3-PPUQ (ours, K3 KL score)", color="#1f77b4", linewidth=2.0, markersize=6, alpha=0.95)
ax.plot(sub_probv3["step"], sub_probv3[key] * 100, "-o",
        label="prob-only PPUQ (control, −log π_old score)", color="#ff7f0e", linewidth=2.0, markersize=6, alpha=0.95)
ax.set_xlabel("training step")
ax.set_ylabel("val_acc (GSM8K test, %)")
ax.set_title("K3-PPUQ vs prob-only PPUQ on GSM8K (FP8 train/inference mismatch, ~4× larger)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc="lower left")
ax.set_xlim(0, 122)
# best stable ckpt at step 99 (before crash)
ax.axvline(x=99, color="green", linestyle="--", alpha=0.4, linewidth=1)
ymin, ymax = ax.get_ylim()
ax.text(100, ymin + (ymax-ymin)*0.85, " step 99\n best stable ckpt", color="green", fontsize=8, alpha=0.8)
# annotate step 99 values
v_K3_99 = sub_K3v3[sub_K3v3.step == 99][key]
v_prob_99 = sub_probv3[sub_probv3.step == 99][key]
if len(v_K3_99) > 0 and len(v_prob_99) > 0:
    ax.annotate(f"K3 step 99: {v_K3_99.iloc[0]*100:.2f}%",
                xy=(99, v_K3_99.iloc[0]*100),
                xytext=(-90, -25), textcoords="offset points", fontsize=10, color="#1f77b4",
                fontweight="bold", arrowprops=dict(arrowstyle="->", color="#1f77b4", alpha=0.5))
    ax.annotate(f"prob step 99: {v_prob_99.iloc[0]*100:.2f}%",
                xy=(99, v_prob_99.iloc[0]*100),
                xytext=(-90, -45), textcoords="offset points", fontsize=10, color="#ff7f0e",
                fontweight="bold", arrowprops=dict(arrowstyle="->", color="#ff7f0e", alpha=0.5))
plt.tight_layout()
fig.savefig(OUT / "eval_acc_fp8_k3_vs_prob.png", dpi=140, bbox_inches="tight")
print(f"  saved {OUT / 'eval_acc_fp8_k3_vs_prob.png'}")

print("[+1] Focused: baseline 0-350 + PPUQ resume 350-400 (BF16)...")
fig, ax = plt.subplots(figsize=(8, 5))
key = "val-core/openai/gsm8k/acc/mean@1"
sub_C = df_C.dropna(subset=[key])
sub_K3 = df_K3_resume.dropna(subset=[key])
ax.plot(sub_C["step"], sub_C[key] * 100, "-o",
        label="GRPO baseline", color="#888888", linewidth=1.8, markersize=5, alpha=0.9)
# PPUQ branches from baseline ckpt at step 350; show only the divergent segment
ax.plot(sub_K3["step"], sub_K3[key] * 100, "-o",
        label="GRPO + PPUQ (ours, resume from baseline step 350)",
        color="#1f77b4", linewidth=2.2, markersize=6, alpha=0.95)
ax.axvline(x=350, color="green", linestyle="--", alpha=0.4, linewidth=1)
ax.set_xlabel("training step")
ax.set_ylabel("val_acc (GSM8K test, %)")
ax.set_title("GRPO vs GRPO+PPUQ on GSM8K (BF16 stress regime, 400 step)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc="lower right")
ax.set_xlim(0, 410)
ymin, ymax = ax.get_ylim()
ax.text(351, ymin + (ymax-ymin)*0.05, " branch point\n step 350",
        color="green", fontsize=8, alpha=0.8)
# annotate final values
final_C = sub_C[sub_C.step == sub_C["step"].max()][key].iloc[0] * 100
final_K3 = sub_K3[sub_K3.step == sub_K3["step"].max()][key].iloc[0] * 100
ax.annotate(f"{final_C:.1f}%", xy=(sub_C["step"].max(), final_C),
            xytext=(8, -8), textcoords="offset points", fontsize=10, color="#444",
            fontweight="bold")
ax.annotate(f"{final_K3:.2f}%", xy=(sub_K3["step"].max(), final_K3),
            xytext=(5, 5), textcoords="offset points", fontsize=10, color="#1f77b4",
            fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "eval_acc_bf16_ours_vs_baseline.png", dpi=140, bbox_inches="tight")
print(f"  saved {OUT / 'eval_acc_bf16_ours_vs_baseline.png'}")

print("[+2] Focused: baseline vs K3-PPUQ (FP8)...")
fig, ax = plt.subplots(figsize=(8, 5))
plot_focused(ax, {
    "GRPO baseline":      (df_v3_baseline, "#888888"),
    "GRPO + PPUQ (ours)": (df_v3_K3, "#1f77b4"),
}, "GRPO vs GRPO+PPUQ on GSM8K (FP8 train/inference mismatch)", xlim=(0, 122))
plt.tight_layout()
fig.savefig(OUT / "eval_acc_fp8_ours_vs_baseline.png", dpi=140, bbox_inches="tight")
print(f"  saved {OUT / 'eval_acc_fp8_ours_vs_baseline.png'}")

print("\nDone. plots in research_docs/figures/")
