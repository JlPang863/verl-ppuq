"""Parse crash-regime C vs D logs, dump CSV + generate comparison plots.

Usage:
    python docs/analyze_crash_compare.py
Outputs:
    docs/crash_compare_C_baseline.csv
    docs/crash_compare_D_tokenrs.csv
    docs/crash_compare_overview.png    (4-panel: val_acc, kl, grad_norm, entropy)
    docs/crash_compare_mismatch.png    (Job D only: PPL Gap evolution + drop fraction)
"""
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
LOGS = REPO / "logs"
OUT = REPO / "docs"

KEYS = [
    "val-core/openai/gsm8k/acc/mean@1",
    "critic/rewards/mean",
    "actor/kl_loss", "actor/grad_norm", "actor/entropy",
    "actor/pg_loss", "actor/lr",
    "response_length/mean",
    "rollout_corr/log_ppl_abs_diff",
    "rollout_corr/log_ppl_diff_max",
    "rollout_corr/chi2_seq", "rollout_corr/chi2_token",
    "rollout_corr/rollout_rs_token_k3_masked_fraction",
    "rollout_corr/rollout_rs_token_k3_max",
    "training/rollout_probs_diff_max",
    "training/rollout_probs_diff_mean",
]


def parse(*logs):
    rows = {}
    for log in logs:
        for line in Path(log).read_text().splitlines():
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


C = parse(LOGS / "crash_baseline_20260421_135250.log",
          LOGS / "crash_baseline_resume_20260421_205750.log")
D = parse(LOGS / "crash_tokenrs_20260421_135250.log",
          LOGS / "crash_tokenrs_resume_20260421_205750.log")
print(f"C: {len(C)} steps, last={C['step'].max()}")
print(f"D: {len(D)} steps, last={D['step'].max()}")

C.to_csv(OUT / "crash_compare_C_baseline.csv", index=False)
D.to_csv(OUT / "crash_compare_D_tokenrs.csv", index=False)
print(f"wrote CSVs to {OUT}")

# ---------- Overview plot ----------
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

def plot(ax, key, ylabel, title, logy=False):
    for df, label, color in [(C, "baseline (no IS/RS)", "#d62728"),
                              (D, "token_rs (IS+K3-RS)", "#1f77b4")]:
        sub = df.dropna(subset=[key])
        ax.plot(sub["step"], sub[key], label=label, color=color, linewidth=1.4, alpha=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    if logy:
        ax.set_yscale("log")

plot(axes[0, 0], "val-core/openai/gsm8k/acc/mean@1", "val_acc", "Val accuracy (GSM8K test)")
plot(axes[0, 1], "actor/kl_loss", "KL(π_θ || π_ref)", "Policy drift from ref (observed; coef=0)")
plot(axes[1, 0], "actor/grad_norm", "grad_norm", "Gradient norm (training stability)")
plot(axes[1, 1], "actor/entropy", "entropy", "Policy entropy (mode-collapse indicator)")

plt.suptitle(
    "Crash-inducing regime comparison  (300 steps, Qwen2.5-3B + LoRA, no-KL, lr=1e-5)\n"
    "baseline (red)  vs  verl token_rs token-IS + token-K3 RS (blue)",
    fontsize=11,
)
plt.tight_layout()
outp = OUT / "crash_compare_overview.png"
plt.savefig(outp, dpi=140, bbox_inches="tight")
print(f"saved {outp}")

# ---------- Mismatch plot (D only) ----------
fig, axes = plt.subplots(2, 2, figsize=(11, 7))

def plotD(ax, key, ylabel, title):
    sub = D.dropna(subset=[key])
    ax.plot(sub["step"], sub[key], color="#2ca02c", linewidth=1.3)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

plotD(axes[0, 0], "rollout_corr/log_ppl_abs_diff", "E[|log π_θ - log π_roll|]", "PPL Gap (mean)")
plotD(axes[0, 1], "rollout_corr/log_ppl_diff_max", "max per-token mismatch", "PPL Gap (max tail)")
plotD(axes[1, 0], "rollout_corr/rollout_rs_token_k3_masked_fraction", "fraction of tokens dropped", "token_rs drop rate")
plotD(axes[1, 1], "rollout_corr/chi2_seq", "chi² divergence (seq-level)", "Sequence-level chi²")
plt.suptitle("Mismatch evolution (token_rs run D, 300 steps, crash regime)", fontsize=11)
plt.tight_layout()
outp = OUT / "crash_compare_mismatch.png"
plt.savefig(outp, dpi=140, bbox_inches="tight")
print(f"saved {outp}")

# Summary table on last 50 steps
print("\n=== last 50 steps (250–300) means ===")
tail_C = C[C["step"] >= 250]
tail_D = D[D["step"] >= 250]
for key in ["val-core/openai/gsm8k/acc/mean@1", "actor/kl_loss", "actor/grad_norm", "actor/entropy", "response_length/mean"]:
    c = tail_C[key].mean() if key in tail_C.columns else None
    d = tail_D[key].mean() if key in tail_D.columns else None
    if c is None or pd.isna(c) or d is None or pd.isna(d):
        continue
    delta = d - c
    pct = 100 * delta / c if abs(c) > 1e-9 else float("nan")
    print(f"  {key:<45s}  C={c:.4f}  D={d:.4f}  Δ={delta:+.4f}  ({pct:+.1f}%)")
