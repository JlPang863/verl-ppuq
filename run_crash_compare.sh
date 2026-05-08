#!/usr/bin/env bash
# Crash-inducing regime 对比(300 步):
#   Job C(GPU 0,1): baseline + no KL + lr×3  —— 看 baseline 会不会崩
#   Job D(GPU 2,3): token_rs + 同上          —— 看 token_rs 能不能救场
#
# 配置(两边完全一致,只差 token_rs 开关):
#   - actor.kl_loss_coef = 0          (去掉 KL 锚,policy 可以自由漂)
#   - actor.optim.lr     = 1e-5       (比 baseline 3e-6 大 3×)
#   - total_training_steps = 300
#
# 预计时间 ~8 小时
#
# 用法:
#   source venv/bin/activate
#   bash run_crash_compare.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
mkdir -p logs

TS=$(date +%Y%m%d_%H%M%S)
export HF_HOME="${HF_HOME:-/mnt/data1/jinlong/hf_cache}"
mkdir -p "$HF_HOME"

# Shared crash-inducing overrides
COMMON_OVERRIDES=(
    "actor_rollout_ref.actor.kl_loss_coef=0.0"
    "actor_rollout_ref.actor.optim.lr=1e-5"
    "trainer.total_training_steps=300"
)

# ----- Job C: baseline(no KL + high lr)on GPU 0,1 -----
export RAY_TMPDIR=/mnt/data1/jinlong/ray_tmp_crash_base
mkdir -p "$RAY_TMPDIR"
LOG_C=logs/crash_baseline_${TS}.log
echo "[Job C] baseline crash-regime on GPU 0,1 → $LOG_C"
GPUS=0,1 \
CKPT_DIR=/mnt/data1/jinlong/ckpts/baseline_crash_noKL_lr1e5 \
nohup bash run_gsm8k_demo.sh \
    "${COMMON_OVERRIDES[@]}" \
    trainer.experiment_name=baseline_crash_noKL_lr1e5 \
    > "$LOG_C" 2>&1 &
PID_C=$!
echo "[Job C] PID=$PID_C"

sleep 3

# ----- Job D: token_rs(no KL + high lr)on GPU 2,3 -----
export RAY_TMPDIR=/mnt/data1/jinlong/ray_tmp_crash_rs
mkdir -p "$RAY_TMPDIR"
LOG_D=logs/crash_tokenrs_${TS}.log
echo "[Job D] token_rs crash-regime on GPU 2,3 → $LOG_D"
GPUS=2,3 \
EXP_NAME=tokenrs_crash_noKL_lr1e5 \
nohup bash run_gsm8k_token_rs.sh \
    "${COMMON_OVERRIDES[@]}" \
    > "$LOG_D" 2>&1 &
PID_D=$!
echo "[Job D] PID=$PID_D"

echo ""
echo "================================================================"
echo "Crash-inducing 对比实验已启动,预计 ~8 小时"
echo "  Job C PID : $PID_C  (baseline, GPU 0,1, log: $LOG_C)"
echo "  Job D PID : $PID_D  (token_rs, GPU 2,3, log: $LOG_D)"
echo ""
echo "Config(两边一致):kl_loss_coef=0, lr=1e-5, total_steps=300"
echo ""
echo "进度:"
echo "  tail -f $LOG_C"
echo "  tail -f $LOG_D"
echo ""
echo "停止:"
echo "  kill -9 $PID_C $PID_D; pkill -9 -f 'verl.trainer.main_ppo'"
echo "================================================================"
