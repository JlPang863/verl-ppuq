#!/usr/bin/env bash
# 今晚双 job 启动器:
#   Job A(GPU 0,1): run_gsm8k_token_rs.sh —— verl token-IS + token-K3 RS,跑 200 步
#   Job B(GPU 2,3): run_gsm8k_mismatch_analysis.sh —— 纯 baseline + measurement,跑 200 步
#
# 用法:
#   source venv/bin/activate
#   bash run_tonight.sh
#
# 完了看:
#   ls -lt logs/tonight_*.log                                 # 两个 log
#   wandb: https://wandb.ai/jlpang863-university-of-california/verl_grpo_gsm8k_demo
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
mkdir -p logs

TS=$(date +%Y%m%d_%H%M%S)

# 环境变量——两个 job 共享 HF cache,但 Ray TMPDIR 分开避免串
export HF_HOME="${HF_HOME:-/mnt/data1/jinlong/hf_cache}"
mkdir -p "$HF_HOME"

# ----- Job A: token_rs (GPU 0,1) -----
export RAY_TMPDIR=/mnt/data1/jinlong/ray_tmp_tokenrs
mkdir -p "$RAY_TMPDIR"
LOG_A=logs/tonight_tokenrs_${TS}.log
echo "[Job A] launching token_rs on GPU 0,1 → $LOG_A"
GPUS=0,1 nohup bash run_gsm8k_token_rs.sh \
    trainer.total_training_steps=200 \
    > "$LOG_A" 2>&1 &
PID_A=$!
echo "[Job A] PID=$PID_A"

sleep 3  # 给 A 一点时间初始化 Ray,避免抢端口

# ----- Job B: mismatch_analysis (GPU 2,3) -----
export RAY_TMPDIR=/mnt/data1/jinlong/ray_tmp_analysis
mkdir -p "$RAY_TMPDIR"
LOG_B=logs/tonight_analysis_${TS}.log
echo "[Job B] launching mismatch_analysis on GPU 2,3 → $LOG_B"
GPUS=2,3 nohup bash run_gsm8k_mismatch_analysis.sh \
    trainer.total_training_steps=200 \
    > "$LOG_B" 2>&1 &
PID_B=$!
echo "[Job B] PID=$PID_B"

echo ""
echo "================================================================"
echo "两个 job 已启动,在后台跑。"
echo "  Job A PID : $PID_A  (token_rs,  GPU 0,1, log: $LOG_A)"
echo "  Job B PID : $PID_B  (mismatch, GPU 2,3, log: $LOG_B)"
echo ""
echo "进度查看:"
echo "  tail -f $LOG_A"
echo "  tail -f $LOG_B"
echo "  nvidia-smi"
echo ""
echo "停止(真需要时):"
echo "  kill -9 $PID_A $PID_B; pkill -9 -f 'verl.trainer.main_ppo'"
echo "================================================================"
