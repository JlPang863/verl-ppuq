#!/usr/bin/env bash
# verl 自带的 token-level rejection sampling(按训推不一致丢 token)对比实验。
# 参考 examples/rollout_correction/run_with_rollout_corr.sh + run_gsm8k_demo.sh
#
# 关键:
#   - actor_rollout_ref.rollout.calculate_log_probs=True        让 vLLM 返回 log_probs
#   - algorithm.rollout_correction.rollout_rs=token_k3           按 K3 KL 估计丢 token
#   - algorithm.rollout_correction.rollout_rs_threshold=$TH      阈值(超过就丢)
#
# 用法:
#   source venv/bin/activate
#   bash run_gsm8k_token_rs.sh                     # 默认 token_k3, threshold=0.02, no IS
#   RS_MODE=token_k2 RS_THRESHOLD=0.01 bash run_gsm8k_token_rs.sh
#   IS_MODE=token IS_THRESHOLD=2.0 bash run_gsm8k_token_rs.sh    # 叠加 token-TIS
#   bash run_gsm8k_token_rs.sh trainer.total_training_steps=20  # 烟雾测试
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"

# --- Rollout Correction 配置 ---
# 默认 token-IS + token-K3 RS 组合,verl 推荐,最 robust
IS_MODE="${IS_MODE:-token}"                # null | token | sequence
IS_THRESHOLD="${IS_THRESHOLD:-2.0}"        # IS weight 上限;IS_MODE=null 时忽略

# RS(rejection sampling)— 按 mismatch 丢 token
RS_MODE="${RS_MODE:-token_k3}"             # token_k1 | token_k2 | token_k3 | seq_*
RS_THRESHOLD="${RS_THRESHOLD:-0.02}"       # K3 推荐 0.001-0.1,0.02 温和丢

EXP_NAME="${EXP_NAME:-qwen2.5_3b_grpo_lora_${RS_MODE}_t${RS_THRESHOLD}}"
CKPT_DIR="${CKPT_DIR:-/mnt/data1/jinlong/ckpts/${EXP_NAME}}"
GPUS="${GPUS:-0,1}"
N_GPUS=$(awk -F',' '{print NF}' <<< "$GPUS")

echo "Data dir     : $DATA_DIR"
echo "Model        : $MODEL_PATH"
echo "Ckpt dir     : $CKPT_DIR"
echo "Exp name     : $EXP_NAME"
echo "RS mode      : $RS_MODE (threshold=$RS_THRESHOLD)"
echo "IS mode      : $IS_MODE (threshold=$IS_THRESHOLD)"
echo "GPUs         : $GPUS (n=$N_GPUS)"

# 1. Data preprocessing
if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/test.parquet" ]]; then
    echo "[1/2] Preprocessing GSM8K -> $DATA_DIR"
    python3 examples/data_preprocess/gsm8k.py --local_save_dir "$DATA_DIR"
else
    echo "[1/2] Found cached parquet at $DATA_DIR, skipping preprocess."
fi

# 2. GRPO + LoRA + verl token_rs
echo "[2/2] Launching GRPO with rollout_rs=$RS_MODE on GPUs=$GPUS"
CUDA_VISIBLE_DEVICES="$GPUS" python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.val_before_train=False \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$N_GPUS" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.rollout_is="$IS_MODE" \
    algorithm.rollout_correction.rollout_is_threshold="$IS_THRESHOLD" \
    algorithm.rollout_correction.rollout_rs="$RS_MODE" \
    algorithm.rollout_correction.rollout_rs_threshold="$RS_THRESHOLD" \
    algorithm.rollout_correction.bypass_mode=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_gsm8k_demo' \
    trainer.experiment_name="$EXP_NAME" \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 "$@"
