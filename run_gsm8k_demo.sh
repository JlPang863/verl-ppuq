#!/usr/bin/env bash
# End-to-end GSM8K GRPO demo: Qwen2.5-3B-Instruct + LoRA on 2 GPUs.
# Usage:
#   source venv/bin/activate
#   bash run_gsm8k_demo.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
CKPT_DIR="${CKPT_DIR:-/mnt/data1/jinlong/ckpts/qwen2.5_3b_grpo_lora}"
GPUS="${GPUS:-0,1}"
N_GPUS=$(awk -F',' '{print NF}' <<< "$GPUS")

echo "Data dir : $DATA_DIR"
echo "Model    : $MODEL_PATH"
echo "Ckpt dir : $CKPT_DIR"
echo "GPUs     : $GPUS (n=$N_GPUS)"

# 1. Data preprocessing (skip if parquet already exists)
if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/test.parquet" ]]; then
    echo "[1/2] Preprocessing GSM8K -> $DATA_DIR"
    python3 examples/data_preprocess/gsm8k.py --local_save_dir "$DATA_DIR"
else
    echo "[1/2] Found cached parquet at $DATA_DIR, skipping preprocess."
fi

# 2. GRPO + LoRA training
echo "[2/2] Launching GRPO training on GPUs=$GPUS"
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
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_gsm8k_demo' \
    trainer.experiment_name='qwen2.5_3b_grpo_lora' \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 "$@"

# Rho-1 token selection (off by default, see docs/rho1_grpo.md):
#   bash run_gsm8k_demo.sh \
#       actor_rollout_ref.actor.rho1.enable=True \
#       actor_rollout_ref.actor.rho1.select_ratio=0.6
