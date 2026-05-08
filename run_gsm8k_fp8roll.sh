#!/usr/bin/env bash
# FSDP + LoRA + FP8 vLLM rollout (BF16 训练 / FP8 推理) on GSM8K.
# 用现有 venv/(FSDP 栈),不动任何代码;只是在 vLLM 这一侧开 quantization=fp8。
# 这会人为放大 train/inference mismatch (~100×),用作 PPUQ vs AR-Lopti 对照的 stress regime。
#
# 用法:
#   bash run_gsm8k_fp8roll.sh                                       # baseline (无 RS)
#   RS_MODE=per_prompt_k3_quantile RS_THRESHOLD=0.95 bash run_gsm8k_fp8roll.sh   # K3-PPUQ
#   RS_MODE=per_prompt_neg_logp_quantile bash run_gsm8k_fp8roll.sh              # prob-only PPUQ
#   RS_MODE=token_k3 bash run_gsm8k_fp8roll.sh                                  # verl token_rs (对照)
#   bash run_gsm8k_fp8roll.sh trainer.total_training_steps=3                    # 烟雾
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

VENV_DIR="${VENV_DIR:-$REPO_DIR/venv}"
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "[preflight] FSDP venv 不存在: $VENV_DIR"; exit 1
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"

IS_MODE="${IS_MODE:-null}"
IS_THRESHOLD="${IS_THRESHOLD:-2.0}"
RS_MODE="${RS_MODE:-null}"                          # 默认 baseline,不做 RS
RS_THRESHOLD="${RS_THRESHOLD:-0.95}"

# rollout precision: fp8 (default, the whole point) or null (退回 BF16 对照)
ROLL_QUANT="${ROLL_QUANT:-fp8}"

# LoRA rank: 0 = full-params (verl FP8 rollout 官方验证路径), >0 = LoRA (跟 FP8 有兼容问题)
LORA_RANK="${LORA_RANK:-0}"
LORA_ALPHA="${LORA_ALPHA:-32}"

_LORA_TAG=$([[ "$LORA_RANK" -gt 0 ]] && echo "lora${LORA_RANK}" || echo "full")
EXP_NAME="${EXP_NAME:-qwen2.5_3b_grpo_${_LORA_TAG}_fp8roll_${RS_MODE//\//_}_q${RS_THRESHOLD}}"
CKPT_DIR="${CKPT_DIR:-/mnt/data1/jinlong/ckpts/${EXP_NAME}}"
GPUS="${GPUS:-0,1}"
N_GPUS=$(awk -F',' '{print NF}' <<< "$GPUS")

QUANT_FLAGS=()
if [[ "$ROLL_QUANT" != "null" && -n "$ROLL_QUANT" ]]; then
    QUANT_FLAGS+=("+actor_rollout_ref.rollout.quantization=${ROLL_QUANT}")
fi

LORA_FLAGS=()
if [[ "$LORA_RANK" -gt 0 ]]; then
    LORA_FLAGS+=(
        "actor_rollout_ref.model.lora_rank=${LORA_RANK}"
        "actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}"
    )
fi

echo "Data dir   : $DATA_DIR"
echo "Model      : $MODEL_PATH"
echo "Ckpt dir   : $CKPT_DIR"
echo "Exp name   : $EXP_NAME"
echo "RS mode    : $RS_MODE (q_keep=$RS_THRESHOLD)"
echo "IS mode    : $IS_MODE"
echo "Roll quant : $ROLL_QUANT"
echo "LoRA rank  : $LORA_RANK ($_LORA_TAG)"
echo "GPUs       : $GPUS (n=$N_GPUS)"

if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/test.parquet" ]]; then
    echo "[1/2] Preprocessing GSM8K -> $DATA_DIR"
    python3 examples/data_preprocess/gsm8k.py --local_save_dir "$DATA_DIR"
else
    echo "[1/2] Found cached parquet at $DATA_DIR, skipping preprocess."
fi

echo "[2/2] Launching FSDP+LoRA+FP8-rollout GRPO on GPUs=$GPUS"
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
    "${LORA_FLAGS[@]}" \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    "${QUANT_FLAGS[@]}" \
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
