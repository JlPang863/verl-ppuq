#!/usr/bin/env bash
# FP8 End-to-End GRPO on Qwen2.5-3B-Instruct (dense) + GSM8K.
#
# 注意事项:
#   1. FP8 E2E 只支持 Megatron 后端(不是 FSDP)。LoRA 在 Megatron 路径是支持的(走
#      Megatron-Bridge 的 PEFT 集成),默认 LORA_RANK=0 = full params;设 LORA_RANK=64 启用 LoRA。
#   2. 需要一个**独立**的 venv: venv_megatron/ (见下面的 setup 说明);不会动你现有的 venv/。
#   3. blockwise FP8 recipe 在 L40s(SM89) + CUDA 12.8 上 verl 官方没验证过,跑不通就用
#      FP8_MODE=rollout_only 退回到 BF16 训练 + FP8 rollout。
#
# 用法:
#   bash run_gsm8k_fp8_e2e.sh                             # 默认: FP8 E2E + full params + 2 卡
#   FP8_MODE=rollout_only bash run_gsm8k_fp8_e2e.sh       # 降级: BF16 train + FP8 rollout
#   LORA_RANK=64 bash run_gsm8k_fp8_e2e.sh                # LoRA + FP8 E2E
#   GPUS=0,1,2,3 bash run_gsm8k_fp8_e2e.sh                # 4 卡
#   bash run_gsm8k_fp8_e2e.sh trainer.total_training_steps=5   # 烟雾测试
#
# --------------- 独立 venv setup(只跑一次,不会污染现有环境)-------------------
#   python3 -m venv venv_megatron
#   source venv_megatron/bin/activate
#   pip install -U pip setuptools wheel
#   pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
#   pip install -e .                                                   # verl
#   pip install "transformers==4.57.6" "vllm==0.11.0" "numpy<2.0"
#   pip install flash-attn==2.8.3 --no-build-isolation
#   pip install nvidia-modelopt nvidia-cuda-nvcc-cu12
#   pip install --no-deps "git+https://github.com/NVIDIA/Megatron-LM.git@core_dev_r0.16.0"
#   pip install --no-deps "git+https://github.com/NVIDIA-NeMo/Megatron-Bridge.git@v0.3.1"  # main 已要求 py3.12
#   pip install mbridge
#   # Transformer Engine: 需 cuDNN 头文件 + libs(走 venv 自带的)
#   export CUDNN_PATH="$PWD/venv_megatron/lib/python3.10/site-packages/nvidia/cudnn"
#   export CPATH="$CUDNN_PATH/include:${CPATH:-}"
#   export LIBRARY_PATH="$CUDNN_PATH/lib:${LIBRARY_PATH:-}"
#   export LD_LIBRARY_PATH="$CUDNN_PATH/lib:${LD_LIBRARY_PATH:-}"
#   export NVTE_FRAMEWORK=pytorch MAX_JOBS=8
#   pip install --no-build-isolation "transformer-engine[pytorch]"     # 编译 ~20 min
#   deactivate
# -----------------------------------------------------------------------------
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

VENV_DIR="${VENV_DIR:-$REPO_DIR/venv_megatron}"
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "[preflight] venv_megatron 不存在: $VENV_DIR"
    echo "  请先按照脚本顶部的 setup 说明创建一个独立 venv(不会动现有 venv/)。"
    exit 1
fi
# 启用独立 venv,避免污染当前 shell 的 venv/
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"

EXP_NAME="${EXP_NAME:-qwen2.5_3b_grpo_fp8_e2e}"
CKPT_DIR="${CKPT_DIR:-/mnt/data1/jinlong/ckpts/${EXP_NAME}}"
GPUS="${GPUS:-0,1}"
N_GPUS=$(awk -F',' '{print NF}' <<< "$GPUS")

# TP 默认 = 卡数,PP/EP/CP 全 1(dense 模型单节点场景)
TRAIN_TP="${TRAIN_TP:-$N_GPUS}"
TRAIN_PP="${TRAIN_PP:-1}"
GEN_TP="${GEN_TP:-$N_GPUS}"

# FP8 mode: "e2e" (train + rollout FP8) or "rollout_only" (BF16 train + FP8 rollout)
FP8_MODE="${FP8_MODE:-e2e}"
# LoRA: 0 disables (= full params); >0 enables LoRA with that rank
LORA_RANK="${LORA_RANK:-0}"
LORA_ALPHA="${LORA_ALPHA:-32}"

FP8_TRAIN_FLAGS=()
if [[ "$FP8_MODE" == "e2e" ]]; then
    FP8_TRAIN_FLAGS+=(
        '+actor_rollout_ref.actor.megatron.override_transformer_config.fp8=hybrid'
        '+actor_rollout_ref.actor.megatron.override_transformer_config.fp8_recipe=blockwise'
        '+actor_rollout_ref.actor.optim.override_optimizer_config.fp8_recipe=blockwise'
        '+actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True'
    )
fi

LORA_FLAGS=()
if [[ "$LORA_RANK" -gt 0 ]]; then
    LORA_FLAGS+=(
        "actor_rollout_ref.model.lora.rank=${LORA_RANK}"
        "actor_rollout_ref.model.lora.alpha=${LORA_ALPHA}"
        'actor_rollout_ref.model.lora.type=lora'
        'actor_rollout_ref.model.lora.merge=False'
    )
fi

# TE blockwise FP8 需要这个;CUDA < 12.9 也设了无害
export NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1
export HF_HOME="${HF_HOME:-/mnt/data1/jinlong/hf_cache}"
export VLLM_USE_V1=1

# 让 TE 在 runtime 找到 venv 自带的 cuDNN(系统没装 cuDNN)
_VENV_SP="$VENV_DIR/lib/python3.10/site-packages"
export CUDNN_PATH="$_VENV_SP/nvidia/cudnn"
export LD_LIBRARY_PATH="$CUDNN_PATH/lib:${LD_LIBRARY_PATH:-}"

echo "Data dir    : $DATA_DIR"
echo "Model       : $MODEL_PATH"
echo "Ckpt dir    : $CKPT_DIR"
echo "Exp name    : $EXP_NAME"
echo "GPUs        : $GPUS (n=$N_GPUS)"
echo "Train TP/PP : $TRAIN_TP / $TRAIN_PP  | Gen TP: $GEN_TP"
echo "Venv        : $VENV_DIR"

if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/test.parquet" ]]; then
    echo "[1/2] Preprocessing GSM8K -> $DATA_DIR"
    python3 examples/data_preprocess/gsm8k.py --local_save_dir "$DATA_DIR"
else
    echo "[1/2] Found cached parquet at $DATA_DIR, skipping preprocess."
fi

echo "[2/2] Launching GRPO + FP8 E2E on GPUs=$GPUS"
CUDA_VISIBLE_DEVICES="$GPUS" python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
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
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="$TRAIN_TP" \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="$TRAIN_PP" \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=null \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.context_parallel_size=1 \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    "${FP8_TRAIN_FLAGS[@]}" \
    "${LORA_FLAGS[@]}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size="$TRAIN_TP" \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size="$TRAIN_PP" \
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=null \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$GEN_TP" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    +actor_rollout_ref.rollout.quantization=fp8 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_gsm8k_demo' \
    trainer.experiment_name="$EXP_NAME" \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 "$@"
