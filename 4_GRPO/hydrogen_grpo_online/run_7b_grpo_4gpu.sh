#!/bin/bash
# ==============================================================================
# Qwen2.5-7B-SFT GRPO LoRA 训练脚本（在线奖励函数版本）
# 框架: VERL v0.7.0
# 硬件: 4x A100 40GB
# 算法: GRPO（无 Critic）
# 微调: LoRA（lora_rank=64）
# 奖励: 纯执行准确率 (0/1)，完全匹配=1，否则=0
#
# 与离线版本的主要差异:
#   - 使用 reward_fn_online.py（纯执行准确率奖励）
#   - kl_loss_coef=0.03（增强 KL 约束防止策略漂移）
#   - n=16（增加采样数确保有足够正样本）
#   - lr=1e-6（降低学习率稳定训练）
#
# 运行前准备:
#   cd /ssd1/zhangyuzhe/verl-release-v0.7.0
#   python hydrogen_grpo_online/precompute_gold_answers.py
#   python hydrogen_grpo_online/data_preprocess.py
#
# 运行方式:
#   bash hydrogen_grpo_online/run_7b_grpo_4gpu.sh
# ==============================================================================

set -x

# 指定使用的 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="/ssd1/zhangyuzhe/LlamaFactory-main/saves/qwen25-7b-hydrogen/full/sft"
TRAIN_DATA="${SCRIPT_DIR}/data/train.parquet"
VAL_DATA="${SCRIPT_DIR}/data/test.parquet"
REWARD_FN_PATH="${SCRIPT_DIR}/reward_fn_online.py"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.val_before_train=False \
    \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    data.train_batch_size=40 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=True \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=40 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.03 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='hydrogen_patent_grpo_online' \
    trainer.experiment_name='qwen25_7b_hydrogen_grpo_online_4gpu' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.total_epochs=2 \
    \
    custom_reward_function.path="${REWARD_FN_PATH}" \
    custom_reward_function.name=compute_score \
    "$@"
