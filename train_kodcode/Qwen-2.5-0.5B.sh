#!/bin/bash
set -x

BASE_MODEL=${1:-"Qwen/Qwen2.5-Coder-0.5B-Instruct"}
MODEL_NICKNAME=$(echo $BASE_MODEL | cut -d'/' -f2)
DATASET=${3:-"kodcode-3k"}
TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)

RUN_NAME=${MODEL_NICKNAME}-${DATASET}-${TIMESTAMP}

# Create logs directory with datetime subdirectory
LOG_DIR="./logs/${TIMESTAMP}" # Define log directory with datetime format
mkdir -p $LOG_DIR # Create the directory if it doesn't exist

GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)



# Force disable Flash Attention globally
train_files="['./data/$DATASET/train.parquet']"
test_files="['./data/$DATASET/test.parquet']"

# Individually add the configuration settings that disable flash attention
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=False \
    data.truncation='left' \
    data.prompt_key=prompt \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$GPUS_PER_NODE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.top_k=50 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    reward_model.reward_manager=prime \
    custom_reward_function.path=evaluate_code/__init__.py \
    custom_reward_function.name=compute_score \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='Qwen2.5-0.5B-Instruct-RL-04-10-2025' \
    trainer.experiment_name='Qwen2.5-0.5B-Instruct_evalplus_optimized' \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.resume_mode=disable \
    trainer.save_freq=20 \
    trainer.test_freq=2 \
    trainer.total_epochs=15 2>&1 | tee "${LOG_DIR}/${RUN_NAME}.log"