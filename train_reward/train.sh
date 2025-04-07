#!/bin/bash
set -x

# Force disable Flash Attention globally
evalplus_train_path="./rl_data/train.parquet"
evalplus_test_path="./rl_data/test.parquet"

train_files="['$evalplus_train_path']"
test_files="['$evalplus_test_path']"

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
    actor_rollout_ref.model.path=models/merged_model \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.top_k=50 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    reward_model.reward_manager=prime \
    custom_reward_function.path=custom_module.py \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='Qwen2.5-3B-Instruct-RL' \
    trainer.experiment_name='Qwen2.5-3B-Instruct_evalplus_optimized' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.resume_mode=disable \
    trainer.save_freq=20 \
    trainer.test_freq=2 \
    trainer.total_epochs=15 $@