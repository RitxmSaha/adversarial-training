#!/bin/bash

set -x

torchrun -m verl.trainer.fsdp_sft_trainer \
    data.train_files=./sft_data/train.parquet \
    data.val_files=./sft_data/test.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    data.train_batch_size=8 \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain=Qwen/Qwen2.5-3B-Instruct \
    model.lora_rank=8 \
    model.lora_alpha=32 \
    model.target_modules=all-linear \
    trainer.default_local_dir=./models \
    trainer.default_hdfs_dir=hdfs://user/verl/experiments/humaneval/qwen2.5-3b-instruct/ \
    trainer.project_name=humaneval-sft \
    trainer.experiment_name=humaneval-sft-qwen2.5-3b-instruct-fixed \
    trainer.total_epochs=12 \
    trainer.logger=['console','wandb'] \
    optim.lr=5e-5 \
    optim.warmup_steps_ratio=0.2 \
    optim.weight_decay=0.0001
