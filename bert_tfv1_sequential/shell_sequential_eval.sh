#!/usr/bin/env bash

python3.6 run_sequential.py \
  --task_name=excavation \
  --data_dir=../data_excavation \
  --output_dir=../model_excavation \
  --gpu_id=0 \
  --num_layers=0 \
  --with_multibatch=0 \
  --save_checkpoints_steps=100 \
  --do_train=false \
  --do_eval=true \
  --max_seq_length=196 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=10
