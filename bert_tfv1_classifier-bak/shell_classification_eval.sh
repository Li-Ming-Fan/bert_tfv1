#!/usr/bin/env bash

python3.6 run_classifier.py \
  --task_name=sim \
  --data_dir=../data_sim \
  --output_dir=../model_sim \
  --gpu_id=0 \
  --with_multibatch=0 \
  --do_train=false \
  --do_eval=true \
  --max_seq_length=20 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=10
