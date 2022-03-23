#!/bin/bash
#
# Reserve four GPUs
#SBATCH --gres=gpu:4
#
# Reserve four CPU cores
#SBATCH -c 4
#
# Reserve 128GB RAM
#SBATCH --mem=128G
#
#SBATCH -p gpus
#
# Run the program
python ./run_clm.py \
	--model_name_or_path google/bigbird-roberta-base \
	--output_dir ./model/bigbird-clm \
	--overwrite_output_dir TRUE \
	--train_file ./data/train_lm.txt \
	--validation_file ./data/valid_lm.txt \
	--do_train TRUE \
	--do_eval TRUE \
	--per_device_train_batch_size 1 \
	--per_device_eval_batch_size 1 \
	--num_train_epochs 10 \
	--gradient_accumulation_steps 8 \
	--eval_accumulation_steps 8 \
	--fp16 TRUE \
	--save_total_limit 10 \
	--block_size 4096 \
	--gradient_checkpointing TRUE \
