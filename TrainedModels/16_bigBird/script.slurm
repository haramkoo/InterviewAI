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
python ./run_summarization.py \
	--model_name_or_path google/bigbird-pegasus-large-arxiv \
	--output_dir ./model/bigbird \
	--overwrite_output_dir TRUE \
	--train_file ./data/train_trunc.json \
	--validation_file ./data/valid_trunc.json \
	--do_train TRUE \
	--do_eval TRUE \
	--per_device_train_batch_size 1 \
	--per_device_eval_batch_size 1 \
	--num_train_epochs 10 \
	--predict_with_generate TRUE \
	--gradient_accumulation_steps 4 \
	--fp16 TRUE
