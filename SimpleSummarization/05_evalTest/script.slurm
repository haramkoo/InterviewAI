#!/bin/bash
#
# Reserve one GPU
#SBATCH --gres=gpu:1
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
	--model_name_or_path facebook/bart-base \
	--output_dir ./model/evalTest5 \
	--overwrite_output_dir TRUE \
	--train_file ./data/transformers_train_small.json \
	--validation_file ./data/transformers_validation_small.json \
	--do_train TRUE \
	--do_eval TRUE \
	--per_device_train_batch_size 4 \
	--per_device_eval_batch_size 4 \
	--num_train_epochs 10 \
	--predict_with_generate TRUE
