#!/bin/bash
#
# Reserve one GPU
#SBATCH --gres=gpu:1
#
# Reserve one CPU core
#SBATCH -c 1
#
# Reserve 8GB RAM
#SBATCH --mem=8G
#
#SBATCH -p gpus
#
# Run the program
python ./run_summarization.py --model_name_or_path facebook/bart-large --output_dir ./output --train_file ./train.json --do_train TRUE
