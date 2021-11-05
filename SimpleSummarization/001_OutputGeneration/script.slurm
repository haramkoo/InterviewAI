#!/bin/bash
#
# Reserve four CPU cores
#SBATCH -c 4
#
# Reserve 16GB RAM
#SBATCH --mem=16G
#
#
# Run the program
python ./generation.py
