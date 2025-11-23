#!/bin/bash
#SBATCH --job-name=vlm_sam3_benchmark
#SBATCH --output=benchmark_val_%j.out
#SBATCH --error=benchmark_val_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00

python benchmark.py --data_dir /media/M2SSD/FSC147 --num_samples 0 --split val
