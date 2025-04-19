#!/bin/bash
#SBATCH --job-name=rank_gpt
#SBATCH --partition=short-unkillable                           # Ask for unkillable job
#SBATCH --cpus-per-task=24                             # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --ntasks-per-node=1                                  # Ask for 1 GPU
#SBATCH --mem=128G           
#SBATCH --time=3:00:00                                    
#SBATCH --output=./slurm_logs/deeprerank/job_output-%j.txt
#SBATCH --error=./slurm_logs/deeprerank/job_error-%j.txt 

module load cudatoolkit/12.6.0
module load miniconda/3
conda init
conda activate ir

python run_evaluation.py