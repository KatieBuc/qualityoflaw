#!/bin/bash
#SBATCH --job-name=mt5-ppl
#SBATCH --account=mg61
#SBATCH --partition=fit
#SBATCH --qos=fitq
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=A100-80G
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cchi0115@student.monash.edu
#SBATCH --output=logs/mt5-ppl-%j.out
#SBATCH --error=logs/mt5-ppl-%j.err

set -euo pipefail

PROJECT_ROOT="/home/cchi0115/mg61_scratch/chunheng/Foundation_model"
HF_ROOT="${HOME}/mg61_scratch/chunheng/chunheng/huggingface"

cd "${PROJECT_ROOT}"
mkdir -p logs "${HF_ROOT}/hub" "${HF_ROOT}/transformers"

module load miniforge3
module load cuda/12.2.0
conda activate mt5-ppl

export HF_HOME="${HF_ROOT}"
export HUGGINGFACE_HUB_CACHE="${HF_ROOT}/hub"
export TRANSFORMERS_CACHE="${HF_ROOT}/transformers"

echo "Job started on $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

python scripts/calculate_mt5_perplexity.py "$@"

echo "Job finished at $(date)"
