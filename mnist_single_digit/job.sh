#!/bin/bash
#SBATCH --job-name=ddpm-train
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/fqureshi/scratch/ddpm-output/logs/%x-%j.out
#SBATCH --error=/home/fqureshi/scratch/ddpm-output/logs/%x-%j.err

# Python Environment
module load python/3.12
module load python-build-bundle/2024a
module load scipy-stack/2025a
module load opencv/4.12.0
module load arrow/21.0.0
source "$HOME/venv/ddpm/bin/activate"

DATA_ROOT="/home/fqureshi/scratch/hf_cache"
OUTDIR="/home/fqureshi/scratch/ddpm-output"

python train.py \
  --data-root="$DATA_ROOT" \
  --outdir="$OUTDIR" \
  --epochs=1002 \
  --auto-resume \
  --amp