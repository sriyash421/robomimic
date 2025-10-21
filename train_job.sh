#!/usr/bin/env bash
# run_robomimic.sbatch
#SBATCH --job-name=mean_flow-robomimic
#SBATCH --account=socialrl
#SBATCH --partition=ckpt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

cd /gscratch/socialrl/sriyash/robomimic
source "/gscratch/weirdlab/sriyash/anaconda3/etc/profile.d/conda.sh"
conda activate robomimic_venv

python robomimic/scripts/train.py \
  --config $1 \
  --name $2 \
  --dataset $3
