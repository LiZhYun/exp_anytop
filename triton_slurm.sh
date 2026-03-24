#!/bin/bash
#SBATCH --job-name=exp_anytop
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err

##SBATCH --gres=gpu:h200_2g.35gb:1
##SBATCH --partition=gpu-h200-35g-ia

module load mamba
module load triton-dev/2025.1-gcc
module load gcc/13.3.0
module load cuda/12.6.2
export HF_HOME=$WRKDIR/.huggingface_cache
export PIP_CACHE_DIR=$WRKDIR/.pip_cache
export CONDA_PKGS_DIRS=$WRKDIR/.conda_pkgs
export CONDA_ENVS_PATH=$WRKDIR/.conda_envs
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_EXTENSIONS_DIR=$WRKDIR/torch_extensions
export WANDB_DIR=$WRKDIR/wandb
export WANDB_CACHE_DIR=$WRKDIR/wandb_cache

source activate anytop

export PYTHONPATH="${PWD}:$PYTHONPATH"

COMMON_ARGS="
  --objects_subset all
  --lambda_geo 1.0
  --overwrite
  --balanced
  --ml_platform_type WandBPlatform
  --wandb_project anytop-conditioned-test
  --wandb_entity zhiyuanli
"

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    srun python -m train.train_anytop $COMMON_ARGS --model_prefix baseline
else
    srun python -m train.train_conditioned $COMMON_ARGS --model_prefix conditioned
fi
