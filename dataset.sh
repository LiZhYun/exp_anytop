#!/bin/bash
#SBATCH --job-name=dataset
#SBATCH --mem=64G
#SBATCH --time=0-05:00:00
#SBATCH --output=logs/data_%j.out
#SBATCH --error=logs/data_%j.err

module load mamba
module load triton-dev/2025.1-gcc
module load gcc/13.3.0
module load cuda/12.6.2
export HF_HOME=/$WRKDIR/.huggingface_cache
export PIP_CACHE_DIR=/$WRKDIR/.pip_cache
export CONDA_PKGS_DIRS=/$WRKDIR/.conda_pkgs
export CONDA_ENVS_PATH=/$WRKDIR/.conda_envs
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_EXTENSIONS_DIR=$WRKDIR/torch_extensions
export WANDB_DIR=/$WRKDIR/wandb
export WANDB_CACHE_DIR=/$WRKDIR/wandb_cache

source activate anytop

export PYTHONPATH="${PWD}:$PYTHONPATH"

srun python -m utils.create_dataset
