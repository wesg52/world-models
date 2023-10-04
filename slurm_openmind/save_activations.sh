#!/bin/bash
#SBATCH -t 03:00:00 
#SBATCH -o log/%j-save_activations.log
#SBATCH -N 1
#SBATCH -c 20
#SBATCH --mem-per-cpu=5G
#SBATCH --gres=gpu:a100:2

# set environment variables
export PATH=$ORDINAL_PROBING_ROOT:$PATH

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export RESULTS_DIR=/om2/user/wesg/world_models/results
export ACTIVATION_DATASET_DIR=/om2/user/wesg/world_models/activation_datasets
export FEATURE_DATASET_DIR=$ORDINAL_PROBING_ROOT/feature_datasets/processed_datasets
export TRANSFORMERS_CACHE=/om/user/wesg/models
export HF_HOME=/om/user/wesg/models

sleep 0.1  # wait for paths to update

# activate environment and load modules
/om2/user/wesg/anaconda/bin/activate interp

python -u save_activations.py --model $1 --entity_type $2 --batch_size $3 --prompt_name $4
