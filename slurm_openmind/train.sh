#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH -o log/%j-train_probes.log
#SBATCH --mem-per-cpu=4G
#SBATCH -N 1
#SBATCH -c 12

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

sleep 0.1  # wait for paths to update

python -u probe_experiment.py \
    --model $1 \
    --entity_type $2 \
    --experiment_name $3 \
    --prompt_name $4 \
    --feature_name $5