#!/bin/bash
#SBATCH -o log/%j-run_kendall_probes.log
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1

# set environment variables
export PATH=$ORDINAL_PROBING_ROOT:$PATH

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export RESULTS_DIR=/home/gridsan/wgurnee/mechint/ordinal-probing/results
export FEATURE_DATASET_DIR=$ORDINAL_PROBING_ROOT/feature_datasets/processed_datasets
export TRANSFORMERS_CACHE=/home/gridsan/groups/maia_mechint/models
export HF_DATASETS_CACHE=/home/gridsan/groups/maia_mechint/ordinal_probing/hf_home
export HF_HOME=/home/gridsan/groups/maia_mechint/ordinal_probing/hf_home

sleep 0.1  # wait for paths to update

# activate environment and load modules
source $ORDINAL_PROBING_ROOT/ord/bin/activate

sleep 0.1  # wait for paths to update

python -u train_probes.py \
    --experiment_name $1 \
    --experiment_type kendall_train \
    --model $2 \
    --entity_type $3 \
    --feature_name $4 \
    --prompt_name $5 \
    --probe_device cuda