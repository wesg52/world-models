#!/bin/bash
#SBATCH -o log/%j-train_probes.log
#SBATCH --mem-per-cpu=4G
#SBATCH -N 1
#SBATCH -c 12

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

python -u probe_experiment.py \
    --model $1 \
    --entity_type $2 \
    --experiment_name $3 \
    --prompt_name $4 \
    --feature_name $5