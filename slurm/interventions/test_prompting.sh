#!/bin/bash
#SBATCH -o log/%j-prompting.log
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

PYTHIA_MODELS=('pythia-70m' 'pythia-160m' 'pythia-410m' 'pythia-1b' 'pythia-1.4b' 'pythia-2.8b' 'pythia-6.9b')

for model in "${PYTHIA_MODELS[@]}"
do
    python -u intervention.py \
        --model $model \
        --entity_type $1 \
        --experiment_name test_prompting \
        --feature_name $2
done