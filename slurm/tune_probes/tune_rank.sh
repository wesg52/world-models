#!/bin/bash
#SBATCH -o log/%j-%a-run_probes.log
#SBATCH --mem-per-cpu=4G
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -a 1-6

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

model_name="pythia-70m"
    
python -u train_probes.py \
    --experiment_name rank_tune_2 \
    --experiment_type rank_tune \
    --model "$model_name" \
    --entity_type hex_color \
    --feature_name red


python -u train_probes.py \
    --experiment_name rank_tune_2 \
    --experiment_type rank_tune \
    --model "$model_name" \
    --entity_type city \
    --feature_name latitude

python -u train_probes.py \
    --experiment_name rank_tune_2 \
    --experiment_type rank_tune \
    --model "$model_name" \
    --entity_type city \
    --feature_name latitude \
    --normalization_type layer_norm


python -u train_probes.py \
    --experiment_name rank_tune_2 \
    --experiment_type rank_tune \
    --model "$model_name" \
    --entity_type token \
    --feature_name num


python -u train_probes.py \
    --experiment_name rank_tune_2 \
    --experiment_type rank_tune \
    --model "$model_name" \
    --entity_type token \
    --feature_name num \
    --normalization_type layer_norm


python -u train_probes.py \
    --experiment_name rank_tune_2 \
    --experiment_type rank_tune \
    --model "$model_name" \
    --entity_type hex_color \
    --feature_name red \
    --normalization_type layer_norm


python -u train_probes.py \
    --experiment_name rank_tune_2 \
    --experiment_type rank_tune \
    --model "$model_name" \
    --entity_type hex_color \
    --feature_name red \
    --normalization_type centering


python -u train_probes.py \
    --experiment_name rank_tune_2 \
    --experiment_type rank_tune \
    --model "$model_name" \
    --entity_type token \
    --feature_name len


python -u train_probes.py \
    --experiment_name rank_tune_2 \
    --experiment_type rank_tune \
    --model "$model_name" \
    --entity_type token \
    --feature_name len \
    --normalization_type layer_norm

python -u train_probes.py \
    --experiment_name rank_tune_2 \
    --experiment_type rank_tune \
    --model "$model_name" \
    --entity_type token \
    --feature_name alpha


python -u train_probes.py \
    --experiment_name rank_tune_2 \
    --experiment_type rank_tune \
    --model "$model_name" \
    --entity_type token \
    --feature_name alpha \
    --normalization_type layer_norm


python -u train_probes.py \
    --experiment_name rank_tune_2 \
    --experiment_type rank_tune \
    --model "$model_name" \
    --entity_type color \
    --feature_name blue


python -u train_probes.py \
    --experiment_name rank_tune_2 \
    --experiment_type rank_tune \
    --model "$model_name" \
    --entity_type color \
    --feature_name blue \
    --normalization_type layer_norm


python -u train_probes.py \
    --experiment_name rank_tune_2 \
    --experiment_type rank_tune \
    --model "$model_name" \
    --entity_type hex_color \
    --feature_name blue \
    --normalization_type centering

#     # testing
# python -u train_probes.py \
#     --experiment_name rank_tune_test \
#     --experiment_type rank_tune \
#     --model pythia-70m \
#     --entity_type hex_color \
#     --feature_name red


# python -u train_probes.py \
#     --experiment_name rank_tune_test \
#     --experiment_type rank_train \
#     --model pythia-70m \
#     --entity_type hex_color \
#     --feature_name red \
#     --normalization_type layer_norm


# python -u train_probes.py \
#     --experiment_name rank_tune_test \
#     --experiment_type rank_tune \
#     --model pythia-70m \
#     --entity_type token \
#     --feature_name num \
#     --normalization_type layer_norm