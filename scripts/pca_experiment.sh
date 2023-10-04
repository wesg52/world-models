#!/bin/bash

MODELS=('Llama-2-7b-hf' 'Llama-2-13b-hf' 'Llama-2-70b-hf')

for model in "${MODELS[@]}"
do
    # Space
    python -u probe_experiment.py \
        --model $model \
        --pca \
        --entity_type world_place \
        --experiment_name full_prompt \
        --prompt_name coords

    python -u probe_experiment.py \
        --model $model \
        --pca \
        --entity_type us_place \
        --experiment_name full_prompt \
        --prompt_name coords

    python -u probe_experiment.py \
        --model $model \
        --pca \
        --entity_type nyc_place \
        --experiment_name full_prompt \
        --prompt_name where_nyc
    
    # Time
    python -u probe_experiment.py \
        --model $model \
        --pca \
        --entity_type art \
        --experiment_name full_prompt \
        --prompt_name release \
        --feature_name release_date

    python -u probe_experiment.py \
        --model $model \
        --pca \
        --entity_type headline \
        --experiment_name full_prompt \
        --prompt_name empty \
        --feature_name pub_date

    python -u probe_experiment.py \
        --model $model \
        --pca \
        --entity_type historical_figure \
        --experiment_name full_prompt \
        --prompt_name when \
        --feature_name death_year
done
