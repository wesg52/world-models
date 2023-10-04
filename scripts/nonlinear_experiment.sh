#!/bin/bash

MODELS=('Llama-2-7b-hf' 'Llama-2-13b-hf' 'Llama-2-70b-hf')

for model in "${MODELS[@]}"
do
    # Space
    python -u nonlinear_experiment.py \
        --model $model \
        --entity_type world_place \
        --experiment_name prompt_full \
        --prompt_name coords

    python -u nonlinear_experiment.py \
        --model $model \
        --entity_type us_place \
        --experiment_name prompt_full \
        --prompt_name coords

    python -u nonlinear_experiment.py \
        --model $model \
        --entity_type nyc_place \
        --experiment_name prompt_full \
        --prompt_name where_nyc
    
    # Time
    python -u nonlinear_experiment.py \
        --model $model \
        --entity_type art \
        --experiment_name prompt_full \
        --prompt_name release \
        --feature_name release_date

    python -u nonlinear_experiment.py \
        --model $model \
        --entity_type headline \
        --experiment_name prompt_full \
        --prompt_name when_w_period \
        --feature_name pub_date

    python -u nonlinear_experiment.py \
        --model $model \
        --entity_type historical_figure \
        --experiment_name prompt_full \
        --prompt_name when \
        --feature_name death_year
done
