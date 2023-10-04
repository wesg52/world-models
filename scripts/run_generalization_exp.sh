#!/bin/bash
export EXPERIMENT_NAME=generalization_full

MODELS=('Llama-2-7b-hf' 'Llama-2-13b-hf' 'Llama-2-70b-hf')

for model in "${MODELS[@]}"
do
    # Art and Entertainment
    python -u generalization_experiment.py \
        --model $model \
        --entity_type art \
        --experiment_name $EXPERIMENT_NAME \
        --prompt_name release \
        --feature_name release_date \
        --test_column decade

    python -u generalization_experiment.py \
        --model $model \
        --entity_type art \
        --experiment_name $EXPERIMENT_NAME \
        --prompt_name release \
        --feature_name release_date \
        --test_column entity_type

    # Headlines
    python -u generalization_experiment.py \
        --model $model \
        --entity_type headline \
        --experiment_name $EXPERIMENT_NAME \
        --prompt_name when_w_period \
        --feature_name pub_date \
        --test_column news_desk

    python -u generalization_experiment.py \
        --model $model \
        --entity_type headline \
        --experiment_name $EXPERIMENT_NAME \
        --prompt_name when_w_period \
        --feature_name pub_date \
        --test_column year

    # Historical Figures
    python -u generalization_experiment.py \
        --model $model \
        --entity_type historical_figure \
        --experiment_name $EXPERIMENT_NAME \
        --prompt_name when \
        --feature_name death_year \
        --test_column occupation

    python -u generalization_experiment.py \
        --model $model \
        --entity_type historical_figure \
        --experiment_name $EXPERIMENT_NAME \
        --prompt_name when \
        --feature_name death_year \
        --test_column death_century

    # World Place
    python -u generalization_experiment.py \
        --model $model \
        --entity_type world_place \
        --experiment_name $EXPERIMENT_NAME \
        --prompt_name coords \
        --feature_name coords \
        --test_column entity_type

    python -u generalization_experiment.py \
        --model $model \
        --entity_type world_place \
        --experiment_name $EXPERIMENT_NAME \
        --prompt_name coords \
        --feature_name coords \
        --test_column country
    

    # US Place
    python -u generalization_experiment.py \
        --model $model \
        --entity_type us_place \
        --experiment_name $EXPERIMENT_NAME \
        --prompt_name coords \
        --feature_name coords \
        --test_column state_id

    python -u generalization_experiment.py \
        --model $model \
        --entity_type us_place \
        --experiment_name $EXPERIMENT_NAME \
        --prompt_name coords \
        --feature_name coords \
        --test_column entity_type

    python -u generalization_experiment.py \
        --model $model \
        --entity_type us_place \
        --experiment_name $EXPERIMENT_NAME \
        --prompt_name coords \
        --feature_name coords \
        --test_column timezone


    # NYC Place
    python -u generalization_experiment.py \
        --model $model \
        --entity_type nyc_place \
        --experiment_name $EXPERIMENT_NAME \
        --prompt_name where_nyc \
        --feature_name coords \
        --test_column facility_t_name

    python -u generalization_experiment.py \
        --model $model \
        --entity_type nyc_place \
        --experiment_name $EXPERIMENT_NAME \
        --prompt_name where_nyc \
        --feature_name coords \
        --test_column borough_name
done