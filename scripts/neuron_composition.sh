#!/bin/bash

MODELS=('Llama-2-70b-hf')

for model in "${MODELS[@]}"
do
    # Time
    python -u model_composition.py \
        --model $model \
        --entity_type art \
        --prompt_name release \
        --feature_name release_date

    python -u model_composition.py \
        --model $model \
        --entity_type headline \
        --prompt_name empty \
        --feature_name pub_date

    python -u model_composition.py \
        --model $model \
        --entity_type historical_figure \
        --prompt_name when \
        --feature_name death_year

    # Space
    python -u model_composition.py \
        --model $model \
        --entity_type world_place \
        --prompt_name coords

    python -u model_composition.py \
        --model $model \
        --entity_type us_place \
        --prompt_name coords

    python -u model_composition.py \
        --model $model \
        --entity_type nyc_place \
        --prompt_name where_nyc
done