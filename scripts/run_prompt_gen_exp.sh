#!/bin/bash
export EXPERIMENT_NAME=full_prompts

MODELS=('Llama-2-7b-hf' 'Llama-2-13b-hf' 'Llama-2-70b-hf')

for model in "${MODELS[@]}"
do
    # Art and Entertainment
    python -u prompt_gen_experiment.py \
        --model $model \
        --entity_type art \
        --experiment_name $EXPERIMENT_NAME \
        --prompt_names empty random release empty_all_caps \
        --feature_name release_date

    # Headlines
    python -u prompt_gen_experiment.py \
        --model $model \
        --entity_type headline \
        --experiment_name $EXPERIMENT_NAME \
        --prompt_names empty empty_wo_period when_w_period when_wo_period \
        --feature_name pub_date 

    # Historical Figures
    python -u prompt_gen_experiment.py \
        --model $model \
        --entity_type historical_figure \
        --experiment_name $EXPERIMENT_NAME \
        --prompt_names empty random when when_all_caps \
        --feature_name death_year

    # World Place
    python -u prompt_gen_experiment.py \
        --model $model \
        --entity_type world_place \
        --experiment_name $EXPERIMENT_NAME \
        --prompt_names empty empty_all_caps random coords \
        --feature_name coords 

    # US Place
    python -u prompt_gen_experiment.py \
        --model $model \
        --entity_type us_place \
        --experiment_name $EXPERIMENT_NAME \
        --prompt_names empty random coords where_us \
        --feature_name coords

    # NYC Place
    python -u prompt_gen_experiment.py \
        --model $model \
        --entity_type nyc_place \
        --experiment_name $EXPERIMENT_NAME \
        --prompt_names empty random where_is where_nyc \
        --feature_name coords
done