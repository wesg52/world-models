#!/bin/bash

export EXPERIMENT_NAME=full_prompts

MODELS=('Llama-2-7b-hf' 'Llama-2-13b-hf')

WORLD_PLACE_PROMPTS=('empty' 'empty_all_caps' 'random' 'coords')
US_PLACE_PROMPTS=('empty' 'random' 'coords' 'where_us')
NYC_PLACE_PROMPTS=('empty' 'random' 'where_is' 'where_nyc')

ART_PROMPTS=('empty' 'random' 'release' 'empty_all_caps')
HEADLINE_PROMPTS=('empty' 'empty_wo_period' 'when_w_period' 'when_wo_period')
HISTORICAL_PROMPTS=('empty' 'random' 'when' 'when_all_caps')

for model in "${MODELS[@]}"
do
    for prompt in "${WORLD_PLACE_PROMPTS[@]}"
    do
        sbatch slurm/train_probes/main.sh $model world_place $EXPERIMENT_NAME $prompt coords
    done

    for prompt in "${US_PLACE_PROMPTS[@]}"
    do
        sbatch slurm/train_probes/main.sh $model us_place $EXPERIMENT_NAME $prompt coords
    done

    for prompt in "${NYC_PLACE_PROMPTS[@]}"
    do
        sbatch slurm/train_probes/main.sh $model nyc_place $EXPERIMENT_NAME $prompt coords
    done

    for prompt in "${ART_PROMPTS[@]}"
    do
        sbatch slurm/train_probes/main.sh $model art $EXPERIMENT_NAME $prompt release_date
    done

    for prompt in "${HEADLINE_PROMPTS[@]}"
    do
        sbatch slurm/train_probes/main.sh $model headline $EXPERIMENT_NAME $prompt pub_date
    done

    for prompt in "${HISTORICAL_PROMPTS[@]}"
    do
        sbatch slurm/train_probes/main.sh $model historical_figure $EXPERIMENT_NAME $prompt death_year
    done
done

