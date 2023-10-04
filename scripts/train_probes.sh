#!/bin/bash

MODELS=('Llama-2-7b-hf' 'Llama-2-13b-hf' 'Llama-2-70b-hf')

WORLD_PLACE_PROMPTS=('empty' 'describe' 'where_is' 'coords')
US_PLACE_PROMPTS=('empty' 'where_us')
NYC_PLACE_PROMPTS=('where_nyc' 'where_is' 'where_nyc_normalized')

ART_PROMPTS=('empty' 'heard' 'release')
HEADLINE_PROMPTS=('article_wo_period' 'article_w_period')
HISTORICAL_PROMPTS=('empty' 'death_date' 'death_date_occupation')



# for model in "${MODELS[@]}"
# do
#     for prompt in "${WORLD_PLACE_PROMPTS[@]}"
#     do
#         python -u probe_experiment.py \
#             --model $model \
#             --entity_type world_place \
#             --experiment_name tuned \
#             --prompt_name $prompt
#     done
# done

for model in "${MODELS[@]}"
do
    for prompt in "${US_PLACE_PROMPTS[@]}"
    do
        python -u probe_experiment.py \
            --model $model \
            --entity_type us_place \
            --experiment_name tuned \
            --prompt_name $prompt
    done
done

for model in "${MODELS[@]}"
do
    for prompt in "${NYC_PLACE_PROMPTS[@]}"
    do
        python -u probe_experiment.py \
            --model $model \
            --entity_type nyc_place \
            --experiment_name tuned \
            --prompt_name $prompt
    done
done

for model in "${MODELS[@]}"
do
    for prompt in "${ART_PROMPTS[@]}"
    do
        python -u probe_experiment.py \
            --model $model \
            --entity_type art \
            --experiment_name tuned \
            --prompt_name $prompt \
            --feature_name release_date
    done
done

for model in "${MODELS[@]}"
do
    for prompt in "${HEADLINE_PROMPTS[@]}"
    do
        python -u probe_experiment.py \
            --model $model \
            --entity_type headline \
            --experiment_name tuned \
            --prompt_name $prompt \
            --feature_name pub_date
    done
done

for model in "${MODELS[@]}"
do
    for prompt in "${HISTORICAL_PROMPTS[@]}"
    do
        python -u probe_experiment.py \
            --model $model \
            --entity_type historical_figure \
            --experiment_name tuned \
            --prompt_name $prompt \
            --feature_name death_year
    done
done

python -u probe_experiment.py \
            --model Llama-2-7b-hf \
            --entity_type us_place \
            --experiment_name tuned \
            --prompt_name where_us