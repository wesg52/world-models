#!/bin/bash

python -u make_prompt_datasets.py --model_family pythia --entity_type art
python -u make_prompt_datasets.py --model_family Llama-2 --entity_type art

python -u make_prompt_datasets.py --model_family pythia --entity_type headline
python -u make_prompt_datasets.py --model_family Llama-2 --entity_type headline

python -u make_prompt_datasets.py --model_family pythia --entity_type historical_figure
python -u make_prompt_datasets.py --model_family Llama-2 --entity_type historical_figure

python -u make_prompt_datasets.py --model_family pythia --entity_type world_place
python -u make_prompt_datasets.py --model_family Llama-2 --entity_type world_place

python -u make_prompt_datasets.py --model_family pythia --entity_type us_place
python -u make_prompt_datasets.py --model_family Llama-2 --entity_type us_place

python -u make_prompt_datasets.py --model_family pythia --entity_type nyc_place
python -u make_prompt_datasets.py --model_family Llama-2 --entity_type nyc_place