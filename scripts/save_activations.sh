#!/bin/bash


python -u save_activations.py --model Llama-2-13b-hf --entity_type world_place --batch_size 48

python -u save_activations.py --model Llama-2-13b-hf --entity_type us_place --batch_size 48

python -u save_activations.py --model Llama-2-13b-hf --entity_type historical_figure --batch_size 48

python -u save_activations.py --model Llama-2-13b-hf --entity_type art --batch_size 48

python -u save_activations.py --model Llama-2-13b-hf --entity_type headline --batch_size 48

python -u save_activations.py --model Llama-2-13b-hf --entity_type nyc_place --batch_size 48

# python -u save_activations.py --model Llama-2-13b-hf --entity_type headline --batch_size 64
# python -u save_activations.py --model Llama-2-13b-hf --entity_type headline --batch_size 48 --device cuda:1

# python -u save_activations.py --model Llama-2-13b-hf --entity_type art --batch_size 48


# python -u save_activations.py --model Llama-2-70b-hf --entity_type world_place --batch_size 48

# python -u save_activations.py --model Llama-2-70b-hf --entity_type us_place --batch_size 48

# python -u save_activations.py --model Llama-2-70b-hf --entity_type historical_figure --batch_size 48

# python -u save_activations.py --model Llama-2-70b-hf --entity_type art --batch_size 48

# python -u save_activations.py --model Llama-2-70b-hf --entity_type headline --batch_size 48

# python -u save_activations.py --model Llama-2-13b-hf --entity_type nyc_place --batch_size 64
# python -u save_activations.py --model Llama-2-13b-hf --entity_type nyc_place --batch_size 48 --device cuda:1
# python -u save_activations.py --model Llama-2-70b-hf --entity_type nyc_place --batch_size 48

# python -u save_activations.py --model Llama-2-13b-hf --entity_type historical_figure --batch_size 64
# python -u save_activations.py --model Llama-2-13b-hf --entity_type historical_figure --batch_size 48 --device cuda:1