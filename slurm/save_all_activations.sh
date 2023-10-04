#!/bin/bash

sbatch slurm/save_activations.sh Llama-2-7b-hf art 48

sbatch slurm/save_activations.sh Llama-2-7b-hf world_place 48

sbatch slurm/save_activations.sh Llama-2-7b-hf us_place 48

sbatch slurm/save_activations.sh Llama-2-7b-hf headline 48

sbatch slurm/save_activations.sh Llama-2-7b-hf nyc_place 48

sbatch slurm/save_activations.sh Llama-2-7b-hf historical_figure 48




sbatch slurm/save_activations.sh Llama-2-13b-hf art 32

sbatch slurm/save_activations.sh Llama-2-13b-hf world_place 32

sbatch slurm/save_activations.sh Llama-2-13b-hf us_place 32

sbatch slurm/save_activations.sh Llama-2-13b-hf headline 32

sbatch slurm/save_activations.sh Llama-2-13b-hf nyc_place 32

sbatch slurm/save_activations.sh Llama-2-13b-hf historical_figure 32


# sbatch slurm/save_activations_large.sh Llama-2-70b-hf art 16

# sbatch slurm/save_activations_large.sh Llama-2-70b-hf world_place 16

# sbatch slurm/save_activations_large.sh Llama-2-70b-hf us_place 16

# sbatch slurm/save_activations_large.sh Llama-2-70b-hf headline 16

# sbatch slurm/save_activations_large.sh Llama-2-70b-hf nyc_place 16

# sbatch slurm/save_activations_large.sh Llama-2-70b-hf historical_figure 16