#!/bin/bash

MODELS=('Llama-2-7b-hf' 'Llama-2-7b-hf')

for model in "${MODELS[@]}"
do
    sbatch slurm/evaluate_probes/entity.sh rank_refactor_test spearman_train oos_generalization $model art release_date
    sbatch slurm/evaluate_probes/entity.sh rank_refactor_test kendall_train oos_generalization $model art release_date
    sbatch slurm/evaluate_probes/entity.sh rank_refactor_test lsq_train oos_generalization $model art release_date

    sbatch slurm/evaluate_probes/entity.sh rank_refactor_test spearman_train oos_generalization $model world_place latitude
    sbatch slurm/evaluate_probes/entity.sh rank_refactor_test kendall_train oos_generalization $model world_place latitude
    sbatch slurm/evaluate_probes/entity.sh rank_refactor_test lsq_train oos_generalization $model world_place latitude

    sbatch slurm/evaluate_probes/entity.sh rank_refactor_test spearman_train oos_generalization $model world_place longitude
    sbatch slurm/evaluate_probes/entity.sh rank_refactor_test kendall_train oos_generalization $model world_place longitude
    sbatch slurm/evaluate_probes/entity.sh rank_refactor_test lsq_train oos_generalization $model world_place longitude

    sbatch slurm/evaluate_probes/entity.sh rank_refactor_test spearman_train oos_generalization $model us_place latitude
    sbatch slurm/evaluate_probes/entity.sh rank_refactor_test kendall_train oos_generalization $model us_place latitude
    sbatch slurm/evaluate_probes/entity.sh rank_refactor_test lsq_train oos_generalization $model us_place latitude

    sbatch slurm/evaluate_probes/entity.sh rank_refactor_test spearman_train oos_generalization $model us_place longitude
    sbatch slurm/evaluate_probes/entity.sh rank_refactor_test kendall_train oos_generalization $model us_place longitude
    sbatch slurm/evaluate_probes/entity.sh rank_refactor_test lsq_train oos_generalization $model us_place longitude
done

# for model in "${MODELS[@]}"
# do
#     sbatch -c 24 slurm/evaluate_probes/zips.sh rank_refactor_test rank_train composition $model
#     sbatch -c 24 slurm/evaluate_probes/phone_numbers.sh rank_refactor_test rank_train composition $model
# done