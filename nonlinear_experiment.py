import argparse
import numpy as np
import pandas as pd
from feature_datasets.common import *
from probe_experiment import *
from sklearn.linear_model import RidgeCV
from probes.mlp import MLPRegressor


MLP_PARAM_DICT = {
    'weight_decay': [0.01, 0.03, 0.1, 0.3]
}


def save_experiment(args, results):
    save_path = os.path.join(
        os.getenv('RESULTS_DIR', 'results'),
        args.experiment_name,
        args.model,
        args.entity_type,
        args.feature_name,
        'nonlinearity_test'
    )
    os.makedirs(save_path, exist_ok=True)

    result_name = f'{args.prompt_name}.p'
    pickle.dump(
        results,
        open(os.path.join(save_path, result_name), 'wb')
    )


def run_experiment(activations, target, is_test, place=False):
    ridge_probe = RidgeCV(alphas=np.logspace(3, 4.5, 12), store_cv_values=True)

    if place:
        probe, ridge_scores, ridge_projection_df = place_probe_experiment(
            activations, target, is_test, probe=ridge_probe)
    else:
        probe, ridge_scores, ridge_projection_df = time_probe_experiment(
            activations, target, is_test, probe=ridge_probe)
    probe_cv_values = probe.cv_values_.mean(axis=((0, 1) if place else 0))

    mlp_results = {}
    val_scores = []
    for wd in MLP_PARAM_DICT['weight_decay']:
        mlp_probe = MLPRegressor(
            input_size=activations.shape[1],
            output_size=2 if place else 1,
            hidden_size=256,
            patience=3,
            learning_rate=1e-3,
            weight_decay=wd
        )

        if place:
            probe, mlp_scores, mlp_projection_df = place_probe_experiment(
                activations, target, is_test, probe=mlp_probe)
        else:
            probe, mlp_scores, mlp_projection_df = time_probe_experiment(
                activations, target, is_test, probe=mlp_probe)

        val_scores.append(min(probe.validation_scores))
        mlp_results[wd] = (mlp_scores, mlp_projection_df)

    best_mlp_wd = MLP_PARAM_DICT['weight_decay'][np.argmin(val_scores)]
    mlp_scores, mlp_projection_df = mlp_results[best_mlp_wd]

    results = {
        'ridge_scores': ridge_scores,
        'mlp_scores': mlp_scores,
        'ridge_prediction_df': ridge_projection_df,
        'mlp_prediction_df': mlp_projection_df,
        'ridge_cv_values': probe_cv_values,
        'mlp_validation_scores': val_scores
    }
    return results


MODEL_LAYER = {
    'Llama-2-7b-hf': 20,
    'Llama-2-13b-hf': 24,
    'Llama-2-70b-hf': 48,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # experiment params
    parser.add_argument(
        '--experiment_name', type=str, help='Name of experiment for save dir')
    parser.add_argument(
        '--model', default='pythia-70m',
        help='Name of model from TransformerLens')
    parser.add_argument(
        '--entity_type',
        help='Name of feature collection (should be dir under processed_datasets/)')
    parser.add_argument(
        '--feature_name', type=str, default='coords',
        help='Name of feature to probe, must be in FEATURE_PROMPT_MAPPINGS')
    parser.add_argument(
        '--prompt_name', type=str,
        help='Name of prompt to use for probing, must key in <ENTITY>_PROMPTS')
    parser.add_argument(
        '--layer', type=float, help='model depth')
    parser.add_argument(
        '--normalization_type', type=str, default='none',
        help='Type of normalization to apply to activations before training')
    parser.add_argument(
        '--label_processing', type=str, default='none',
        help='Type of weighting to apply to labels before training')
    parser.add_argument(
        '--activation_aggregation', default='last',
        help='Average activations across all tokens in a sequence')

    args = parser.parse_args()

    entity_df = common.load_entity_data(args.entity_type)
    is_place = args.entity_type.endswith('place')

    print(timestamp(),
          f'running probe on {args.model} {args.feature_name}.{args.prompt_name}')

    # layers = args.layers if args.layers[0] >= 0 \
    #     else list(range(MODEL_N_LAYERS[args.model]))
    layers = [MODEL_LAYER[args.model]]

    results = {}
    for layer in layers:
        # load data
        activations = load_activation_probing_dataset_args(
            args, args.prompt_name, layer).dequantize()

        if activations.isnan().any():
            print(timestamp(), 'WARNING: nan activations, skipping layer', layer)
            continue

        target = get_target_values(entity_df, args.feature_name)

        is_test = entity_df.is_test.values

        layer_results = run_experiment(
            activations, target, is_test, place=is_place)

        results[layer] = layer_results

    save_experiment(args, results)
