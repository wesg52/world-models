import os
import pickle
import argparse

from utils import timestamp, MODEL_N_LAYERS, get_model_family
from make_prompt_datasets import ENTITY_PROMPTS
from save_activations import load_activation_probing_dataset, load_activation_probing_dataset_args
from probes.evaluation import *
from feature_datasets import common
from probe_experiment import *

import warnings
warnings.filterwarnings(
    'ignore', category=UserWarning,
    message='TypedStorage is deprecated')


def save_experiment(args, results, layer):
    save_path = os.path.join(
        os.getenv('RESULTS_DIR', 'results'),
        args.experiment_name,
        args.model,
        args.entity_type,
        args.feature_name,
        'prompt_gen_eval'
    )
    os.makedirs(save_path, exist_ok=True)

    result_name = f'{layer}.p'
    pickle.dump(
        results,
        open(os.path.join(save_path, result_name), 'wb')
    )


def place_probe_generalization_experiment(activation_dict, probe_dict, target, is_test):
    train_target = target[~is_test]
    test_target = target[is_test]

    y_mean = train_target.mean(axis=0)
    y_std = train_target.std(axis=0)

    scores = {}
    for prompt_a, activations_a in activation_dict.items():
        for prompt_b, (probe_b, bias_b) in probe_dict.items():
            activations_b = activation_dict[prompt_b]

            a_minus_b_dir = activations_a.mean(
                axis=0) - activations_b.mean(axis=0)

            a_proj_b = activations_a @ probe_b + bias_b
            a_proj_b_centered = (
                activations_a - a_minus_b_dir) @ probe_b + bias_b

            a_proj_b_unnorm = a_proj_b * y_std + y_mean
            a_proj_b_centered_unnorm = a_proj_b_centered * y_std + y_mean

            train_scores = score_place_probe(
                train_target, a_proj_b_unnorm[~is_test], use_haversine=True)
            test_scores = score_place_probe(
                test_target, a_proj_b_unnorm[is_test], use_haversine=True)

            train_scores_centered = score_place_probe(
                train_target, a_proj_b_centered_unnorm[~is_test], use_haversine=True)
            test_scores_centered = score_place_probe(
                test_target, a_proj_b_centered_unnorm[is_test], use_haversine=True)

            scores[prompt_a, prompt_b, 'train', 'uncentered'] = train_scores
            scores[prompt_a, prompt_b, 'test', 'uncentered'] = test_scores
            scores[prompt_a, prompt_b, 'train',
                   'centered'] = train_scores_centered
            scores[prompt_a, prompt_b, 'test',
                   'centered'] = test_scores_centered

    return scores


def time_probe_generalization_experiment(activation_dict, probe_dict, target, is_test):
    train_target = target[~is_test]
    test_target = target[is_test]

    y_mean = train_target.mean()
    y_std = train_target.std()

    scores = {}
    for prompt_a, activations_a in activation_dict.items():
        for prompt_b, (probe_b, bias_b) in probe_dict.items():
            activations_b = activation_dict[prompt_b]

            a_minus_b_dir = activations_a.mean(
                axis=0) - activations_b.mean(axis=0)

            a_proj_b = activations_a @ probe_b + bias_b
            a_proj_b_centered = (
                activations_a - a_minus_b_dir) @ probe_b + bias_b

            a_proj_b_unnorm = a_proj_b * y_std + y_mean
            a_proj_b_centered_unnorm = a_proj_b_centered * y_std + y_mean

            train_scores = score_time_probe(
                train_target, a_proj_b_unnorm[~is_test])
            test_scores = score_time_probe(
                test_target, a_proj_b_unnorm[is_test])

            train_scores_centered = score_time_probe(
                train_target, a_proj_b_centered_unnorm[~is_test])
            test_scores_centered = score_time_probe(
                test_target, a_proj_b_centered_unnorm[is_test])

            scores[prompt_a, prompt_b, 'train', 'uncentered'] = train_scores
            scores[prompt_a, prompt_b, 'test', 'uncentered'] = test_scores
            scores[prompt_a, prompt_b, 'train',
                   'centered'] = train_scores_centered
            scores[prompt_a, prompt_b, 'test',
                   'centered'] = test_scores_centered

    return scores


MODEL_LAYER = {
    'Llama-2-7b-hf': 20,
    'Llama-2-13b-hf': 22,
    'Llama-2-70b-hf': 50,
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
        '--prompt_names', type=str, nargs='+',
        help='Name of prompt to use for probing, must key in <ENTITY>_PROMPTS')
    parser.add_argument(
        '--normalization_type', type=str, default='none',
        help='Type of normalization to apply to activations before training')
    parser.add_argument(
        '--label_processing', type=str, default='none',
        help='Type of weighting to apply to labels before training')
    parser.add_argument(
        '--activation_aggregation', default='last',
        help='Average activations across all tokens in a sequence')
    parser.add_argument(
        '--layer', type=int, default=20)

    args = parser.parse_args()

    # n_layers = MODEL_N_LAYERS[args.model]
    layer = MODEL_LAYER[args.model]

    entity_df = common.load_entity_data(args.entity_type)
    is_test = entity_df.is_test.values

    print(timestamp(),
          f'running prompt exp on {args.model} {args.entity_type}.{args.feature_name}.{layer}')

    target = get_target_values(entity_df, args.feature_name)

    activation_dict = {}
    probe_dict = {}
    for prompt in args.prompt_names:
        # load data
        activations = load_activation_probing_dataset_args(
            args, prompt, layer).dequantize().numpy()

        probe_result = load_probe_results(
            args.experiment_name, args.model, args.entity_type, args.feature_name, prompt)

        activation_dict[prompt] = activations
        probe_dict[prompt] = (
            probe_result['probe_directions'][layer],
            probe_result['probe_biases'][layer]
        )

    is_place = args.entity_type in set(
        ['world_place', 'us_place', 'nyc_place'])

    if is_place:
        results = place_probe_generalization_experiment(
            activation_dict, probe_dict, target, is_test)
    else:
        results = time_probe_generalization_experiment(
            activation_dict, probe_dict, target, is_test)

    save_experiment(args, results, layer)
