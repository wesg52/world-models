import os
import pickle
import argparse
import einops
import torch
import numpy as np
from load import load_model
from sklearn.metrics import *
from utils import timestamp, MODEL_N_LAYERS, adjust_precision
from make_prompt_datasets import DATASET_MANAGERS, FEATURE_PROMPT_MAPPINGS
from train_probes import load_probe_result, load_all_probes, load_supervised_data
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, kendalltau, pearsonr
from analysis.weight_composition import *


def save_evaluation(args, eval_result):
    save_path = os.path.join(
        os.getenv('RESULTS_DIR', 'results'),
        args.experiment_name,
        args.model,
        args.entity_type,
        args.feature_name,
        'evaluations'
    )
    os.makedirs(save_path, exist_ok=True)

    evaluation_metadata = [
        args.evaluation_type,
        args.experiment_type,
        args.normalization_type,
        args.label_processing,
        args.activation_aggregation,
    ]

    eval_name = '.'.join(evaluation_metadata) + '.p'

    pickle.dump(
        eval_result,
        open(os.path.join(save_path, eval_name), 'wb')
    )


def load_evaluation(
        experiment_name, model, entity_type, feature_name,
        evaluation_type, experiment_type, normalization_type='none',
        label_processing='none', activation_aggregation='last'):
    save_path = os.path.join(
        os.getenv('RESULTS_DIR', 'results'),
        experiment_name,
        model,
        entity_type,
        feature_name,
        'evaluations'
    )
    evaluation_metadata = [
        evaluation_type,
        experiment_type,
        normalization_type,
        label_processing,
        activation_aggregation,
    ]
    eval_name = '.'.join(evaluation_metadata) + '.p'
    save_file = os.path.join(save_path, eval_name)
    return pickle.load(open(save_file, 'rb'))


def load_all_probe_evals(experiment_name, model_name, entity_type, feature_name):
    probe_path = os.path.join(
        os.getenv('RESULTS_DIR', 'results'),
        experiment_name,
        model_name,
        entity_type,
        feature_name,
        'evaluations'
    )
    experiment_metadata = (experiment_name, model_name, entity_type)
    probe_files = os.listdir(probe_path)
    probe_files = [f for f in probe_files if f.endswith('.p')]
    probes = {}
    for probe_file in probe_files:
        probe_metadata = tuple(probe_file.split('.')[:-1])
        probe_results = pickle.load(
            open(os.path.join(probe_path, probe_file), 'rb'))
        probes[experiment_metadata + probe_metadata] = probe_results
    return probes


def evaluate_ranking(args, probe, prompt_name, layer, save_projections=False):
    # load data
    layer_activations, entity_values, test_ixs = load_supervised_data(
        args, manager, args.feature_name, prompt_name, layer)

    test_entity_values = entity_values[test_ixs]

    feature_proj = (torch.tensor(layer_activations) @ probe).numpy()
    all_spearman = spearmanr(feature_proj, entity_values)
    test_spearman = spearmanr(feature_proj[test_ixs], test_entity_values)
    train_spearman = spearmanr(
        feature_proj[~test_ixs], entity_values[~test_ixs])

    all_kendall = kendalltau(feature_proj, entity_values)
    test_kendall = kendalltau(feature_proj[test_ixs], test_entity_values)
    train_kendall = kendalltau(
        feature_proj[~test_ixs], entity_values[~test_ixs])

    all_pearson = pearsonr(feature_proj, entity_values)
    test_pearson = pearsonr(feature_proj[test_ixs], test_entity_values)
    train_pearson = pearsonr(
        feature_proj[~test_ixs], entity_values[~test_ixs])

    # TODO: add weights

    result_dict = {
        'all_spearman_coef': all_spearman.correlation,
        'train_spearman_coef': train_spearman.correlation,
        'test_spearman_coef': test_spearman.correlation,
        'all_kendall_coef': all_kendall.correlation,
        'train_kendall_coef': train_kendall.correlation,
        'test_kendall_coef': test_kendall.correlation,
        'all_pearson_coef': all_pearson.correlation,
        'train_pearson_coef': train_pearson.correlation,
        'test_pearson_coef': test_pearson.correlation,
        'all_spearman_p': all_spearman.pvalue,
        'train_spearman_p': train_spearman.pvalue,
        'test_spearman_p': test_spearman.pvalue,
        'all_kendall_p': all_kendall.pvalue,
        'train_kendall_p': train_kendall.pvalue,
        'test_kendall_p': test_kendall.pvalue,
        'all_pearson_p': all_pearson.pvalue,
        'train_pearson_p': train_pearson.pvalue,
        'test_pearson_p': test_pearson.pvalue,
        'norm': torch.norm(probe).item(),
    }
    if save_projections:
        result_dict['feature_projection'] = feature_proj.astype(np.float16)

    return result_dict


def evaluate_layer_and_prompt_generalization(args, probe):
    transfer_results = {}
    for layer in range(MODEL_N_LAYERS[args.model]):
        prompt_key_list = FEATURE_PROMPT_MAPPINGS[args.entity_type][args.feature_name]
        for transfer_prompt in prompt_key_list:
            transfer_results[(layer, transfer_prompt)] = evaluate_ranking(
                args, probe, transfer_prompt, layer)
    return transfer_results


def evaluate_inter_feature_generalization():
    raise NotImplementedError


def evaluate_probe(args, probe_results, layer):
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # experiment params
    parser.add_argument(
        '--experiment_name', type=str, help='Name of experiment for save dir')
    parser.add_argument(
        '--experiment_type', type=str, default='spearman_train',
        help='Type of experiment to evaluate')
    parser.add_argument(
        '--evaluation_type', type=str, default='oos_generalization',
        choices=['oos_generalization', 'composition',
                 'layer_and_prompt_generalization', 'cross_feature_generalization'])
    parser.add_argument(
        '--model', default='pythia-70m',
        help='Name of model from TransformerLens')
    parser.add_argument(
        '--entity_type',
        help='Name of feature collection (should be dir under processed_datasets/)')
    parser.add_argument(
        '--feature_name', type=str,
        help='Name of feature to probe, must be in FEATURE_PROMPT_MAPPINGS')

    # never changed from defaults
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

    n_layers = MODEL_N_LAYERS[args.model]

    if int(os.getenv('SLURM_CPUS_PER_TASK', -1)) > 0:
        torch.set_num_threads(int(os.getenv('SLURM_CPUS_PER_TASK', 1)))

    manager = DATASET_MANAGERS[args.entity_type]
    if args.evaluation_type == 'composition':
        torch.set_grad_enabled(False)
        model = load_model(args.model)

    probes = load_all_probes(
        args.experiment_name, args.model, args.entity_type, args.feature_name)

    eval_results = {}
    for key in probes.keys():
        experiment_type = key[3]
        prompt_name = key[-1]

        if experiment_type != args.experiment_type:
            continue

        for layer in probes[key].keys():

            print(timestamp(),
                  f'running evaluation on {args.model} {args.evaluation_type}.{args.feature_name}.L{layer}')

            probe_key = 'rank_probe' if experiment_type == 'lsq_train' else 'probe'
            probe = torch.tensor(
                probes[key][layer][probe_key]).to(torch.float32)

            # run evaluations
            if args.evaluation_type == 'oos_generalization':
                probe_results = evaluate_ranking(
                    args, probe, prompt_name, layer, save_projections=True)
                eval_results[(layer, prompt_name)] = probe_results

            elif args.evaluation_type == 'layer_and_prompt_generalization':
                probe_results = evaluate_layer_and_prompt_generalization(
                    args, probe)
                eval_results[(layer, prompt_name)] = probe_results

            elif args.evaluation_type == 'composition':
                composition_results = evaluate_probe_composition(
                    model, probe)
                eval_results[(layer, prompt_name)] = composition_results

            elif args.evaluation_type == 'cross_feature_generalization':
                generalization_results = evaluate_inter_feature_generalization(
                    args, probe)
                eval_results[(layer, prompt_name)] = generalization_results

    save_evaluation(args, eval_results)
