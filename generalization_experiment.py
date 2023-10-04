import argparse
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from feature_datasets.common import *
from probe_experiment import *


TEST_SETS = {
    'world_place': {
        'entity_type': ['structure', 'natural_place'],
        'country': [
            'United_States', 'United_Kingdom', 'India', 'Australia', 'Italy',
            'Canada', 'South_Africa', 'Syria', 'Japan', 'Poland',
            'China', 'Philippines', 'Brazil', 'Mexico', 'Sweden', 'Indonesia',
            'Switzerland', 'Pakistan', 'Argentina', 'Netherlands'
        ],
    },
    'us_place': {
        'state_id': [
            'CA', 'TX', 'PA', 'NY', 'FL', 'OH', 'IL', 'NJ', 'NC', 'MI', 'GA', 'VA',
            'WA', 'WI', 'MN', 'MO', 'MA', 'IN', 'MD', 'LA', 'AL', 'TN', 'AZ', 'OK',
            'CO', 'SC', 'KY', 'OR', 'CT', 'IA', 'KS', 'AR', 'MS', 'NM', 'UT', 'WV',
            'ID', 'NE', 'NV', 'SD', 'MT', 'NH', 'ME', 'ND', 'WY', 'RI', 'DE', 'VT',
            'DC'
        ],
        'entity_type': ['city', 'zip', 'county', 'college', 'structure', 'natural_place'],
        'timezone': [
            'America/Chicago', 'America/Los_Angeles',
            'America/Denver', 'America/Detroit', 'America/Phoenix'
        ]
    },
    'nyc_place': {
        'facility_t_name': [
            'Recreational Facility', 'Education Facility', 'Residential',
            'Religious Institution', 'Social Services', 'Transportation Facility',
            'Commercial', 'Government Facility (non public safety)',
            'Miscellaneous', 'Public Safety', 'Cultural Facility', 'Water',
            'Health Services'],
        'borough_name': ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
    },
    'art': {
        'entity_type': ['movie', 'book'],
        'creator': [
            'The Beatles', 'Bob Dylan', 'Pink Floyd', 'The Beach Boys',
            'Kanye West', 'The Rolling Stones', 'Elvis Presley', 'U2',
            'Céline Dion', 'Madonna', 'Prince', 'Michael Jackson',
            'George Harrison', 'The Kinks', 'Eminem'
        ],
        'decade': [1950, 1960, 1970, 1980, 1990, 2000, 2010]
    },
    'headline': {
        'news_desk': ['National', 'Washington', 'Obits', 'Politics'],
        'year': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
    },
    'historical_figure': {
        'occupation': [
            'artist', 'politician', 'researcher', 'religious figure',
            'military personnel', 'aristocrat', 'monarch', 'teacher', 'athlete',
            'sovereign', 'ruler', 'architect', 'feudatory', 'physician', 'nun',
            'monk', 'condottiero', 'engineer', 'philosopher', 'explorer'
        ],
        'death_century': list(range(0, 2000, 100))
    }
}


def save_experiment(args, results):
    save_path = os.path.join(
        os.getenv('RESULTS_DIR', 'results'),
        args.experiment_name,
        args.model,
        args.entity_type,
        args.feature_name,
        'generalization_eval'
    )
    os.makedirs(save_path, exist_ok=True)

    result_name = f'{args.test_column}.p'
    pickle.dump(
        results,
        open(os.path.join(save_path, result_name), 'wb')
    )


def load_generalization_experiment(experiment_name, model, entity_type, feature_name, test_column):
    load_path = os.path.join(
        os.getenv('RESULTS_DIR', 'results'),
        experiment_name,
        model,
        entity_type,
        feature_name,
        'generalization_eval',
        f'{test_column}.p'
    )
    return pickle.load(open(load_path, 'rb'))


def compute_prox_error(target, projection, is_test, is_place=False):
    distance_fn = pairwise_haversine_distance if is_place else pairwise_abs_distance_fn
    error_matrix = compute_proximity_error_matrix(
        target, projection, distance_fn)

    train_error, test_error, combined_error = proximity_scores(
        error_matrix, is_test)
    return train_error, test_error, combined_error


def run_experiment(activations, target, test_sets, test_column, place=False):
    if test_sets[0] != 'baseline':
        test_sets = ['baseline'] + test_sets

    proj_cols = ['x', 'y'] if place else 'projection'
    opt_alpha = MODEL_ALPHAS[args.model]
    # alphas = MODEL_ALPHA * np.array([1, 1.75, 2.5])

    test_set_results = {}
    for test_set in test_sets:
        if test_set == 'baseline':
            is_test = entity_df.is_test.values
            # probe = RidgeCV(alphas=alphas, store_cv_values=True)
            probe = Ridge(alpha=opt_alpha)
        else:
            is_test = entity_df[test_column].values == test_set
            probe = Ridge(alpha=opt_alpha)

        if place:
            probe, scores, projection_df = place_probe_experiment(
                activations, target, is_test, probe=probe)
        else:
            probe, scores, projection_df = time_probe_experiment(
                activations, target, is_test, probe=probe)

        probe_direction = probe.coef_.T.astype(np.float16)

        # if test_set == 'baseline':
        #     probe_alphas = probe.cv_values_.mean(axis=(0, 1) if place else 0)
        #     opt_alpha = alphas[np.argmin(probe_alphas)]

        train_prox_error, test_prox_error, combined_prox_error = compute_prox_error(
            target, projection_df[proj_cols].values, is_test, is_place=place)

        scores['train', 'prox_error'] = train_prox_error.mean()
        scores['test', 'prox_error'] = test_prox_error.mean()
        projection_df['prox_error'] = combined_prox_error

        test_set_results[test_set] = {
            'scores': scores,
            'projection_df': projection_df,
            'probe_direction': probe_direction
        }

    gen_results = {}
    baseline_df = test_set_results['baseline']['projection_df']
    is_baseline_test = baseline_df.is_test.values
    for test_set in test_sets[1:]:
        is_generalization_test = entity_df[test_column].values == test_set

        base_gen_test_ixs = is_baseline_test & is_generalization_test
        baseline_generalization_target = target[base_gen_test_ixs]
        baseline_generalization_pred = baseline_df[proj_cols].values[base_gen_test_ixs]

        generalization_target = target[is_generalization_test]
        generalization_pred = test_set_results[test_set]['projection_df'][proj_cols].values[
            is_generalization_test]

        if place:
            baseline_score = score_place_probe(
                baseline_generalization_target, baseline_generalization_pred, use_haversine=True)
            generalization_score = score_place_probe(
                generalization_target, generalization_pred, use_haversine=True)
        else:
            baseline_score = score_time_probe(
                baseline_generalization_target, baseline_generalization_pred)
            generalization_score = score_time_probe(
                generalization_target, generalization_pred)

        baseline_prox_score = baseline_df['prox_error'][
            base_gen_test_ixs].mean()
        generalization_prox_score = test_set_results[test_set]['projection_df']['prox_error'][
            is_generalization_test].mean()

        baseline_score['prox_error'] = baseline_prox_score
        generalization_score['prox_error'] = generalization_prox_score

        gen_results[test_set, 'baseline'] = baseline_score
        gen_results[test_set, 'generalization'] = generalization_score

    rdf = pd.DataFrame(gen_results).T
    rdf.index.names = ['held_out', 'experiment']

    return test_set_results, rdf


MODEL_LAYER = {
    'Llama-2-7b-hf': 22,
    'Llama-2-13b-hf': 25,
    'Llama-2-70b-hf': 50,
}

MODEL_ALPHAS = {
    'Llama-2-7b-hf': 3000,
    'Llama-2-13b-hf': 6000,
    'Llama-2-70b-hf': 18000,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # experiment params
    parser.add_argument(
        '--experiment_name', type=str, help='Name of experiment for save dir')
    parser.add_argument(
        '--experiment_type', type=str, default='lsq_train',
        help='Type of experiment: spearman_train, kendall_train, lsq_train')
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
    parser.add_argument(
        '--test_column', type=str,
        help='Column to use for test set (see TEST_SETS above)')

    args = parser.parse_args()

    entity_df = common.load_entity_data(args.entity_type)
    is_place = args.entity_type.endswith('place')

    print(timestamp(),
          f'running probe on {args.model} {args.experiment_type}.{args.feature_name}.{args.prompt_name}.{args.test_column}')

    # layers = args.layers if args.layers[0] >= 0 \
    #     else list(range(MODEL_N_LAYERS[args.model]))
    layers = [MODEL_LAYER[args.model]]

    results = {}
    for layer in tqdm.tqdm(layers):
        # load data
        activations = load_activation_probing_dataset_args(
            args, args.prompt_name, layer).dequantize()

        if activations.isnan().any():
            print(timestamp(), 'WARNING: nan activations, skipping layer', layer)
            continue

        target = get_target_values(entity_df, args.feature_name)

        test_column = args.test_column
        test_sets = TEST_SETS[args.entity_type][test_column]

        test_set_results, rdf = run_experiment(
            activations, target, test_sets, test_column, place=is_place)

        results[layer] = {
            'full': test_set_results,
            'metric_df': rdf
        }

    save_experiment(args, results)
