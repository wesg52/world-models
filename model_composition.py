import gc
import os
import einops
import torch
import argparse
import numpy as np
import pandas as pd
from utils import timestamp
from scipy.stats import spearmanr
from save_activations import load_activation_probing_dataset
from feature_datasets.common import load_entity_data
from probe_experiment import get_target_values
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_top_neurons(learned_probes, model, k=50, W_in=True):
    if W_in:
        W_norm = (model.W_in / model.W_in.norm(dim=1,
                  keepdim=True)).swapaxes(1, 2)
    else:
        W_norm = (model.W_out / model.W_out.norm(dim=-1, keepdim=True))

    n_layers, d_mlp, d_model = W_norm.shape

    W_comp = einops.einsum(W_norm, learned_probes.float(),
                           'l1 m d, l2 d-> l2 l1 m')

    top_neurons = W_comp.flatten().abs().argsort()
    top_acts, top_layers, top_neurons = np.unravel_index(
        top_neurons[-k:], (learned_probes.shape[0], n_layers, d_mlp))

    top_cos = W_comp[top_acts, top_layers, top_neurons]

    return top_layers, top_neurons, top_cos


def make_correlation_df(model, features, entity_activations, top_layers, top_neurons, W_in=True):
    neuron_corr = {}
    for l, n in zip(top_layers, top_neurons):
        if W_in:
            neuron_probe = model.W_in[l, :, n]
        else:
            neuron_probe = model.W_out[l, n, :]

        for activation_layer, activations in entity_activations.items():
            neuron_probe_projection = activations @ neuron_probe
            corr = spearmanr(neuron_probe_projection, features).correlation
            neuron_corr[(l, n, activation_layer)] = corr

    corr_df = pd.DataFrame({'corr': pd.Series(neuron_corr)})
    corr_df.index.names = ['neuron_layer', 'neuron', 'activation_layer']
    corr_df = corr_df.reset_index()
    return corr_df


def place_neuron_correlations(model, probe_result, place_df, entity_activations, top_k=50, start_layer=15):
    n_layers = model.config.num_hidden_layers
    lon_probes = torch.stack([
        torch.tensor(probe_result['probe_directions'][layer][:, 1])
        for layer in range(start_layer, n_layers)
    ])
    lon_probes = lon_probes / lon_probes.norm(dim=1, keepdim=True)

    lat_probes = torch.stack([
        torch.tensor(probe_result['probe_directions'][layer][:, 1])
        for layer in range(start_layer, n_layers)
    ])
    lat_probes = lat_probes / lat_probes.norm(dim=1, keepdim=True)

    top_neuron_dict = {
        ('lon', 'in'): get_top_neurons(lon_probes, model, k=top_k, W_in=True),
        ('lon', 'out'): get_top_neurons(lon_probes, model, k=top_k, W_in=False),
        ('lat', 'in'): get_top_neurons(lat_probes, model, k=top_k, W_in=True),
        ('lat', 'out'): get_top_neurons(lat_probes, model, k=top_k, W_in=False),
    }

    corr_dfs = []
    for (feature, neuron_weight), (top_layers, top_neurons, top_cos) in top_neuron_dict.items():
        feature_col = 'latitude' if feature == 'lat' else 'longitude'
        use_Win = neuron_weight == 'in'
        feature_values = place_df[feature_col].values

        corr_df = make_correlation_df(
            model, feature_values, entity_activations, top_layers, top_neurons, W_in=use_Win)
        corr_df['probe_cos'] = top_cos
        corr_df['feature'] = feature
        corr_df['neuron_weight'] = neuron_weight
        corr_dfs.append(corr_df)
    return pd.concat(corr_dfs)


def RMSnorm(x, eps=1e-6):
    mean_sq = (x ** 2).mean(dim=1, keepdim=True)
    x = x / torch.sqrt(mean_sq + eps)
    return x


def pearson_correlation(matrix, target):
    n, d = matrix.size()
    target = target.view(-1, 1)  # reshape target to a column vector

    # Calculate the sums
    sum_x = matrix.sum(dim=0)
    sum_y = target.sum()
    sum_xy = (matrix * target).sum(dim=0)
    sum_xx = (matrix * matrix).sum(dim=0)
    sum_yy = (target * target).sum()

    # Compute the Pearson correlation for each column
    numerator = n * sum_xy - sum_x * sum_y
    denominator = torch.sqrt((n * sum_xx - sum_x ** 2)
                             * (n * sum_yy - sum_y ** 2))

    correlation = numerator / denominator
    return correlation


def spearman_correlation(matrix, target):
    n, d = matrix.size()
    target = target.view(-1, 1)  # reshape target to a column vector

    # Chunk neurons to reduce memory overhead
    chunk_size = 1024
    num_rows = matrix.size(0)
    matrix_ranks = torch.zeros_like(
        matrix, dtype=torch.float, device=matrix.device)
    for i in range(0, num_rows, chunk_size):
        chunk = matrix[:, i:i + chunk_size]
        rank_chunk = chunk.argsort(dim=0).argsort(dim=0).float() + 1.0
        matrix_ranks[:, i:i + chunk_size] = rank_chunk

    target_ranks = target.argsort(dim=0).argsort(
        dim=0).float() + 1.0  # convert to 1-indexed ranks

    # Calculate the sums
    sum_x = matrix_ranks.sum(dim=0)
    sum_y = target_ranks.sum()
    sum_xy = (matrix_ranks * target_ranks).sum(dim=0)
    sum_xx = (matrix_ranks * matrix_ranks).sum(dim=0)
    sum_yy = (target_ranks * target_ranks).sum()

    # Compute the Spearman correlation for each column
    numerator = n * sum_xy - sum_x * sum_y
    denominator = torch.sqrt((n * sum_xx - sum_x ** 2)
                             * (n * sum_yy - sum_y ** 2))

    correlation = numerator / denominator
    return correlation


def neuron_full_correlations(target_values, entity_activations, model, weight='W_in', use_spearman=True):
    target = torch.tensor(target_values).cuda()
    corrs = []
    for layer in range(model.config.num_hidden_layers - 1):
        acts = entity_activations[layer].cuda()
        acts = RMSnorm(acts)
        if weight == 'W_in':
            weights = model.model.layers[layer+1].mlp.up_proj.weight
        elif weight == 'W_gate':
            weights = model.model.layers[layer+1].mlp.gate_proj.weight
        elif weight == 'W_out':
            weights = model.model.layers[layer].mlp.down_proj.weight.T
        else:
            raise ValueError(f'Invalid weight type: {weight}')

        weights = weights.cuda().to(torch.float32)
        neuron_acts = weights @ acts.T

        del acts
        del weights
        gc.collect()
        torch.cuda.empty_cache()

        if use_spearman:
            corr = spearman_correlation(neuron_acts.T, target).detach().cpu()
        else:
            corr = pearson_correlation(neuron_acts.T, target).detach().cpu()

        corrs.append(corr)

    full_corr = torch.stack(corrs, dim=0)
    return full_corr


def place_all_neuron_correlations(place_df, entity_activations, model, top_k=50):
    lat = place_df.latitude.values
    lon = place_df.longitude.values

    targets = {
        'lat': lat,
        'lon': lon,
    }
    if 'country' in place_df.columns:
        targets['abs_lat'] = np.abs(lat)
        targets['abs_lon'] = np.abs(lon)

    weights = ['W_in', 'W_gate', 'W_out']
    neuron_dfs = []
    for target_name, target_values in targets.items():
        for weight in weights:
            full_corr = neuron_full_correlations(
                target_values, entity_activations, model, weight=weight)
            top_ixs = full_corr.flatten().abs().argsort()[-top_k:]
            top_layers, top_neurons = np.unravel_index(
                top_ixs, full_corr.shape)
            df = pd.DataFrame({
                'feature': [target_name for _ in range(top_k)],
                'weight': [weight for _ in range(top_k)],
                # +1 because we skip the first layer
                'layer': top_layers + (1 if weight != 'W_out' else 0),
                'neuron': top_neurons,
                'corr': full_corr[top_layers, top_neurons],
                'abs_corr': full_corr[top_layers, top_neurons].abs()

            })
            neuron_dfs.append(df)
    return pd.concat(neuron_dfs)


def time_neuron_correlations(target, entity_activations, model, top_k=50):

    weights = ['W_in', 'W_gate', 'W_out']
    neuron_dfs = []
    for weight in weights:
        full_corr = neuron_full_correlations(
            target, entity_activations, model, weight=weight)
        top_ixs = full_corr.flatten().abs().argsort()[-top_k:]
        top_layers, top_neurons = np.unravel_index(
            top_ixs, full_corr.shape)
        df = pd.DataFrame({
            'feature': ['time' for _ in range(top_k)],
            'weight': [weight for _ in range(top_k)],
            # +1 because we skip the first layer
            'layer': top_layers + (1 if weight != 'W_out' else 0),
            'neuron': top_neurons,
            'corr': full_corr[top_layers, top_neurons],
            'abs_corr': full_corr[top_layers, top_neurons].abs()

        })
        neuron_dfs.append(df)
    return pd.concat(neuron_dfs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model', default='Llama-2-7b-hf',
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

    args = parser.parse_args()

    torch.set_grad_enabled(False)
    model = AutoModelForCausalLM.from_pretrained(
        f"meta-llama/{args.model}")

    n_layers = model.config.num_hidden_layers
    entity_activations = {l: load_activation_probing_dataset(
        args.model, args.entity_type, args.prompt_name, l).dequantize()
        for l in range(n_layers)
    }

    entity_df = load_entity_data(args.entity_type)
    target = get_target_values(entity_df, args.feature_name)

    print(timestamp(),
          f'running neuron composition on {args.model} {args.entity_type}')

    if args.entity_type.endswith('place'):
        neuron_df = place_all_neuron_correlations(
            entity_df, entity_activations, model)
    else:
        neuron_df = time_neuron_correlations(
            target, entity_activations, model)

    save_path = os.path.join('results', 'top_neurons', args.model)
    os.makedirs(save_path, exist_ok=True)
    neuron_df.to_csv(os.path.join(save_path, args.entity_type), index=False)
