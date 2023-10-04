import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns


def performance_by_prompt(rdf, prompts, models, normalize_layer=True, metric='test_r2'):
    # plot model r^2 for each prompt
    fig, axs = plt.subplots(1, len(prompts), figsize=(15, 5), sharey=True)
    for i, prompt in enumerate(prompts):
        ax = axs[i]
        for model in models:
            data_df = rdf[(rdf.model == model) & (rdf.prompt == prompt)]
            layer = data_df.layer.values
            if normalize_layer:
                layer = layer / layer.max()
            ax.plot(layer, data_df[metric], label=model)
        ax.set_title(prompt)
        ax.legend()
        ax.grid()
    return fig, axs


def performance_by_model(rdf, prompts, models, normalize_layer=True, metric='test_r2'):
    fig, axs = plt.subplots(1, len(models), figsize=(15, 5), sharey=True)
    for i, model in enumerate(models):
        ax = axs[i]
        for prompt in prompts:
            data_df = rdf[(rdf.model == model) & (rdf.prompt == prompt)]
            layer = data_df.layer.values
            if normalize_layer:
                layer = layer / layer.max()
            ax.plot(layer, data_df[metric], label=prompt)
        ax.set_title(model)
        ax.legend()
        ax.grid()
    return fig, axs


def performance_by_model_multiple_metrics(rdf, prompts, models, normalize_layer=True, metrics=('test_r2', )):
    # plot test r^2 for each model
    fig, axs = plt.subplots(len(metrics), 3, figsize=(15, 3*len(metrics)))
    for j, metric in enumerate(metrics):
        for i, model in enumerate(models):
            ax = axs[j, i]
            for prompt in prompts:
                data_df = rdf[(rdf.model == model) & (rdf.prompt == prompt)]
                layer = data_df.layer.values
                if normalize_layer:
                    layer = layer / layer.max()
                ax.plot(layer, data_df[metric], label=prompt)
            ax.legend()
            ax.grid()
            if i == 0:
                ax.set_ylabel(metric)
            if j == 0:
                ax.set_title(model)
    plt.tight_layout()


def performance_by_prompt_multiple_metrics(rdf, prompts, models, normalize_layer=True, metrics=('test_r2', )):
    fig, axs = plt.subplots(len(metrics), len(
        prompts), figsize=(15, 3*len(metrics)))
    for j, metric in enumerate(metrics):
        for i, prompt in enumerate(prompts):
            ax = axs[j, i]
            for model in models:
                data_df = rdf[(rdf.model == model) & (rdf.prompt == prompt)]
                layer = data_df.layer.values
                if normalize_layer:
                    layer = layer / layer.max()
                ax.plot(layer, data_df[metric], label=model)
            ax.legend()
            ax.grid()
            if i == 0:
                ax.set_ylabel(metric)
            if j == 0:
                ax.set_title(prompt)
    plt.tight_layout()
    # OLD


def plot_generalization_gap(rdf, prompts, models, normalize_layer=True, metric='r2'):
    fig, axs = plt.subplots(1, len(prompts), figsize=(15, 5), sharey=True)
    for i, prompt in enumerate(prompts):
        ax = axs[i]
        for model in models:
            data_df = rdf[(rdf.model == model) & (rdf.prompt == prompt)]
            layer = data_df.layer.values
            if normalize_layer:
                layer = layer / layer.max()
            ax.plot(layer, (data_df[f'train_{metric}'] - data_df[f'test_{metric}']
                            ) / data_df[f'test_{metric}'], label=model)
        ax.set_title(prompt)
        ax.legend()
        ax.grid()


def plot_probe_cosine_sim(model_directions, models, prompts, feature_dim=None):
    n_rows, n_cols = len(models), len(prompts)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    for m_ix, model in enumerate(models):
        for p_jx, prompt in enumerate(prompts):
            ax = axs[m_ix, p_jx]
            directions = model_directions[model, prompt]
            if feature_dim is not None:
                directions = directions[:, :, feature_dim]
            similarities = cosine_similarity(directions)
            sns.heatmap(similarities, ax=ax)
            ax.set_title(f'{model} {prompt}')
            ax.set_xlabel('layer')
            ax.set_ylabel('layer')
    plt.tight_layout()


def colorFader(c1='red', c2='blue', mix=0):
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


def plot_rank_probe_train_metadata(probe_results, prompts, metrics=('weight_norm', 'train_spearman', 'pred_max', 'pred_min')):
    fig, axs = plt.subplots(len(metrics), len(
        prompts), figsize=(3*len(prompts), 3*len(metrics)))
    n_layers = max([int(k[-1]) for k in probe_results.keys()]) + 1
    layer_results = {int(k[-1]): v for k, v in probe_results.items()}
    if n_layers != len(layer_results):
        print('Warning missing or extra layers in probe results')

    for ix, prompt in enumerate(prompts):
        for layer in range(n_layers):
            layer_df = pd.DataFrame(
                layer_results[layer][prompt]['train_results'])
            for jx, metric in enumerate(metrics):
                ax = axs[jx, ix]
                layer_df.plot(x='epoch', y=metric, ax=ax, label=layer,
                              color=colorFader(mix=layer/n_layers))
                ax.set_title(f'{prompt} {metric}')
                # don't plot legend
                ax.get_legend().remove()
    plt.tight_layout()


def plot_rank_probe_train_metadata_plotly(probe_results, prompts, metrics=('weight_norm', 'train_spearman', 'pred_max', 'pred_min')):
    n_layers = max([int(k[-1]) for k in probe_results.keys()]) + 1
    layer_results = {int(k[-1]): v for k, v in probe_results.items()}
    if n_layers != len(layer_results):
        print('Warning missing or extra layers in probe results')

    fig = make_subplots(rows=len(metrics), cols=len(prompts),
                        subplot_titles=[f'{prompt} {metric}' for metric in metrics for prompt in prompts])

    for ix, prompt in enumerate(prompts):
        for layer in range(n_layers):
            layer_df = pd.DataFrame(
                layer_results[layer][prompt]['train_results'])
            for jx, metric in enumerate(metrics):
                fig.add_trace(
                    go.Scatter(x=layer_df['epoch'], y=layer_df[metric],
                               mode='lines', name=str(layer),
                               line=dict(color=colorFader(mix=layer/n_layers))),
                    row=jx + 1, col=ix + 1
                )

    fig.update_layout(height=300 * len(metrics), width=300 *
                      len(prompts), showlegend=False)
    fig.show()
