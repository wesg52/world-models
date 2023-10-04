import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_per_model_lat_lon_correlation(lat_df, lon_df, models):
    prompts = lat_df.prompt.unique()
    n_rows = len(models)
    n_cols = 2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    for ix, model in enumerate(models):
        # plot lat in first column
        model_lat_df = lat_df[lat_df.model == model]
        model_lon_df = lon_df[lon_df.model == model]
        for prompt in prompts:
            prompt_df = model_lat_df[model_lat_df.prompt == prompt]
            prompt_df.plot(x='layer', y='test_spearman_coef',
                           ax=axs[ix, 0], label=prompt)

            prompt_df = model_lon_df[model_lon_df.prompt == prompt]
            prompt_df.plot(x='layer', y='test_spearman_coef',
                           ax=axs[ix, 1], label=prompt)

            axs[ix, 0].set_title(f'{model} latitude')
            axs[ix, 1].set_title(f'{model} longitude')

    plt.tight_layout()


def per_prompt_lat_lon_correlation(lat_df, lon_df, models, prompts, normalize_layer=False):
    n_rows = len(prompts)
    n_cols = 2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))

    for ix, prompt in enumerate(prompts):
        # Setting a flag to know if we've already set the title
        set_title = False

        for model in models:
            model_lat_df = lat_df[lat_df.model == model]
            model_lon_df = lon_df[lon_df.model == model]

            # Filter dataframes by prompt and model
            prompt_lat_df = model_lat_df[model_lat_df.prompt == prompt]
            prompt_lon_df = model_lon_df[model_lon_df.prompt == prompt]

            if normalize_layer:
                layers = (prompt_lat_df['layer'] /
                          prompt_lat_df['layer'].max()).values
            else:
                layers = prompt_lat_df['layer'].values

            lat_spearman = prompt_lat_df['test_spearman_coef'].values
            lon_spearman = prompt_lon_df['test_spearman_coef'].values
            # Plot for latitude
            axs[ix, 0].plot(layers, lat_spearman, label=model)

            # Plot for longitude
            axs[ix, 1].plot(layers, lon_spearman, label=model)

            # Set the title once
            if not set_title:
                axs[ix, 0].set_title(f'{prompt} latitude')
                axs[ix, 1].set_title(f'{prompt} longitude')
                set_title = True

    # Adjusting the layout and adding legends
    for i in range(n_rows):
        axs[i, 0].legend()
        axs[i, 1].legend()
    plt.tight_layout()
