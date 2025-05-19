import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .constants import metric_map
from .utils import paretoset_naive


def plot_scaling_curve(df, fig=None, models=None, benchmark=None, save=False, save_path='figures/loss_curve_merged.pdf',
                       xlim=None, ylim=None, figsize=(10, 6)):
    
    if models is None:
        valid_modes = ['mammut', 'clip', 'coca']
    else:
        valid_modes = models
    df_filtered = df[df['model_type'].isin(valid_modes)].copy()

    # Get unique model sizes
    params_list = df_filtered.model.unique()

    model_ranking = [
        "S-", "M-", "B-", "L-", "H-"
    ]

    def get_model_size(model):
        # Extract the model size from the model name
        for size in model_ranking:
            if size in model:
                return size
        return model
    
    # sort by model ranking
    params_list = sorted(params_list, key=lambda x: (model_ranking.index(get_model_size(x)) if get_model_size(x) in model_ranking else len(model_ranking), x))
    # Create color schemes for each mode
    # Use distinct color maps for better differentiation
    clip_cmap = plt.cm.viridis_r
    mammut_cmap = plt.cm.viridis_r
    coca_cmap = plt.cm.viridis_r

    colors_mapping = {
        'clip': clip_cmap,
        'mammut': mammut_cmap,
        'coca': coca_cmap,
        'siglip': plt.cm.viridis_r
    }

    colors_dict = dict()

    if fig is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor='w')

    with plt.rc_context({
        'text.usetex': False,
        'font.family': 'sans-serif',
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 20,
    }):

        legend_handles = []

        for mode in valid_modes:
            colors_dict[mode] = dict()
            mode_df = df_filtered[df_filtered['model_type'] == mode].copy()

            # Get unique model sizes for this mode and assign colors
            params_list = mode_df.model.unique()
            params_list = sorted(params_list, key=lambda x: (model_ranking.index(get_model_size(x)) if get_model_size(x) in model_ranking else len(model_ranking), x))

            mode_colors = colors_mapping[mode](np.linspace(0.4, 0.9, len(params_list)))


            for i, param_size in enumerate(params_list):
                model_df = mode_df[mode_df.model == param_size].copy()

                # Calculate compute
                model_df['compute'] = model_df["gflops_total"]
                model_df = model_df.sort_values('compute')
                pareto_df = []
                for param in params_list:
                    param_df = model_df[model_df.model == param].copy()
                    X = param_df['compute'].values
                    y = param_df[metric_map[benchmark]].values
                    pareto_front = paretoset_naive(np.array([X, y]).T)
                    pareto_df.append(param_df.iloc[pareto_front])
                pareto_df = pd.concat(pareto_df)
                # Sort by compute (which corresponds to more training steps)
            
                line = plt.plot(
                    pareto_df['compute'],
                    pareto_df[metric_map[benchmark]],
                    'o-',  # Circles with lines
                    color=mode_colors[i],
                    label=param_size,
                    lw=0.8,
                    alpha=0.4
                )[0]

                legend_handles.append(line)
                colors_dict[mode][param_size] = mode_colors[i]

      
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel('Loss', fontsize=13)
        plt.xlabel('Compute $C$ [FLOPs]', fontsize=13)
        # plt.title('Scaling Curves: Loss vs. Compute', fontsize=18)


        plt.legend(handles=legend_handles, fontsize=14, loc='best', framealpha=0.9)

        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)

        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()

        if save:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

    return fig, colors_dict


