import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from .utils import paretoset_naive, get_model_type, lower_convexhull, pareto_frontier_bins
from .functions import func_map
from .constants import colors, dataset_colors
from .configs import (
    FigureConfig,
    ScalingLawConfig,
    SaveConfig,
    LargeFlopsMapping,
    ScalingFunction,
)
import argparse
import warnings
from .fitting_utils import fit_bootstrap
from .saving_utils import save_figure
from .loss_curve_only_plot import plot_scaling_curve
import json
from scipy import stats

warnings.filterwarnings("ignore")

def extract_number(s):
    # extract number from a string like 1.28B, 1.28M, 1.28K
    if "B" in s:
        return float(s.replace("B", "").replace(",", "")) * 1e9
    elif "M" in s:
        return float(s.replace("M", "").replace(",", "")) * 1e6
    elif "K" in s:
        return float(s.replace("K", "").replace(",", "")) * 1e3
    else:
        return float(s.replace(",", ""))

def filter_data(
        df, 
        benchmark, 
        metric_map, 
        lr_scheduler, 
        min_flops=None, 
        max_samples_seen=None,
        x_col=None,
        max_x=None,
        eval_type=None,
        ):
    """
    Filter the data for the given benchmark and lr_scheduler
    """
    # print(df.columns)
    if benchmark == "ade20k":
        df["gflops_total" ]= df["gflops"]
        df["total_samples_seen"] = df["samples_seen"]
    
    if max_x is not None:
        df = df[df[x_col] <= max_x]
        
    df["model_type"] = df.model.apply(lambda x: get_model_type(x))
    if eval_type is not None:
        df = df[df.eval_type == eval_type]
    df = df[df.downstream_dataset == benchmark]
    df = df[df.lr_scheduler == lr_scheduler]
    if benchmark != "ade20k":
        if lr_scheduler in ["cosine", "const-cooldown"]:
            df = df[df[["epoch", "total_epochs"]].apply(lambda x: x[0] == x[1], axis=1)]
        else:
            df["smaples_per_epoch"] = df["total_samples_seen"] / df["total_epochs"]
            df["total_smaples_seen"] = df["epoch"] * df["smaples_per_epoch"]
            df["gflops_total"] = df["gflops"] * df["total_smaples_seen"]
            df["total_steps"] = df["total_samples_seen"] / df["global_batch_size"]
            df = df[df[["warmup", "total_steps", "epoch", "total_epochs"]].apply(lambda x: (x[0]/x[1]) <= (x[2]/x[3]), axis=1)]


    if "loss" not in metric_map[benchmark].lower():
        df[metric_map[benchmark]] = df[metric_map[benchmark]].apply(
            lambda x: 1 - x if x < 1 else 100 - x
        )

    
    if min_flops is not None:
        df = df[df.gflops_total >= min_flops]

  
    
    if metric_map[benchmark] == "contrastive_loss":
        df = df[~df.contrastive_loss.isna()]
       

    if max_samples_seen is not None:
        df = df[df.total_samples_seen <= max_samples_seen]

    if benchmark == "imagenet1k":
        df = df[df[["gflops_total", "acc1"]].apply(lambda x: x[1] < 0.7 if x[0] > 1e11 else True, axis=1)]
    if benchmark == "mscoco_captions":
        df = df[df[["gflops_total", metric_map[benchmark]]].apply(lambda x: x[1] < 0.7 if x[0] > 1e11 else True, axis=1)]

    if x_col == "total_samples_seen":
        df = df[~df.model.str.contains("H-")]
    

    print(df.pretrain_dataset.unique())
    return df



def create_figure(
    X,
    Y,
    pareto_frontier,
    x_pred,
    y_pred_all,
    large_flops_mapping,
    popts,
    popt_mean,
    model_type,
    pretrain_dataset,
    func,
    i,
    hue=None,
    show_ci=True,
    pretrain_datasets=None,
    x_col=None,
    X_flops=None,
    higher_pareto_frontier=None,
    approach3_params_path=None,
    pcov=None,
    max_flops=1e12,
    color_values=None,
    plot_scaling_curve_by_mode=False,
):
    """
    Create the figure for the results
    """
    x_all = X[pareto_frontier]
    y_all = Y[pareto_frontier]
    func_str = str(func)
    func_str = f"{model_type.upper()}: {func_str}"

    if isinstance(x_col, list) and len(x_col) > 1:
        x_to_show = X_flops
    else:
        x_to_show = X



    
    flop_grid = np.geomspace(
        np.exp(x_to_show.min()) if func.func_type in ["line", "line_correction", "simple_line"] else x_to_show.min(),
        max_flops, num=300
    )


    if func.func_type in ["line", "line_correction", "simple_line"]:
        flop_grid = np.log(flop_grid)
        # x_to_show = np.log(x_to_show)
        # x_all = np.log(x_all)
        # x_pred = np.log(x_pred)
        # y_all = np.log(y_all)
        # y_pred_all = np.log(y_pred_all)

    X_flops_pareto = X_flops[pareto_frontier]
    x_to_show_pareto = x_to_show[pareto_frontier]
    x_pred = x_to_show[higher_pareto_frontier]

    func_to_show = func(flop_grid, *popt_mean)
    x_to_show_pareto_lower = x_to_show_pareto[X_flops_pareto <= large_flops_mapping[model_type]]
    x_to_show_pareto_lower = np.geomspace(
        np.exp(x_to_show_pareto_lower.min()) if func.func_type in ["line", "line_correction", "simple_line"] else x_to_show_pareto_lower.min(),
        np.exp(x_to_show_pareto.max()) if func.func_type in ["line", "line_correction", "simple_line"] else x_to_show_pareto.max(),
        num=300,
    ) # to make it smoother
    if func.func_type in ["line", "line_correction", "simple_line"]:
        x_to_show_pareto_lower = np.log(x_to_show_pareto_lower)
        func_to_show_lower = func(x_to_show_pareto_lower, *popt_mean)
    else:
        func_to_show_lower = func(x_to_show_pareto_lower, *popt_mean)

    
    if show_ci:
       
        lower, upper = compute_ci(
            x_all, func, popts, model_type, pcov, flop_grid, x_to_show_pareto_lower, alpha=0.05
        ) 

        # if func.func_type in ["line", "line_correction", "simple_line"]:
        #     lower = np.exp(lower)
        #     upper = np.exp(upper)
        plt.fill_between(
            np.exp(flop_grid) if func.func_type in ["line", "line_correction", "simple_line"] else flop_grid,
            lower, upper, alpha=0.2, color=colors[model_type] if len(pretrain_datasets) == 1 else dataset_colors[pretrain_dataset])
        
    if func.func_type in ["line", "line_correction", "simple_line"]:
        x_to_show = np.exp(x_to_show)
        x_all = np.exp(x_all)
        x_pred = np.exp(x_pred)
        x_to_show_pareto = np.exp(x_to_show_pareto)
        x_all = np.exp(x_all)
        func_to_show = np.exp(func_to_show)
        func_to_show_lower = np.exp(func_to_show_lower)
        y_pred_all = np.exp(y_pred_all)
        y_all = np.exp(y_all)
        x_to_show_pareto_lower = np.exp(x_to_show_pareto_lower)
        Y = np.exp(Y)
        flop_grid = np.exp(flop_grid)

    
    sns.lineplot(
        x=flop_grid,
        y=func_to_show,
        color=colors[model_type] if len(pretrain_datasets) == 1 else dataset_colors[pretrain_dataset],
        linestyle="--",
        linewidth=1,
    )
    
    sns.lineplot(
        x=x_to_show_pareto_lower,
        y=func_to_show_lower,
        linewidth=1,
        label=func_str,
        legend=False,
        color=colors[model_type] if len(pretrain_datasets) == 1 else dataset_colors[pretrain_dataset],
    )



    sns.scatterplot(
        x=x_to_show_pareto,
        y=y_all,
        palette=color_values,
        color=colors[model_type] if len(pretrain_datasets) == 1 else dataset_colors[pretrain_dataset] if not color_values else None,
        alpha=0.5,
        s=30,
        # marker=markers[pretrain_dataset] if len(pretrain_datasets) > 1 else "o",
        legend=False,
    )

    print(color_values)

    if not plot_scaling_curve_by_mode:
        sns.lineplot(
            x=x_to_show,
            y=Y,
            # alpha=0 if color_values is not None else 0.01,
            alpha=0.01 if model_type == "clip" else 0.1,
            # palette=colors[model_type] if hue is None else "viridis",
            palette=color_values,
            color=colors[model_type] if len(pretrain_datasets) == 1 else dataset_colors[pretrain_dataset] if not color_values else None,
            marker="o",
            linewidth=0,
            errorbar=None,
            label=pretrain_dataset if len(pretrain_datasets) > 1 else None,
            legend=False
        )

    sns.scatterplot(
        x=x_pred,
        y=y_pred_all,
        color=colors[model_type] if len(pretrain_datasets) == 1 else dataset_colors[pretrain_dataset],
        alpha=0.7,
        s=60,
        marker="X",
    )

    def logpow(x, a, b):
        # x = np.exp(x)
        return a * np.power(x, b)


    if approach3_params_path is not None:
        with open(approach3_params_path, "r") as f:
            params = json.load(f)

        G_inv = params[model_type]["G_inv"]
        b = params[model_type]["b"]
        print(G_inv, np.power(1e9 /6, b))
        G_inv = G_inv * np.power(6, b) 
        # G_inv = np.log(1/G_inv)

        y_pow = logpow(x_to_show, a=G_inv, b=b)
        # print("x_to_show", x_to_show, y_pow)
    
        # if func.func_type in ["line", "line_correction", "simple_line"]:
        #     y_pow = np.exp(y_pow)
        
        sns.lineplot(
            x=x_to_show,
            y=y_pow,
            label=f"approach 3 ({model_type}): ${G_inv:.2f}*x^{{{b:.3f}}}$",
            markers="x",
            linestyle="-.",
            color=colors[model_type]
        )

def compute_ci(
    x_all, func, popts, model_type, pcov, flop_grid, x_to_show_pareto_lower, alpha=0.05,
):
    """
    Compute the confidence interval for the predictions
    """
    if len(popts) > 1:
            lower, upper = compute_confidence_interval_boostrap(
                flop_grid, func, popts, alpha=0.05
            )
    elif pcov is not None:
        # Compute the confidence interval for the predictions
        
        popt_mean = popts[0]
    
        J = func.jacobian(params=popt_mean, x=flop_grid)
        y_std_pred = np.sqrt(np.diag(J @ pcov @ J.T))
        # Compute the t-value for the confidence interval
        alpha = 0.05
        tval = stats.t.ppf(1 - alpha / 2, len(x_to_show_pareto_lower) - len(popt_mean))
        tval = np.log(tval)


        lower = func(flop_grid, *popt_mean) - tval * y_std_pred
        upper = func(flop_grid, *popt_mean) + tval * y_std_pred

        if func.func_type in ["line", "line_correction", "simple_line"]:
            lower = np.exp(lower)
            upper = np.exp(upper)

    return lower, upper

def compute_confidence_interval_boostrap(x_pred, func, popts, alpha=0.05):
    """
    Compute the confidence interval for the predictions
    """

    preds = []
    for popt in popts:
        preds.append(func(x_pred, *popt))
    preds = np.array(preds)
    # np.sort(preds, axis=0)
    # preds = np.log(preds)
    lower = np.percentile(preds, 100 * alpha / 2, axis=0)
    upper = np.percentile(preds, 100 * (1 - alpha / 2), axis=0)
    # lower = np.exp(lower)
    # upper = np.exp(upper)
    return lower, upper

def fit_scaling_law_and_make_figure(
    bootstrap_results: dict,
    i: int,
    group: pd.DataFrame,
    benchmark: str,
    metric_map: dict,
    large_flops_mapping: dict,
    x_col: str,
    x_col_pareto: str,
    y_col: str,
    model_type: str,
    figure_config: FigureConfig = None,
    extra_cols: list = None,
    scaling_law_config: ScalingLawConfig = None,
    save_config: SaveConfig = None,
    max_x: float = 1e12,
    colors_dict=None,
    pretrain_dataset=None,
) -> None:


    func_type = scaling_law_config.func_type
    method = scaling_law_config.method
    n_bootstraps = scaling_law_config.n_bootstraps
    show_ci = scaling_law_config.show_ci
    pretrain_datasets = scaling_law_config.pretrain_datasets
    pareto_method = scaling_law_config.pareto_method

    hue = scaling_law_config.hue
    color_values = None
    # if colors_dict is not None:
    #     group["color"] = group["model"].apply(lambda x: colors_dict[x])
    #     color_values = group["color"].values
    # else:
    #     color_values = None

    func = ScalingFunction(func_type)
    print(f"Processing {model_type} model")
    if func_type in ["line", "line_correction", "simple_line"]:
        group[x_col] = group[x_col].apply(np.log)
        group[y_col] = group[y_col].apply(np.log)

        large_flops_mapping[model_type] = np.log(large_flops_mapping[model_type])

    X = group[x_col_pareto].values
    Y = group[metric_map[benchmark]].values


    # sort the values
    sorted_indices = np.lexsort((Y, X))
    X = X[sorted_indices]
    Y = Y[sorted_indices]
    color_values = color_values[sorted_indices] if color_values is not None else None
    X_flops = X.copy()
    lower_inidices = np.where(X <= large_flops_mapping[model_type])[0]
    if hue is not None:
        hue = group[hue].values

    if func_type not in ["power_func_multi", "power_func_multi3d", "power_func_multi4d"]:
        if pareto_method == "convexhull":
            pareto_frontier = lower_convexhull(X,Y)
        elif pareto_method == "pareto_naive":
            pareto_frontier = paretoset_naive(np.array([X, Y]).T)
        elif pareto_method == "pareto_bin":
            pareto_frontier = pareto_frontier_bins(
                np.array([X, Y]).T,
                num_bins=None,  # automatically compute the number of bins
                n_to_keep=1,
            )
        else:
            raise ValueError(f"Unknown pareto method: {pareto_method}")
    else:
        pareto_frontier = np.ones(len(X), dtype=bool)

    if save_config.save_pareto_to_df:
        group_pareto = group.iloc[sorted_indices][pareto_frontier]
        group_pareto.to_csv(
            os.path.join(save_config.save_pareto_to_df_path, f"{model_type}_pareto.csv"),
            index=False,
        )
    

        
    lower_pareto_frontier = [idx for idx in lower_inidices if pareto_frontier[idx]]
    higher_pareto_frontier = [idx for idx in range(len(X)) if X[idx] > large_flops_mapping[model_type] and pareto_frontier[idx]]
    X = group[x_col].values[sorted_indices]
    if y_col is not None:
        Y = group[y_col].values[sorted_indices]

    
    
    x_all = X[lower_pareto_frontier]
    y_all = Y[lower_pareto_frontier]

    x_pred = X[higher_pareto_frontier]
    y_pred_all = Y[higher_pareto_frontier]
    extra_data = group[extra_cols].values[sorted_indices][higher_pareto_frontier] if extra_cols is not None else None
    extra_data = dict(zip(extra_cols, extra_data.T)) if extra_data is not None else None
    # Fit the model to the data
    preds, popts, pcov = fit_bootstrap(
        x_all, y_all, x_pred, func=func, n_bootstrap=n_bootstraps, method=method
    )

    preds = np.array(preds)
    median = np.median(preds, axis=0)

    popt_mean = np.median(popts, axis=0)
    print(f"popt_mean: {popt_mean}")
    func.params = popt_mean

    if isinstance(x_col, str):
        pred = func(x_pred, *popt_mean)
        if scaling_law_config.predict_for_vals is not None:
            if func_type in ["line", "line_correction", "simple_line"]:
                x_pred = np.log(scaling_law_config.predict_for_vals)
            else:
                x_pred = np.array(scaling_law_config.predict_for_vals)
            pred = func(x_pred, *popt_mean)
        ci = compute_ci(
            x_all, func, popts, model_type, pcov, x_pred, x_all, alpha=0.05
        )
        if func_type in ["line", "line_correction", "simple_line"]:
            pred = np.exp(pred)
            # ci = np.exp(ci)
        predictions = {
            x_col: x_pred,
            f"{y_col}_predicted": pred,
            y_col: y_pred_all,
        }
        predictions["extra_data"] = extra_data
        predictions["ci"] = ci
    else:
        predictions = {}
        for j, x_col_i in enumerate(x_col):
            predictions[x_col_i] = x_pred[:, j]

        predictions[y_col] = y_pred_all
        pred = func(x_pred, *popt_mean)
        ci = compute_ci(
            x_all, func, popts, model_type, pcov, x_pred, x_all[lower_pareto_frontier], alpha=0.05
        )
        if func_type in ["line", "line_correction", "simple_line"]:
            pred = np.exp(pred)
            ci = np.exp(ci)
        predictions[f"{y_col}_predicted"] = pred
        predictions["extra_data"] = extra_data
        predictions["ci"] = ci


    bootstrap_results[model_type] = {
        "median": median,
        "popt_mean": popt_mean,
        "popt_std": np.std(popts, axis=0),
        "popts": popts,
        "func": func,
        "predictions": predictions,
    }
    if scaling_law_config.create_figure:

        create_figure(
            X,
            Y,
            pareto_frontier,
            x_pred=x_pred,
            y_pred_all=y_pred_all,
            large_flops_mapping=large_flops_mapping,
            popts=popts,
            popt_mean=popt_mean,
            model_type=model_type,
            func=func,
            i=i,
            hue=hue,
            show_ci=show_ci,
            pretrain_dataset=pretrain_dataset,
            pretrain_datasets=pretrain_datasets,
            X_flops=X_flops,
            higher_pareto_frontier=higher_pareto_frontier,
            x_col=x_col,
            approach3_params_path=scaling_law_config.approach3_params_path,
            pcov=pcov,
            max_flops=max_x,
            color_values=color_values,
            plot_scaling_curve_by_mode=scaling_law_config.plot_scaling_curve_by_mode,
        )
    
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(x_col if figure_config.x_label is None else figure_config.x_label, fontsize=13)
    plt.ylabel(y_col if figure_config.y_label is None else figure_config.y_label, fontsize=13)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    if scaling_law_config.plot_scaling_curve_by_mode:
        # move legend out of the plot
        plt.legend(
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )
    else:
        plt.legend(
            loc="best",
        )
    plt.tight_layout()


def plot_scaling_law_bootstrap(
    df: pd.DataFrame, 
    scaling_law_config: ScalingLawConfig, 
    save_config: SaveConfig = None,
    figure_config: FigureConfig = None,
) -> None:
    """
    Plot the scaling law using bootstrapping
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the data
    scaling_law_config : ScalingLawConfig
        Configuration for the scaling law
    save_config : SaveConfig
        Configuration for saving the results
    figure_config : FigureConfig
        Configuration for the figure
    """

    hue = figure_config.hue

    # if figure_config is not None:
    #     hue = figure_config.hue
    #     show_ci = figure_config.show_ci
    #     fig_width = figure_config.fig_width
    #     fig_height = figure_config.fig_height
    # else:
    #     hue = None
    #     show_ci = True
    #     fig_width = 8
    #     fig_height = 6

    benchmark = scaling_law_config.benchmark
    metric_map = scaling_law_config.metric_map
    large_flops_mapping = scaling_law_config.large_flops_mapping
    n_bootstraps = scaling_law_config.n_bootstraps
    lr_scheduler = scaling_law_config.lr_scheduler
    y_col = scaling_law_config.y_col
    x_col = scaling_law_config.x_col
    if len(x_col) == 1:
        x_col = x_col[0]
    x_col_pareto = scaling_law_config.x_col_pareto
    func_type = scaling_law_config.func_type
    method = scaling_law_config.method
    models = scaling_law_config.models
    extra_cols = save_config.save_extra_cols
    pretrain_datasets = scaling_law_config.pretrain_datasets
    
    if func_type not in func_map:
        raise ValueError(f"Function type {func_type} not in {func_map.keys()}")
    if method not in ["curve_fit", "huber", "isoflop_curves"]:
        raise ValueError(f"Method {method} not in ['curve_fit', 'huber', 'isoflop_curves']")
    if save_config is not None:
        if not os.path.exists(save_config.save_dir):
            os.makedirs(save_config.save_dir)

    if large_flops_mapping is None:
        large_flops_mapping = {"mammut": np.inf, "coca": np.inf, "clip": np.inf}
    df_clean = filter_data(
        df, 
        benchmark, 
        metric_map, 
        lr_scheduler, 
        scaling_law_config.min_flops, 
        scaling_law_config.max_samples_seen,
        x_col=x_col,
        max_x=figure_config.max_x,
    )
    if models is not None:
        df_clean = df_clean[df_clean.model_type.isin(models)]
    if pretrain_datasets and len(pretrain_datasets) > 0:
        df_clean = df_clean[df_clean.pretrain_dataset.isin(pretrain_datasets)]
    else:
        pretrain_datasets = df_clean.pretrain_dataset.unique()



    if y_col is None:
        y_col = metric_map[benchmark]

    bootstrap_results = {}
    i = 1

    figure = plt.figure(figsize=(figure_config.fig_width, figure_config.fig_height))
    colors_dict = {}

    if scaling_law_config.plot_scaling_curve_by_mode:
        figure, colors_dict = plot_scaling_curve(
            df_clean,
            models=models,
            benchmark=benchmark,
            fig=figure,
            save=False
        )

    if len(pretrain_datasets) > 1:
        if hue is None:
            hue = "pretrain_dataset"



    for pretrain_dataset in pretrain_datasets:
        print(f"Processing {pretrain_dataset} dataset")
        df_clean_dataset = df_clean[df_clean.pretrain_dataset == pretrain_dataset]
        for model_type, group in df_clean_dataset.groupby("model_type"):

            if method == "isoflop_curves":
                isoflop_curves(df_clean, scaling_law_config, model_type)
                continue
        
            fit_scaling_law_and_make_figure(
                bootstrap_results,
                i,
                group,
                benchmark,
                metric_map,
                large_flops_mapping,
                x_col,
                x_col_pareto,
                y_col,
                model_type,
                figure_config=figure_config,
                extra_cols=extra_cols,
                scaling_law_config=scaling_law_config,
                max_x=figure_config.max_x,
                pretrain_dataset=pretrain_dataset,
                colors_dict=colors_dict,
                save_config=save_config,
            )
            i += 1
        
    if save_config is not None:
        save_figure(
            figure,
            models,
            save_config,
            benchmark,
            func_type,
            method,
            large_flops_mapping,
            x_col,
            y_col,
            bootstrap_results,
        )
    
    
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot scaling laws for CLIP models")
    parser.add_argument(
        "--benchmark", type=str, required=True, help="Benchmark to plot"
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Directory to save the results"
    )
    parser.add_argument(
        "--n_bootstraps", type=int, default=100, help="Number of bootstraps"
    )
    parser.add_argument(
        "--lr_scheduler", type=str, default="cosine", help="Learning rate scheduler"
    )
    parser.add_argument(
        "--y_col",
        type=str,
        default=None,
        help="Column to use for y-axis by default it is the metric",
    )
    parser.add_argument(
        "--x_col",
        type=str,
        default="gflops_total",
        nargs="+",
        help="Column to use for x-axis, default is gflops_total",
    )
    parser.add_argument(
        "--func_type", type=str, default="saturation", help="Function type to use"
    )
    parser.add_argument("--hue", type=str, default=None, help="Column to use for hue")
    parser.add_argument(
        "--show_ci", action="store_true", help="Show confidence intervals"
    )
    parser.add_argument(
        "--method", type=str, default="curve_fit", help="Method to use for fitting"
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=None, help="Models to use"
    )
    parser.add_argument(
        "--pretrain_datasets",
        type=str,
        nargs="+",
        default=None,
        help="Pretrain datasets to use",
    )
    parser.add_argument(
        "--save_extra_cols",
        type=str,
        nargs="+",
        default=None,
        help="Extra columns to save in the results",
    )
    parser.add_argument(
        "--save_latex", action="store_true", help="Save figure in LaTeX format"
    )
    parser.add_argument(
        "--save_json", action="store_true", help="Save results in JSON format"
    )
    parser.add_argument(
        "--save_csv", action="store_true", help="Save results in CSV format"
    )
    parser.add_argument(
        "--save_pdf", action="store_true", help="Save figure in PDF format"
    )
    parser.add_argument(
        "--create_figure", action="store_true", help="Create figure"
    )
    parser.add_argument(
        "--min_flops", type=float, default=None, help="Minimum flops to use"
    )
    parser.add_argument(
        "--approach3_params_path", type=str, default=None, help="Path to the approach 3 params"
    )
    parser.add_argument("--fig_width", type=int, default=8, help="Figure width")
    parser.add_argument("--fig_height", type=int, default=6, help="Figure height")
    parser.add_argument(
        "--large_flops_mapping", type=str, default=None, help="Mapping for large flops"
    )  # "mammut:1e8,coca:1e8,clip:1e8"
    parser.add_argument(
        "--pareto_method",
        type=str,
        default="pareto_naive",
        help="Method to use for pareto frontiers",
    )
    parser.add_argument(
        "--x_label", type=str, default=None, help="X-axis label"
    )
    parser.add_argument(
        "--y_label", type=str, default=None, help="Y-axis label"
    )
    parser.add_argument(
        "--max_x", type=float, default=1e12, help="Maximum x value for the plot"
    )
    parser.add_argument(
        "--x_col_pareto", type=str, default="gflops_total", help="Column to use for pareto frontiers"
    )
    parser.add_argument(
        "--max_samples_seen", type=float, default=None, help="Maximum samples seen to use"
    )
    parser.add_argument(
        "--plot_scaling_curve_by_mode", action="store_true", help="Plot scaling curve by mode"
    )
    parser.add_argument(
        "--predict_for_vals", type=float, nargs="+", default=None, help="Values to predict for"
    )
    parser.add_argument(
        "--save_pareto_to_df", action="store_true", help="Save pareto frontiers to dataframe"
    )
    parser.add_argument(
        "--eval_type", type=str, default=None, help="Evaluation type to use"
    )


    args = parser.parse_args()

    current_path = os.path.dirname(os.path.abspath(__file__))

    if args.benchmark == "ade20k":
        data_path = os.path.join(current_path, "data/segmentation_results.parquet")
        df = pd.read_parquet(data_path, engine="fastparquet")
    else:
        data_path = os.path.join(current_path, "data/all_results.csv.gz")
        df = pd.read_csv(data_path, compression="gzip")

    # Define the large flops mapping
    large_flops_mapping = LargeFlopsMapping()
    if args.large_flops_mapping is not None:
        large_flops_mapping.parse_str(args.large_flops_mapping)
    # Convert the large flops mapping to a dictionary

    # Define the scaling law config
    scaling_law_config = ScalingLawConfig(
        benchmark=args.benchmark,
        large_flops_mapping=large_flops_mapping,
        n_bootstraps=args.n_bootstraps,
        lr_scheduler=args.lr_scheduler,
        y_col=args.y_col,
        x_col=args.x_col,
        func_type=args.func_type,
        hue=args.hue,
        show_ci=args.show_ci,
        method=args.method,
        models=args.models,
        create_figure=args.create_figure,
        pretrain_datasets=args.pretrain_datasets,
        min_flops=args.min_flops,
        x_col_pareto=args.x_col_pareto,
        pareto_method=args.pareto_method,
        approach3_params_path=args.approach3_params_path,
        max_samples_seen=args.max_samples_seen,
        plot_scaling_curve_by_mode=args.plot_scaling_curve_by_mode,
        predict_for_vals=args.predict_for_vals,
        eval_type=args.eval_type,
    )
    # Define the save config
    save_config = SaveConfig(
        save_dir=args.save_dir,
        save_latex=args.save_latex,
        save_json=args.save_json,
        save_csv=args.save_csv,
        save_pdf=args.save_pdf,
        save_extra_cols=args.save_extra_cols,
        save_pareto_to_df=args.save_pareto_to_df,
    )
    # Define the figure config
    figure_config = FigureConfig(
        hue=args.hue,
        show_ci=args.show_ci,
        fig_width=args.fig_width,
        fig_height=args.fig_height,
        x_label=args.x_label,
        y_label=args.y_label,
        max_x=args.max_x
    )

    # Plot the scaling law
    plot_scaling_law_bootstrap(
        df, scaling_law_config, save_config=save_config, figure_config=figure_config
    )
