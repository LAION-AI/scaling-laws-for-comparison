from decimal import Decimal
import json
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from .constants import metric_map


def save_results(
        bootstrap_results: Dict[str, Dict[str, List[float]]],
        output_dir: str,
        benchmark: str,
        ):
    """
    Save the results to a csv file
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for model_type, result in bootstrap_results.items():
        # Save the results to a csv file
        df = pd.DataFrame(result)
        df.to_csv(
            os.path.join(output_dir, f"{benchmark}_{model_type}.csv"), index=False
        )

    # Save the results to a json file
    with open(os.path.join(output_dir, f"{benchmark}_results.json"), "w") as f:
        json.dump(bootstrap_results, f)


def save_latex(
        bootstrap_results: Dict[str, Dict[str, List[float]]],
        filename: str,
        ) -> None:
    """
    Save the results to a latex file
    """
    coefficients = {}
    for model_type, result in bootstrap_results.items():
        func = result["func"]
        func_repr = str(func)
        param_dict = func.param_to_display_dict()
        coefficients[model_type] = {"func_repr": func_repr, "param_dict": param_dict}
    num_params = len(list(coefficients.values())[0]["param_dict"].keys())

    with open(filename, "w") as f:
        # make a table with cole model_type, param1, param2, ...
        f.write("\\begin{table}\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{|c|" + "|c" * num_params + "|}\n")
        f.write("\\hline\n")
        f.write(
            "\\textbf{Model} & "
            + " & ".join(
                [
                    f"\\textbf{{{k}}}"
                    for k in coefficients[list(coefficients.keys())[0]][
                        "param_dict"
                    ].keys()
                ]
            )
            + " \\\\\n"
        )
        f.write("\\hline\n")
        for model_type, coeffs in coefficients.items():
            f.write(
                f"{model_type.upper()} & "
                + " & ".join(
                    [
                        f"{coeffs['param_dict'][k]:.6f}"
                        for k in coeffs["param_dict"].keys()
                    ]
                )
                + " \\\\\n"
            )
            f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Model parameters}\n")
        f.write("\\label{tab:coefficients}\n")
        f.write("\\end{table}\n")

def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{Decimal(value):.2e}" if value > 1 else f"{1-value:.3f}"
    elif isinstance(value, int):
        return f"{value}"
    elif isinstance(value, str):
        return value
    else:
       return str(value)

def compute_mse(
    y_true: List[float],
    y_pred: List[float],
):
    """
    Compute the mean squared error between y_true and y_pred
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    return mse




def save_predictions(
        bootstrap_results: Dict[str, Dict[str, List[float]]],
        filename: str,
        x_col: str,
        y_col: str = "y",
        extra_cols: Optional[List[str]] = None,
        method: Optional[str] = "curve_fit",
        func_type: Optional[str] = "simple",
        benchmark: str = None,
        ):
    """
    Save the predictions to a csv file
    """
    # Save the results to latex
    base_num_cols = 4
    num_cols = base_num_cols + len(extra_cols) if extra_cols is not None else base_num_cols
    with open(filename, "w") as f:
        f.write("\\begin{table}\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{|" + "|c" * num_cols + "|}\n")
        f.write("\\hline\n")
        if isinstance(x_col, str):
            header = f"Model & {x_col} & {y_col} & {y_col}_predicted \\\\\n" if extra_cols is None else f"Model &" +  " & ".join(extra_cols) + f" & {x_col} & {y_col} & {y_col}_predicted \\\\\n"
        else:
            header = f"Model & " + " & ".join(x_col) + f" & {y_col} & {y_col}_predicted \\\\\n" if extra_cols is None else f"Model & " + " & ".join(extra_cols) + f" & " + " & ".join(x_col) + f" & {y_col} & {y_col}_predicted \\\\\n"
        f.write(header)
        f.write("\\hline\n")

        for model_type, result in bootstrap_results.items():
            extra_data = result["predictions"]["extra_data"]
            ci_list = result["predictions"]["ci"]
            ci_list = np.array(ci_list).T

            if isinstance(x_col, str):
                for i in range(len(result["predictions"][x_col])):
                    x = result["predictions"][x_col][i]

                    if len(result["predictions"][y_col]) > 0:
                        y = result["predictions"][y_col][i]
                    else:
                        y = "-"

                    if func_type in ["line", "line_correction", "line_correction_both"]:
                        x = np.exp(x)
                        if y != "-":
                            y = np.exp(y)
            
                    

                    y_pred = result["predictions"][f"{y_col}_predicted"][i]
                    # print(result["predictions"]["ci"])
                    ci =  ci_list[i]
                    ci = [format_value(ci[0]), format_value(ci[1])]
                    if metric_map[benchmark] == "acc1":
                        ci = [ci[1], ci[0]]


                    
                    x = format_value(x)
                    y = format_value(y)
                    y_pred = format_value(y_pred)
                    y_pred = f"{y_pred} ({ci[0]}, {ci[1]})"
                    if extra_cols is not None:
                        if all([len(extra_data[col]) for col in extra_cols]) > 0:
                            extra_data_str = " & ".join(
                                format_value(extra_data[col][i]) for col in extra_cols
                            )
                        else:
                            extra_data_str = " & ".join(
                                "-" for col in extra_cols
                            )
                        
                        f.write(
                            f"{model_type.upper()} & {extra_data_str} & {x} & {y} & {y_pred} \\\\\n"
                        )
                    else:
                        f.write(
                            f"{model_type.upper()} & {x} & {y} & {y_pred} \\\\\n"
                        )
            else:
                xs = []
                y = result["predictions"][y_col]
                ci = result["predictions"]["ci"]
                if metric_map[benchmark] == "acc1":
                    ci = [ci[1], ci[0]]
                y_pred = result["predictions"][f"{y_col}_predicted"]
                mse = compute_mse(y, y_pred)
                mse = format_value(mse)
                for x_col_i in x_col:
                    xs.append(result['predictions'][x_col_i])
                xs.append(y)
                xs.append(y_pred)
                xs = np.array(xs).T
                for i in range(len(xs)):
                    x = xs[i]
                    x = [format_value(xi) for xi in x]
                    if extra_cols is not None:
                        extra_data_str = " & ".join(
                            format_value(extra_data[col][i]) for col in extra_cols
                        )
                        f.write(
                            f"{model_type.upper()} & {extra_data_str} & " + " & ".join(x) + f" \\\\\n"
                        )
                    else:
                        f.write(
                            f"{model_type.upper()} & " + " & ".join(x) + f" \\\\\n"
                        )

            y_true_list, y_pred_list = result["predictions"][y_col], result["predictions"][f"{y_col}_predicted"]
            if len(result["predictions"][y_col]) != 0:
                mse = compute_mse(y_true_list, y_pred_list)
                mse = f"{np.sqrt(mse):.2e}"
            else:
                mse = "-"
            f.write(
                f"{model_type.upper()}  & MSE: {mse} \\\\\n"
            )
            f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Model predictions}\n")
        f.write("\\label{tab:predictions}\n")
        f.write("\\end{table}\n")


def save_figure(
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
):
    """
    Save the figure to a file
    """

    if not os.path.exists(save_config.save_dir):
        os.makedirs(save_config.save_dir)
    if not os.path.exists(save_config.save_dir_figures):
        os.makedirs(save_config.save_dir_figures)
    if not os.path.exists(save_config.save_dir_tables):
        os.makedirs(save_config.save_dir_tables)

    if save_config.save_pdf:
        filename = save_config.save_filename_template.format(
            benchmark=benchmark,
            method=method,
            func_type=func_type,
            large_flops_mapping=str(large_flops_mapping),
            y_col=y_col,
            ext="pdf",
            models="_".join(models),
        )
        filename = filename.replace(" ", "_").replace(":", "_").replace(",", "_").replace("+", "_")
        print(f"Saving figure to {os.path.join(save_config.save_dir_figures, f'{filename}')}")
        figure.savefig(
            os.path.join(save_config.save_dir_figures, filename), bbox_inches="tight", dpi=300
        )

    if save_config.save_latex:
        filename = save_config.save_filename_template.format(
            benchmark=benchmark,
            method=method,
            func_type=func_type,
            large_flops_mapping=str(large_flops_mapping),
            y_col=y_col,
            ext="tex",
            models="_".join(models),
        )
        print(f"Saving table to {os.path.join(save_config.save_dir_tables, f'{filename}')}")
        # save latex table with coefficients (popt_mean) for each model
        save_latex(bootstrap_results, os.path.join(save_config.save_dir_tables, filename))
        perdictions_filename = save_config.save_filename_template.format(
            benchmark=benchmark,
            method=method,
            func_type=func_type,
            large_flops_mapping=str(large_flops_mapping),
            y_col=y_col,
            ext="_predictions.tex",
            models="_".join(models),
        )
        print(
            f"Saving predictions to {os.path.join(save_config.save_dir_tables, f'{perdictions_filename}')}"
        )
        save_predictions(
            bootstrap_results,
            os.path.join(save_config.save_dir_tables, perdictions_filename),
            x_col,
            y_col,
            extra_cols=save_config.save_extra_cols,
            method=method,
            func_type=func_type,
            benchmark=benchmark
        )
