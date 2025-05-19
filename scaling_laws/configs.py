from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
from .constants import metric_map
from .functions import func_map
from decimal import Decimal
import os


@dataclass
class FigureConfig:
    """
    Configuration for the figure.
    """

    hue: str = None
    show_ci: bool = True
    fig_width: int = 8
    fig_height: int = 6
    x_label: str = None
    y_label: str = None
    max_x: float = None


@dataclass
class ScalingLawConfig:
    """
    Configuration for the scaling law.
    """

    benchmark: str
    metric_map: Dict[str, str] = field(default_factory=lambda: metric_map)
    large_flops_mapping: Dict[str, float] = None
    n_bootstraps: int = 100
    lr_scheduler: str = "cosine"
    y_col: str = None
    x_col: List[str]|str = "gflops_total"
    x_col_pareto: str = "gflops_total"
    func_type: str = "saturation"
    hue: str = None
    show_ci: bool = True
    create_figure: bool = True
    method: str = "curve_fit" # curve_fit, huber, isoflop_curves
    models: List[str] = field(default_factory=list)
    pretrain_datasets: List[str] = field(default_factory=list)
    min_flops: float = None
    pareto_method: str = "pareto_naive" # pareto_naive, convexhull
    approach3_params_path: str = None 
    max_samples_seen: int = None
    plot_scaling_curve_by_mode: bool = False
    predict_for_vals: list = None
    eval_type: str = None


    def __post_init__(self):
        self.y_col_pareto: str = metric_map[self.benchmark]


@dataclass
class SaveConfig:
    """
    Configuration for saving the results.
    """

    save_dir: str

    save_latex: bool = False
    save_json: bool = True
    save_csv: bool = True
    save_pdf: bool = True
    save_filename_template: str = (
        "{benchmark}_{method}_{func_type}_{large_flops_mapping}_{y_col}_{models}.{ext}"
    )
    save_extra_cols: List[str] = field(default_factory=list)
    save_pareto_to_df: bool = None

    def __init__(self, save_dir, **kwargs):
        self.save_dir = save_dir
        self.save_dir_figures = os.path.join(save_dir, "figures")
        self.save_dir_tables = os.path.join(save_dir, "tables")
        self.save_pareto_to_df_path = os.path.join(save_dir, "pareto")
        os.makedirs(self.save_dir_figures, exist_ok=True)
        os.makedirs(self.save_dir_tables, exist_ok=True)
        print(self.save_pareto_to_df)
        
        for key, value in kwargs.items():
            setattr(self, key, value)
        if self.save_pareto_to_df:
            print(
                f"Saving pareto to df at {self.save_pareto_to_df_path}"
            )
            os.makedirs(self.save_pareto_to_df_path, exist_ok=True)


@dataclass
class LargeFlopsMapping:
    """
    Mapping for large flops.
    """

    mammut: float = np.inf
    coca: float = np.inf
    clip: float = np.inf

    def __post_init__(self):
        self.mapping = {"mammut": self.mammut, "coca": self.coca, "clip": self.clip}

    def parse_str(self, str_mapping):
        """
        Parse the string mapping to a dictionary.
        """
        mapping = {}
        for item in str_mapping.split(","):
            key, value = item.split(":")
            mapping[key] = float(value)
            setattr(self, key, float(value))
        self.mapping = mapping

    def __getitem__(self, key):
        """
        Get the value for a given key.
        """
        return self.mapping.get(key, np.inf)
    
    def __setitem__(self, key, value):
        """
        Set the value for a given key.
        """
        self.mapping[key] = value
        setattr(self, key, value)

    def __str__(self):
        """
        String representation of the large flops mapping. e.g. coca:1e2,mammut:1e4
        """
        return "_".join(
            [f"{key}:{Decimal(value):.2e}" for key, value in self.mapping.items()]
        )


class ScalingFunction:
    """
    Configuration for the scaling function.
    """

    func_type: str = "saturation"
    params: List[float] = field(default_factory=list)
    param_names: List[str] = field(default_factory=list)

    def __init__(self, func_type="saturation"):
        self.func_type = func_type
        self.param_names = self.get_param_names(self.func_type)
        self.params = [0] * len(self.param_names)
        self.func = func_map[self.func_type]
        self.num_params = len(self.param_names)

    def coeffs(self, params):
        """
        Get the coefficients for the scaling function.
        """
        if self.func_type == "saturation":
            return [params[0]]
        elif self.func_type == "saturation_both":
            return [params[0], params[1]]
        elif self.func_type in ["line", "simple"]:
            return [params[0], params[1]]
        elif self.func_type == "power_func_multi":
            return [params[0], params[1]]
        elif self.func_type == "power_func_multi3d":
            return [params[0], params[1], params[2]]
        elif self.func_type == "power_func_multi4d":
            return [params[0], params[1], params[2], params[3]]
        elif self.func_type == "power_func_multi3d_no_gamma":
            return [params[0], params[1], params[2]]
        else:
            raise ValueError(f"Unknown function type: {self.func_type}")
        
    def exponents(self, params):
        """
        Get the exponents for the scaling function.
        """
        if self.func_type == "saturation":
            return [params[1]]
        elif self.func_type == "saturation_both":
            return [params[2], params[3]]
        elif self.func_type in ["line", "simple"]:
            return [params[1]]
        elif self.func_type == "power_func_multi":
            return [params[2], params[3]]
        elif self.func_type == "power_func_multi3d":
            return [params[3], params[4], params[5]]
        elif self.func_type == "power_func_multi4d":
            return [params[4], params[5], params[6], params[7]]
        elif self.func_type == "power_func_multi3d_no_gamma":
            return [params[3], params[4]]
        else:
            raise ValueError(f"Unknown function type: {self.func_type}")
    
    def free_param(self, params):
        """
        Get the free parameters for the scaling function.
        """
        if self.func_type == "saturation":
            return params[1]
        elif self.func_type == "saturation_both":
            return params[3]
        elif self.func_type in ["line", "simple"]:
            return None

    def initial_guess_param_grid(self):
        """
        Get the initial guess for the parameters.
        """
        if self.func_type in ["saturation", "power_func_multi_saturation", "power_func1d", "simple_line"]:
            return None
        if self.func_type == "saturation_both":
            return [np.arange(0, 100, 5), np.arange(0, 100, 5), np.arange(-0.2, 0, 0.05), np.arange(0, 0.001, 0.0005)]
            # return [a1, a2, b1, b2]
        elif self.func_type in ["line", "simple", "line_correction", "line_correction_both"]:
            return None
        elif self.func_type in  ["power_func_multi", "power_func_multi_log"]:
            return [np.exp(np.arange(0, 30, 5)), np.exp(np.arange(0, 30, 5)), np.arange(-2.5, 0, 0.3), np.arange(-2.5, 0, 0.3), np.arange(0, 1e-3, 0.5e-3)]
        else:
            raise ValueError(f"Unknown function type: {self.func_type}")

    def jacobian(self, params, x):
        """
        Get the Jacobian matrix for the scaling function.
        """
        if self.func_type == "saturation":
            J = np.zeros((len(x), 3))
            J[:, 0] = np.power(x + params[1], params[2])
            J[:, 1] = params[0] * params[2] * np.power(x + params[1], params[2] - 1)
            J[:, 2] = params[0] * np.log(x + params[1]) * np.power(x + params[1], params[2])
            return J
        elif self.func_type == "saturation_both":
            J = np.zeros((len(x), 4))
            J[:, 0] = np.power(x + params[1], params[2])
            J[:, 1] = params[0] * params[2] * np.power(x + params[1], params[2] - 1)
            J[:, 2] = params[0] * np.log(x + params[1]) * np.power(x + params[1], params[2])
            J[:, 3] = 1
            return J
        elif self.func_type == "simple_line":
            J = np.zeros((len(x), 1))
            J[:, 0] = np.ones_like(x)
            return J
        elif self.func_type in ["line"]:  
            J = np.zeros((len(x), 2))
            J[:, 0] = x
            J[:, 1] = np.ones_like(x)
            return J
        elif self.func_type in ["simple"]:
            J = np.zeros((len(x), 2))
            J[:, 0] = np.power(x, params[1])
            J[:, 1] = params[0] * np.log(x) * np.power(x, params[1])
            return J
        else:
            return None

    def get_param_names(self, func_type):
        """
        Get the parameter names for the scaling function.
        """
        if func_type in ["saturation"]:
            return ["a", "b", "c"]
        elif func_type == "simple_line":
            return ["a"]
        elif func_type == "line_correction":
            return ["a", "b", "k"]
        elif func_type == "line_correction_both":
            return ["a", "b", "c", "k"]
        elif func_type in ["saturation_both", "power_func1d"]:
            return ["a", "b", "c", "d"]
        elif func_type in ["line", "simple"]:
            return ["a", "b"]
        else:
            raise ValueError(f"Unknown function type: {func_type}")

    def to_dict(self):
        """
        Convert the scaling function to a dictionary.
        """
        return {
            "func_type": self.func_type,
            "params": self.params,
            "param_names": self.param_names,
        }
    
    def param_dict(self):
        """
        Convert the scaling function parameters to a dictionary.
        """
        return {name: param for name, param in zip(self.param_names, self.params)}

    def param_to_display_dict(self):
        """
        Convert the scaling function parameters to a dictionary for display.
        """
        param_dict = self.param_dict()
        if self.func_type in ["saturation", "saturation_both"]:
            param_dict["b"] = np.log(param_dict["b"])
        if self.func_type == "line":
            param_dict["b"] = np.exp(param_dict["b"])
        if self.func_type == "power_func1d":
            param_dict["a"] = np.exp(param_dict["a"]/ param_dict["b"])
        if self.func_type == "simple_line":
            param_dict["a"] = np.exp(param_dict["a"])
        return {name: param for name, param in param_dict.items()}

    def func_str(self):
        """
        Get the string representation of the scaling function.
        """
        if self.func_type == "saturation":
            return "${a:.2f} * ({x} + {b:.2f})^{{{c:.2f}}}$"
        elif self.func_type == "saturation_both":
            b = np.log(self.param_dict()["b"])
            return "${a:.2f} * ({x} + exp({{{b:.2f}}}))^{{{c:.3}}} + {d:.2f}$"
        elif self.func_type == "simple_line":
            return "${a:.2f} * {x}$"
        elif self.func_type in ["line"]:
            return "${b:.2f} * {x}^{{{a:.2f}}}$"
        elif self.func_type == "simple":
            return "${a:.2f} * {x}^{{{b:.2f}}}$"
        elif self.func_type == "power_func1d":
            return "${a:.2f} * ({x}^{d:.2f} + {b:.2f})^{{{c:.2f}}}$"
        elif self.func_type == "line_correction":
            return "${a:.2f} * ({x}^{{{k:.2f}}} + {b:.2f}) $"
        elif self.func_type == "line_correction_both":
            return "${a:.2f} * {x} + {b:.2f} * \exp({k:.2f} * {x}) + {c:.2f}$"
        
    def __repr__(self, x="x"):
        """
        String representation of the scaling function.
        """
        func_str = self.func_str()
        func_str = func_str.replace("{x}", x)
        func_str = func_str.replace("{x1}", x)
        func_str = func_str.replace("{x2}", x)
        func_str = func_str.replace("{x3}", x)
        func_str = func_str.replace("{x4}", x)
        params_to_display_dict = self.param_to_display_dict()
        params_to_display = [
            params_to_display_dict.get(name, name) for name in self.param_names
        ]
        return f"{func_str.format(**dict(zip(self.param_names, params_to_display)))}"

    def __call__(self, x, a, b=None, c=None, d=None):
    # def __call__(self, x, a, b=None):
        """
        Call the scaling function.
        """
        if self.func_type == "simple_line":
            return self.func(x, a)
        if self.func_type in ["simple", "line"]:
            return self.func(x, a, b)
        if self.func_type == "saturation_both":
            return self.func(x, a, b, c, d)
        return self.func(x, a, b, c)
