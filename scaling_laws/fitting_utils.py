import numpy as np
from scipy.optimize import curve_fit, minimize
from sklearn.utils import resample
from .utils import huber_loss_objective
from .configs import ScalingFunction
from itertools import product
import multiprocessing as mp
from tqdm import tqdm

def parabola_func(x, a1, a2, b1=0, b2=0, c=0):
    """
    Parabola function x is (N,2)
    """
    x1 = x[:, 0]
    x2 = x[:, 1]
    return a1 * (x1 + b1)**2 + a2 * (x2 + b2)**2 + c

def parabola_func1d(x, a1, b1):
    """
    Parabola function x is (N,)
    """
    return a1 * (x + b1)**2 

def fit_parabola(
    x: np.ndarray,
    y: np.ndarray,
):
    """
    Fit a parabola to the data
    """
    # Fit the parabola to the data
    popt, pcov = curve_fit(parabola_func1d, x, y, p0=[1, 1], maxfev=100000)
    # Get the parameters
    return popt, pcov



def fit_curve_fit(x, y, func: ScalingFunction, initial_params=None, method="curve_fit"):
    """
    Fit the data using curve_fit

    """
    num_params = func.num_params
    if func.func_type == "saturation":
        bounds = [(0, 0, -np.inf), (np.inf, np.inf, 0)]
        initial_params = [1e1, 1e1, -0.1]
    elif func.func_type == "saturation_both":
        bounds = [(0, np.inf), (0, np.inf), (-np.inf, 0), (0, np.inf)]
        bounds = np.array(bounds)
        bounds = bounds.T.tolist()
    else:
        bounds = [[-np.inf] * func.num_params, [np.inf] * func.num_params]

    popt, pcov = curve_fit(
        func, 
        x, 
        y, 
        maxfev=1000000, 
        bounds=bounds, 
        p0=initial_params,
        )

    return OptimizeResult(popt, np.mean(func(x, *popt)), pcov)

class OptimizeResult:
    def __init__(self, x, fun, pcov=None):
        self.x = x
        self.fun = fun
        self.pcov = pcov

    def __repr__(self):
        return f"OptimizeResult(x={self.x})"


def fit_huber(x, y, func: ScalingFunction, initial_params=None, delta=1e-3):
    """
    Fit the data using Huber loss

    """

    if initial_params is None:
        initial_params = np.ones(func.num_params) * 0.5
    # Define the function to fit
    num_params = func.num_params
    result = minimize(
        huber_loss_objective,
        x0=initial_params,
        method="L-BFGS-B",
        args=(x, y, func, delta),
    )

    return OptimizeResult(result.x, result.fun)

def fit_func(args):
    initial_params, x, y, func, method = args
    if method == "curve_fit":
        return fit_curve_fit(x, y, func, initial_params)
    elif method == "huber":
        return fit_huber(x, y, func, initial_params)
    else:
        raise ValueError(f"Unknown method: {method}")

def fit_parallel(
    x,
    y,
    n_bootstrap,
    func: ScalingFunction,
    initial_params=None,
    method="curve_fit",
    n_jobs=1,
):
    
    """
    Fit the data using parallel processing

    """
    # Create a list to store the results
    results = []
    initial_params = list(initial_params)

    tasks = [
        (initial_params[i], x, y, func, method) for i in range(len(initial_params))
    ]

    
    # Create a pool of workers
    with mp.Pool(processes=n_jobs) as pool:
        # Use the pool to map the function to the initial parameters
        results = list(
            tqdm(
                pool.imap(
                    fit_func,
                    tasks,
                    chunksize=1,
                ),
                total=len(tasks),
                desc="Fitting",

            )
        )
    return results


def fit_with_initial_params_grid(
    x,
    y,
    n_bootstrap,
    func: ScalingFunction,
    initial_params_grid=None,
    method="curve_fit",
):
    """
    Fit the data using a grid of initial parameters

    """
    # Create a list to store the results
    results = []
    popts = []
    min_mse = np.inf
    best_params = None
    best_pcov = None
    
    # initial_params_grid is a list of lists with dimension (num_guesses, num_params)
    # Create a grid of initial parameters

    if initial_params_grid is not None:

        initial_params_grid = product(*initial_params_grid)



    min_loss = np.inf
    # Iterate over the grid of initial parameters

    results = fit_parallel(
        x,
        y,
        n_bootstrap,
        func,
        initial_params=initial_params_grid,
        method=method,
        n_jobs=24,
    )



    for i, res in enumerate(results):
        # Predict the values for the original x values
        popt = res.x
        pred = func(x, *popt)
        mse = np.mean((y - pred) ** 2)
        if mse < min_loss:
            min_loss = mse
            best_params = popt
            best_pcov = res.pcov
      

    return best_params, best_pcov



def fit_bootstrap(
    pareto_x,
    pareto_y,
    x_all,
    func: ScalingFunction,
    n_bootstrap=100,
    method="curve_fit",
):
    """
    Fit the data using bootstrapping

    """
    # Create a list to store the results
    results = []
    popts = []

    initial_params = func.initial_guess_param_grid()
    if initial_params is not None:

        print(
            f"Fitting with {len(initial_params[0])} initial parameter guesses"
        )
        
    pcov = None
    # Create a bootstrapped sample
    for i in range(n_bootstrap):
        # Resample the data with replacement
        if n_bootstrap == 1:
            x_resampled = pareto_x
            y_resampled = pareto_y
        else:
            x_resampled, y_resampled = resample(pareto_x, pareto_y)
            
        if func.initial_guess_param_grid() is not None:
            popt, pcov = fit_with_initial_params_grid(
                x_resampled, y_resampled, n_bootstrap, func, initial_params, method
            )
        elif method == "curve_fit":
            result = fit_curve_fit(x_resampled, y_resampled, func, method=method)
            popt = result.x
            if n_bootstrap == 1:
                pcov = result.pcov


       
        elif method == "huber":
            if func.initial_guess_param_grid() is not None:
                popt = fit_with_initial_params_grid(
                    x_resampled, y_resampled, n_bootstrap, func, initial_params, method
                )
            else:
                popt = fit_huber(x_resampled, y_resampled, func).x

        popts.append(popt)
        print(popt)
        # Predict the values for the original x values
        y_pred = func(x_all, *popt)
        results.append(y_pred)

    return results, popts, pcov