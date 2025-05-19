import numpy as np
from scipy.special import logsumexp, huber
from scipy.spatial import ConvexHull
import alphashape


def log_pred(
    exps, coefs, e, data
):
    """Predict the log-loss from the power law given the parameter values for exponents and coefficients, and irreducible error e
    data: 1 or 2D array"""


    arr_1 = np.log(coefs) + (exps * np.log(data))
    arr_2 = np.full((arr_1.shape[0], 1), np.log(e))
    cat = np.concatenate((arr_1, arr_2), axis=1)
    return logsumexp(cat, axis=-1)

def pareto_frontier_bins(arr, num_bins=None, n_to_keep=1):
    """
    Compute the pareto frontier for the given array of values.
    1. Bin the x-axis values using a geometric scale.
    2. For each bin, find the values with minimum error.
    3. Sort the values in the bin by x-axis value.
    4. Keep the first n_to_keep values in the bin.
    Parameters
    ----------
    arr : np.ndarray
        The array of values to compute the pareto frontier for.
        The first column is the x-axis values and the second column is the y-axis values.
    num_bins : int
        The number of bins to use for the pareto frontier.
    Returns
    -------
    np.ndarray
        The pareto frontier for the given array of values.
    """
    x_val = arr[:, 0]
    y_val = arr[:, 1]
    if num_bins is None:
        flop_val = 1500
        num_bins = int(np.log(x_val.max() / flop_val) / np.log(2))
    bins = np.geomspace(x_val.min(), x_val.max(), num_bins)
    pareto = []

    for i in range(len(bins) - 1):
        mask = (x_val >= bins[i]) & (x_val < bins[i + 1])

        x = x_val[mask]
        y = y_val[mask]
        idx = np.argsort(y)
        x_sorted = x[idx][:n_to_keep]
        y_sorted = y[idx][:n_to_keep]
        idx = np.argsort(x_sorted)
        x_sorted = x_sorted[idx]
        y_sorted = y_sorted[idx]
        pareto.extend(list(zip(x_sorted, y_sorted)))


    pareto = np.array(pareto)

    new_pareto = []
    min_y = pareto[0][1]
    new_pareto.append(pareto[0])

    for i in range(len(pareto)):

        if pareto[i][1] < min_y:
            new_pareto.append(pareto[i])
            min_y = pareto[i][1]

    x_max = x_val.max()
    x_max_index = np.where(x_val == x_max)[0][0]
    y_max = y_val[x_max_index]
    if y_max < min_y:
        new_pareto.append((x_max, y_max))
    
    pareto = np.array(new_pareto)
    pareto_mask = np.zeros(len(arr), dtype=bool)
    for i in range(len(arr)):
        if arr[i][0] in pareto[:, 0] and arr[i][1] in pareto[:, 1]:
            pareto_mask[i] = True
    return pareto_mask


def lower_convexhull(x, y):
    vantage_point = [
        x[0],
        y[0],
    ]

    points = np.log(
        np.vstack(
            (
                vantage_point,  # Lower vantage point for boundary
                np.array(
                    [x, y]
                ).T,  # Actual points
            )
        )
    )

    alpha = 0.1

    concave_hull = alphashape.alphashape(points, alpha).exterior.xy
    concave_hull = np.array(concave_hull)
    lower_hull_indicies = paretoset_naive(
        np.array([concave_hull[0], concave_hull[1]]).T
    )

    concave_hull = concave_hull[0][lower_hull_indicies], concave_hull[1][
        lower_hull_indicies
    ]

    valid_hull_indices_mask = np.zeros(len(x), dtype=bool)
    for i in range(len(x)):
        if (
            np.min(
                np.sqrt(
                    (concave_hull[0] - np.log(x[i])) ** 2
                    + (concave_hull[1] - np.log(y[i])) ** 2
                )
            )
            < 1e-3
        ):
            valid_hull_indices_mask[i] = True

    return valid_hull_indices_mask

def paretoset_naive(costs, distinct=True):
    """Naive implementation.

    Parameters
    ----------
    costs : (np.ndarray) Array of shape (n_costs, n_objectives).

    Returns
    -------
    mask : (np.ndarray) Boolean array indicating the paretoset.

    """
    n_costs, _ = costs.shape

    # Assume all points/costs are inefficient
    is_efficient = np.zeros(n_costs, dtype=np.bool_)

    for i in range(n_costs):
        this_cost = costs[i, :]

        # Here `NOT(ANY(a_i > b_i))` is equal to ALL(a_i <= b_i), but faster.
        at_least_as_good = np.logical_not(np.any(costs > this_cost, axis=1))
        any_better = np.any(costs <= this_cost, axis=1)

        dominated_by = np.logical_and(at_least_as_good, any_better)

        # If we're looking for distinct points and it's already in the
        # pareto set, disregard this value.
        if distinct and np.any(is_efficient):
            if np.any(np.all(costs[is_efficient] == this_cost, axis=1)):
                continue

        if not (np.any(dominated_by[:i]) or np.any(dominated_by[i + 1 :])):
            is_efficient[i] = True

    return is_efficient

def huber_loss(delta, residuals):
    abs_res = np.abs(residuals)
    quad = 0.5 * residuals**2
    linear = delta * (abs_res - 0.5 * delta)
    return np.where(
        abs_res <= delta,
        quad,
        linear,
    )



def huber_loss_objective(params, x, y, func, delta=2e-2):
    # predictions = func(x, *params)
    exps = func.exponents(params)
    coefs = func.coeffs(params)
    e = func.free_param(params)
    predictions = log_pred(
        data=x,
        exps=exps,
        coefs=coefs,
        e=e,
    )

    return np.sum(
        huber_loss(delta=delta, residuals=np.log(y) - predictions)
    )


def get_model_type(model):
    if "siglip" in model.lower():
        return "siglip"
    if "cap" in model.lower():
        return "cap"
    if "mammut" in model:
        return "mammut"
    if "coca" in model:
        return "coca"
    return "clip"
