import numpy as np
from scipy.special import logsumexp


def power_func(x, a, b, c):
    return a * np.power(x + b, c)


def simple_power_func(x, a, b):
    return a * np.power(x, b)

def simple_power_correction(x, a, b, k):
    return (a * np.power(x, b)) + k * x

def power_law_saturation_both(x, a, b, c, d):
    return a * np.power(x + b, c) + d

def simple_line(x, a):
    return a


def line(x, a, b):
    return a * x + b

def line_correction(x, a, b, k):
    return a + np.log(np.exp(k * x) + b)

def line_correction_both(x, a, b, c, k):
    return c + np.log(a * np.exp(k * x) + b)



func_map = {
    "line": line, 
    "simple_line": simple_line,
    "line_correction": line_correction,
    "line_correction_both": line_correction_both,
    "saturation": power_func, 
    "saturation_both": power_law_saturation_both,
    "simple": simple_power_func,
    }
