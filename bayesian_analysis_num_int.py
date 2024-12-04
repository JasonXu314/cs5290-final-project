import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import norm, poisson

# Load the Pima Indians Diabetes dataset
data = pd.read_csv('diabetes.csv')

# 1. Two-sample (one-tailed) test using Numeric Integration
def two_sample_num_int(x1, x2):
    n1, n2 = len(x1), len(x2)
    mean1, mean2 = np.mean(x1), np.mean(x2)
    var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_sd = np.sqrt(pooled_var * (1/n1 + 1/n2))
    posterior_mean = mean1 - mean2
    posterior_var = pooled_sd**2

    res, err = quad(lambda p: norm.pdf(p, loc=posterior_mean, scale=np.sqrt(posterior_var)), -np.inf, 0)
    return res

# Example usage
x1 = data[data['Outcome'] == 0]['Glucose']
x2 = data[data['Outcome'] == 1]['Glucose']
two_sample_result = two_sample_num_int(x1, x2)
print("Two-sample test (Numerical Integration):", two_sample_result)
