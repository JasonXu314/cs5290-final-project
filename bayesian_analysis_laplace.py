import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import minimize

# Load the Pima Indians Diabetes dataset
data = pd.read_csv('diabetes.csv')

# 1. Two-sample test using Laplace approximation
def two_sample_laplace(x1, x2):
    n1, n2 = len(x1), len(x2)
    mean1, mean2 = np.mean(x1), np.mean(x2)
    var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_sd = np.sqrt(pooled_var * (1/n1 + 1/n2))
    posterior_mean = mean1 - mean2
    posterior_var = pooled_sd**2
    return norm.cdf(0, loc=posterior_mean, scale=np.sqrt(posterior_var))

# Example usage
x1 = data[data['Outcome'] == 0]['Glucose']
x2 = data[data['Outcome'] == 1]['Glucose']
two_sample_result = two_sample_laplace(x1, x2)
print("Two-sample test (Laplace):", two_sample_result)

# 2. Linear Regression with Laplace approximation
def linear_regression_laplace(X, y):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

X = data[['Glucose', 'BMI', 'Age']]
y = data['Insulin']
linear_model = linear_regression_laplace(X, y)
print(linear_model.summary())

# 3. Analysis of Variance (ANOVA)
def anova_laplace(df, formula):
    model = smf.ols(formula, data=df).fit()
    anova_result = sm.stats.anova_lm(model, typ=2)
    return anova_result

anova_result = anova_laplace(data, 'BMI ~ C(Outcome)')
print("ANOVA result:\n", anova_result)

# 4. Multiple Regression
multiple_reg_model = linear_regression_laplace(X, y)
print("Multiple regression:\n", multiple_reg_model.summary())

# 5. Linear Models
linear_model_full = smf.ols('Insulin ~ Glucose + BMI + Age', data=data).fit()
print(linear_model_full.summary())

# 6. Logistic Regression
def logistic_regression_laplace(X, y):
    X = sm.add_constant(X)
    model = sm.Logit(y, X).fit(disp=0)
    return model

X = data[['Glucose', 'BMI', 'Age']]
y = data['Outcome']
logistic_model = logistic_regression_laplace(X, y)
print("Logistic regression:\n", logistic_model.summary())

# 7. Poisson Regression
def poisson_regression_laplace(X, y):
    X = sm.add_constant(X)
    model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    return model

poisson_model = poisson_regression_laplace(X, data['Pregnancies'])
print("Poisson regression:\n", poisson_model.summary())

# 8. Adjustment for Confounding
adjusted_model = smf.ols('BMI ~ Glucose + Age', data=data).fit()
print("Adjustment for confounding:\n", adjusted_model.summary())
