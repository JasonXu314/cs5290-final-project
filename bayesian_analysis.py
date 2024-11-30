import numpy as np
import pandas as pd
import pymc3 as pm
from scipy import stats

# Read the data
data = pd.read_csv('diabetes.csv')

# 1. Two-sample Bayesian test
def two_sample_test(data):
    with pm.Model() as model:
        # Priors
        mu_0 = pm.Normal('mu_0', mu=0, sd=10)
        mu_1 = pm.Normal('mu_1', mu=0, sd=10)
        sigma = pm.HalfNormal('sigma', sd=10)
        
        # Likelihood
        y_0 = pm.Normal('y_0', mu=mu_0, sd=sigma, observed=data[data['Outcome']==0]['Glucose'])
        y_1 = pm.Normal('y_1', mu=mu_1, sd=sigma, observed=data[data['Outcome']==1]['Glucose'])
        
        # Compute difference
        diff = pm.Deterministic('diff', mu_1 - mu_0)
        
        # Sample
        trace = pm.sample(2000, return_inferencedata=False)
    
    return np.mean(trace['diff'] > 0)

# 2. Bayesian Linear Regression
def linear_regression(data, response='BloodPressure'):
    with pm.Model() as model:
        # Priors
        alpha = pm.Normal('alpha', mu=0, sd=10)
        beta = pm.Normal('beta', mu=0, sd=10)
        sigma = pm.HalfNormal('sigma', sd=10)
        
        # Likelihood
        mu = alpha + beta * data['Glucose']
        y = pm.Normal('y', mu=mu, sd=sigma, observed=data[response])
        
        # Sample
        trace = pm.sample(2000, return_inferencedata=False)
    
    return np.percentile(trace['beta'], [2.5, 97.5])

# 3. Variance Analysis
def variance_analysis(data):
    with pm.Model() as model:
        # Split data by BMI median
        bmi_groups = data['BMI'] > data['BMI'].median()
        
        # Priors
        mu = pm.Normal('mu', mu=0, sd=10)
        sigma_1 = pm.HalfNormal('sigma_1', sd=10)
        sigma_2 = pm.HalfNormal('sigma_2', sd=10)
        
        # Likelihoods
        y1 = pm.Normal('y1', mu=mu, sd=sigma_1, observed=data[bmi_groups]['Glucose'])
        y2 = pm.Normal('y2', mu=mu, sd=sigma_2, observed=data[~bmi_groups]['Glucose'])
        
        # Sample
        trace = pm.sample(2000, return_inferencedata=False)
    
    return np.mean(trace['sigma_1'] > trace['sigma_2'])

# 4. Multiple Regression
def multiple_regression(data):
    with pm.Model() as model:
        # Priors
        alpha = pm.Normal('alpha', mu=0, sd=10)
        beta1 = pm.Normal('beta1', mu=0, sd=10)
        beta2 = pm.Normal('beta2', mu=0, sd=10)
        sigma = pm.HalfNormal('sigma', sd=10)
        
        # Likelihood
        mu = alpha + beta1 * data['Glucose'] + beta2 * data['BMI']
        y = pm.Normal('y', mu=mu, sd=sigma, observed=data['BloodPressure'])
        
        # Sample
        trace = pm.sample(2000, return_inferencedata=False)
    
    return np.percentile(trace['beta1'], [2.5, 97.5]), np.percentile(trace['beta2'], [2.5, 97.5])

# 5. Logistic Regression
def logistic_regression(data):
    with pm.Model() as model:
        # Priors
        alpha = pm.Normal('alpha', mu=0, sd=10)
        beta = pm.Normal('beta', mu=0, sd=10)
        
        # Likelihood
        p = pm.math.sigmoid(alpha + beta * data['Glucose'])
        y = pm.Bernoulli('y', p=p, observed=data['Outcome'])
        
        # Sample
        trace = pm.sample(2000, return_inferencedata=False)
    
    return np.percentile(trace['beta'], [2.5, 97.5])

# 6. Poisson Regression
def poisson_regression(data):
    with pm.Model() as model:
        # Priors
        alpha = pm.Normal('alpha', mu=0, sd=10)
        beta = pm.Normal('beta', mu=0, sd=10)
        
        # Likelihood
        mu = pm.math.exp(alpha + beta * data['Glucose'])
        y = pm.Poisson('y', mu=mu, observed=data['Pregnancies'])
        
        # Sample
        trace = pm.sample(2000, return_inferencedata=False)
    
    return np.percentile(trace['beta'], [2.5, 97.5])

# 7. Adjustment for Confounding
def confounding_adjustment(data):
    with pm.Model() as model:
        # Priors
        alpha = pm.Normal('alpha', mu=0, sd=10)
        beta_glucose = pm.Normal('beta_glucose', mu=0, sd=10)
        beta_bp = pm.Normal('beta_bp', mu=0, sd=10)
        sigma = pm.HalfNormal('sigma', sd=10)
        
        # Likelihood
        mu = alpha + beta_glucose * data['Glucose'] + beta_bp * data['BloodPressure']
        y = pm.Normal('y', mu=mu, sd=sigma, observed=data['Insulin'])
        
        # Sample
        trace = pm.sample(2000, return_inferencedata=False)
    
    return np.percentile(trace['beta_glucose'], [2.5, 97.5])

# Run analyses
results = {
    'Two-sample test': two_sample_test(data),
    'Linear regression': linear_regression(data),
    'Variance analysis': variance_analysis(data),
    'Multiple regression': multiple_regression(data),
    'Logistic regression': logistic_regression(data),
    'Poisson regression': poisson_regression(data),
    'Confounding adjustment': confounding_adjustment(data)
}