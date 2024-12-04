import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from scipy import stats

def preprocess_data(data):
    """Clean physiologically impossible values and prepare data."""
    clean = data.copy()
    # Remove impossible glucose values
    clean = clean[clean['Glucose'] > 30]
    # Remove impossible BMI values
    clean = clean[clean['BMI'] > 13]
    # Remove impossible blood pressure values
    clean = clean[clean['BloodPressure'] > 40]
    clean = clean[clean['BloodPressure'] < 250]
    # Remove impossible insulin values
    clean = clean[clean['Insulin'] < 1000]
    
    return clean

def two_sample_test(data):
    """Robust two-sample comparison of glucose levels between outcome groups."""
    with pm.Model() as model:
        pooled_mean = data['Glucose'].mean()
        pooled_std = data['Glucose'].std()
        
        mu_0 = pm.Normal('mu_0', mu=pooled_mean, sigma=pooled_std)
        mu_1 = pm.Normal('mu_1', mu=pooled_mean, sigma=pooled_std)
        
        sigma = pm.HalfNormal('sigma', sigma=pooled_std)
        
        nu = pm.Exponential('nu', 1/10)  
        
        y_0 = pm.StudentT('y_0', nu=nu, mu=mu_0, sigma=sigma, 
                         observed=data[data['Outcome']==0]['Glucose'])
        y_1 = pm.StudentT('y_1', nu=nu, mu=mu_1, sigma=sigma, 
                         observed=data[data['Outcome']==1]['Glucose'])
        
        diff = pm.Deterministic('diff', mu_1 - mu_0)
        
        # Increased samples for better convergence
        idata = pm.sample(5000, tune=2000, return_inferencedata=True)
    
    diff_samples = idata.posterior['diff'].values.flatten()
    hdi_bounds = pm.hdi(diff_samples)
    
    return {
        'probability': float(np.mean(diff_samples > 0)),
        'diff_mean': float(np.mean(diff_samples)),
        'diff_hdi': hdi_bounds
    }

def robust_regression(data, response='BloodPressure'):
    """Robust regression using Student's t distribution."""
    with pm.Model() as model:
        glucose_std = (data['Glucose'] - data['Glucose'].mean()) / data['Glucose'].std()
        
        alpha = pm.Normal('alpha', mu=data[response].mean(), sigma=20)
        beta = pm.Normal('beta', mu=0, sigma=2)
        sigma = pm.HalfNormal('sigma', sigma=20)
        nu = pm.Exponential('nu', 1/30)
        
        mu = alpha + beta * glucose_std
        y = pm.StudentT('y', nu=nu, mu=mu, sigma=sigma, observed=data[response])
        
        idata = pm.sample(4000, tune=2000, return_inferencedata=True)
    
    beta_samples = idata.posterior['beta'].values.flatten() / data['Glucose'].std()
    hdi_bounds = pm.hdi(beta_samples)
    
    return {
        'beta_mean': float(np.mean(beta_samples)),
        'beta_hdi': hdi_bounds
    }

def heteroscedastic_analysis(data):
    """Analysis allowing for different variances between groups."""
    bmi_median = data['BMI'].median()
    high_bmi = data['BMI'] > bmi_median
    
    with pm.Model() as model:
        
        mu_1 = pm.Normal('mu_1', mu=data['Glucose'].mean(), sigma=20)
        mu_2 = pm.Normal('mu_2', mu=data['Glucose'].mean(), sigma=20)
        sigma_1 = pm.HalfNormal('sigma_1', sigma=20)
        sigma_2 = pm.HalfNormal('sigma_2', sigma=20)
    
        y1 = pm.Normal('y1', mu=mu_1, sigma=sigma_1, observed=data[high_bmi]['Glucose'])
        y2 = pm.Normal('y2', mu=mu_2, sigma=sigma_2, observed=data[~high_bmi]['Glucose'])
        
        ratio = pm.Deterministic('ratio', sigma_1 / sigma_2)
        
        idata = pm.sample(4000, tune=2000, return_inferencedata=True)
    
    ratio_samples = idata.posterior['ratio'].values.flatten()
    hdi_bounds = pm.hdi(ratio_samples)
    
    return {
        'ratio_mean': float(np.mean(ratio_samples)),
        'ratio_hdi': hdi_bounds,
        'prob_greater': float(np.mean(ratio_samples > 1))
    }

def robust_multiple_regression(data):
    """Multiple regression with robust likelihood."""
    with pm.Model() as model:
        
        glucose_std = (data['Glucose'] - data['Glucose'].mean()) / data['Glucose'].std()
        bmi_std = (data['BMI'] - data['BMI'].mean()) / data['BMI'].std()
        
       
        alpha = pm.Normal('alpha', mu=data['BloodPressure'].mean(), sigma=20)
        beta_glucose = pm.Normal('beta_glucose', mu=0, sigma=2)
        beta_bmi = pm.Normal('beta_bmi', mu=0, sigma=2)
        sigma = pm.HalfNormal('sigma', sigma=20)
        nu = pm.Exponential('nu', 1/30)
        
        
        mu = alpha + beta_glucose * glucose_std + beta_bmi * bmi_std
        y = pm.StudentT('y', nu=nu, mu=mu, sigma=sigma, observed=data['BloodPressure'])
        
        
        idata = pm.sample(4000, tune=2000, return_inferencedata=True)
    
    glucose_samples = idata.posterior['beta_glucose'].values.flatten() / data['Glucose'].std()
    bmi_samples = idata.posterior['beta_bmi'].values.flatten() / data['BMI'].std()
    
    return {
        'glucose_effect': pm.hdi(glucose_samples),
        'bmi_effect': pm.hdi(bmi_samples)
    }

def robust_logistic_regression(data):
    """Logistic regression with regularizing priors."""
    with pm.Model() as model:
        
        glucose_std = (data['Glucose'] - data['Glucose'].mean()) / data['Glucose'].std()
        
        
        alpha = pm.Normal('alpha', mu=0, sigma=2)
        beta = pm.Normal('beta', mu=0, sigma=2)
        
        
        p = pm.math.invlogit(alpha + beta * glucose_std)
        y = pm.Bernoulli('y', p=p, observed=data['Outcome'])
        
        
        idata = pm.sample(4000, tune=2000, return_inferencedata=True)
    
    beta_samples = idata.posterior['beta'].values.flatten() / data['Glucose'].std()
    hdi_bounds = pm.hdi(beta_samples)
    
    return {
        'beta_mean': float(np.mean(beta_samples)),
        'beta_hdi': hdi_bounds
    }

def robust_poisson_regression(data):
    """Poisson regression for count data with robust priors."""
    with pm.Model() as model:
        # Standardize predictor
        glucose_std = (data['Glucose'] - data['Glucose'].mean()) / data['Glucose'].std()
        
        # Weakly informative priors
        alpha = pm.Normal('alpha', mu=0, sigma=2)
        beta = pm.Normal('beta', mu=0, sigma=2)
        
        # Model with link function
        mu = pm.math.exp(alpha + beta * glucose_std)
        y = pm.Poisson('y', mu=mu, observed=data['Pregnancies'])
        
        # Sample
        idata = pm.sample(4000, tune=2000, return_inferencedata=True)
    
    beta_samples = idata.posterior['beta'].values.flatten() / data['Glucose'].std()
    hdi_bounds = pm.hdi(beta_samples)
    
    return {
        'beta_mean': float(np.mean(beta_samples)),
        'beta_hdi': hdi_bounds
    }

def robust_confounding_adjustment(data):
    """Multiple regression adjusting for confounding with robust priors."""
    # First run unadjusted model
    with pm.Model() as model_unadj:
        # Standardize predictor
        glucose_std = (data['Glucose'] - data['Glucose'].mean()) / data['Glucose'].std()
        
        # Priors
        alpha = pm.Normal('alpha', mu=data['Insulin'].mean(), sigma=20)
        beta_glucose = pm.Normal('beta_glucose', mu=0, sigma=2)
        sigma = pm.HalfNormal('sigma', sigma=20)
        nu = pm.Exponential('nu', 1/30)
        
        # Model
        mu = alpha + beta_glucose * glucose_std
        y = pm.StudentT('y', nu=nu, mu=mu, sigma=sigma, observed=data['Insulin'])
        
        # Sample
        idata_unadj = pm.sample(4000, tune=2000, return_inferencedata=True)
    
    # Then run adjusted model
    with pm.Model() as model_adj:
        # Standardize predictors
        glucose_std = (data['Glucose'] - data['Glucose'].mean()) / data['Glucose'].std()
        bp_std = (data['BloodPressure'] - data['BloodPressure'].mean()) / data['BloodPressure'].std()
        
        # Priors
        alpha = pm.Normal('alpha', mu=data['Insulin'].mean(), sigma=20)
        beta_glucose = pm.Normal('beta_glucose', mu=0, sigma=2)
        beta_bp = pm.Normal('beta_bp', mu=0, sigma=2)
        sigma = pm.HalfNormal('sigma', sigma=20)
        nu = pm.Exponential('nu', 1/30)
        
        # Model
        mu = alpha + beta_glucose * glucose_std + beta_bp * bp_std
        y = pm.StudentT('y', nu=nu, mu=mu, sigma=sigma, observed=data['Insulin'])
        
        # Sample
        idata_adj = pm.sample(4000, tune=2000, return_inferencedata=True)
    
    # Calculate HDIs for both models
    unadj_samples = idata_unadj.posterior['beta_glucose'].values.flatten() / data['Glucose'].std()
    adj_samples = idata_adj.posterior['beta_glucose'].values.flatten() / data['Glucose'].std()
    
    return {
        'unadjusted_mean': float(np.mean(unadj_samples)),
        'unadjusted_hdi': pm.hdi(unadj_samples),
        'adjusted_mean': float(np.mean(adj_samples)),
        'adjusted_hdi': pm.hdi(adj_samples)
    }

def run_analysis(data_path):
    """Run complete analysis with diagnostics."""
    # Load and clean data
    data = pd.read_csv(data_path)
    cleaned_data = preprocess_data(data)
    
    # Run analyses
    results = {
        'two_sample': two_sample_test(cleaned_data),
        'linear_regression': robust_regression(cleaned_data),
        'variance_analysis': heteroscedastic_analysis(cleaned_data),
        'multiple_regression': robust_multiple_regression(cleaned_data),
        'logistic_regression': robust_logistic_regression(cleaned_data),
        'poisson_regression': robust_poisson_regression(cleaned_data),
        'confounding_adjustment': robust_confounding_adjustment(cleaned_data)
    }
    
    # Generate summary
    summary = {
        'Two-sample test': {
            'test': 'Posterior probability of difference in Glucose levels between Outcome groups',
            'result': f"Posterior probability = {results['two_sample']['probability']:.3f} " +
                     f"(mean difference = {results['two_sample']['diff_mean']:.1f} " +
                     f"[{float(results['two_sample']['diff_hdi'][0]):.1f}, {float(results['two_sample']['diff_hdi'][1]):.1f}])"
        },
        'Linear regression': {
            'test': 'Credible interval for effect of Glucose on Blood Pressure',
            'result': f"95% HDI: [{float(results['linear_regression']['beta_hdi'][0]):.3f}, " +
                     f"{float(results['linear_regression']['beta_hdi'][1]):.3f}]"
        },
        'Variance analysis': {
            'test': 'Posterior probability of higher variance in high-BMI group',
            'result': f"Posterior probability = {results['variance_analysis']['prob_greater']:.3f} " +
                     f"(variance ratio = {results['variance_analysis']['ratio_mean']:.2f})"
        },
        'Multiple regression': {
            'test': 'Credible intervals for effects of Glucose and BMI on Blood Pressure',
            'result': f"Glucose effect: 95% HDI: [{float(results['multiple_regression']['glucose_effect'][0]):.3f}, " +
                     f"{float(results['multiple_regression']['glucose_effect'][1]):.3f}], " +
                     f"BMI effect: 95% HDI: [{float(results['multiple_regression']['bmi_effect'][0]):.3f}, " +
                     f"{float(results['multiple_regression']['bmi_effect'][1]):.3f}]"
        },
        'Logistic regression': {
            'test': 'Credible interval for effect of Glucose on Outcome probability',
            'result': f"95% HDI: [{float(results['logistic_regression']['beta_hdi'][0]):.3f}, " +
                     f"{float(results['logistic_regression']['beta_hdi'][1]):.3f}]"
        },
        'Poisson regression': {
            'test': 'Credible interval for effect of Glucose on Pregnancy count',
            'result': f"95% HDI: [{float(results['poisson_regression']['beta_hdi'][0]):.3f}, " +
                     f"{float(results['poisson_regression']['beta_hdi'][1]):.3f}]"
        },
        'Confounding adjustment': {
            'test': 'Credible interval for Glucose effect on Insulin, adjusting for Blood Pressure',
            'result': f"Unadjusted: 95% HDI: [{float(results['confounding_adjustment']['unadjusted_hdi'][0]):.3f}, " +
                f"{float(results['confounding_adjustment']['unadjusted_hdi'][1]):.3f}], " +
                f"Adjusted: 95% HDI: [{float(results['confounding_adjustment']['adjusted_hdi'][0]):.3f}, " +
                f"{float(results['confounding_adjustment']['adjusted_hdi'][1]):.3f}]"
}
    }
    
    return results, summary

if __name__ == '__main__':
    results, summary = run_analysis('diabetes.csv')
    
    
    for analysis, details in summary.items():
        print(f"\n* Analysis: {analysis}")
        print(f"   * Test— {details['test']}")
        print(f"   * Result— {details['result']}")