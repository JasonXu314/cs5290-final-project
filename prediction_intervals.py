import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm

# Read the data
data = pd.read_csv('diabetes.csv')

# Create BMI test points
bmi_test = np.linspace(data['BMI'].min(), data['BMI'].max(), 100)

# Fit Bayesian logistic regression
with pm.Model() as model:
    # Priors
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10)
    
    # Likelihood
    p = pm.math.sigmoid(alpha + beta * data['BMI'])
    y = pm.Bernoulli('y', p=p, observed=data['Outcome'])
    
    # Generate predictions for test points
    p_pred = pm.Deterministic('p_pred', pm.math.sigmoid(alpha + beta * bmi_test[:, None]))
    
    # Sample from posterior
    trace = pm.sample(2000, return_inferencedata=False)

# Calculate prediction intervals for probabilities
pred_probs = trace['p_pred']
mean_prob = pred_probs.mean(axis=1)
prob_pi = np.percentile(pred_probs, [2.5, 97.5], axis=1)

# Create visualization
plt.figure(figsize=(10, 6))
plt.scatter(data['BMI'], data['Outcome'], alpha=0.3, label='Observed Data')
plt.plot(bmi_test, mean_prob, 'r-', label='Predicted Probability')
plt.fill_between(bmi_test, prob_pi[0], prob_pi[1], color='r', alpha=0.2, 
                 label='95% Prediction Interval')
plt.xlabel('BMI')
plt.ylabel('Probability of Diabetes')
plt.title('Predicted Probability of Diabetes by BMI')
plt.legend()
plt.grid(True)
plt.show()

# Calculate predictions for specific BMI values
test_bmis = [20, 25, 30, 35, 40]
test_indices = [np.abs(bmi_test - bmi).argmin() for bmi in test_bmis]
predictions = {
    'BMI': test_bmis,
    'Pred_Prob': [mean_prob[i] for i in test_indices],
    'PI_Lower': [prob_pi[0][i] for i in test_indices],
    'PI_Upper': [prob_pi[1][i] for i in test_indices]
}

print("\nPrediction Intervals for probability of diabetes at different BMI values:")
for i, bmi in enumerate(test_bmis):
    print(f"\nBMI {bmi}:")
    print(f"Predicted Probability: {predictions['Pred_Prob'][i]:.3f}")
    print(f"95% Prediction Interval: [{predictions['PI_Lower'][i]:.3f}, {predictions['PI_Upper'][i]:.3f}]")