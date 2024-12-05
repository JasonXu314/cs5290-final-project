import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import minimize

# Load the Pima Indians Diabetes dataset
data = pd.read_csv('diabetes.csv')

# 1. Two-sample test using Laplace approximation
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('pima-indians-diabetes.csv')

# Define the two groups (for example, Outcome = 0 vs Outcome = 1)
group1 = data[data['Outcome'] == 0]['Glucose']
group2 = data[data['Outcome'] == 1]['Glucose']

# Calculate the sample means and variances
mean1, mean2 = np.mean(group1), np.mean(group2)
var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
n1, n2 = len(group1), len(group2)

# Calculate pooled variance (assuming equal variance for both groups)
pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
pooled_sd = np.sqrt(pooled_var * (1/n1 + 1/n2))

# Posterior mean and variance for the difference in means (Laplace Approximation)
posterior_mean_diff = mean1 - mean2
posterior_var_diff = pooled_sd**2

# Compute the probability that the difference is greater than 0
probability_diff_greater_than_zero = 1 - norm.cdf(0, loc=posterior_mean_diff, scale=np.sqrt(posterior_var_diff))

# Print the results
print(f"Posterior Mean of Difference: {posterior_mean_diff}")
print(f"Posterior Variance of Difference: {posterior_var_diff}")
print(f"Probability that Group 1 mean > Group 2 mean: {probability_diff_greater_than_zero}")

# Visualize the distributions and the test result
x = np.linspace(min(group1.min(), group2.min()), max(group1.max(), group2.max()), 500)

# Plotting the normal approximations for both groups
plt.figure(figsize=(8, 6))

# Group 1 (Outcome = 0)
plt.plot(x, norm.pdf(x, loc=mean1, scale=np.sqrt(var1/n1)), label="Group 1 (Outcome=0)", color='blue')

# Group 2 (Outcome = 1)
plt.plot(x, norm.pdf(x, loc=mean2, scale=np.sqrt(var2/n2)), label="Group 2 (Outcome=1)", color='red')

# Plot the difference in means as a normal distribution (Laplace approximation)
plt.plot(x, norm.pdf(x, loc=posterior_mean_diff, scale=np.sqrt(posterior_var_diff)), label="Difference in Means (Laplace)", color='green', linestyle='--')

plt.axvline(x=0, color='black', linestyle=':', label='Difference = 0')

# Adding labels and title
plt.title('Two-Sample Test Using Laplace Approximation')
plt.xlabel('Glucose Level')
plt.ylabel('Density')
plt.legend()
plt.show()


# 2. Linear Regression with Laplace approximation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Load the dataset (replace 'diabetes.csv' with your actual dataset path)
data = pd.read_csv(path)

# Select BMI as predictor and Insulin as response
X = data['BMI'].values
y = data['Insulin'].values

# Handle missing or zero values in Insulin (replace zeros with NaN and drop them)
data = data[data['Insulin'] != 0]  # Drop rows with zero insulin values
X = data['BMI'].values
y = data['Insulin'].values

# Add an intercept term to X
X_design = np.c_[np.ones(X.shape[0]), X]

# Laplace Approximation for Linear Regression
def laplace_linear_regression(X, y):
    # Fit linear regression coefficients using least squares
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    residuals = y - X @ beta_hat
    sigma_squared = np.var(residuals, ddof=X.shape[1])
    
    # Posterior mean and variance of beta coefficients
    beta_var = sigma_squared * np.linalg.inv(X.T @ X)
    return beta_hat, beta_var

# Fit model
beta_mean, beta_var = laplace_linear_regression(X_design, y)

# Predictions
X_pred = np.linspace(X.min(), X.max(), 100)
X_pred_design = np.c_[np.ones(X_pred.shape[0]), X_pred]
y_pred = X_pred_design @ beta_mean
y_std = np.sqrt(np.sum((X_pred_design @ beta_var) * X_pred_design, axis=1))

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.7, label='Data (BMI vs Insulin)')
plt.plot(X_pred, y_pred, color='red', label='Regression Line')
plt.fill_between(X_pred, y_pred - 2 * y_std, y_pred + 2 * y_std, color='red', alpha=0.2, label='95% Confidence Interval')
plt.xlabel('BMI')
plt.ylabel('Insulin')
plt.title('Linear Regression (BMI vs Insulin) using Laplace Approximation')
plt.legend()
plt.show()


# 3. Analysis of Variance (ANOVA)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Function to compute posterior mean and variance (Laplace Approximation)
def laplace_variance_analysis(data, response, factor):
    # Group the data by the factor (e.g., age or pregnancy categories)
    groups = data.groupby(factor)[response]
    
    posterior_means = {}
    posterior_vars = {}
    
    # For each group, calculate the posterior mean and variance
    for group, values in groups:
        n = len(values)
        mean = np.mean(values)
        var = np.var(values, ddof=1)
        
        # Posterior mean is the sample mean for Laplace approximation
        posterior_mean = mean
        
        # Posterior variance is the sample variance adjusted by sample size
        posterior_variance = var / n
        
        posterior_means[group] = posterior_mean
        posterior_vars[group] = posterior_variance
    
    return posterior_means, posterior_vars

# Group dataset into Age categories
data['Age_Category'] = pd.cut(
    data['Age'], bins=[20, 30, 40, 50, np.inf], labels=['20-30', '31-40', '41-50', '50+']
)

# Perform variance analysis on Blood Pressure levels by Age categories
posterior_means_bp, posterior_vars_bp = laplace_variance_analysis(data, response='BloodPressure', factor='Age_Category')

# Display results
print("Posterior Means (Blood Pressure by Age Category):")
print(posterior_means_bp)
print("\nPosterior Variances (Blood Pressure by Age Category):")
print(posterior_vars_bp)

# Visualization of posterior means with 95% credible intervals
categories_bp = list(posterior_means_bp.keys())
means_bp = list(posterior_means_bp.values())
errors_bp = [1.96 * np.sqrt(posterior_vars_bp[cat]) for cat in categories_bp]

plt.figure(figsize=(10, 6))
plt.bar(categories_bp, means_bp, yerr=errors_bp, color=['blue', 'green', 'orange', 'red'], alpha=0.7, capsize=5)
plt.xlabel('Age Category')
plt.ylabel('Posterior Mean (Blood Pressure)')
plt.title('Variance Analysis (Blood Pressure Levels by Age Category) using Laplace Approximation')
plt.show()


# 4. Multiple Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Load the dataset (adjust path as necessary)
data = pd.read_csv(path)

# Define predictors and target variable for multiple regression
X = data[['BMI', 'Age', 'BloodPressure']]  # Predictors
y = data['Insulin']  # Target variable

# Add a constant (bias term) for intercept in the regression model
X = np.c_[np.ones(X.shape[0]), X]  # Add a column of ones for the intercept term

# Function to perform multiple regression with Laplace approximation
def multiple_regression_laplace(X, y):
    # Calculate the posterior mean using Maximum Likelihood Estimation (MLE)
    # For Linear Regression, the posterior mean is the Ordinary Least Squares (OLS) estimate
    X_transpose = X.T
    beta = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)

    # Calculate the residuals (errors)
    residuals = y - X.dot(beta)
    
    # Calculate the variance of the residuals (estimate of sigma^2)
    sigma_squared = np.var(residuals, ddof=X.shape[1])  # Degrees of freedom = number of predictors
    var_beta = sigma_squared * np.linalg.inv(X_transpose.dot(X))  # Covariance matrix of the regression coefficients

    # Posterior mean of the coefficients (Laplace approximation is similar to MLE for linear regression)
    posterior_mean = beta
    posterior_var = var_beta

    return posterior_mean, posterior_var, residuals, sigma_squared

# Fit the model using Laplace approximation
posterior_mean, posterior_var, residuals, sigma_squared = multiple_regression_laplace(X, y)

# Print the posterior means (coefficients) and variances
print("Posterior Means (Coefficients):")
print(posterior_mean)
print("\nPosterior Variances (Coefficients):")
print(np.diag(posterior_var))  # Diagonal of the covariance matrix gives variances

# Visualize the regression results
# We will plot the observed data vs predicted values for one of the predictors, say BMI
y_pred = X.dot(posterior_mean)

plt.figure(figsize=(10, 6))
plt.scatter(data['BMI'], y, color='blue', alpha=0.6, label='Observed Data')
plt.plot(data['BMI'], y_pred, color='red', label='Regression Line (Predicted)')
plt.fill_between(data['BMI'], y_pred - 1.96*np.sqrt(sigma_squared), y_pred + 1.96*np.sqrt(sigma_squared),
                 color='red', alpha=0.2, label='95% Confidence Interval')
plt.title("Multiple Regression (BMI, Age, BloodPressure vs Insulin) using Laplace Approximation")
plt.xlabel('BMI')
plt.ylabel('Insulin')
plt.legend()
plt.show()



# 6. Logistic Regression
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('pima-indians-diabetes.csv')

# Let's compare 'Age' and 'BMI' with 'Outcome'
X = data[['Age', 'BMI']].values
y = data['Outcome'].values

# Add intercept to X
X = np.c_[np.ones(X.shape[0]), X]

# Logistic regression log-likelihood function
def log_likelihood(beta, X, y):
    # Sigmoid function
    p = expit(X.dot(beta))
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

# Logistic regression gradient (first derivative)
def log_likelihood_grad(beta, X, y):
    p = expit(X.dot(beta))
    return X.T.dot(p - y)

# Logistic regression Hessian (second derivative)
def log_likelihood_hess(beta, X, y):
    p = expit(X.dot(beta))
    diag_p = p * (1 - p)  # Elementwise multiplication
    return X.T.dot(np.diag(diag_p)).dot(X)

# Perform optimization to find the MAP estimate (maximum likelihood)
initial_beta = np.zeros(X.shape[1])  # Initial guess for beta
result = minimize(log_likelihood, initial_beta, args=(X, y), jac=log_likelihood_grad, hess=log_likelihood_hess)

# MAP estimate of the coefficients
posterior_mean = result.x

# Calculate the Hessian at the MAP estimate
hessian = log_likelihood_hess(posterior_mean, X, y)

# The posterior covariance is the inverse of the negative Hessian (Laplace approximation)
posterior_covariance = np.linalg.inv(hessian)

# Print results
print("Posterior Mean (Coefficients):", posterior_mean)
print("Posterior Covariance Matrix:", posterior_covariance)

# Visualizing the results

# Plotting the logistic regression surface
# We will generate a grid of 'Age' and 'BMI' values to visualize the model
age_range = np.linspace(data['Age'].min(), data['Age'].max(), 100)
bmi_range = np.linspace(data['BMI'].min(), data['BMI'].max(), 100)
age_grid, bmi_grid = np.meshgrid(age_range, bmi_range)

# Prepare the grid for predictions
X_grid = np.c_[np.ones(age_grid.size), age_grid.ravel(), bmi_grid.ravel()]
predicted_probabilities_grid = expit(X_grid.dot(posterior_mean))  # Apply sigmoid to get probabilities

# Reshape the probabilities for the surface plot
predicted_probabilities_grid = predicted_probabilities_grid.reshape(age_grid.shape)

# Plot the 3D surface of predicted probabilities
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Surface plot of the predicted probabilities
ax.plot_surface(age_grid, bmi_grid, predicted_probabilities_grid, cmap='viridis', edgecolor='none')

ax.set_xlabel('Age')
ax.set_ylabel('BMI')
ax.set_zlabel('Predicted Probability of Diabetes')
ax.set_title('Bayesian Logistic Regression with Laplace Approximation (Age vs. BMI)')

plt.show()
