import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import special
from scipy.stats import poisson

full_data = pd.read_csv("diabetes.csv")
data = full_data.Pregnancies.to_numpy()

y_count = len(data)
y_sum = data.sum()

print(f'n: {y_count}')
print(f'sum: {y_sum}')

def post(mu):
	if mu == 0:
		return 0
	
	lognum = -768 * mu + y_sum * math.log(mu)
	logdenom = (y_sum + 1) * -math.log(y_count) + special.loggamma(y_sum + 1)

	return math.exp(lognum - logdenom)

mu_space = np.linspace(0, 10, 100)
mu_results = np.zeros((100), np.float64)
for i, mu in enumerate(mu_space):
	mu_results[i] = post(mu)
mu_results /= mu_results.sum()

plt.figure()
plt.title('Normalized Posterior Distribution of $\mu$')
plt.plot(mu_space, mu_results, color="blue")
plt.xlabel('$\mu$')
plt.ylabel('Likelihood')
plt.savefig("preg_ca_pois_mu.jpg")

mu_hat = mu_space[np.argmax(mu_results)]

print(f'mu MLE: {mu_hat}')

pred_space = np.linspace(0, 20, 21)
pred_results = poisson.pmf(pred_space, mu_hat)
data_vals = np.zeros((21), np.int32)
for count in data:
    data_vals[count] += 1
	
plt.figure()
plt.title(f'Posterior Predictions using $\mu$={mu_hat:.3f}')
plt.plot(pred_space, pred_results, color="blue")
plt.bar(pred_space, data_vals / len(data), color="green")
plt.xlabel('Pregnancies')
plt.ylabel('Percentage/Likelihood')
plt.savefig("preg_ca_pois_pred.jpg")