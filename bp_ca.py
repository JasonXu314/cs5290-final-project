import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import special
from scipy.stats import norm

full_data = pd.read_csv("diabetes.csv")
data = full_data.BloodPressure.to_numpy()

y_count = len(data)
y_mean = data.mean()
y_var = data.var()
y_sqr_dev = ((data - y_mean) ** 2).sum()

print(f'n: {y_count}')
print(f'mean: {y_mean}')
print(f'square deviance: {y_sqr_dev}')

def post(sig_sqr):
	if sig_sqr == 0:
		return 0
	
	lognum = -(y_count / 2) * math.log(sig_sqr) - ((y_count * y_sqr_dev) / (2 * (y_count - 1) * sig_sqr))
	logdenom = special.loggamma((y_count / 2) - 1) - (y_count / 2 - 1) * math.log((y_count * y_sqr_dev) / (2 * (y_count - 1)))

	return math.exp(lognum - logdenom)

sig_sqr_space = np.linspace(200, 500, 1000)
sig_sqr_results = np.zeros((1000), np.float64)
for i, sig_sqr in enumerate(sig_sqr_space):
	sig_sqr_results[i] = post(sig_sqr)
sig_sqr_results /= sig_sqr_results.sum()

plt.figure()
plt.title('Normalized Posterior Distribution of $\sigma^2$')
plt.plot(sig_sqr_space, sig_sqr_results, color="blue")
plt.xlabel('$\sigma^2$')
plt.ylabel('Likelihood')
plt.savefig("bp_ca_sig.jpg")

sig_sqr_hat = sig_sqr_space[np.argmax(sig_sqr_results)]

print(f'variance MLE: {sig_sqr_hat}')

pred_space = np.linspace(20, 180, 1000)
pred_results = norm.pdf(pred_space, y_mean, math.sqrt(sig_sqr_hat))
data_space = np.linspace(20, 180, 161)
data_vals = np.zeros((161), np.int32)
for bp in data:
    data_vals[bp - 20] += 1

sf = (data_vals / len(data)).max() / pred_results.max()
pred_results *= sf

plt.figure()
plt.title(f'Posterior Predictions using $\sigma^2$={sig_sqr_hat:.3f}')
plt.plot(pred_space, pred_results, color="blue")
plt.bar(data_space, data_vals / len(data), color="green")
plt.xlabel('Blood Pressure')
plt.ylabel('Percentage/Likelihood')
plt.savefig("bp_ca_pred.jpg")

# Model verification

log_bf_m1 = np.log(np.exp(-((data - y_mean) ** 2) / (2 * sig_sqr_hat)) / (math.sqrt(2 * math.pi * sig_sqr_hat))).sum()
log_bf_m2 = y_count * math.log(1 / 160)

bf = math.exp(log_bf_m1 - log_bf_m2)
print(f'Bayes\' Factor: {bf}')