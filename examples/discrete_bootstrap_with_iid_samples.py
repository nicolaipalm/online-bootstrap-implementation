"""Example script for applying the discrete bootstrap to independent and identically distributed samples


"""

import numpy as np
from matplotlib import pyplot as plt

from bstrapping.bootstrap_procedures.discrete_bootstrap import DiscreteBootstrap

# specify variance, mean and number of the samples
variance = 10
mean = 4
number_sample_points = 3000

# generate samples from a normal distribution
samples = np.random.multivariate_normal(
    mean=mean * np.ones(number_sample_points),
    cov=variance * np.identity(number_sample_points))

# Perform the discrete bootstrap
bootstrap = DiscreteBootstrap(samples=samples)

# Print bootstrapped variance of the empirical mean along with the true variance
print(f'Bootstrapped variance: \n {bootstrap.bootstrapped_variance}')

print(f'True variance of empirical mean: {variance / number_sample_points}')

print(f'Confidence interval: {bootstrap.bootstrapped_confidence_interval(alpha=0.05)}')

plt.title('iid process')
x = range(number_sample_points)
plt.plot(x, samples)
plt.show()
# bootstrap.save_to_csv('discrete_bootstrap_iid_samples')
